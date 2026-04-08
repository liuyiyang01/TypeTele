"""
Task Planner Module for TypeTele

Implements VLM-based multi-step task planning as described in the paper:
- Decomposes complex tasks into sequential steps
- Determines manipulation type for each step
- Supports vision input for context-aware planning
"""

import os
import json
import base64
import io
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from threading import Thread, Lock
import time
import numpy as np
from openai import OpenAI


@dataclass
class TaskStep:
    """Represents a single step in a task plan."""
    step_number: int
    description: str
    manipulation_type: str
    hand: str = "right"  # "right", "left", or "both"
    objects_involved: List[str] = None
    interaction_type: str = ""  # e.g., "grasp", "place", "pour", "press"
    duration_estimate: float = 0.0  # Estimated duration in seconds
    
    def __post_init__(self):
        if self.objects_involved is None:
            self.objects_involved = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TaskPlan:
    """Represents a complete task plan with multiple steps."""
    task_description: str
    steps: List[TaskStep]
    total_steps: int
    estimated_duration: float
    
    def to_dict(self) -> Dict:
        return {
            "task_description": self.task_description,
            "total_steps": self.total_steps,
            "estimated_duration": self.estimated_duration,
            "steps": [step.to_dict() for step in self.steps]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskPlan':
        steps = [TaskStep(**step_data) for step_data in data.get("steps", [])]
        return cls(
            task_description=data.get("task_description", ""),
            steps=steps,
            total_steps=data.get("total_steps", len(steps)),
            estimated_duration=data.get("estimated_duration", 0.0)
        )


class TaskPlanner:
    """
    VLM-based Task Planner for TypeTele.
    
    Implements the multi-step planning described in Section 3.2 of the paper:
    "We then prompt the MLLM to sequentially reason through two sub-questions:
    (1) How many steps are required to complete the task?
    (2) Which type of manipulation should be assigned to each hand at each step?"
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o",
        category: str = "leap",
        enable_vision: bool = True
    ):
        """
        Initialize the Task Planner.
        
        Args:
            api_key: API key for the VLM service
            base_url: Base URL for the VLM API
            model: Model name to use (default: gpt-4o as mentioned in paper)
            category: Hand category for type library
            enable_vision: Whether to enable vision input
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.category = category
        self.enable_vision = enable_vision
        
        self.type_library = []
        self.type_files = []
        
        # Threading for async planning
        self._planning_thread = None
        self._planning_lock = Lock()
        self._current_plan: Optional[TaskPlan] = None
        self._planning_in_progress = False
        self._new_plan_available = False
        
    def load_type_library(self, type_library_path: Optional[str] = None):
        """
        Load the dexterous manipulation type library.
        
        Args:
            type_library_path: Path to type library. If None, uses default path.
        """
        if type_library_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            type_library_path = os.path.join(current_dir, "..", "TypeLibrary", self.category)
        
        json_path = os.path.join(type_library_path, "_type_info.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.type_library = data
                    self.type_files = [t.get('id') for t in data if 'id' in t]
                    print(f"[TaskPlanner] Loaded {len(self.type_files)} types from library")
                    return
            except Exception as e:
                print(f"[TaskPlanner] Failed to read _type_info.json: {e}")
        
        # Fallback to txt files
        if os.path.isdir(type_library_path):
            self.type_files = [f[:-4] for f in os.listdir(type_library_path) if f.endswith('.txt')]
            self.type_library = [{"id": name} for name in self.type_files]
            print(f"[TaskPlanner] Loaded {len(self.type_files)} types from txt files")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 string for VLM API.
        
        Args:
            image: BGR or RGB image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        import cv2
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    
    def _build_type_catalog(self) -> str:
        """Build a catalog of available manipulation types."""
        catalog_lines = []
        for t in self.type_library:
            type_id = t.get('id', '')
            name = t.get('name_en', '')
            pose = t.get('pose', '')
            usage = t.get('usage', '')
            intents = ', '.join(t.get('intents', [])[:3])
            
            catalog_lines.append(
                f"- {type_id}: {name}\n"
                f"  Pose: {pose}\n"
                f"  Usage: {usage}\n"
                f"  Intents: {intents}"
            )
        return "\n".join(catalog_lines)
    
    def plan_task(
        self,
        task_description: str,
        image: Optional[np.ndarray] = None,
        timeout: float = 60.0
    ) -> Optional[TaskPlan]:
        """
        Plan a multi-step task using VLM.
        
        This implements the two-step reasoning from the paper:
        1. Decompose task into steps
        2. Assign manipulation type to each step
        
        Args:
            task_description: Natural language description of the task
            image: Optional camera image for visual context
            timeout: Timeout for API call
            
        Returns:
            TaskPlan object or None if planning failed
        """
        if not self.type_library:
            print("[TaskPlanner] Type library not loaded. Call load_type_library() first.")
            return None
        
        type_catalog = self._build_type_catalog()
        
        # Build the prompt following the paper's approach
        system_prompt = """You are a robotic task planning expert for dexterous manipulation.
Your task is to decompose complex manipulation tasks into sequential steps and assign appropriate manipulation types from a predefined library.

You must reason through two sub-questions:
1. How many steps are required to complete the task?
2. Which type of manipulation should be assigned to each hand at each step?

For each step, identify:
- The specific action description
- Objects involved
- The manipulation type from the provided catalog
- The hand to use (right/left/both)
- Type of interaction (grasp, place, pour, press, etc.)

Output your response as a JSON object with the following structure:
{
    "task_description": "original task",
    "total_steps": N,
    "estimated_duration": seconds,
    "steps": [
        {
            "step_number": 1,
            "description": "what to do in this step",
            "manipulation_type": "type_id_from_catalog",
            "hand": "right/left/both",
            "objects_involved": ["object1", "object2"],
            "interaction_type": "grasp/place/pour/etc",
            "duration_estimate": seconds
        }
    ]
}"""

        user_prompt = f"""Task: {task_description}

Available Manipulation Types:
{type_catalog}

Analyze the task and provide a step-by-step plan using the manipulation types above.
Only use manipulation types from the provided catalog.
If the task is simple and requires only one step, that's acceptable.
"""

        try:
            print(f"[TaskPlanner] Sending request to {self.model}...")
            start_time = time.time()
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Add image if provided and vision is enabled
            if self.enable_vision and image is not None:
                base64_image = self._encode_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_prompt})
            
            # Call VLM API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=timeout,
                stream=False
            )
            
            elapsed = time.time() - start_time
            print(f"[TaskPlanner] API response received in {elapsed:.2f}s")
            
            # Parse response
            content = response.choices[0].message.content
            plan_data = json.loads(content)
            
            # Validate and create TaskPlan
            steps_data = plan_data.get("steps", [])
            steps = []
            for step_data in steps_data:
                step = TaskStep(
                    step_number=step_data.get("step_number", len(steps) + 1),
                    description=step_data.get("description", ""),
                    manipulation_type=step_data.get("manipulation_type", ""),
                    hand=step_data.get("hand", "right"),
                    objects_involved=step_data.get("objects_involved", []),
                    interaction_type=step_data.get("interaction_type", ""),
                    duration_estimate=step_data.get("duration_estimate", 0.0)
                )
                steps.append(step)
            
            plan = TaskPlan(
                task_description=plan_data.get("task_description", task_description),
                steps=steps,
                total_steps=len(steps),
                estimated_duration=plan_data.get("estimated_duration", 0.0)
            )
            
            print(f"[TaskPlanner] Generated plan with {plan.total_steps} steps")
            for step in plan.steps:
                print(f"  Step {step.step_number}: {step.description} -> {step.manipulation_type}")
            
            return plan
            
        except Exception as e:
            print(f"[TaskPlanner] Error during task planning: {e}")
            return None
    
    def plan_task_async(
        self,
        task_description: str,
        image: Optional[np.ndarray] = None
    ):
        """
        Start async task planning in background thread.
        
        Args:
            task_description: Natural language description of the task
            image: Optional camera image for visual context
        """
        if self._planning_in_progress:
            print("[TaskPlanner] Planning already in progress, ignoring new request")
            return
        
        self._planning_in_progress = True
        self._planning_thread = Thread(
            target=self._planning_worker,
            args=(task_description, image)
        )
        self._planning_thread.start()
    
    def _planning_worker(self, task_description: str, image: Optional[np.ndarray]):
        """Worker thread for async planning."""
        try:
            plan = self.plan_task(task_description, image)
            with self._planning_lock:
                self._current_plan = plan
                self._new_plan_available = True
        except Exception as e:
            print(f"[TaskPlanner] Async planning error: {e}")
        finally:
            self._planning_in_progress = False
    
    def has_new_plan(self) -> bool:
        """Check if a new plan is available from async planning."""
        with self._planning_lock:
            return self._new_plan_available
    
    def get_plan(self) -> Optional[TaskPlan]:
        """
        Get the latest plan from async planning.
        
        Returns:
            TaskPlan or None if no new plan available
        """
        with self._planning_lock:
            if self._new_plan_available:
                self._new_plan_available = False
                return self._current_plan
            return None
    
    def validate_plan(self, plan: TaskPlan) -> Tuple[bool, List[str]]:
        """
        Validate a task plan against the type library.
        
        Args:
            plan: TaskPlan to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        valid_type_ids = set(self.type_files)
        
        for step in plan.steps:
            if step.manipulation_type not in valid_type_ids:
                errors.append(
                    f"Step {step.step_number}: Invalid manipulation type '{step.manipulation_type}'"
                )
        
        return len(errors) == 0, errors
    
    def get_current_step(self, plan: TaskPlan, step_number: int) -> Optional[TaskStep]:
        """
        Get a specific step from a plan.
        
        Args:
            plan: TaskPlan
            step_number: Step number (1-indexed)
            
        Returns:
            TaskStep or None if not found
        """
        for step in plan.steps:
            if step.step_number == step_number:
                return step
        return None
