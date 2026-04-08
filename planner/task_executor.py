"""
Task Executor Module for TypeTele

Handles sequential execution of multi-step task plans,
including step transitions and completion detection.
"""

import time
from typing import Optional, Callable, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass

from .task_planner import TaskPlan, TaskStep


class ExecutionState(Enum):
    """States for task execution state machine."""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING_FOR_COMPLETION = auto()
    STEP_COMPLETE = auto()
    TASK_COMPLETE = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class ExecutionContext:
    """Context for task execution."""
    current_plan: Optional[TaskPlan] = None
    current_step_number: int = 0
    total_steps: int = 0
    start_time: float = 0.0
    step_start_time: float = 0.0
    state: ExecutionState = ExecutionState.IDLE
    error_message: str = ""
    
    # Step completion detection
    step_progress: float = 0.0  # 0.0 to 1.0
    step_timeout: float = 10.0  # seconds
    
    def reset(self):
        """Reset execution context."""
        self.current_plan = None
        self.current_step_number = 0
        self.total_steps = 0
        self.start_time = 0.0
        self.step_start_time = 0.0
        self.state = ExecutionState.IDLE
        self.error_message = ""
        self.step_progress = 0.0


class TaskExecutor:
    """
    Executes multi-step task plans sequentially.
    
    Coordinates with the teleoperation system to:
    1. Switch manipulation types for each step
    2. Monitor step completion
    3. Handle step transitions
    4. Provide execution feedback
    """
    
    def __init__(
        self,
        type_change_callback: Callable[[str], None],
        completion_check_callback: Optional[Callable[[], bool]] = None,
        step_complete_callback: Optional[Callable[[int, TaskStep], None]] = None,
        task_complete_callback: Optional[Callable[[TaskPlan], None]] = None,
        default_step_timeout: float = 15.0
    ):
        """
        Initialize the Task Executor.
        
        Args:
            type_change_callback: Function to call when type needs to change
            completion_check_callback: Function to check if current step is complete
            step_complete_callback: Function called when a step completes
            task_complete_callback: Function called when task completes
            default_step_timeout: Default timeout for step completion
        """
        self.type_change_callback = type_change_callback
        self.completion_check_callback = completion_check_callback
        self.step_complete_callback = step_complete_callback
        self.task_complete_callback = task_complete_callback
        self.default_step_timeout = default_step_timeout
        
        self.context = ExecutionContext()
        
        # Step completion detection parameters
        self.completion_threshold = 0.95
        self.min_step_duration = 2.0  # Minimum time before checking completion
        
    def start_task(self, plan: TaskPlan) -> bool:
        """
        Start executing a task plan.
        
        Args:
            plan: TaskPlan to execute
            
        Returns:
            True if task started successfully
        """
        if self.context.state not in [ExecutionState.IDLE, ExecutionState.TASK_COMPLETE, ExecutionState.ERROR]:
            print("[TaskExecutor] Cannot start new task, already executing")
            return False
        
        if not plan.steps:
            print("[TaskExecutor] Cannot execute empty plan")
            return False
        
        self.context.reset()
        self.context.current_plan = plan
        self.context.total_steps = plan.total_steps
        self.context.start_time = time.time()
        self.context.state = ExecutionState.EXECUTING
        
        print(f"[TaskExecutor] Starting task: {plan.task_description}")
        print(f"[TaskExecutor] Total steps: {plan.total_steps}")
        
        # Start first step
        self._start_step(1)
        return True
    
    def _start_step(self, step_number: int):
        """Start a specific step."""
        step = self.context.current_plan.steps[step_number - 1]
        self.context.current_step_number = step_number
        self.context.step_start_time = time.time()
        self.context.step_progress = 0.0
        
        # Set step timeout
        self.context.step_timeout = max(
            step.duration_estimate * 2 if step.duration_estimate > 0 else self.default_step_timeout,
            self.min_step_duration
        )
        
        print(f"[TaskExecutor] Starting step {step_number}/{self.context.total_steps}: {step.description}")
        print(f"[TaskExecutor] Using manipulation type: {step.manipulation_type}")
        
        # Trigger type change
        if self.type_change_callback:
            self.type_change_callback(step.manipulation_type)
        
        self.context.state = ExecutionState.WAITING_FOR_COMPLETION
    
    def update(self) -> ExecutionState:
        """
        Update execution state - should be called in main loop.
        
        Returns:
            Current execution state
        """
        if self.context.state == ExecutionState.IDLE:
            return self.context.state
        
        if self.context.state == ExecutionState.WAITING_FOR_COMPLETION:
            self._check_step_completion()
        
        elif self.context.state == ExecutionState.STEP_COMPLETE:
            self._handle_step_complete()
        
        return self.context.state
    
    def _check_step_completion(self):
        """Check if current step is complete."""
        elapsed = time.time() - self.context.step_start_time
        step = self.context.current_plan.steps[self.context.current_step_number - 1]
        
        # Check for timeout
        if elapsed > self.context.step_timeout:
            print(f"[TaskExecutor] Step {self.context.current_step_number} timeout, proceeding")
            self.context.state = ExecutionState.STEP_COMPLETE
            return
        
        # Wait for minimum duration
        if elapsed < self.min_step_duration:
            return
        
        # Check external completion signal if provided
        if self.completion_check_callback:
            if self.completion_check_callback():
                print(f"[TaskExecutor] Step {self.context.current_step_number} completed (external signal)")
                self.context.state = ExecutionState.STEP_COMPLETE
                return
        
        # Estimate progress based on time
        expected_duration = max(step.duration_estimate, self.min_step_duration)
        self.context.step_progress = min(elapsed / expected_duration, 1.0)
        
        # Auto-complete if progress reaches threshold
        if self.context.step_progress >= self.completion_threshold:
            print(f"[TaskExecutor] Step {self.context.current_step_number} completed (time-based)")
            self.context.state = ExecutionState.STEP_COMPLETE
    
    def _handle_step_complete(self):
        """Handle step completion and transition to next step."""
        step = self.context.current_plan.steps[self.context.current_step_number - 1]
        
        # Notify callback
        if self.step_complete_callback:
            self.step_complete_callback(self.context.current_step_number, step)
        
        # Check if all steps complete
        if self.context.current_step_number >= self.context.total_steps:
            self._complete_task()
        else:
            # Start next step
            self._start_step(self.context.current_step_number + 1)
    
    def _complete_task(self):
        """Handle task completion."""
        total_time = time.time() - self.context.start_time
        print(f"[TaskExecutor] Task completed in {total_time:.2f}s")
        
        if self.task_complete_callback:
            self.task_complete_callback(self.context.current_plan)
        
        self.context.state = ExecutionState.TASK_COMPLETE
    
    def manual_step_complete(self) -> bool:
        """
        Manually mark current step as complete.
        
        Returns:
            True if step was marked complete
        """
        if self.context.state == ExecutionState.WAITING_FOR_COMPLETION:
            print(f"[TaskExecutor] Manual completion of step {self.context.current_step_number}")
            self.context.state = ExecutionState.STEP_COMPLETE
            return True
        return False
    
    def skip_to_step(self, step_number: int) -> bool:
        """
        Skip to a specific step.
        
        Args:
            step_number: Step number to skip to (1-indexed)
            
        Returns:
            True if successful
        """
        if self.context.state not in [ExecutionState.EXECUTING, ExecutionState.WAITING_FOR_COMPLETION]:
            return False
        
        if step_number < 1 or step_number > self.context.total_steps:
            return False
        
        print(f"[TaskExecutor] Skipping to step {step_number}")
        self._start_step(step_number)
        return True
    
    def pause(self):
        """Pause execution."""
        if self.context.state == ExecutionState.WAITING_FOR_COMPLETION:
            self.context.state = ExecutionState.PAUSED
            print("[TaskExecutor] Execution paused")
    
    def resume(self):
        """Resume execution."""
        if self.context.state == ExecutionState.PAUSED:
            self.context.state = ExecutionState.WAITING_FOR_COMPLETION
            print("[TaskExecutor] Execution resumed")
    
    def stop(self):
        """Stop execution."""
        print("[TaskExecutor] Stopping execution")
        self.context.reset()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current execution status.
        
        Returns:
            Dictionary with execution status
        """
        return {
            "state": self.context.state.name,
            "current_step": self.context.current_step_number,
            "total_steps": self.context.total_steps,
            "progress": self._calculate_overall_progress(),
            "step_progress": self.context.step_progress,
            "elapsed_time": time.time() - self.context.start_time if self.context.start_time > 0 else 0,
            "error": self.context.error_message
        }
    
    def _calculate_overall_progress(self) -> float:
        """Calculate overall task progress."""
        if self.context.total_steps == 0:
            return 0.0
        
        completed_steps = self.context.current_step_number - 1
        current_step_contribution = self.context.step_progress / self.context.total_steps
        
        return (completed_steps / self.context.total_steps) + current_step_contribution
    
    def is_executing(self) -> bool:
        """Check if a task is currently being executed."""
        return self.context.state in [
            ExecutionState.EXECUTING,
            ExecutionState.WAITING_FOR_COMPLETION,
            ExecutionState.STEP_COMPLETE,
            ExecutionState.PAUSED
        ]
    
    def get_current_step_info(self) -> Optional[TaskStep]:
        """Get information about the current step."""
        if not self.context.current_plan or self.context.current_step_number < 1:
            return None
        
        try:
            return self.context.current_plan.steps[self.context.current_step_number - 1]
        except IndexError:
            return None
