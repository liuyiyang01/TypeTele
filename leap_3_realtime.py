# from asr.tencent_asr import AsrServer
from hand_detect.detectFinger import FingerDetector
from retrieve.retrieve import Retrieve
from leap_hand_utils.leap_node import LeapNode
from planner.task_planner import TaskPlanner, TaskPlan
from planner.task_executor import TaskExecutor, ExecutionState

import os
import time
import numpy as np
import cv2
import threading

from asr.typing_asr import KeyboardAsrServer

# Joint reordering indices for LEAP Hand consistency
_REORDER_INDEX = np.array([9, 8, 10, 11, 5, 4, 6, 7, 1, 0, 2, 3, 12, 13, 14, 15])
_INVERSE_INDEX = np.argsort(_REORDER_INDEX)

class RealTimeRunner:
    """
    Main loop integrating ASR, Hand Detection, Retrieval, VLM Planning, and LEAP Hand control.
    Components run in separate threads/processes managed via start()/stop().
    
    Enhanced with VLM-based multi-step task planning as described in the TypeTele paper.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Load initial grasp primitive (absolute positions)
        self.category = cfg["type"]["category"]
        self.curr_type = cfg["type"]["type_name"]
        self.open_pos, self.close_pos = self.load_type(self.curr_type)

        # Initialize ASR
        self.asr_type = cfg["asr"].get("type", "typing")
        if self.asr_type == "tencent":
            print("[Info] Tencent ASR enabled.")
            self.asr = AsrServer(cfg["asr"])
        else:
            print("[Info] ASR disabled. Using keyboard input.")
            self.asr = KeyboardAsrServer(cfg["asr"])

        # Initialize other components
        self.finger_detector = FingerDetector(cfg["detector"])
        r_cfg = cfg["retriever"]
        self.retriever = Retrieve(
            api_key=r_cfg["api_key"],
            base_url=r_cfg["base_url"],
            category=self.category,
            model=r_cfg.get("model"),
            enable_vision=r_cfg.get("enable_vision", True)
        )
        self.leap_node = LeapNode(self.cfg["leap_cfg"])
        
        # Initialize VLM Task Planner (Section 3.2 of paper)
        planner_cfg = cfg.get("planner", {})
        self.task_planner = TaskPlanner(
            api_key=planner_cfg.get("api_key", r_cfg["api_key"]),
            base_url=planner_cfg.get("base_url", r_cfg["base_url"]),
            model=planner_cfg.get("model", "gpt-4o"),  # Paper uses GPT-4o
            category=self.category,
            enable_vision=planner_cfg.get("enable_vision", True)
        )
        
        # Initialize Task Executor for multi-step plan execution
        self.task_executor = TaskExecutor(
            type_change_callback=self._on_executor_type_change,
            step_complete_callback=self._on_step_complete,
            task_complete_callback=self._on_task_complete,
            default_step_timeout=cfg.get("step_timeout", 15.0)
        )
        
        # Current camera frame for vision input
        self.current_frame = None
        self._frame_lock = threading.Lock()
        
        # Planning state
        self._planning_in_progress = False
        self._current_plan: Optional[TaskPlan] = None
        
        # Keyboard grasp control (alternative to camera)
        self._use_keyboard_grasp = cfg.get("use_keyboard_grasp", False)
        self._grasp_fraction = 0.0  # 0.0 = open, 1.0 = close
        self._grasp_step = 0.05  # 5% per step

    def _apply_keyboard_grasp(self):
        """Apply keyboard-controlled grasp fraction to LEAP hand."""
        # Clamp fraction
        self._grasp_fraction = max(0.0, min(1.0, self._grasp_fraction))
        
        # Interpolate between open and close positions
        type_pos = self.open_pos * (1 - self._grasp_fraction) + self.close_pos * self._grasp_fraction
        self.leap_node.set_leap(type_pos)
        
        # Print status bar
        bar_length = 20
        filled = int(self._grasp_fraction * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\rGrasp: [{bar}] {self._grasp_fraction*100:.0f}%", end="", flush=True)

    def start(self):
        """Start all components and enter main loop."""
        self.asr.start()
        
        # Start finger detector only if not using keyboard grasp
        if not self._use_keyboard_grasp:
            try:
                self.finger_detector.start()
                print("[Info] Camera hand tracking enabled")
            except Exception as e:
                print(f"[Warning] Failed to start camera: {e}")
                print("[Info] Falling back to keyboard grasp control")
                self._use_keyboard_grasp = True
        else:
            print("[Info] Keyboard grasp control enabled (camera disabled)")
            print("[Info] Controls: 'o'=open, 'c'=close, 'a'=decrease, 'd'=increase")
        
        self.retriever.load_type_library()
        self.retriever.start()
        self.task_planner.load_type_library()
        print("[Info] VLM Task Planner initialized with multi-step planning support")
        self.main_loop()

    def stop(self):
        self.asr.stop()
        if not self._use_keyboard_grasp:
            try:
                self.finger_detector.stop()
            except:
                pass
        self.retriever.stop()
        cv2.destroyAllWindows()

    def change_type(self, new_type: str):
        print(f"[Info] Switching grasp type: {self.curr_type} -> {new_type}")
        self.curr_type = new_type
        self.open_pos, self.close_pos = self.load_type(self.curr_type)
        # Reset grasp fraction when type changes
        self._grasp_fraction = 0.0

    def load_type(self, type_name: str):
        """Load and decode grasp primitive (open/close poses) from file."""
        def parse_line(line: str):
            line = line.strip().strip('[]')
            parts = [p for p in line.replace(',', ' ').split() if p]
            vals = [float(p) for p in parts]
            if len(vals) != 16:
                raise ValueError(f"Invalid line length for {type_name}: {len(vals)}")
            return np.array(vals, dtype=float)

        def _decode_saved(vec: np.ndarray) -> np.ndarray:
            """Convert saved format to hardware absolute position."""
            joints = np.zeros(16, dtype=float)
            joints[_INVERSE_INDEX] = vec
            return joints + 3.14159
        
        # Resolve file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        type_file = os.path.join(current_dir, "TypeLibrary", self.category, f"{type_name}.txt")
        
        if not os.path.exists(type_file):
            raise FileNotFoundError(f"Type file not found: {type_file}")
        
        with open(type_file, 'r', encoding='utf-8') as f:
            open_abs = _decode_saved(parse_line(f.readline()))
            close_abs = _decode_saved(parse_line(f.readline()))
            
        return open_abs, close_abs

    def _on_executor_type_change(self, type_name: str):
        """Callback when task executor needs to change type."""
        try:
            self.change_type(type_name)
        except Exception as e:
            print(f"[Error] Failed to change type in executor: {e}")
    
    def _on_step_complete(self, step_number: int, step):
        """Callback when a step completes."""
        print(f"[TaskExecutor] Step {step_number} completed: {step.description}")
    
    def _on_task_complete(self, plan):
        """Callback when entire task completes."""
        print(f"[TaskExecutor] Task completed: {plan.task_description}")
        self._current_plan = None
        self._planning_in_progress = False
    
    def _is_planning_query(self, query: str) -> bool:
        """
        Determine if a query requires multi-step planning.
        
        Planning queries typically contain:
        - Multiple actions (and, then, after)
        - Complex tasks (pick up X and do Y)
        - Sequential operations
        """
        planning_keywords = [
            'and then', 'then', 'after', 'first', 'next', 'finally',
            'pour', 'place', 'move', 'transfer', 'stack', 'open and',
            'pick up', 'put down', 'give me'
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in planning_keywords)
    
    def main_loop(self):
        try:
            while True:
                # 1. ASR -> Retrieve or Plan
                if self.asr and self.asr.has_new_result():
                    new_query = self.asr.get()
                    if new_query:
                        print(f"[Command] New input: {new_query}")
                        # Direct switch via command
                        if new_query.startswith('/'):
                            try:
                                self.change_type(new_query[1:])
                                continue
                            except Exception as e:
                                print(f"[Error] Failed to switch type: {e}")
                        # Check for special commands
                        elif new_query.lower() in ['next', 'continue', 'done']:
                            # Manual step completion
                            if self.task_executor.is_executing():
                                self.task_executor.manual_step_complete()
                            continue
                        elif new_query.lower() in ['stop', 'cancel', 'abort']:
                            # Stop current task
                            if self.task_executor.is_executing():
                                self.task_executor.stop()
                                self._planning_in_progress = False
                                self._current_plan = None
                                print("[Info] Task execution stopped")
                            continue
                        elif new_query.lower() in ['status', 'progress']:
                            # Show execution status
                            if self.task_executor.is_executing():
                                status = self.task_executor.get_status()
                                print(f"[Status] Step {status['current_step']}/{status['total_steps']}, "
                                      f"Progress: {status['progress']*100:.1f}%")
                            else:
                                print("[Status] No task currently executing")
                            continue
                        # Keyboard grasp control commands - ALWAYS handle these, don't send to VLM
                        elif new_query.lower() in ['o', 'open']:
                            # Open hand fully
                            self._grasp_fraction = 0.0
                            self._apply_keyboard_grasp()
                            print("\n[Grasp] Hand opened")
                            continue
                        elif new_query.lower() in ['c', 'close']:
                            # Close hand fully
                            self._grasp_fraction = 1.0
                            self._apply_keyboard_grasp()
                            print("\n[Grasp] Hand closed")
                            continue
                        elif new_query.lower() == 'a':
                            # Decrease grasp (more open)
                            self._grasp_fraction -= self._grasp_step
                            self._apply_keyboard_grasp()
                            continue
                        elif new_query.lower() == 'd':
                            # Increase grasp (more close)
                            self._grasp_fraction += self._grasp_step
                            self._apply_keyboard_grasp()
                            continue
                        # Multi-step planning query (Section 3.2 of paper)
                        elif self._is_planning_query(new_query):
                            print("[Planner] Multi-step task detected, starting VLM planning...")
                            self._planning_in_progress = True
                            # Get current frame for vision context
                            current_image = None
                            with self._frame_lock:
                                if self.current_frame is not None:
                                    current_image = self.current_frame.copy()
                            
                            # Plan task asynchronously
                            self.task_planner.plan_task_async(new_query, current_image)
                        else:
                            # Simple single-step retrieval
                            # Pass current image for vision-aware retrieval
                            current_image = None
                            with self._frame_lock:
                                if self.current_frame is not None:
                                    current_image = self.current_frame.copy()
                            self.retriever.retrieve(new_query, current_image)

                # 2. Check for completed task plans
                if self._planning_in_progress:
                    # Check if planning is complete
                    if self.task_planner.has_new_plan():
                        plan = self.task_planner.get_plan()
                        if plan:
                            print(f"[Planner] Plan received with {plan.total_steps} steps")
                            self._current_plan = plan
                            # Start executing the plan
                            self.task_executor.start_task(plan)
                        else:
                            print("[Planner] Planning failed, falling back to simple retrieval")
                            self._planning_in_progress = False
                
                # 3. Update task executor
                if self.task_executor.is_executing():
                    self.task_executor.update()

                # 4. Retriever Result -> Switch Type (only if not executing a plan)
                if not self.task_executor.is_executing() and self.retriever.has_new_result():
                    result = self.retriever.get()
                    if result and result != self.curr_type:
                        self.change_type(result)

                # 5. Hand Detection -> Robot Control (only if camera is enabled)
                if not self._use_keyboard_grasp:
                    try:
                        result = self.finger_detector.get()
                    except Exception as e:
                        result = None
                else:
                    result = None
                
                if result:
                    ratio, bgr = result
                    
                    # Store frame for vision input
                    with self._frame_lock:
                        self.current_frame = bgr.copy() if bgr is not None else None
                    
                    # Define masks for finger groups
                    thumb_mask = np.array([0]*12 + [1]*4)
                    index_mask = np.array([1]*4 + [0]*12)
                    middle_mask = np.array([0]*4 + [1]*4 + [0]*8)
                    ring_mask = np.array([0]*8 + [1]*4 + [0]*4)
                    
                    # Interpolate pose based on finger flexion ratios
                    # Note: LEAP Hand typically lacks independent pinky control
                    type_pos = (
                        (self.open_pos * (1 - ratio['thumb']) + self.close_pos * ratio['thumb']) * thumb_mask +
                        (self.open_pos * (1 - ratio['index']) + self.close_pos * ratio['index']) * index_mask +
                        (self.open_pos * (1 - ratio['middle']) + self.close_pos * ratio['middle']) * middle_mask +
                        (self.open_pos * (1 - ratio['ring']) + self.close_pos * ratio['ring']) * ring_mask
                    )

                    self.leap_node.set_leap(type_pos)

                    if bgr is not None:
                        # Display execution status on frame
                        if self.task_executor.is_executing():
                            status = self.task_executor.get_status()
                            step_info = self.task_executor.get_current_step_info()
                            status_text = f"Step {status['current_step']}/{status['total_steps']}: {step_info.description if step_info else ''}"
                            cv2.putText(bgr, status_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            progress_text = f"Progress: {status['progress']*100:.0f}%"
                            cv2.putText(bgr, progress_text, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cv2.imshow("Hand Detection", bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("Stopping...")
            self.stop()
            print("Closed.")


def run_leap():
    cfg = {
        "asr": {
            "type": "typing",  # Options: "typing" (keyboard) or "tencent" (API)
            
            # --- Tencent ASR Config ---
            "verbose": True,
            'credentials': {
                'secret_id': "your_secret_id",
                'secret_key': "your_secret_key"
            },
            'audio': {
                'channels': 1,
                'sample_rate': 16000,
                'chunk_duration': 0.1
            },
            'vad': {
                'silence_threshold': 500,
                'min_audio_length': 0.5,
                'max_silence_duration': 2.0
            },
            'tencent': {
                'endpoint': "asr.tencentcloudapi.com",
                'region': "ap-guangzhou",
                'engine_service_type': "16k_zh"
            },
            'test_microphone': True
        },
        "retriever": {
            # --- LLM / Retrieval：智谱 OpenAI 兼容接口 ---
            "api_key": os.getenv("BIGMODEL_API_KEY", ""),
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "model": "glm-4.5-air",
            "category": "leap",
            "enable_vision": True,  # Enable vision input for retrieval
        },
        "planner": {
            # --- VLM Task Planner Configuration (Section 3.2 of paper) ---
            "api_key": os.getenv("BIGMODEL_API_KEY", ""),
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "model": os.getenv("PLANNER_MODEL", "glm-4.6v"),  # GLM-4.6V for vision-language planning
            "enable_vision": True,  # Enable vision for context-aware planning
        },
        "detector": {
            "camera": {
                "camera_id": 4,  # Camera index
                "width": 640,
                "height": 480,
                "fps": 30,
                "queue_size": 1
            },
            "hand_type": "Left",
            "selfie": False,
        },
        "type": {
            "type_name": "box",
            "category": "leap"
        },
        "leap_cfg": {
            "curr_lim": 150,
            "kP": 100,
            "kI": 0,
            "kD": 150
        },
        "step_timeout": 15.0,  # Default timeout for each step in multi-step tasks
        "use_keyboard_grasp": True,  # Use keyboard 'a'/'d'/'o'/'c' for grasp control instead of camera
    }
    runner = RealTimeRunner(cfg)
    runner.start()


if __name__ == "__main__":
    run_leap()