from openai import OpenAI
from threading import Thread, Lock
import time
import os
import json
import difflib
import re
import base64
from typing import Optional
import numpy as np


class Retrieve:
    def __init__(
        self,
        api_key,
        base_url,
        category: str = "papert",
        model: str | None = None,
        enable_vision: bool = True
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Z.AI: use e.g. "glm-5" / "glm-4.5-air"; BigModel/DeepSeek: e.g. "deepseek-chat"
        self.model = model or os.getenv("RETRIEVER_MODEL", "deepseek-chat")

        self.category = category or "papert"
        self.enable_vision = enable_vision

        self.running = False
        self.retrieve_thread = None

        self.type_files = [] 
        self.types = []       

        self._result_lock = Lock()
        self.have_new_result = False
        self.result = ""

        self._input_lock = Lock()
        self.have_new_input = False
        self.input = ""
        
        # Vision input support
        self._image_lock = Lock()
        self.current_image: Optional[np.ndarray] = None
        self.has_new_image = False

    def start(self):
        self.running = True
        self.retrieve_thread = Thread(target=self.spin)
        self.retrieve_thread.start()

    def spin(self):
        while self.running:
            with self._input_lock:
                if self.have_new_input:
                    current_input = self.input
                    self.have_new_input = False
                else:
                    current_input = None
            
            if current_input:
                print("Start retrieving...")
                type_name = self._retrieve(current_input)
                with self._result_lock:
                    self.result = type_name
                    self.have_new_result = True
            time.sleep(1)

    def stop(self):
        self.running = False
        if self.retrieve_thread is not None:
            self.retrieve_thread.join()
            self.retrieve_thread = None

    def load_type_library(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_type_library_path = os.path.join(current_dir, "../TypeLibrary")
        type_library_path = os.path.join(base_type_library_path, self.category)

        if not os.path.isdir(type_library_path):
            print(f"[Retrieve] Directory {type_library_path} does not exist, falling back to {base_type_library_path}")
            type_library_path = base_type_library_path

        json_path = os.path.join(type_library_path, "_type_info.json")

        loaded_from_json = False
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.types = data
                    self.type_files = [t.get('id') for t in self.types if 'id' in t]
                    loaded_from_json = True
            except Exception as e:
                print(f"Failed to read _type_info.json: {e}")

        if not loaded_from_json:
            self.type_files = [f[:-4] for f in os.listdir(type_library_path) if f.endswith('.txt')]
            self.types = [{"id": name} for name in self.type_files]
        print(f"Loaded types (category={self.category}): {self.type_files}")

    def _local_score(self, query: str, gesture: dict) -> float:
        q = query.lower().strip()
        gid = gesture.get('id', '') or ''
        name = gesture.get('name', '') or ''
        usage = gesture.get('usage', '') or ''
        intents = gesture.get('intents', []) or []

        base = difflib.SequenceMatcher(None, q, gid).ratio()

        bonus_name = 0.15 if name and name in q else 0.0

        bonus_intent = 0.0
        for it in intents:
            if isinstance(it, str) and it and it.lower() in q:
                bonus_intent += 0.1
        bonus_intent = min(bonus_intent, 0.3)

        tokens = re.findall(r"[a-zA-Z]+", usage.lower())
        token_hit = sum(1 for t in tokens if len(t) > 3 and t in q)
        bonus_usage = min(token_hit * 0.05, 0.15)

        score = base + bonus_name + bonus_intent + bonus_usage
        return score

    def _local_retrieve(self, query: str):
        best = None
        best_score = 0.0
        for g in self.types:
            s = self._local_score(query, g)
            if s > best_score:
                best_score = s
                best = g.get('id')
        return best, best_score

    def _retrieve(self, query: str):
        if not self.type_files:
            return None

        local_id, score = self._local_retrieve(query)
        if local_id and score >= 0.75:
            return local_id

        brief_lines = []
        for t in self.types:
            intents = ','.join(t.get('intents', [])[:3]) if isinstance(t.get('intents'), list) else ''
            brief_lines.append(f"{t.get('id')}: {t.get('pose','')}; intents={intents}")
        catalog = "\n".join(brief_lines)

        # Get current image if available
        current_image = None
        with self._image_lock:
            if self.has_new_image and self.current_image is not None:
                current_image = self.current_image.copy()
                self.has_new_image = False

        # Build prompt with vision support
        system_prompt = (
            "You are a gesture type selector for dexterous manipulation. "
            "Given a natural language user query and optionally a camera image, "
            "choose the best gesture id from the catalog that matches the task and visual context. "
            "If nothing fits, answer None. Just output the id or None."
        )
        
        user_prompt = (
            f"Catalog:\n{catalog}\n\n"
            f"Query: {query}\n\n"
            f"Answer with just the gesture id or None:"
        )

        try:
            # Prepare messages with optional vision
            messages = [{"role": "system", "content": system_prompt}]
            
            if self.enable_vision and current_image is not None:
                # Vision-enabled retrieval as described in paper
                base64_image = self._encode_image(current_image)
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
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            candidate = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM retrieval exception: {e}")
            candidate = None

        if candidate in self.type_files:
            return candidate
        
        if local_id and score >= 0.55:
            return local_id
        return None
        
    def set_image(self, image: np.ndarray):
        """
        Set current camera image for vision-aware retrieval.
        
        Args:
            image: BGR or RGB image as numpy array
        """
        with self._image_lock:
            self.current_image = image.copy() if image is not None else None
            self.has_new_image = True
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 string for VLM API.
        
        Args:
            image: BGR or RGB image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        import cv2
        # Convert to RGB if needed (assume BGR from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image

    def retrieve(self, query: str, image: Optional[np.ndarray] = None):
        """
        Submit a retrieval query.
        
        Args:
            query: Natural language query
            image: Optional camera image for vision context
        """
        with self._input_lock:
            self.input = query
            self.have_new_input = True
        
        # Update image if provided
        if image is not None:
            self.set_image(image)

    def retrieve_sync(self, query: str):
        """Run retrieval in-process (no background thread). For testing without hardware."""
        return self._retrieve(query)

    def has_new_result(self):
        with self._result_lock:
            return self.have_new_result

    def get(self):
        try:
            with self._result_lock:
                if self.have_new_result:
                    self.have_new_result = False
                    return self.result
                else:
                    return None
        except Exception as e:
            print(f"Error getting result: {e}")
            return None