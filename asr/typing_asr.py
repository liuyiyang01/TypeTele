import sys
import time
import threading
from queue import Queue, Empty
import os

class KeyboardAsrServer:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.verbose = cfg.get("verbose", True) if cfg else True
        
        self.is_running = False
        self.input_thread = None
        
        self.result_queue = Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self.have_new_result = False
        
        print("=== Keyboard Input Mode ===")
        print("Enter commands in the terminal, press Enter to send")
        print("Supports: arrow keys, copy/paste, backspace, etc.")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 30)

    def _input_loop(self):
        print("Keyboard input ready, start typing commands...")
        print()
        
        while self.is_running:
            try:
                # Use standard input() which supports arrow keys, copy/paste, etc.
                text = input("Enter command: ").strip()
                
                if not text:
                    continue
                
                if text.lower() in ['quit', 'exit']:
                    print(f"Exit command detected: {text}")
                    self.is_running = False
                    break
                
                self._put_result(text)
                print(f"[Sent] {text}")
                print()
                
            except EOFError:
                # Handle Ctrl+D
                print("\nEOF received, exiting...")
                self.is_running = False
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\nCtrl+C received, exiting...")
                self.is_running = False
                break

    def _put_result(self, text):
        with self._result_lock:
            if not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except:
                    pass
            
            self.result_queue.put(text)
            self.have_new_result = True
    
    def start(self):
        if self.is_running:
            if self.verbose:
                print("Keyboard input is already running")
            return True
        
        self.is_running = True
        
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        if self.verbose:
            print("Keyboard input started")
        
        return True
    
    def stop(self):
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        if self.verbose:
            print("Keyboard input stopped")
    
    def get(self):
        try:
            with self._result_lock:
                if self.have_new_result:
                    result = self.result_queue.get_nowait()
                    self.have_new_result = False
                    return result
                return None
        except:
            return None
    
    def has_new_result(self):
        with self._result_lock:
            return self.have_new_result
    
    def is_running_status(self):
        return self.is_running
    
    def __del__(self):
        self.stop()
