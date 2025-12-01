"""
Progress indicator utilities for cleaner console output.
"""

import sys
import time
from typing import Optional


class ProgressSpinner:
    """Simple spinner for showing progress without verbose output."""
    
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.running = False
        self.frame_idx = 0
        
    def start(self):
        """Start the spinner."""
        self.running = True
        self._update()
        
    def _update(self):
        """Update spinner frame."""
        if self.running:
            frame = self.FRAMES[self.frame_idx % len(self.FRAMES)]
            sys.stdout.write(f'\r{frame} {self.message}...')
            sys.stdout.flush()
            self.frame_idx += 1
            
    def stop(self, final_message: Optional[str] = None):
        """Stop the spinner and optionally print final message."""
        self.running = False
        if final_message:
            sys.stdout.write(f'\r✓ {final_message}\n')
        else:
            sys.stdout.write(f'\r✓ {self.message} 完成\n')
        sys.stdout.flush()


class ProgressBar:
    """Simple progress bar for showing completion percentage."""
    
    def __init__(self, total: int, message: str = "Progress", width: int = 40):
        self.total = total
        self.message = message
        self.width = width
        self.current = 0
        
    def update(self, current: Optional[int] = None):
        """Update progress bar."""
        if current is not None:
            self.current = current
        else:
            self.current += 1
            
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        sys.stdout.write(f'\r{self.message}: [{bar}] {percent*100:.0f}%')
        sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
            
    def finish(self):
        """Complete the progress bar."""
        self.update(self.total)


def show_progress(message: str, duration: float = 1.0):
    """
    Show a simple progress message for a brief duration.
    
    Args:
        message: Message to display
        duration: How long to show the message (seconds)
    """
    spinner = ProgressSpinner(message)
    spinner.start()
    time.sleep(duration)
    spinner.stop()


def with_progress(func, message: str = "Processing"):
    """
    Decorator to show progress while executing a function.
    
    Args:
        func: Function to execute
        message: Progress message
        
    Returns:
        Function result
    """
    def wrapper(*args, **kwargs):
        spinner = ProgressSpinner(message)
        spinner.start()
        try:
            result = func(*args, **kwargs)
            spinner.stop(f"{message} 完成")
            return result
        except Exception as e:
            spinner.stop(f"{message} 失敗: {e}")
            raise
    return wrapper
