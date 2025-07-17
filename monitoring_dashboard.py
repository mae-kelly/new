import json
import time
import psutil
import threading
from datetime import datetime
import subprocess
import os

class BrilliantMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.training_active = True
        
    def monitor_training(self):
        while self.training_active:
            try:
                # Check if model files exist
                model_exists = os.path.exists('models/insane_results.json')
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Training progress estimation
                elapsed = (datetime.now() - self.start_time).total_seconds()
                
                status = {
                    'elapsed_minutes': elapsed / 60,
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'training_complete': model_exists,
                    'estimated_completion': '15-30 minutes' if not model_exists else 'COMPLETE!',
                    'status': 'TRAINING INSANE ACCURACY MODEL 🚀' if not model_exists else 'TRAINING COMPLETE! 🎯'
                }
                
                # Live dashboard
                print(f"\r🧠 LIVE TRAINING MONITOR | "
                      f"⏰ {elapsed/60:.1f}min | "
                      f"🖥️ CPU:{cpu_percent:.1f}% | "
                      f"💾 RAM:{memory.percent:.1f}% | "
                      f"Status: {status['status']}", end='', flush=True)
                
                # Save status
                with open('training_status.json', 'w') as f:
                    json.dump(status, f, indent=2, default=str)
                    
                if model_exists:
                    print("\n🎉 TRAINING COMPLETED! Check results.")
                    break
                    
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n⏸️ Monitoring stopped")
                break
                
    def start_monitoring(self):
        monitor_thread = threading.Thread(target=self.monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread

if __name__ == '__main__':
    monitor = BrilliantMonitor()
    monitor.monitor_training()
