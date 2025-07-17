import json
import os
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_file='scan_checkpoint.json'):
        self.checkpoint_file = checkpoint_file
        self.load_checkpoint()
    
    def load_checkpoint(self):
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint = json.load(f)
            else:
                self.checkpoint = {
                    'completed_projects': [],
                    'completed_datasets': [],
                    'completed_tables': [],
                    'last_project': None,
                    'last_dataset': None,
                    'scan_start_time': None,
                    'total_mappings': 0
                }
        except Exception:
            self.checkpoint = {
                'completed_projects': [],
                'completed_datasets': [],
                'completed_tables': [],
                'last_project': None,
                'last_dataset': None,
                'scan_start_time': None,
                'total_mappings': 0
            }
    
    def save_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def is_project_completed(self, project_id):
        return project_id in self.checkpoint.get('completed_projects', [])
    
    def is_dataset_completed(self, project_id, dataset_id):
        key = f"{project_id}.{dataset_id}"
        return key in self.checkpoint.get('completed_datasets', [])
    
    def is_table_completed(self, project_id, dataset_id, table_id):
        key = f"{project_id}.{dataset_id}.{table_id}"
        return key in self.checkpoint.get('completed_tables', [])
    
    def mark_project_completed(self, project_id):
        if project_id not in self.checkpoint['completed_projects']:
            self.checkpoint['completed_projects'].append(project_id)
        self.checkpoint['last_project'] = project_id
        self.save_checkpoint()
    
    def mark_dataset_completed(self, project_id, dataset_id):
        key = f"{project_id}.{dataset_id}"
        if key not in self.checkpoint['completed_datasets']:
            self.checkpoint['completed_datasets'].append(key)
        self.checkpoint['last_dataset'] = key
        self.save_checkpoint()
    
    def mark_table_completed(self, project_id, dataset_id, table_id):
        key = f"{project_id}.{dataset_id}.{table_id}"
        if key not in self.checkpoint['completed_tables']:
            self.checkpoint['completed_tables'].append(key)
        self.save_checkpoint()
    
    def start_scan(self):
        self.checkpoint['scan_start_time'] = datetime.now().isoformat()
        self.save_checkpoint()
    
    def add_mappings(self, count):
        self.checkpoint['total_mappings'] = self.checkpoint.get('total_mappings', 0) + count
        self.save_checkpoint()
    
    def get_progress(self):
        return {
            'projects_completed': len(self.checkpoint.get('completed_projects', [])),
            'datasets_completed': len(self.checkpoint.get('completed_datasets', [])),
            'tables_completed': len(self.checkpoint.get('completed_tables', [])),
            'total_mappings': self.checkpoint.get('total_mappings', 0),
            'last_project': self.checkpoint.get('last_project'),
            'scan_start_time': self.checkpoint.get('scan_start_time')
        }
    
    def clear_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        self.load_checkpoint()
