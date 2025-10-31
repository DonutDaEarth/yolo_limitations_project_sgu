"""
Utility helper functions for the project
"""

import os
import yaml
import json
import numpy as np
from typing import Dict, Any


def load_config(config_path: str = './config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    converted_data = convert_types(data)
    
    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"Saved: {filepath}")


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def format_results_table(results: Dict[str, Dict]) -> str:
    """Format results as a markdown table"""
    lines = []
    lines.append("| Metric | " + " | ".join(results.keys()) + " |")
    lines.append("|" + "---|" * (len(results) + 1))
    
    # Get all metric keys
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    for metric in sorted(all_metrics):
        row = f"| {metric} |"
        for model_name in results.keys():
            value = results[model_name].get(metric, 'N/A')
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {value} |"
        lines.append(row)
    
    return "\n".join(lines)


def calculate_percentage_change(value1: float, value2: float) -> float:
    """Calculate percentage change from value1 to value2"""
    if value1 == 0:
        return 0
    return ((value2 - value1) / value1) * 100


def ensure_dir(directory: str):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)


def get_coco_category_name(category_id: int) -> str:
    """Get COCO category name from ID"""
    # COCO 80 class names
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
        'toothbrush'
    ]
    
    if 0 <= category_id < len(coco_names):
        return coco_names[category_id]
    return f"class_{category_id}"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test load config
    try:
        config = load_config('./config/config.yaml')
        print(f"✓ Config loaded: {config['project']['name']}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
    
    # Test JSON save/load
    test_data = {'test': 123, 'array': np.array([1, 2, 3])}
    save_json(test_data, './test_output.json')
    loaded_data = load_json('./test_output.json')
    print(f"✓ JSON save/load: {loaded_data}")
    
    # Clean up
    if os.path.exists('./test_output.json'):
        os.remove('./test_output.json')
    
    print("\n✓ All utility tests passed!")
