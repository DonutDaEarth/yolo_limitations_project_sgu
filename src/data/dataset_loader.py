"""
Dataset Loader for COCO and other datasets
Handles loading, filtering, and preprocessing
"""

import os
import json
import numpy as np
from pycocotools.coco import COCO
from typing import List, Dict, Tuple
import cv2


class COCODatasetLoader:
    """Loader for COCO dataset with filtering capabilities"""
    
    def __init__(self, data_root: str, split: str = 'val', year: int = 2017, 
                 annotations_file: str = None):
        """
        Initialize COCO dataset loader
        
        Args:
            data_root: Root directory of COCO dataset
            split: Dataset split ('train', 'val', 'test')
            year: COCO dataset year (2014, 2017)
            annotations_file: Path to custom annotations file (overrides default)
        """
        self.data_root = data_root
        self.split = split
        self.year = year
        
        # Setup paths
        self.images_dir = os.path.join(data_root, f'{split}{year}')
        
        if annotations_file is not None:
            self.annotations_file = annotations_file
        else:
            self.annotations_file = os.path.join(
                data_root, 
                'annotations', 
                f'instances_{split}{year}.json'
            )
        
        # Load COCO API
        print(f"Loading COCO {split}{year} dataset...")
        self.coco = COCO(self.annotations_file)
        
        # Get categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = {cat['id']: cat['name'] for cat in self.categories}
        
        print(f"Dataset loaded: {len(self.coco.getImgIds())} images, "
              f"{len(self.categories)} categories")
    
    def get_image_ids(self, filter_by_area: str = None) -> List[int]:
        """
        Get image IDs, optionally filtered by object size
        
        Args:
            filter_by_area: Filter by object area ('small', 'medium', 'large', None)
                           small: area < 32^2, medium: 32^2 < area < 96^2, large: area > 96^2
        
        Returns:
            List of image IDs
        """
        img_ids = self.coco.getImgIds()
        
        if filter_by_area is None:
            return img_ids
        
        # Filter images that contain objects of specified size
        filtered_ids = []
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                area = ann['area']
                
                if filter_by_area == 'small' and area < 32 * 32:
                    filtered_ids.append(img_id)
                    break
                elif filter_by_area == 'medium' and 32*32 <= area < 96*96:
                    filtered_ids.append(img_id)
                    break
                elif filter_by_area == 'large' and area >= 96*96:
                    filtered_ids.append(img_id)
                    break
        
        print(f"Filtered {len(filtered_ids)} images with '{filter_by_area}' objects")
        return filtered_ids
    
    def load_image(self, image_id: int) -> Tuple[np.ndarray, Dict]:
        """
        Load image and its metadata
        
        Args:
            image_id: COCO image ID
            
        Returns:
            Tuple of (image, metadata)
        """
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        return image, img_info
    
    def load_annotations(self, image_id: int) -> List[Dict]:
        """
        Load annotations for an image
        
        Args:
            image_id: COCO image ID
            
        Returns:
            List of annotation dictionaries
        """
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Convert to standard format
        annotations = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            annotations.append({
                'bbox': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2]
                'category_id': ann['category_id'],
                'category_name': self.category_names[ann['category_id']],
                'area': ann['area'],
                'iscrowd': ann['iscrowd']
            })
        
        return annotations
    
    def get_small_object_subset(self, max_images: int = 500) -> List[int]:
        """
        Get a subset of images that predominantly contain small objects
        
        Args:
            max_images: Maximum number of images to return
            
        Returns:
            List of image IDs
        """
        img_ids = self.coco.getImgIds()
        small_object_images = []
        
        for img_id in img_ids:
            if len(small_object_images) >= max_images:
                break
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Count small objects
            small_count = sum(1 for ann in anns if ann['area'] < 32 * 32)
            total_count = len(anns)
            
            # If more than 30% of objects are small, include this image
            if total_count > 0 and (small_count / total_count) > 0.3:
                small_object_images.append(img_id)
        
        print(f"Found {len(small_object_images)} images with predominantly small objects")
        return small_object_images
    
    def get_dense_cluster_subset(self, max_images: int = 500, 
                                  min_objects: int = 10) -> List[int]:
        """
        Get images with dense object clusters
        
        Args:
            max_images: Maximum number of images
            min_objects: Minimum number of objects per image
            
        Returns:
            List of image IDs
        """
        img_ids = self.coco.getImgIds()
        dense_images = []
        
        for img_id in img_ids:
            if len(dense_images) >= max_images:
                break
            
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) >= min_objects:
                dense_images.append(img_id)
        
        print(f"Found {len(dense_images)} images with {min_objects}+ objects")
        return dense_images
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        img_ids = self.coco.getImgIds()
        
        total_objects = 0
        small_objects = 0
        medium_objects = 0
        large_objects = 0
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                total_objects += 1
                area = ann['area']
                
                if area < 32 * 32:
                    small_objects += 1
                elif area < 96 * 96:
                    medium_objects += 1
                else:
                    large_objects += 1
        
        return {
            'num_images': len(img_ids),
            'num_categories': len(self.categories),
            'total_objects': total_objects,
            'small_objects': small_objects,
            'medium_objects': medium_objects,
            'large_objects': large_objects,
            'small_ratio': small_objects / total_objects if total_objects > 0 else 0
        }


if __name__ == "__main__":
    # Test the loader
    loader = COCODatasetLoader(
        data_root="./data/coco",
        split='val',
        year=2017
    )
    
    # Get statistics
    stats = loader.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get small object subset
    small_img_ids = loader.get_small_object_subset(max_images=100)
    print(f"\nSmall object subset: {len(small_img_ids)} images")
    
    # Load sample image
    if len(small_img_ids) > 0:
        img_id = small_img_ids[0]
        image, img_info = loader.load_image(img_id)
        annotations = loader.load_annotations(img_id)
        print(f"\nSample image: {img_info['file_name']}")
        print(f"  Size: {image.shape}")
        print(f"  Objects: {len(annotations)}")
