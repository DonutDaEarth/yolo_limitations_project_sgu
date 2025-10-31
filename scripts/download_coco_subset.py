"""
Download a subset of COCO dataset for quick testing
"""
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO


def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))


def download_coco_subset(output_dir='data/coco', num_images=500, year='2017'):
    """
    Download a subset of COCO dataset
    
    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to download
        year: COCO dataset year (2017)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / f'val{year}'
    annotations_dir = output_dir / 'annotations'
    
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    # Download annotations (small file ~250MB)
    print(f"\n📥 Downloading COCO {year} annotations...")
    annotations_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'
    annotations_zip = output_dir / f'annotations_trainval{year}.zip'
    
    if not (annotations_dir / f'instances_val{year}.json').exists():
        if not annotations_zip.exists():
            print("Downloading annotations archive...")
            download_file(annotations_url, annotations_zip)
        
        print("Extracting annotations...")
        import zipfile
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("✓ Annotations extracted")
    else:
        print("✓ Annotations already exist")
    
    # Load COCO annotations
    ann_file = annotations_dir / f'instances_val{year}.json'
    print(f"\n📝 Loading COCO annotations from {ann_file}...")
    coco = COCO(str(ann_file))
    
    # Get image IDs
    img_ids = coco.getImgIds()
    
    # Select subset
    if num_images < len(img_ids):
        import random
        random.seed(42)
        img_ids = random.sample(img_ids, num_images)
    
    print(f"\n📥 Downloading {len(img_ids)} images...")
    
    # Download images
    downloaded = 0
    skipped = 0
    failed = 0
    
    for img_id in tqdm(img_ids, desc="Downloading images"):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = images_dir / img_info['file_name']
        
        # Skip if already exists
        if img_path.exists():
            skipped += 1
            continue
        
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n⚠️  Failed to download {img_info['file_name']}: {e}")
    
    # Create subset annotation file
    print(f"\n📝 Creating subset annotation file...")
    subset_ann = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'categories': coco.dataset['categories'],
        'images': coco.loadImgs(img_ids),
        'annotations': []
    }
    
    # Get annotations for selected images
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        subset_ann['annotations'].extend(anns)
    
    # Save subset annotations
    subset_ann_file = annotations_dir / f'instances_val{year}_subset{num_images}.json'
    with open(subset_ann_file, 'w') as f:
        json.dump(subset_ann, f)
    
    print(f"\n✅ Dataset download complete!")
    print(f"   Downloaded: {downloaded} images")
    print(f"   Skipped (already exist): {skipped} images")
    print(f"   Failed: {failed} images")
    print(f"   Annotations: {len(subset_ann['annotations'])}")
    print(f"\n📁 Dataset location:")
    print(f"   Images: {images_dir}")
    print(f"   Annotations: {subset_ann_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download COCO subset')
    parser.add_argument('--num_images', type=int, default=500,
                        help='Number of images to download (default: 500)')
    parser.add_argument('--output_dir', type=str, default='data/coco',
                        help='Output directory (default: data/coco)')
    parser.add_argument('--year', type=str, default='2017',
                        help='COCO dataset year (default: 2017)')
    
    args = parser.parse_args()
    
    download_coco_subset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        year=args.year
    )
