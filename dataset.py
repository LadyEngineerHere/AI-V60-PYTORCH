import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as F

class CustomDataset(Dataset):
    def __init__(self, json_file, img_dirs, transform=None):
        self.img_dirs = img_dirs
        self.transform = transform
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = self.data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}
        
        # Create a mapping from image_id to its annotations
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_file_name = img_info['file_name']
        
        # Find the correct image directory
        img_path = None
        for img_dir in self.img_dirs:
            potential_path = os.path.join(img_dir, img_file_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_file_name} not found in any of the directories.")
        
        image = Image.open(img_path).convert("RGB")
        
        # Resize the image to 512x512 while maintaining aspect ratio
        image = self.resize_image(image, target_size=(512, 512))
        
        # Get annotations for the image
        annotations = self.image_to_annotations.get(img_id, [])
        if len(annotations) == 0:
            # Skip images with no annotations
            return None
        
        bboxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']
            # COCO format: [x, y, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            # Adjust bounding boxes according to the new image size
            x_min, y_min, x_max, y_max = self.adjust_bbox(image.size, (x_min, y_min, x_max, y_max), img_info['width'], img_info['height'])
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        
        # Convert lists to tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dict
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

    def resize_image(self, image, target_size):
        """Resize image while maintaining aspect ratio and add padding if necessary."""
        original_size = image.size
        image = F.resize(image, target_size)
        return image

    def adjust_bbox(self, new_size, bbox, original_width, original_height):
        """Adjust bounding box according to the new image size."""
        x_min, y_min, x_max, y_max = bbox
        width_ratio = new_size[0] / original_width
        height_ratio = new_size[1] / original_height
        x_min *= width_ratio
        y_min *= height_ratio
        x_max *= width_ratio
        y_max *= height_ratio
        return x_min, y_min, x_max, y_max

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))
