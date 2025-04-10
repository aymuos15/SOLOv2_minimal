import os
import json
import random

import numpy as np
import cv2
from PIL import Image

def create_square(center, side_length):
    """Create a square as a polygon from center point and side length."""
    half_side = side_length / 2
    x, y = center
    # Create points in clockwise order
    points = [
        (x - half_side, y - half_side),
        (x + half_side, y - half_side),
        (x + half_side, y + half_side),
        (x - half_side, y + half_side)
    ]
    return points

def create_binary_mask(shape, polygon_points):
    """Create a binary mask from polygon points."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    # Convert points to integer for drawing
    points_array = np.array([[int(round(p[0])), int(round(p[1]))] for p in polygon_points], dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask

def add_noise(image, noise_level=0.1):
    """Add random noise to an image."""
    noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def create_dummy_dataset(output_dir, split='train', num_images=None, img_size=None, shape_size_range=None, num_shapes=None, add_noise_to_images=False, noise_level=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the dataset JSON structure
    dataset_info = {
        'info': {
            'description': 'Dummy Dataset with Squares',
            'url': None,
            'version': '1.0',
            'year': 2024,
            'contributor': None,
            'date_created': '2024-08-28'
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'square', 'supercategory': 'object'}
        ]
    }

    annotation_id = 1
    category_counts = {1: 0}  # Track count of square category only
    
    for img_id in range(num_images):
        image_filename = f'image_{img_id:04d}.jpg'
        if split == 'test':
            img_output_dir = os.path.join(output_dir)
        else:
            img_output_dir = os.path.join(output_dir)
        
        # Ensure directory exists
        if not os.path.exists(img_output_dir):
            os.makedirs(img_output_dir)
            
        image_path = os.path.join(img_output_dir, image_filename)
        
        # Create a black background image
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        
        # Determine number of shapes for this image
        shapes_in_this_image = random.randint(1, num_shapes)
        
        # Draw shapes and create annotations
        for shape_idx in range(shapes_in_this_image):
            # Always use square (category ID 1)
            shape_type = 1
            category_counts[shape_type] += 1
            # color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            color = (255, 255, 255)  # White color for square
            
            # Square
            # side_length = random.randint(250, 260)
            side_length = random.randint(shape_size_range[0], shape_size_range[1])
            half_side = side_length / 2
            
            # Ensure square is fully within image bounds
            center_x = random.randint(int(half_side) + 1, img_size[0] - int(half_side) - 1)
            center_y = random.randint(int(half_side) + 1, img_size[1] - int(half_side) - 1)
            center = (center_x, center_y)
            
            # Create square vertices with floating-point coordinates
            square_points = create_square(center, side_length)
            
            # Create a binary mask from the polygon
            mask = create_binary_mask(img.shape, square_points)
            
            # Draw the square directly on the image using the mask
            img[mask == 1] = color
            
            # Use mask for contour extraction
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().astype(float).tolist()
                segmentation.append(contour)
            
            # Calculate bounding box from mask
            pos = np.where(mask)
            x_min = np.min(pos[1])
            y_min = np.min(pos[0])
            x_max = np.max(pos[1])
            y_max = np.max(pos[0])
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            # Create annotation
            annotation = {
                'id': annotation_id,
                'image_id': img_id,
                'category_id': 1,  # Changed to 1 for square
                'segmentation': segmentation,
                'area': float(np.sum(mask)),  # Exact pixel count
                'bbox': [float(x_min), float(y_min), float(width), float(height)],
                'iscrowd': 0
            }
        
            dataset_info['annotations'].append(annotation)
            annotation_id += 1
    
        # Add noise if specified - AFTER annotations are created
        if add_noise_to_images:
            img = add_noise(img, noise_level)
            
        # Save image
        Image.fromarray(img).save(image_path)
        
        # Image info
        dataset_info['images'].append({
            'id': img_id,
            'width': img_size[0],
            'height': img_size[1],
            'file_name': image_filename
        })
    
    # Print category statistics
    print(f"Category distribution - Squares: {category_counts[1]}")

    # make the annotations dir if it does not exist
    if not os.path.exists('annotations'):
        os.makedirs('annotations')
    
    # Save dataset JSON for non-test splits
    if split == 'test':
        with open(os.path.join('annotations/annotations_test.json'), 'w') as f:
            json.dump(dataset_info, f)
    else:
        with open(os.path.join('annotations/annotations_train.json'), 'w') as f:
            json.dump(dataset_info, f)
    
    print(f'Dummy dataset created at {output_dir}')

# Example usage
create_dummy_dataset('train', split='train', num_images=200, img_size=(512, 512), shape_size_range=(315, 320), num_shapes=1, add_noise_to_images=True, noise_level=0.1)
create_dummy_dataset('test', split='test', num_images=50, img_size=(511, 512), shape_size_range=(310, 320), num_shapes=1, add_noise_to_images=True, noise_level=0.11)
