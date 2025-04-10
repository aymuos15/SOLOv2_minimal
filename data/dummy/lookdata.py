# Check the label of a single image
import mmengine

annotation = mmengine.load('annotations/annotations_train.json')

# First, let's examine the structure of the annotation file
print("Annotation keys:", annotation.keys())

# Try to access annotations more safely
image_id = 0  # Change this to the index of the image you want to check

# Check if the expected keys exist
has_annotations = 'annotations' in annotation
has_images = 'images' in annotation

if has_annotations and has_images:
    # Original code path
    image_annotations = annotation['annotations']
    image_info = annotation['images'][image_id]
    image_name = image_info['file_name']
    image_width = image_info['width']
    image_height = image_info['height']
    
    # Print the image information
    print(f"Image ID: {image_id}")
    print(f"Image Name: {image_name}")
    print(f"Image Width: {image_width}")
    print(f"Image Height: {image_height}")  

    # Filter annotations for this specific image
    image_annotations = [ann for ann in annotation['annotations'] if ann.get('image_id') == image_id]
    
    # Check how many annotations are in the image
    num_annotations = len(image_annotations)
    print(f"Number of annotations in the image: {num_annotations}")

    # How many classes are in the image
    classes = set()
    for ann in image_annotations:
        classes.add(ann.get('category_id'))
    print(f"Number of classes in the image: {len(classes)}")
    # Print the class names
    for class_id in classes:
        print(f"Class ID: {class_id}")

    # How many images are in the dataset
    num_images = len(annotation['images'])
    print(f"Number of images in the dataset: {num_images}")

print("\n\n")

# Checking with pycocotools
from pycocotools.coco import COCO

# Path to load the COCO annotation file
annotation_file = 'annotations/annotations_train.json'

# Initialise the COCO object
coco = COCO(annotation_file)

# Get all category tags and corresponding category IDs
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Print all category IDs and corresponding category names
for category_id, category_name in category_id_to_name.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")
