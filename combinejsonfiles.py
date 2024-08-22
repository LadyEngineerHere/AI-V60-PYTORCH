import os
import json
from lxml import etree
from PIL import Image

def convert_xml_to_coco_format(xml_folder, image_folder, output_json):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_map = {}
    category_id = 1

    annotation_id = 1
    
    # Iterate through subfolders in xml_folder
    for subfolder in os.listdir(xml_folder):
        subfolder_path = os.path.join(xml_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for xml_file in os.listdir(subfolder_path):
                if xml_file.endswith('.xml'):
                    xml_path = os.path.join(subfolder_path, xml_file)
                    
                    # Parse the XML file
                    try:
                        tree = etree.parse(xml_path)
                        root = tree.getroot()
                    except Exception as e:
                        print(f"Error parsing XML file {xml_path}: {e}")
                        continue
                    
                    image_name = root.find('filename').text
                    img_path = os.path.join(image_folder, subfolder, image_name)
                    
                    # Get image dimensions
                    try:
                        with Image.open(img_path) as img:
                            image_width, image_height = img.size
                    except Exception as e:
                        print(f"Error opening image {img_path}: {e}")
                        continue
                    
                    # Add image info to COCO format
                    coco_format["images"].append({
                        "id": len(coco_format["images"]) + 1,
                        "file_name": image_name,
                        "width": image_width,
                        "height": image_height
                    })
                    
                    for obj in root.findall('object'):
                        label = obj.find('name').text
                        bbox = obj.find('bndbox')
                        
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        
                        # Convert bounding box to COCO format
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        # Add category if not already present
                        if label not in category_map:
                            category_map[label] = category_id
                            coco_format["categories"].append({
                                "id": category_id,
                                "name": label
                            })
                            category_id += 1
                        
                        category_id = category_map[label]
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": len(coco_format["images"]),
                            "category_id": category_id,
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        annotation_id += 1
    
    # Save the annotations to a JSON file
    with open(output_json, 'w') as out_file:
        json.dump(coco_format, out_file, indent=4)

# Call the conversion function
convert_xml_to_coco_format(
    '/Users/amandanassar/Desktop/V60 NOK AI/annotations', 
    '/Users/amandanassar/Desktop/V60 NOK AI/Data images', 
    '/Users/amandanassar/Desktop/V60 NOK AI/annotations.json'
)
