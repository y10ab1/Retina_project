import cv2
import os
import sys
import numpy as np
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm

def process_image(image_path):
    original_image = cv2.imread(image_path)
    grad_cam_image = original_image.copy()  # Assuming you process this to get grad_cam_image
    gray_grad_cam = cv2.cvtColor(grad_cam_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_grad_cam, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_box = original_image.copy()

    coordinate_list = []
    img_width, img_height = original_image.shape[1], original_image.shape[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w >= 30 and h >= 30) and (30 <= x <= img_width - 30) and (30 <= y <= img_height - 30):
            coordinate_list.append((x, y, x+w, y+h))
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    if len(coordinate_list) == 0:
        # whole image is considered as a bounding box
        coordinate_list.append((0, 0, original_image.shape[1], original_image.shape[0]))

    return image_with_box, coordinate_list


def save_images_and_csv(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for image_name, data in results.items():
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, data['image_with_box'])
        records.append({'image_path': image_path, 'coordinate_list': data['coordinates']})

    pd.DataFrame(records).to_csv(os.path.join(output_dir, 'coordinates.csv'), index=False)

def main(args):
    results = {}
    if args.image_path:
        image_with_box, coordinates = process_image(args.image_path)
        results[os.path.basename(args.image_path)] = {'image_with_box': image_with_box, 'coordinates': coordinates}
    elif args.image_dir:
        bar = tqdm(os.listdir(args.image_dir))
        for image_file in bar:
            image_path = os.path.join(args.image_dir, image_file)
            image_with_box, coordinates = process_image(image_path)
            results[image_file] = {'image_with_box': image_with_box, 'coordinates': coordinates}

    save_images_and_csv(results, './coordinate_output')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path','-p', type=str, help='Path to a single image')
    parser.add_argument('--image_dir','-d', type=str, help='Directory containing images')
    args = parser.parse_args()
    main(args)
