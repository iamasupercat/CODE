'''
원본이미지 증강: 노이즈, 색상반전, 좌우반전


사용법:
    python augmentation.py 0616 0718 0721 0725 0728 0729 0731 0801 0804 0805 0806 \
    --subfolder frontfender hood trunklid


target_dir/
    ├── subfolder/
    │   ├── bad/
    │   │   ├── images/          # 원본 이미지들
    │   │   └──  labels/          # YOLO 라벨 파일들
    │   │   
    │   └── good/
    │       ├── images/          # 원본 이미지들
    │       └── labels/          # YOLO 라벨 파일들
    │       
    ├── subfolder_aug/
    │   ├── bad/
    │   │   ├── images/          # 증강 이미지들
    │   │   └──  labels/          # 증강 라벨 파일들
    │   │   
    │   └── good/
    │       ├── images/          # 증강 이미지들
    │       └── labels/          # 증강 라벨 파일들
    │ 

'''


from PIL import Image, ImageOps
import numpy as np
import os
import random
from pathlib import Path
import shutil
import sys
import argparse

def load_yolo_labels(label_path):
    """YOLO 라벨 파일을 읽어서 바운딩 박스 정보를 반환"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([class_id, x_center, y_center, width, height])
    return labels

def save_yolo_labels(label_path, labels):
    """YOLO 라벨을 파일에 저장"""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")



def flip_horizontal_and_bbox(image, labels):
    """이미지를 좌우 반전시키고 바운딩 박스 좌표를 변환"""
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    new_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        # x 좌표만 반전 (1 - x_center)
        new_x_center = 1.0 - x_center
        new_labels.append([class_id, new_x_center, y_center, width, height])
    
    return flipped_image, new_labels

def add_noise(image):
    """이미지에 가우시안 노이즈 추가"""
    # PIL 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, 25, img_array.shape).astype(np.int16)
    
    # 노이즈 추가
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # numpy 배열을 PIL 이미지로 변환
    return Image.fromarray(noisy_array)

def invert_colors(image):
    """색상 반전"""
    return ImageOps.invert(image.convert('RGB'))

def augment_image(image_path, label_path, output_dir, category):
    """단일 이미지에 대해 모든 증강 적용"""
    # 원본 이미지 로드
    try:
        image = Image.open(image_path)
        # EXIF orientation 정보를 자동으로 적용하여 올바른 방향으로 회전
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
    except:
        print(f"Failed to load image: {image_path}")
        return
    
    # 라벨 로드
    labels = load_yolo_labels(label_path)
    
    # 파일명에서 확장자 제거
    base_name = Path(image_path).stem
    
    # 1. 노이즈 추가
    noisy_img = add_noise(image)
    noisy_img.save(f"{output_dir}/{category}/images/{base_name}_noise.jpg", 'JPEG')
    save_yolo_labels(f"{output_dir}/{category}/labels/{base_name}_noise.txt", labels)
    
    # 2. 색상 반전
    inverted_img = invert_colors(image)
    inverted_img.save(f"{output_dir}/{category}/images/{base_name}_invert.jpg", 'JPEG')
    save_yolo_labels(f"{output_dir}/{category}/labels/{base_name}_invert.txt", labels)
    
    # 3. 좌우 반전
    flipped_img, flipped_labels = flip_horizontal_and_bbox(image, labels)
    flipped_img.save(f"{output_dir}/{category}/images/{base_name}_flip.jpg", 'JPEG')
    save_yolo_labels(f"{output_dir}/{category}/labels/{base_name}_flip.txt", flipped_labels)

def process_category(input_dir, output_dir, category):
    """특정 카테고리(good/bad)의 모든 이미지 처리"""
    images_dir = f"{input_dir}/{category}/images"
    labels_dir = f"{input_dir}/{category}/labels"
    
    # 폴더가 존재하지 않으면 건너뛰기
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} not found, skipping {category} category")
        return
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Processing {len(image_files)} images in {category} category...")
    
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        print(f"[{i}/{len(image_files)}] Processing {image_file}")
        
        try:
            augment_image(image_path, label_path, output_dir, category)
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

def create_output_folders(output_dir):
    """출력 폴더 구조 생성"""
    folders_to_create = [
        f"{output_dir}",
        f"{output_dir}/good",
        f"{output_dir}/good/images",
        f"{output_dir}/good/labels",
        f"{output_dir}/bad",
        f"{output_dir}/bad/images", 
        f"{output_dir}/bad/labels"
    ]
    
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")



def main():
    print("=== Image Augmentation Tool ===")
    print("This tool will augment images with 3 methods:")
    print("1. Gaussian noise")
    print("2. Color inversion")
    print("3. Horizontal flip")
    print()
    
    # argparse를 사용하여 인자 파싱
    parser = argparse.ArgumentParser(description='Image augmentation tool for YOLO dataset')
    parser.add_argument('folders', nargs='+', help='Input folder names (e.g., 0718 0721)')
    parser.add_argument('--subfolder', nargs='+', help='Specific subfolders to process (e.g., frontdoor hood)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    input_dirs = [os.path.join(base_path, date) for date in args.folders]
    target_subfolders = args.subfolder
    
    # 모든 입력 폴더가 존재하는지 확인
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Error: Folder '{input_dir}' not found.")
            return
    
    print(f"Selected input folders: {', '.join(args.folders)}")
    print(f"절대경로: {', '.join(input_dirs)}")
    if target_subfolders:
        print(f"Target subfolders: {', '.join(target_subfolders)}")
    else:
        print("Processing all subfolders with good/bad structure")
    
    # 모든 폴더에 대해 처리
    total_processed_folders = 0
    total_processed_subfolders = 0
    
    for input_dir in input_dirs:
        print(f"\n{'='*50}")
        print(f"Processing folder: {input_dir}")
        print(f"{'='*50}")
        
        # good/bad 구조를 가진 하위폴더 찾기
        subfolders = []
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                # good 또는 bad 폴더가 있는지 확인
                good_path = os.path.join(item_path, "good")
                bad_path = os.path.join(item_path, "bad")
                
                if os.path.exists(good_path) or os.path.exists(bad_path):
                    subfolders.append(item)
        
        if not subfolders:
            print(f"No subfolders with good/bad structure found in '{input_dir}'")
            print("Please make sure you have subfolders containing 'good' and/or 'bad' subfolders.")
            continue
        
        # 특정 하위폴더만 처리하도록 필터링
        if target_subfolders:
            filtered_subfolders = [sf for sf in subfolders if sf in target_subfolders]
            if not filtered_subfolders:
                print(f"None of the specified subfolders {target_subfolders} found in '{input_dir}'")
                print(f"Available subfolders: {', '.join(subfolders)}")
                continue
            subfolders = filtered_subfolders
        
        print(f"Found {len(subfolders)} subfolders with good/bad structure: {', '.join(subfolders)}")
        
        # 각 하위폴더에 대해 증강 처리
        folder_processed = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_dir, subfolder)
            output_dir = f"{subfolder_path}_aug"
            
            print(f"\n=== Processing subfolder: {subfolder} ===")
            print(f"Output folder will be: {output_dir}")
            
            # 출력 폴더가 이미 존재하는 경우 확인
            if os.path.exists(output_dir):
                response = input(f"Output folder '{output_dir}' already exists. Overwrite? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Skipping {subfolder}")
                    continue
                shutil.rmtree(output_dir)
            
            # 출력 폴더 구조 생성
            print(f"Creating output folder structure...")
            create_output_folders(output_dir)
            
            # classes.txt 파일이 있으면 복사
            classes_file = os.path.join(subfolder_path, "classes.txt")
            if os.path.exists(classes_file):
                shutil.copy(classes_file, f"{output_dir}/classes.txt")
                print(f"Copied classes.txt to output directory")
            
            # good과 bad 카테고리 처리
            categories_processed = 0
            for category in ['good', 'bad']:
                category_path = os.path.join(subfolder_path, category)
                if os.path.exists(category_path):
                    print(f"\n=== Processing {category} category in {subfolder} ===")
                    process_category(subfolder_path, output_dir, category)
                    categories_processed += 1
            
            if categories_processed > 0:
                folder_processed += 1
                total_processed_subfolders += 1
                print(f"Completed augmentation for {subfolder}")
            else:
                print(f"No 'good' or 'bad' folders found in {subfolder}")
        
        if folder_processed > 0:
            total_processed_folders += 1
            print(f"Completed augmentation for folder: {input_dir}")
    
    print(f"\n{'='*50}")
    print(f"=== Image augmentation completed! ===")
    print(f"Processed {total_processed_folders} folders")
    print(f"Processed {total_processed_subfolders} subfolders")
    
    # 결과 통계 출력
    try:
        total_images = 0
        for input_dir in input_dirs:
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    good_path = os.path.join(item_path, "good")
                    bad_path = os.path.join(item_path, "bad")
                    
                    if os.path.exists(good_path) or os.path.exists(bad_path):
                        # 특정 하위폴더만 처리하는 경우 필터링
                        if target_subfolders and item not in target_subfolders:
                            continue
                            
                        output_dir = f"{item_path}_aug"
                        good_count = len([f for f in os.listdir(f"{output_dir}/good/images") if f.endswith('.jpg')]) if os.path.exists(f"{output_dir}/good/images") else 0
                        bad_count = len([f for f in os.listdir(f"{output_dir}/bad/images") if f.endswith('.jpg')]) if os.path.exists(f"{output_dir}/bad/images") else 0
                        total_images += good_count + bad_count
        print(f"Total augmented images: {total_images}")
    except:
        pass

if __name__ == "__main__":
    main() 