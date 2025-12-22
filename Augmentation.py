'''
원본이미지 증강: noise, invert, flip
이미 aug가 있는 경우: 덮어씌우기 (image, txt 모두)
기본 경로와 OBB 폴더 모두 검색됨


# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python augmentation.py --date-range 0616 1109 \
    --subfolder frontdoor





폴더 구조:
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
import math

def collect_date_range_folders(base_path: str, start: str, end: str, include_obb: bool = True):
    """
    base_path 아래 날짜 폴더 중 start~end 범위(포함)의 절대경로 리스트 반환.
    - 지원 포맷: 4자리(MMDD) 또는 8자리(YYYYMMDD)
    - 입력 길이에 맞는 폴더만 비교 대상으로 포함
    - include_obb가 True이면 OBB 폴더도 함께 검색
    """
    if not (start.isdigit() and end.isdigit()):
        raise ValueError("date-range는 숫자만 가능합니다. 예: 0715 0805 또는 20240715 20240805")
    if len(start) != len(end) or len(start) not in (4, 8):
        raise ValueError("date-range는 4자리(MMDD) 또는 8자리(YYYYMMDD)로 동일 길이여야 합니다.")

    s_val, e_val = int(start), int(end)
    if s_val > e_val:
        s_val, e_val = e_val, s_val

    found = []
    
    # 기본 경로 검색
    try:
        for name in os.listdir(base_path):
            full = os.path.join(base_path, name)
            if not os.path.isdir(full):
                continue
            if not (name.isdigit() and len(name) == len(start)):
                continue
            val = int(name)
            if s_val <= val <= e_val:
                found.append(os.path.abspath(full))
    except FileNotFoundError:
        print(f"기본 경로가 존재하지 않습니다: {base_path}")
    
    # OBB 폴더도 검색
    if include_obb:
        obb_path = os.path.join(base_path, "OBB")
        try:
            if os.path.exists(obb_path):
                for name in os.listdir(obb_path):
                    full = os.path.join(obb_path, name)
                    if not os.path.isdir(full):
                        continue
                    if not (name.isdigit() and len(name) == len(start)):
                        continue
                    val = int(name)
                    if s_val <= val <= e_val:
                        found.append(os.path.abspath(full))
        except FileNotFoundError:
            pass  # OBB 폴더가 없어도 무시

    # 중복 제거 후 정렬
    found = list(set(found))
    found.sort(key=lambda p: int(os.path.basename(p)))
    return found


def load_yolo_labels_raw(label_path):
    """YOLO 라벨 파일을 그대로 읽어서 반환 (파싱하지 않음)"""
    lines = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                lines.append(line.rstrip('\n\r'))
    return lines

def save_yolo_labels_raw(label_path, lines):
    """YOLO 라벨을 그대로 저장"""
    with open(label_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def convert_to_obb_format(lines):
    """라벨을 OBB 형식(6개 값)으로 변환: 5개 값이면 0 추가, 6개 값이면 그대로 유지"""
    obb_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            obb_lines.append('')
            continue
        
        parts = line.split()
        if len(parts) == 5:
            # 5개 값이면 0 추가 (BB → OBB)
            obb_lines.append(' '.join(parts) + ' 0')
        elif len(parts) >= 6:
            # 이미 6개 이상이면 그대로 유지
            obb_lines.append(line)
        else:
            # 5개 미만이면 그대로 유지
            obb_lines.append(line)
    return obb_lines

def load_yolo_labels_for_flip(label_path):
    """flip을 위한 라벨 파싱 (5개 또는 6개 값 모두 지원)"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        # 6개 값이 있으면 추가 값도 저장
                        extra = parts[5:] if len(parts) > 5 else []
                        labels.append([class_id, x_center, y_center, width, height, extra])
    return labels

def save_yolo_labels_for_flip(label_path, labels):
    """flip 변환된 라벨 저장 (5개 또는 6개 값 모두 지원)"""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id, x_center, y_center, width, height, extra = label
            if extra:
                # 6개 값인 경우
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {' '.join(extra)}\n")
            else:
                # 5개 값인 경우
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def flip_horizontal_and_bbox(image, labels):
    """이미지를 좌우 반전시키고 바운딩 박스 좌표를 변환"""
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    new_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height, extra = label
        # x 좌표만 반전 (1 - x_center)
        new_x_center = 1.0 - x_center
        # 각도(6번째 값)가 있는 OBB의 경우, 좌우 반전 시 각도 보정: angle' = pi - angle
        new_extra = extra
        if isinstance(extra, list) and len(extra) >= 1:
            try:
                angle = float(extra[0])
                new_angle = math.pi - angle
                # 문자열로 다시 저장하며, 나머지 추가 필드는 보존
                new_extra = [f"{new_angle:.6f}"] + extra[1:]
            except Exception:
                # 각도 파싱 실패 시 원본 유지
                new_extra = extra
        new_labels.append([class_id, new_x_center, y_center, width, height, new_extra])
    
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

def detect_label_format(label_lines):
    """라벨 형식 자동 감지: OBB(6개 값) 또는 BB(5개 값)"""
    has_obb = False
    has_bb = False
    for line in label_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 6:
            has_obb = True
        elif len(parts) == 5:
            has_bb = True
    # OBB가 하나라도 있으면 OBB 형식으로 간주
    return 'obb' if has_obb else 'bb'


def augment_image(image_path, label_path, output_dir, category):
    """단일 이미지에 대해 모든 증강 적용
    
    Args:
        image_path: 원본 이미지 경로
        label_path: 원본 라벨 경로
        output_dir: 출력 디렉토리
        category: 카테고리 (good/bad)
    """
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
    label_lines = load_yolo_labels_raw(label_path)
    
    # 라벨 형식 자동 감지 (OBB 또는 BB)
    label_format = detect_label_format(label_lines)
    
    # 파일명에서 확장자 제거
    base_name = Path(image_path).stem
    
    # 1. 노이즈 추가 - 라벨 그대로 복사
    noisy_img = add_noise(image)
    noisy_img.save(f"{output_dir}/{category}/images/{base_name}_noise.jpg", 'JPEG')
    save_yolo_labels_raw(f"{output_dir}/{category}/labels/{base_name}_noise.txt", label_lines)
    
    # 2. 색상 반전 - 라벨 그대로 복사
    inverted_img = invert_colors(image)
    inverted_img.save(f"{output_dir}/{category}/images/{base_name}_invert.jpg", 'JPEG')
    save_yolo_labels_raw(f"{output_dir}/{category}/labels/{base_name}_invert.txt", label_lines)
    
    # 3. 좌우 반전 - x 좌표만 변환
    # 라벨 형식에 따라 flip 처리 (OBB면 각도 보정, BB면 각도 보정 없음)
    labels_for_flip = []
    for line in label_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            # OBB 형식이면 extra(각도) 포함, BB 형식이면 빈 리스트
            extra = parts[5:] if len(parts) > 5 else []
            labels_for_flip.append([class_id, x_center, y_center, width, height, extra])
    
    flipped_img, flipped_labels = flip_horizontal_and_bbox(image, labels_for_flip)
    flipped_img.save(f"{output_dir}/{category}/images/{base_name}_flip.jpg", 'JPEG')
    save_yolo_labels_for_flip(f"{output_dir}/{category}/labels/{base_name}_flip.txt", flipped_labels)

def process_category(input_dir, output_dir, category):
    """특정 카테고리(good/bad)의 모든 이미지 처리
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        category: 카테고리 (good/bad)
    """
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
    parser.add_argument('folders', nargs='*', help='Input folder names (e.g., 0718 0721)')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='날짜 구간 선택 (MMDD 또는 YYYYMMDD). 예: --date-range 0715 0805')
    parser.add_argument('--subfolder', nargs='+', help='Specific subfolders to process (e.g., frontdoor hood)')
    parser.add_argument('--obb-only', action='store_true', 
                        help='OBB 폴더만 처리 (기본 경로는 제외하고 /home/work/datasets/OBB/ 아래만 검색)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    if args.date_range:
        start, end = args.date_range
        if args.obb_only:
            # OBB 폴더만 검색
            input_dirs = collect_date_range_folders(base_path, start, end, include_obb=True)
            # 기본 경로 결과 제거 (OBB 경로만 남김)
            input_dirs = [d for d in input_dirs if "OBB" in d]
        else:
            # 기본 경로와 OBB 경로 모두 검색
            input_dirs = collect_date_range_folders(base_path, start, end, include_obb=True)
        print(f"날짜 구간: {start} ~ {end}")
    else:
        # 직접 폴더 지정 시
        input_dirs = []
        for date in args.folders:
            if args.obb_only:
                # OBB 경로만 확인
                obb_path = os.path.join(base_path, "OBB", date)
                if os.path.exists(obb_path):
                    input_dirs.append(os.path.abspath(obb_path))
            else:
                # 기본 경로와 OBB 경로 모두 확인
                default_path = os.path.join(base_path, date)
                if os.path.exists(default_path):
                    input_dirs.append(os.path.abspath(default_path))
                obb_path = os.path.join(base_path, "OBB", date)
                if os.path.exists(obb_path):
                    input_dirs.append(os.path.abspath(obb_path))
    target_subfolders = args.subfolder
    
    # 모든 입력 폴더가 존재하는지 확인
    if not input_dirs:
        print("Error: No valid folders found.")
        return
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Folder '{input_dir}' not found, skipping...")
            continue
    
    # 존재하는 폴더만 필터링
    input_dirs = [d for d in input_dirs if os.path.exists(d)]
    
    if not input_dirs:
        print("Error: No valid folders found.")
        return
    
    display_dates = [os.path.basename(p) for p in input_dirs]
    print(f"Selected input folders: {', '.join(display_dates)}")
    print(f"절대경로: {', '.join(input_dirs)}")
    
    # OBB 폴더 포함 여부 표시
    if args.obb_only:
        print(f"모드: OBB 폴더만 처리 ({len(input_dirs)}개)")
    else:
        obb_folders = [d for d in input_dirs if "OBB" in d]
        if obb_folders:
            print(f"OBB 폴더 포함: {len(obb_folders)}개")
    if target_subfolders:
        print(f"Target subfolders: {', '.join(target_subfolders)}")
    else:
        print("Processing all subfolders with good/bad structure")
    print("라벨 형식 자동 감지: 원본이 OBB 형식(6개 값)이면 OBB로, BB 형식(5개 값)이면 BB로 처리합니다.")
    
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
            
            # 출력 폴더가 이미 존재하는 경우 무조건 덮어쓰기
            if os.path.exists(output_dir):
                print(f"Output folder '{output_dir}' already exists. Overwriting...")
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