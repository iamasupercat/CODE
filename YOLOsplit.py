#!/usr/bin/env python3
"""
YOLO 훈련/검증/테스트 split만 생성하는 스크립트
원본 이미지 기준으로 분할하고, 증강 이미지는 train에만 포함합니다.

폴더 구조 예시:
    base_path/
    ├── 0718/
    │   ├── frontfender/
    │   │   ├── bad/
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   ├── good/
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   ├── frontfender_aug/
    │   │   ├── bad/images/
    │   │   └── good/images/
    └── ...

사용법:
    python YOLOsplit.py \
          --folders 0718 0721 \
          --subfolders frontfender hood trunklid \
        --name bolt_obb

    # 날짜 구간으로 선택 (예: 0715부터 0805까지, 해당 범위 폴더 자동 선택)
    python YOLOsplit.py \
        --date-range 0807 0821 \
        --subfolders frontfender hood trunklid \
        --name bolt_obb

결과:
    TXT/train_bolt_obb.txt
    TXT/val_bolt_obb.txt
    TXT/test_bolt_obb.txt
"""

import os
import glob
import random
import argparse
from pathlib import Path
from collections import defaultdict

random.seed(42)

SPLIT_RATIO = [0.7, 0.2, 0.1]  # train, val, test
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def extract_image_id(img_name: str) -> str:
    """이미지명에서 UUID까지 포함한 고유 ID 추출 (bad_XXXX_..._UUID)"""
    aug_suffixes = ['_invert', '_blur', '_bright', '_contrast', '_flip', '_gray', '_noise', '_rot']
    img_name_clean = img_name

    for suffix in aug_suffixes:
        if img_name_clean.endswith(suffix + '.jpg') or img_name_clean.endswith(suffix + '.png'):
            img_name_clean = img_name_clean[:-len(suffix)] + ('.jpg' if img_name_clean.endswith('.jpg') else '.png')
            break

    parts = img_name_clean.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and i + 4 < len(parts):
            if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and
                len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                return '_'.join(parts[:i+5])
    return os.path.splitext(img_name_clean)[0]


def collect_yolo_images_from_folders(base_folders, subfolder_names):
    """YOLO 훈련용 원본/증강 이미지들을 수집"""
    all_images = []

    for base_folder in base_folders:
        if not os.path.isdir(base_folder):
            print(f"기본 폴더가 존재하지 않습니다: {base_folder}")
            continue

        print(f"\n=== {base_folder}에서 YOLO용 이미지 수집 ===")

        for subfolder_name in subfolder_names:
            subfolder_path = os.path.join(base_folder, subfolder_name)

            if not os.path.isdir(subfolder_path):
                print(f"  하위폴더가 존재하지 않습니다: {subfolder_name}")
                continue

            print(f"  하위폴더: {subfolder_name}")

            for quality in ['bad', 'good']:
                quality_path = os.path.join(subfolder_path, quality)
                if not os.path.isdir(quality_path):
                    print(f"    {quality} 폴더가 존재하지 않습니다")
                    continue

                images_path = os.path.join(quality_path, 'images')
                labels_path = os.path.join(quality_path, 'labels')

                if not os.path.isdir(images_path):
                    print(f"    {quality}/images 폴더가 존재하지 않습니다")
                    continue

                # 원본 이미지
                img_files = glob.glob(os.path.join(images_path, '*'))
                img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                print(f"    {quality} 원본 이미지: {len(img_files)}개")

                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    folder_name = os.path.basename(base_folder.rstrip(os.sep))
                    absolute_path = os.path.abspath(img_file)
                    img_info = {
                        'path': absolute_path,
                        'subfolder': subfolder_name,
                        'quality': quality,
                        'base_folder': folder_name,
                        'img_name': img_name,
                        'image_id': extract_image_id(img_name),
                        'is_augmented': False
                    }
                    all_images.append(img_info)

                # 증강 이미지 (예: frontfender_aug/bad/images)
                aug_subfolder_path = os.path.join(base_folder, f"{subfolder_name}_aug", quality, 'images')
                if os.path.isdir(aug_subfolder_path):
                    aug_img_files = glob.glob(os.path.join(aug_subfolder_path, '*'))
                    aug_img_files = [f for f in aug_img_files if os.path.splitext(f)[1].lower() in IMG_EXTS]
                    print(f"    {quality} 증강 이미지: {len(aug_img_files)}개")

                    for img_file in aug_img_files:
                        img_name = os.path.basename(img_file)
                        folder_name = os.path.basename(base_folder.rstrip(os.sep))
                        absolute_path = os.path.abspath(img_file)
                        image_id = extract_image_id(img_name)
                        img_info = {
                            'path': absolute_path,
                            'subfolder': subfolder_name,
                            'quality': quality,
                            'base_folder': folder_name,
                            'img_name': img_name,
                            'image_id': image_id,
                            'is_augmented': True
                        }
                        all_images.append(img_info)

    return all_images


def yolo_stratified_split(yolo_images, ratios):
    """YOLO 이미지를 일관되게 분할 (원본 기준), 증강은 train에만 포함"""
    # 분할 키: base_folder/subfolder/quality/image_id
    yolo_groups = defaultdict(list)
    for img in yolo_images:
        group_key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}"
        yolo_groups[group_key].append(img)

    train_keys, val_keys, test_keys = set(), set(), set()

    for group_key, group_images in yolo_groups.items():
        # 원본만 분할에 사용
        originals = [i for i in group_images if not i.get('is_augmented', False)]
        if not originals:
            continue
        random.shuffle(originals)
        n_total = len(originals)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])
        n_test = n_total - n_train - n_val

        for img in originals[:n_train]:
            key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
            train_keys.add(key)
        for img in originals[n_train:n_train+n_val]:
            key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
            val_keys.add(key)
        for img in originals[n_train+n_val:]:
            key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
            test_keys.add(key)

    print(f"\n=== YOLO 분할 키 ===")
    print(f"train 키: {len(train_keys)}개, val 키: {len(val_keys)}개, test 키: {len(test_keys)}개")

    # 최종 리스트 구성
    yolo_train, yolo_val, yolo_test = [], [], []

    # 원본: 키 기준으로 분배
    for img in [i for i in yolo_images if not i.get('is_augmented', False)]:
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
        if key in train_keys:
            yolo_train.append(img)
        elif key in val_keys:
            yolo_val.append(img)
        elif key in test_keys:
            yolo_test.append(img)

    # 증강: train에만 포함 (원본 train 키와 image_id 동일한 것만)
    aug_to_train = 0
    for img in [i for i in yolo_images if i.get('is_augmented', False)]:
        key = f"{img['base_folder']}/{img['subfolder']}/{img['quality']}/{img['image_id']}"
        if key in train_keys:
            yolo_train.append(img)
            aug_to_train += 1
    print(f"train에 추가된 증강 이미지: {aug_to_train}개")

    random.shuffle(yolo_train)
    random.shuffle(yolo_val)
    random.shuffle(yolo_test)
    return yolo_train, yolo_val, yolo_test


def write_yolo_split_files(yolo_splits, name=''):
    """YOLO 분할 결과를 파일로 저장 (경로만 기록)"""
    txt_dir = Path('TXT')
    txt_dir.mkdir(parents=True, exist_ok=True)

    if name:
        yolo_train_file = txt_dir / f"train_{name}.txt"
        yolo_val_file = txt_dir / f"val_{name}.txt"
        yolo_test_file = txt_dir / f"test_{name}.txt"
    else:
        yolo_train_file = txt_dir / "train.txt"
        yolo_val_file = txt_dir / "val.txt"
        yolo_test_file = txt_dir / "test.txt"

    yolo_train, yolo_val, yolo_test = yolo_splits
    missing = []

    def write_paths(file_path, imgs):
        written = 0
        with open(file_path, 'w') as f:
            for img in imgs:
                p = img['path']
                if os.path.isfile(p):
                    f.write(f"{p}\n")
                    written += 1
                else:
                    missing.append(p)
        return written

    ntr = write_paths(yolo_train_file, yolo_train)
    nva = write_paths(yolo_val_file, yolo_val)
    nts = write_paths(yolo_test_file, yolo_test)

    if missing:
        miss_file = txt_dir / f"missing_yolo_{name if name else 'default'}.txt"
        with open(miss_file, 'w') as mf:
            for p in missing:
                mf.write(p + '\n')

    print(f"\n=== YOLO 분할 결과 ===")
    print(f"  train: {len(yolo_train)}개 (실제 기록: {ntr}) -> {yolo_train_file}")
    print(f"  val:   {len(yolo_val)}개 (실제 기록: {nva}) -> {yolo_val_file}")
    print(f"  test:  {len(yolo_test)}개 (실제 기록: {nts}) -> {yolo_test_file}")


def collect_date_range_folders(base_path: str, start: str, end: str):
    """
    base_path 아래 날짜 폴더 중 start~end 범위(포함)의 절대경로 리스트 반환.
    - 지원 포맷: 4자리(MMDD) 또는 8자리(YYYYMMDD)
    - 입력 길이에 맞는 폴더만 비교 대상으로 포함
    """
    if not (start.isdigit() and end.isdigit()):
        raise ValueError("date-range는 숫자만 가능합니다. 예: 0715 0805 또는 20240715 20240805")
    if len(start) != len(end) or len(start) not in (4, 8):
        raise ValueError("date-range는 4자리(MMDD) 또는 8자리(YYYYMMDD)로 동일 길이여야 합니다.")

    s_val, e_val = int(start), int(end)
    if s_val > e_val:
        s_val, e_val = e_val, s_val

    found = []
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
        return []

    found.sort(key=lambda p: int(os.path.basename(p)))
    return found


def main():
    parser = argparse.ArgumentParser(description='YOLO 훈련/검증/테스트 split만 생성합니다.')
    parser.add_argument('--folders', nargs='+', default=['0616', '0718', '0721', '0725', '0728', '0729', '0731', '0801', '0804', '0805', '0806'],
                        help='분석할 기본 폴더 날짜들')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='날짜 구간 선택 (MMDD 또는 YYYYMMDD). 예: --date-range 0715 0805')
    parser.add_argument('--subfolders', nargs='+', required=True,
                        help='찾을 하위폴더들 (여러 개 가능)')
    parser.add_argument('--name', type=str, default='',
                        help='출력 파일명에 사용할 이름 접미사')
    args = parser.parse_args()

    base_path = "/home/work/datasets"
    if args.date_range:
        start, end = args.date_range
        target_folders = collect_date_range_folders(base_path, start, end)
        print(f"날짜 구간: {start} ~ {end}")
    else:
        target_folders = [os.path.join(base_path, date) for date in args.folders]

    display_dates = [os.path.basename(p) for p in target_folders]
    print(f"분석할 기본 폴더들: {display_dates}")
    print(f"절대경로: {target_folders}")
    print(f"찾을 하위폴더들: {args.subfolders}")
    print(f"출력 파일 이름 접미사: {args.name if args.name else '(기본)'}")

    yolo_images = collect_yolo_images_from_folders(target_folders, args.subfolders)
    if not yolo_images:
        print("수집된 YOLO용 이미지가 없습니다.")
        return
    print(f"\n총 {len(yolo_images)}개 YOLO용 이미지 수집 완료")

    yolo_splits = yolo_stratified_split(yolo_images, SPLIT_RATIO)
    write_yolo_split_files(yolo_splits, args.name)


if __name__ == '__main__':
    main()


