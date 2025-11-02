#!/usr/bin/env python3
import os
from pathlib import Path

# datasets 폴더의 절대 경로
base_dir = Path("/home/work/datasets")

# 0807부터 0923까지의 폴더 리스트
date_folders = [
    "0807", "0808", "0809", "0810", "0811", "0813", "0818", "0819", "0820", "0821",
    "0824", "0825", "0827", "0903", "0904", "0905", "0908", "0909", "0910", "0911",
    "0915", "0917", "0918", "0919", "0922", "0923"
]

# 이미지 확장자
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}

# 모든 이미지 경로 수집
image_paths = []

for date_folder in date_folders:
    frontdoor_bad_dir = base_dir / date_folder / "frontdoor" / "bad"
    
    if frontdoor_bad_dir.exists() and frontdoor_bad_dir.is_dir():
        # frontdoor/bad 디렉토리 내의 모든 파일 확인
        for file_path in frontdoor_bad_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # 절대 경로 추가
                image_paths.append(str(file_path.absolute()))

# 경로 정렬 (일관성을 위해)
image_paths.sort()

# txt 파일로 저장
output_file = base_dir / "CODE" / "frontdoor_bad_images.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    for path in image_paths:
        f.write(path + '\n')

print(f"총 {len(image_paths)}개의 이미지 경로를 {output_file}에 저장했습니다.")

