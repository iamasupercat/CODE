#!/usr/bin/env python3
"""
YOLO 라벨을 사용하여 이미지에서 객체를 크롭하는 스크립트
라벨 번호 0,1,2에 대해 크롭 수행
라벨 번호) 0: high 1: mid 2: low
영역별 크롭 후 각각 crop_high, crop_mid, crop_low 폴더에 저장
크롭이미지 파일명 규칙) 원본이미지파일명_라벨번호.jpg
high: 원본이미지파일명_0.jpg
mid: 원본이미지파일명_1.jpg
low: 원본이미지파일명_2.jpg

엑셀파일을 읽어와 각 영역별 클래스 번호를 찾아 각 영역별 폴더에 0,1,2,3 하위 폴더 생성
엑셀파일 영역) '상단': high, '중간': mid, '하단': low
엑셀파일 클래스번호) 1: 실링없음 2: 작업실링 3: 출고실링(양품) 4: 테이프실링
하위폴더 번호) 0: 출고실링(양품) 1: 실링없음 2: 작업실링 3: 테이프실링




사용법:
    python sealing_crop_for_resnet.py \
        --target_dir 0616 0721 0728 0729 0731 0801 0804 0805 0806 \
        --subfolders frontdoor \
        --set_types good bad

    python sealing_crop_for_resnet.py \
        --target_dir 0616 \
        --subfolders frontdoor \
        --set_types good bad


옵션 설명:
    --target_dir: 대상 폴더 경로들 (기본값: 현재 디렉토리)
    --subfolders: 처리할 서브폴더들 (기본값: frontdoor)
    --set_types: 처리할 set 타입들 (기본값: bad good)

폴더 구조:
    datasets/
    ├── 0616/
    │   ├── frontdoor/
    │   │   ├── good/
    │   │   │   ├── crop_high/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_mid/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_low/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_high_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_mid_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_low_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   ├── bad/
    │   │   │   ├── crop_high/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_mid/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_low/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_high_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_mid_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── crop_low_aug/
    │   │   │   │   ├── 0/ (good)
    │   │   │   │   ├── 1/ (bad_NoSealing)
    │   │   │   │   ├── 2/ (bad_SealingDiffers)
    │   │   │   │   └── 3/ (bad_TapeSealing)
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   ├── Y/
    │   │   └── frontdoor.xlsx


"""

import os
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

def yolo_to_bbox(yolo_line, img_width, img_height, img_name=None, line_num=None):
    parts = yolo_line.strip().split()
    cls = int(parts[0])
    vals = list(map(float, parts[1:]))
    # 값이 0~1 범위가 아니면 경고
    for i, v in enumerate(vals):
        if not (0.0 <= v <= 1.0):
            print(f"[경고] {img_name or ''} 라벨 {line_num or ''} 값 {v}가 0~1 범위가 아님: {yolo_line.strip()}")
    x_center = vals[0] * img_width
    y_center = vals[1] * img_height
    w = vals[2] * img_width
    h = vals[3] * img_height
    x1 = int(round(x_center - w / 2))
    y1 = int(round(y_center - h / 2))
    x2 = int(round(x_center + w / 2))
    y2 = int(round(y_center + h / 2))
    # 이미지 경계로 clipping
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return cls, x1, y1, x2, y2

def load_excel_data(excel_path):
    """엑셀 파일에서 이미지별 영역별 클래스 정보를 로드하는 함수"""
    try:
        df = pd.read_excel(excel_path)
        # 이미지파일명과 영역별 클래스 번호 매핑 생성
        class_mapping = {}
        
        # quality가 bad인 행들을 먼저 처리
        bad_rows = df[df['quality'] == 'bad'].copy()
        
        for _, row in bad_rows.iterrows():
            img_name = row['이미지파일명']
            vehicle_number = row['차량번호']
            
            # 상단, 중간, 하단 정보 확인
            high_class = row['상단'] if pd.notna(row['상단']) else None
            mid_class = row['중간'] if pd.notna(row['중간']) else None
            low_class = row['하단'] if pd.notna(row['하단']) else None
            
            # 정보가 없는 경우, 같은 차량번호이고 라벨링여부가 Y인 행 찾기
            if high_class is None or mid_class is None or low_class is None:
                # 같은 차량번호이고 라벨링여부가 Y인 행 찾기
                matching_row = df[(df['차량번호'] == vehicle_number) & 
                                (df['라벨링여부'].str.contains('Y', na=False))]
                
                if not matching_row.empty:
                    ref_row = matching_row.iloc[0]  # 첫 번째 매칭 행 사용
                    print(f"  {img_name}: bad 행의 정보가 없어서 Y 행의 정보 사용")
                    
                    # 기존 값이 None인 경우에만 참조 행의 값 사용
                    if high_class is None:
                        high_class = ref_row['상단'] if pd.notna(ref_row['상단']) else 3
                    if mid_class is None:
                        mid_class = ref_row['중간'] if pd.notna(ref_row['중간']) else 3
                    if low_class is None:
                        low_class = ref_row['하단'] if pd.notna(ref_row['하단']) else 3
                else:
                    # 매칭되는 Y 행이 없는 경우 기본값 사용
                    print(f"  {img_name}: 매칭되는 Y 행이 없어서 기본값 사용")
                    high_class = 3 if high_class is None else high_class
                    mid_class = 3 if mid_class is None else mid_class
                    low_class = 3 if low_class is None else low_class
            
            class_mapping[img_name] = {
                'high': high_class,
                'mid': mid_class,
                'low': low_class
            }
        
        # quality가 good인 행들도 처리
        good_rows = df[df['quality'] == 'good'].copy()
        for _, row in good_rows.iterrows():
            img_name = row['이미지파일명']
            high_class = row['상단'] if pd.notna(row['상단']) else 3
            mid_class = row['중간'] if pd.notna(row['중간']) else 3
            low_class = row['하단'] if pd.notna(row['하단']) else 3
            
            class_mapping[img_name] = {
                'high': high_class,
                'mid': mid_class,
                'low': low_class
            }
        
        return class_mapping
    except Exception as e:
        print(f"엑셀 파일 로드 실패: {e}")
        return {}

def get_class_folder(class_num):
    """클래스 번호를 폴더 번호로 변환하는 함수"""
    # 엑셀파일 클래스번호) 1: 실링없음 2: 작업실링 3: 출고실링(양품) 4: 테이프실링
    # 하위폴더 번호) 0: 출고실링(양품) 1: 실링없음 2: 작업실링 3: 테이프실링
    mapping = {
        1: 1,  # 실링없음 -> 1
        2: 2,  # 작업실링 -> 2
        3: 0,  # 출고실링(양품) -> 0
        4: 3   # 테이프실링 -> 3
    }
    return mapping.get(class_num, 0)

def process_set(set_type, excel_mapping=None):
    base_dir = set_type
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    
    # 영역별 크롭 폴더 생성
    crop_areas = ['crop_high', 'crop_mid', 'crop_low']
    crop_dirs = {}
    
    for area in crop_areas:
        area_dir = os.path.join(base_dir, area)
        os.makedirs(area_dir, exist_ok=True)
        # 0, 1, 2, 3 하위 폴더 생성
        for i in range(4):
            sub_dir = os.path.join(area_dir, str(i))
            os.makedirs(sub_dir, exist_ok=True)
        crop_dirs[area] = area_dir
    
    debug_dir = os.path.join(base_dir, 'debug_crop')
    os.makedirs(debug_dir, exist_ok=True)
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(label_path):
            continue
        
        # 엑셀에서 영역별 클래스 정보 가져오기
        default_class = 3  # 기본값: 출고실링(양품)
        high_class = default_class
        mid_class = default_class
        low_class = default_class
        
        if excel_mapping and img_name in excel_mapping:
            high_class = excel_mapping[img_name].get('high', default_class)
            mid_class = excel_mapping[img_name].get('mid', default_class)
            low_class = excel_mapping[img_name].get('low', default_class)
        
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # EXIF 회전 보정 추가
        img_width, img_height = img.size
        
        # 디버그용 복사본
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        
        with open(label_path, 'r') as f:
            for idx, line in enumerate(f):  # 라벨 파일의 각 라인(각 박스)마다 반복
                if not line.strip():
                    continue
                
                cls, x1, y1, x2, y2 = yolo_to_bbox(line, img_width, img_height, img_name, idx+1)
                
                # 라벨 번호에 따라 영역별 크롭
                area_name = None
                class_num = None
                if cls == 0:
                    area_name = 'crop_high'
                    class_num = high_class
                elif cls == 1:
                    area_name = 'crop_mid'
                    class_num = mid_class
                elif cls == 2:
                    area_name = 'crop_low'
                    class_num = low_class
                else:
                    continue  # 0, 1, 2가 아닌 경우 건너뛰기
                
                # 유효한 crop 영역만 저장
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    save_name = f"{os.path.splitext(img_name)[0]}_{cls}.jpg"
                    
                    # 해당 영역의 클래스 폴더에 저장
                    class_folder = get_class_folder(class_num)
                    save_dir = os.path.join(crop_dirs[area_name], str(class_folder))
                    save_path = os.path.join(save_dir, save_name)
                    crop.save(save_path)
                    
                    # 디버그용 박스 그리기
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 디버그 이미지 저장
        debug_save_path = os.path.join(debug_dir, img_name)
        debug_img.save(debug_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO 라벨을 사용하여 이미지에서 객체를 크롭합니다.')
    parser.add_argument('--target_dir', nargs='+', default=['0616', '0721', '0728', '0729', '0731', '0801', '0804', '0805', '0806'], 
                       help='대상 폴더 날짜들 (기본값: 0616 0721 0728 0729 0731 0801 0804 0805 0806)')
    parser.add_argument('--subfolders', nargs='+', 
                       default=['frontdoor'],
                       help='처리할 서브폴더들 (기본값: frontdoor)')
    parser.add_argument('--set_types', nargs='+', 
                       default=['bad', 'good'],
                       help='처리할 set 타입들 (기본값: bad good)')
    
    args = parser.parse_args()
    
    # 날짜를 절대경로로 변환
    base_path = "/home/work/datasets"
    target_dirs = [os.path.join(base_path, date) for date in args.target_dir]
    
    print(f"대상 폴더들: {args.target_dir}")
    print(f"절대경로: {target_dirs}")
    print(f"처리할 서브폴더들: {args.subfolders}")
    print(f"처리할 set 타입들: {args.set_types}")
    
    for target_dir in target_dirs:
        print(f"\n처리 중인 대상 폴더: {target_dir}")
        
        for part in args.subfolders:
            # 엑셀 파일 경로 - frontdoor.xlsx 파일 읽기
            excel_path = os.path.join(target_dir, part, "frontdoor.xlsx")
            excel_mapping = {}
            
            if os.path.exists(excel_path):
                print(f"엑셀 파일 로드 중: {excel_path}")
                excel_mapping = load_excel_data(excel_path)
                print(f"로드된 클래스 매핑: {len(excel_mapping)}개")
            else:
                print(f"엑셀 파일이 존재하지 않음: {excel_path}")
            
            for set_type in args.set_types:
                base_path = os.path.join(target_dir, part, set_type)
                if os.path.exists(base_path):
                    print(f"처리 중: {base_path}")
                    process_set(base_path, excel_mapping)
                else:
                    print(f"경로가 존재하지 않음: {base_path}")
