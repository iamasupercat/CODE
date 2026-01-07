#!/usr/bin/env python3
"""
XML 라벨 파일을 YOLO OBB 형식 TXT 파일로 변환하는 스크립트 (roLabelImg로 작업한 라벨을 변환)



# 이 밖의 자세한 사용법은 USAGE.md 파일을 참조하세요.
사용법:
    python xml_to_txt_obb.py --date-range 0718 0718 --subfolder hood --category good bad
"""

import os
import sys
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from collections import defaultdict
import re
import html

# 클래스명 정규화 매핑 (한글/영어 클래스명 -> 표준 영어 클래스명)
CLASS_NAME_MAPPING = {
    '볼트 정측면': 'bolt_frontside',
    '볼트 측면': 'bolt_side',
    '볼트 정면': 'bolt_frontside',
    'sedan (trunklid)': 'sedan',
    'sedan': 'sedan',
    'suv (trunklid)': 'suv',
    'suv': 'suv',
    'hood': 'hood',
    'long (frontfender)': 'long',
    'long': 'long',
    'mid (frontfender)': 'mid',
    'mid': 'mid',
    'short (frontfender)': 'short',
    'short': 'short',
}

# Door 모드 전용 클래스 매핑 (high/mid/low는 Door에서만 사용)
DOOR_CLASS_MAPPING = {
    'high': 0,
    'mid': 1,
    'low': 2,
}

# 일반 클래스 매핑 (영어 클래스 이름 -> ID)
FIXED_CLASS_MAPPING = {
    'bolt_frontside': 0,
    'bolt_side': 1,
    'sedan': 2,  # trunklid
    'suv': 3,  # trunklid
    'hood': 4,
    'long': 5,  # frontfender
    'mid': 6,  # frontfender (Door가 아닐 때)
    'short': 7,  # frontfender
    'bolt_front': 8,
}


def parse_xml_to_yolo_obb(xml_path, class_name_to_id=None, is_door_mode=False):
    """
    XML 파일을 파싱하여 YOLO OBB 형식으로 변환
    
    Args:
        xml_path: XML 파일 경로
        class_name_to_id: 클래스 이름을 ID로 매핑하는 딕셔너리 (None이면 자동 생성)
        is_door_mode: Door 모드 여부 (True면 high=0, mid=1, low=2 사용)
    
    Returns:
        (lines, class_mapping): YOLO 형식 라인 리스트와 클래스 매핑 딕셔너리
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"XML 파싱 오류: {xml_path} - {e}")
        return [], {}
    
    # 이미지 크기 가져오기
    size = root.find('size')
    if size is None:
        print(f"경고: {xml_path}에 size 정보가 없습니다.")
        return [], {}
    
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    if img_width == 0 or img_height == 0:
        print(f"경고: {xml_path}의 이미지 크기가 0입니다.")
        return [], {}
    
    # 클래스 매핑 초기화 (로깅용)
    if class_name_to_id is None:
        class_name_to_id = {}
    
    lines = []
    found_classes = set()
    
    # 모든 object 처리
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        
        class_name = name_elem.text.strip()
        
        # HTML 엔티티 디코딩 (예: &#48380;&#53944; -> 볼트)
        try:
            class_name = html.unescape(class_name)
        except:
            pass
        
        # 괄호와 그 안의 내용 제거 (예: "suv (trunklid)" -> "suv")
        class_name_clean = re.sub(r'\s*\([^)]*\)\s*', '', class_name).strip()
        
        # 클래스명 정규화 (한글/영어 클래스명을 표준 영어 클래스명으로 변환)
        if class_name_clean in CLASS_NAME_MAPPING:
            english_class_name = CLASS_NAME_MAPPING[class_name_clean]
        elif class_name in CLASS_NAME_MAPPING:
            english_class_name = CLASS_NAME_MAPPING[class_name]
        else:
            # 한글 매핑이 없으면 원본 그대로 사용 (이미 영어일 수 있음)
            english_class_name = class_name_clean
        
        # Door 모드인 경우 high/mid/low를 Door 전용 매핑으로 처리
        if is_door_mode and english_class_name in DOOR_CLASS_MAPPING:
            class_id = DOOR_CLASS_MAPPING[english_class_name]
        elif is_door_mode and class_name_clean in DOOR_CLASS_MAPPING:
            class_id = DOOR_CLASS_MAPPING[class_name_clean]
        # 일반 클래스 매핑 사용
        elif english_class_name in FIXED_CLASS_MAPPING:
            class_id = FIXED_CLASS_MAPPING[english_class_name]
        elif class_name_clean in FIXED_CLASS_MAPPING:
            class_id = FIXED_CLASS_MAPPING[class_name_clean]
        elif class_name in FIXED_CLASS_MAPPING:
            class_id = FIXED_CLASS_MAPPING[class_name]
        else:
            # 매핑에 없는 클래스는 경고 후 건너뛰기
            print(f"경고: 알 수 없는 클래스 '{class_name}' (정리 후: '{class_name_clean}', 영어 변환: '{english_class_name}') (파일: {xml_path}). 건너뜁니다.")
            continue
        
        # class_name_to_id에 기록 (로깅용)
        if class_name not in class_name_to_id:
            class_name_to_id[class_name] = class_id
        found_classes.add(class_name)
        
        # robndbox 정보 가져오기
        robndbox = obj.find('robndbox')
        if robndbox is None:
            continue
        
        try:
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w = float(robndbox.find('w').text)
            h = float(robndbox.find('h').text)
            angle = float(robndbox.find('angle').text)
        except (ValueError, AttributeError) as e:
            print(f"경고: {xml_path}의 robndbox 파싱 오류 - {e}")
            continue
        
        # 정규화 (0~1 범위로)
        x_center = cx / img_width
        y_center = cy / img_height
        width = w / img_width
        height = h / img_height
        
        # YOLO OBB 형식으로 변환
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle:.6f}"
        lines.append(line)
    
    return lines, class_name_to_id


def convert_xml_folder(xml_folder_path, output_folder_path=None, class_mapping=None, is_door_mode=False):
    """
    폴더 내의 모든 XML 파일을 TXT로 변환
    
    Args:
        xml_folder_path: XML 파일들이 있는 폴더
        output_folder_path: 출력 폴더 (None이면 XML 폴더와 동일)
        class_mapping: 클래스 매핑 딕셔너리 (None이면 자동 생성)
        is_door_mode: Door 모드 여부 (True면 high=0, mid=1, low=2 사용)
    
    Returns:
        (converted_count, class_mapping): 변환된 파일 수와 클래스 매핑
    """
    if output_folder_path is None:
        output_folder_path = xml_folder_path
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xml_folder_path) if f.endswith('.xml')]
    
    if not xml_files:
        print(f"XML 파일이 없습니다: {xml_folder_path}")
        return 0, class_mapping or {}
    
    converted_count = 0
    all_class_mapping = class_mapping.copy() if class_mapping else {}
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder_path, xml_file)
        txt_file = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_folder_path, txt_file)
        
        lines, updated_mapping = parse_xml_to_yolo_obb(xml_path, all_class_mapping, is_door_mode)
        
        # 클래스 매핑 업데이트
        all_class_mapping.update(updated_mapping)
        
        if lines:
            # TXT 파일 저장
            with open(txt_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            converted_count += 1
        else:
            # 빈 파일 생성 (라벨이 없는 경우)
            with open(txt_path, 'w') as f:
                f.write('')
            converted_count += 1
    
    return converted_count, all_class_mapping


def collect_date_range_folders(base_path: str, start: str, end: str):
    """
    base_path 아래 날짜 폴더 중 start~end 범위(포함)의 절대경로 리스트 반환.
    - 지원 포맷: 4자리(MMDD) 또는 8자리(YYYYMMDD)
    - 입력 길이에 맞는 폴더만 비교 대상으로 포함
    - 일반 폴더와 OBB 폴더 모두 검색
    """
    if not (start.isdigit() and end.isdigit()):
        raise ValueError("date-range는 숫자만 가능합니다. 예: 0715 0805 또는 20240715 20240805")
    if len(start) != len(end) or len(start) not in (4, 8):
        raise ValueError("date-range는 4자리(MMDD) 또는 8자리(YYYYMMDD)로 동일 길이여야 합니다.")

    s_val, e_val = int(start), int(end)
    if s_val > e_val:
        s_val, e_val = e_val, s_val

    found = []
    
    # 일반 폴더 검색
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
    
    # OBB 폴더 검색
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
        pass

    found.sort(key=lambda p: int(os.path.basename(p)))
    return found


def main():
    parser = argparse.ArgumentParser(description='XML 라벨을 YOLO OBB TXT 형식으로 변환')
    parser.add_argument('folder', nargs='?', help='XML 파일이 있는 폴더 경로')
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help='날짜 구간 선택 (MMDD). 예: --date-range 0718 0718')
    parser.add_argument('--subfolder', nargs='+', help='서브폴더 이름 (예: hood frontdoor)')
    parser.add_argument('--category', nargs='+', default=['good', 'bad'],
                        help='카테고리 (good, bad). 기본값: good bad')
    
    args = parser.parse_args()
    
    base_path = "/home/ciw/work/datasets"
    
    if args.folder:
        # 단일 폴더 처리
        xml_folder = args.folder
        if not os.path.exists(xml_folder):
            print(f"오류: 폴더가 존재하지 않습니다: {xml_folder}")
            return
        
        # Door 모드 판단 (경로에 frontdoor가 포함되어 있으면 Door 모드)
        is_door_mode = 'frontdoor' in xml_folder.lower() or 'door' in xml_folder.lower()
        
        print(f"변환 중: {xml_folder}")
        count, mapping = convert_xml_folder(xml_folder, is_door_mode=is_door_mode)
        print(f"변환 완료: {count}개 파일")
        if mapping:
            print("\n클래스 매핑:")
            for name, id in sorted(mapping.items(), key=lambda x: x[1]):
                print(f"  {id}: {name}")
    
    elif args.date_range:
        # 날짜 범위 처리
        start, end = args.date_range
        date_folders = collect_date_range_folders(base_path, start, end)
        
        if not date_folders:
            print(f"날짜 범위 {start}~{end}에 해당하는 폴더가 없습니다.")
            return
        
        print(f"날짜 범위: {start} ~ {end}")
        print(f"처리할 폴더: {len(date_folders)}개\n")
        
        total_converted = 0
        all_mapping = {}
        
        for date_folder in date_folders:
            date_name = os.path.basename(date_folder)
            print(f"=== {date_name} ===")
            
            subfolders = args.subfolder if args.subfolder else []
            
            # 서브폴더 찾기
            if not subfolders:
                for item in os.listdir(date_folder):
                    item_path = os.path.join(date_folder, item)
                    if os.path.isdir(item_path):
                        # good 또는 bad 폴더가 있는지 확인
                        for cat in args.category:
                            cat_path = os.path.join(item_path, cat, 'labels')
                            if os.path.exists(cat_path):
                                subfolders.append(item)
                                break
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(date_folder, subfolder)
                print(f"  서브폴더: {subfolder}")
                
                # Door 모드 판단 (서브폴더 이름에 frontdoor가 포함되어 있으면 Door 모드)
                is_door_mode = 'frontdoor' in subfolder.lower() or 'door' in subfolder.lower()
                
                for cat in args.category:
                    labels_path = os.path.join(subfolder_path, cat, 'labels')
                    if not os.path.exists(labels_path):
                        continue
                    
                    xml_files = [f for f in os.listdir(labels_path) if f.endswith('.xml')]
                    if not xml_files:
                        continue
                    
                    print(f"    {cat}: {len(xml_files)}개 XML 파일")
                    count, mapping = convert_xml_folder(labels_path, labels_path, all_mapping, is_door_mode)
                    all_mapping.update(mapping)
                    total_converted += count
                    print(f"      → {count}개 TXT 파일 생성")
        
        print(f"\n총 변환: {total_converted}개 파일")
        if all_mapping:
            print("\n전체 클래스 매핑:")
            for name, id in sorted(all_mapping.items(), key=lambda x: x[1]):
                print(f"  {id}: {name}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

