#!/usr/bin/env python3
"""
UnifiedSplit.py로 생성된 YOLO와 DINO용 TXT 파일들의 고유 ID 일치 여부를 검증하는 스크립트

사용법:
    # Bolt 모드
    python verify_unified_split.py --name Bolt

    # Door 모드 (high 영역만)
    python verify_unified_split.py --name Door --area high

    # Door 모드 (모든 영역)
    python verify_unified_split.py --name Door --all-areas

    # 특정 파일 경로 지정
    python verify_unified_split.py \
        --yolo-train TXT/train_Bolt.txt \
        --yolo-val TXT/val_Bolt.txt \
        --yolo-test TXT/test_Bolt.txt \
        --dino-train TXT/train_dino_Bolt.txt \
        --dino-val TXT/val_dino_Bolt.txt \
        --dino-test TXT/test_dino_Bolt.txt
    
    python verify_unified_split.py \
        --yolo-train TXT/train_Door.txt \
        --yolo-val TXT/val_Door.txt \
        --yolo-test TXT/test_Door.txt \
        --dino-train TXT/train_dino_Door_high.txt \
        --dino-val TXT/val_dino_Door_high.txt \
        --dino-test TXT/test_dino_Door_high.txt
    
    python verify_unified_split.py \
        --yolo-train TXT/train_Door.txt \
        --yolo-val TXT/val_Door.txt \
        --yolo-test TXT/test_Door.txt \
        --dino-train TXT/train_dino_Door_midtxt \
        --dino-val TXT/val_dino_Door_mid.txt \
        --dino-test TXT/test_dino_Door_mid.txt
    
    python verify_unified_split.py \
        --yolo-train TXT/train_Door.txt \
        --yolo-val TXT/val_Door.txt \
        --yolo-test TXT/test_Door.txt \
        --dino-train TXT/train_dino_Door_low.txt \
        --dino-val TXT/val_dino_Door_low.txt \
        --dino-test TXT/test_dino_Door_low.txt
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict


def extract_image_id(img_name: str, is_dino: bool = False) -> str:
    """
    이미지명에서 UUID까지 포함한 고유 ID 추출 (UnifiedSplit.py와 동일한 로직)
    
    Args:
        img_name: 이미지 파일명
        is_dino: DINO 파일인 경우 True (크롭 인덱스 제거)
    """
    aug_suffixes = ['_invert', '_blur', '_bright', '_contrast', '_flip', '_gray', '_noise', '_rot']
    img_name_clean = img_name

    for suffix in aug_suffixes:
        for ext in ('.jpg', '.png'):
            full_suffix = suffix + ext
            if img_name_clean.endswith(full_suffix):
                img_name_clean = img_name_clean[:-len(full_suffix)] + ext
                break

    # 확장자 제거
    img_name_without_ext = os.path.splitext(img_name_clean)[0]
    
    # UUID 패턴 찾기 (하이픈 포함 또는 하이픈 없이)
    # 예: e072699c-6ef8-4f72-a172-179f561f3732 또는 8자리_4자리_4자리_4자리_12자리
    uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
    match = re.search(uuid_pattern, img_name_without_ext, re.IGNORECASE)
    
    if match:
        # UUID를 찾았으면 UUID까지 포함한 ID 반환
        uuid_end_pos = match.end()
        base_id = img_name_without_ext[:uuid_end_pos]
        
        # DINO 파일인 경우: UUID 뒤의 크롭 인덱스(_숫자_숫자) 제거
        if is_dino:
            remaining = img_name_without_ext[uuid_end_pos:]
            # _숫자_숫자 패턴 확인 및 제거
            if remaining.startswith('_'):
                remaining_parts = remaining[1:].split('_')  # 첫 번째 _ 제거 후 split
                # 최소 2개 파트가 있고 모두 숫자인지 확인
                if len(remaining_parts) >= 2 and remaining_parts[0].isdigit() and remaining_parts[1].isdigit():
                    # 크롭 인덱스 제거 (이미 base_id에 UUID까지 포함되어 있음)
                    pass
        
        return base_id
    
    # UUID 패턴을 찾지 못한 경우: 기존 로직 사용 (하이픈 없는 형식)
    parts = img_name_without_ext.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and i + 4 < len(parts):
            if (len(parts[i+1]) == 4 and len(parts[i+2]) == 4 and
                len(parts[i+3]) == 4 and len(parts[i+4]) == 12):
                # UUID까지 포함한 ID 추출
                base_id = '_'.join(parts[:i+5])
                
                # DINO 파일인 경우: UUID 뒤의 크롭 인덱스(_숫자_숫자) 제거
                if is_dino and i + 5 < len(parts):
                    remaining_parts = parts[i+5:]
                    # _숫자_숫자 패턴 확인 (예: _0_34)
                    if len(remaining_parts) >= 2 and remaining_parts[0].isdigit() and remaining_parts[1].isdigit():
                        # 크롭 인덱스 제거 (이미 base_id에 UUID까지 포함되어 있음)
                        pass
                
                return base_id
    
    # UUID 패턴을 찾지 못한 경우
    # DINO 파일인 경우: 크롭 인덱스 제거 시도
    if is_dino and len(parts) >= 2:
        # 볼트 크롭: 마지막 두 파트가 숫자인지 확인 (_cls_idx)
        if parts[-1].isdigit() and parts[-2].isdigit():
            # _숫자_숫자 패턴 제거 (예: _0_0)
            return '_'.join(parts[:-2])
        # 도어 크롭: 마지막 파트가 0,1,2 중 하나인지 확인 (_cls)
        elif parts[-1] in ['0', '1', '2']:
            # _숫자 패턴 제거 (예: _0)
            return '_'.join(parts[:-1])
    
    return img_name_without_ext


def extract_image_id_door(img_name: str) -> str:
    """Door 모드: 이미지명에서 원본 이미지 파일명 추출 (UnifiedSplit.py와 동일한 로직)"""
    img_name_without_ext = os.path.splitext(img_name)[0]
    crop_aug_types = ['bright', 'contrast', 'flip', 'gray', 'noise', 'rot']
    
    # 크롭 증강 이미지: 원본파일명_라벨링번호_증강기법
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 3 and parts[-1] in crop_aug_types and parts[-2].isdigit():
            return '_'.join(parts[:-2])
    
    # 크롭 이미지: 원본파일명_라벨링번호 (0, 1, 2)
    if '_' in img_name_without_ext:
        parts = img_name_without_ext.split('_')
        if len(parts) >= 2 and parts[-1] in ['0', '1', '2']:
            return '_'.join(parts[:-1])
    
    return img_name_without_ext


def extract_ids_from_txt(txt_file: Path, is_dino: bool = False, mode: str = 'bolt'):
    """
    TXT 파일에서 고유 ID 추출
    
    Args:
        txt_file: TXT 파일 경로
        is_dino: DINO 파일인지 여부 (DINO는 경로 뒤에 라벨이 있음)
        mode: 'bolt' 또는 'door'
    
    Returns:
        set: 고유 ID 집합
    """
    if not txt_file.exists():
        print(f"⚠️  파일이 존재하지 않습니다: {txt_file}")
        return set()
    
    ids = set()
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # DINO 파일: 경로 + 라벨 형식
            if is_dino:
                parts = line.split()
                if len(parts) < 1:
                    continue
                img_path = parts[0]
            else:
                # YOLO 파일: 경로만
                img_path = line
            
            # 파일명 추출
            img_name = os.path.basename(img_path)
            
            # 고유 ID 추출
            if mode == 'door' and is_dino:
                # Door 모드 DINO: extract_image_id_door 사용 후 원본 이미지 ID로 변환
                crop_id = extract_image_id_door(img_name)
                # crop_id에서 원본 이미지명 추출 (extract_image_id 사용)
                original_id = extract_image_id(crop_id + '.jpg', is_dino=False)
            else:
                # Bolt 모드 또는 YOLO: extract_image_id 사용 (is_dino 플래그 전달)
                original_id = extract_image_id(img_name, is_dino=is_dino)
            
            ids.add(original_id)
    
    return ids


def compare_ids(yolo_ids: set, dino_ids: set, split_name: str):
    """
    YOLO와 DINO의 고유 ID 집합 비교
    
    Returns:
        dict: 비교 결과
    """
    yolo_only = yolo_ids - dino_ids
    dino_only = dino_ids - yolo_ids
    common = yolo_ids & dino_ids
    
    result = {
        'split_name': split_name,
        'yolo_count': len(yolo_ids),
        'dino_count': len(dino_ids),
        'common_count': len(common),
        'yolo_only_count': len(yolo_only),
        'dino_only_count': len(dino_only),
        'yolo_only': sorted(list(yolo_only))[:20],  # 최대 20개만 표시
        'dino_only': sorted(list(dino_only))[:20],  # 최대 20개만 표시
        'is_match': len(yolo_only) == 0 and len(dino_only) == 0
    }
    
    return result


def print_comparison_result(result: dict):
    """비교 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[{result['split_name'].upper()}] 고유 ID 비교 결과")
    print(f"{'='*60}")
    print(f"YOLO 고유 ID 수: {result['yolo_count']}")
    print(f"DINO 고유 ID 수: {result['dino_count']}")
    print(f"공통 고유 ID 수: {result['common_count']}")
    print(f"YOLO에만 있는 ID 수: {result['yolo_only_count']}")
    print(f"DINO에만 있는 ID 수: {result['dino_only_count']}")
    
    if result['is_match']:
        print(f"\n✅ {result['split_name']}: YOLO와 DINO의 고유 ID가 완전히 일치합니다!")
    else:
        print(f"\n❌ {result['split_name']}: YOLO와 DINO의 고유 ID가 일치하지 않습니다.")
        
        if result['yolo_only_count'] > 0:
            print(f"\nYOLO에만 있는 ID (최대 20개):")
            for id_val in result['yolo_only']:
                print(f"  - {id_val}")
            if result['yolo_only_count'] > 20:
                print(f"  ... 외 {result['yolo_only_count'] - 20}개")
        
        if result['dino_only_count'] > 0:
            print(f"\nDINO에만 있는 ID (최대 20개):")
            for id_val in result['dino_only']:
                print(f"  - {id_val}")
            if result['dino_only_count'] > 20:
                print(f"  ... 외 {result['dino_only_count'] - 20}개")


def verify_by_name(name: str, area: str = None, all_areas: bool = False):
    """이름으로 파일을 찾아서 검증"""
    txt_dir = Path('TXT')
    
    if not txt_dir.exists():
        print(f"❌ TXT 디렉토리가 존재하지 않습니다: {txt_dir}")
        return
    
    # YOLO 파일 경로
    yolo_train_file = txt_dir / f'train_{name}.txt'
    yolo_val_file = txt_dir / f'val_{name}.txt'
    yolo_test_file = txt_dir / f'test_{name}.txt'
    
    # Door 모드인지 확인
    mode = 'door' if area or all_areas else 'bolt'
    
    if all_areas:
        # 모든 영역 검증
        areas = ['high', 'mid', 'low']
        all_match = True
        
        for area in areas:
            print(f"\n{'#'*60}")
            print(f"# [{area.upper()} 영역] 검증")
            print(f"{'#'*60}")
            
            dino_train_file = txt_dir / f'train_dino_{name}_{area}.txt'
            dino_val_file = txt_dir / f'val_dino_{name}_{area}.txt'
            dino_test_file = txt_dir / f'test_dino_{name}_{area}.txt'
            
            # YOLO는 high 영역에서만 사용
            if area == 'high':
                match = verify_files(
                    yolo_train_file, yolo_val_file, yolo_test_file,
                    dino_train_file, dino_val_file, dino_test_file,
                    mode=mode
                )
            else:
                # mid/low는 DINO만 검증 (YOLO는 없음)
                match = verify_dino_only(
                    dino_train_file, dino_val_file, dino_test_file,
                    mode=mode
                )
            
            if not match:
                all_match = False
        
        if all_match:
            print(f"\n{'='*60}")
            print("✅ 모든 영역의 고유 ID가 일치합니다!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("❌ 일부 영역에서 고유 ID가 일치하지 않습니다.")
            print(f"{'='*60}")
    else:
        # 단일 영역 검증
        if area:
            dino_train_file = txt_dir / f'train_dino_{name}_{area}.txt'
            dino_val_file = txt_dir / f'val_dino_{name}_{area}.txt'
            dino_test_file = txt_dir / f'test_dino_{name}_{area}.txt'
        else:
            dino_train_file = txt_dir / f'train_dino_{name}.txt'
            dino_val_file = txt_dir / f'val_dino_{name}.txt'
            dino_test_file = txt_dir / f'test_dino_{name}.txt'
        
        verify_files(
            yolo_train_file, yolo_val_file, yolo_test_file,
            dino_train_file, dino_val_file, dino_test_file,
            mode=mode
        )


def verify_dino_only(dino_train_file: Path, dino_val_file: Path, dino_test_file: Path, mode: str = 'bolt'):
    """DINO 파일만 검증 (YOLO 파일이 없는 경우)"""
    print(f"\n[DINO 전용 검증]")
    print(f"Train: {dino_train_file}")
    print(f"Val: {dino_val_file}")
    print(f"Test: {dino_test_file}")
    
    dino_train_ids = extract_ids_from_txt(dino_train_file, is_dino=True, mode=mode)
    dino_val_ids = extract_ids_from_txt(dino_val_file, is_dino=True, mode=mode)
    dino_test_ids = extract_ids_from_txt(dino_test_file, is_dino=True, mode=mode)
    
    # 각 split 간 중복 확인
    train_val_overlap = dino_train_ids & dino_val_ids
    train_test_overlap = dino_train_ids & dino_test_ids
    val_test_overlap = dino_val_ids & dino_test_ids
    
    all_match = True
    
    if train_val_overlap:
        print(f"\n❌ Train과 Val에 중복된 ID가 있습니다: {len(train_val_overlap)}개")
        all_match = False
    if train_test_overlap:
        print(f"\n❌ Train과 Test에 중복된 ID가 있습니다: {len(train_test_overlap)}개")
        all_match = False
    if val_test_overlap:
        print(f"\n❌ Val과 Test에 중복된 ID가 있습니다: {len(val_test_overlap)}개")
        all_match = False
    
    if all_match:
        print(f"\n✅ Train, Val, Test 간 ID가 겹치지 않습니다.")
        print(f"  Train: {len(dino_train_ids)}개")
        print(f"  Val: {len(dino_val_ids)}개")
        print(f"  Test: {len(dino_test_ids)}개")
    
    return all_match


def verify_files(yolo_train_file: Path, yolo_val_file: Path, yolo_test_file: Path,
                 dino_train_file: Path, dino_val_file: Path, dino_test_file: Path,
                 mode: str = 'bolt'):
    """파일 경로를 직접 지정하여 검증"""
    print(f"\n[파일 검증]")
    print(f"YOLO Train: {yolo_train_file}")
    print(f"YOLO Val: {yolo_val_file}")
    print(f"YOLO Test: {yolo_test_file}")
    print(f"DINO Train: {dino_train_file}")
    print(f"DINO Val: {dino_val_file}")
    print(f"DINO Test: {dino_test_file}")
    
    # 고유 ID 추출
    yolo_train_ids = extract_ids_from_txt(yolo_train_file, is_dino=False, mode=mode)
    yolo_val_ids = extract_ids_from_txt(yolo_val_file, is_dino=False, mode=mode)
    yolo_test_ids = extract_ids_from_txt(yolo_test_file, is_dino=False, mode=mode)
    
    dino_train_ids = extract_ids_from_txt(dino_train_file, is_dino=True, mode=mode)
    dino_val_ids = extract_ids_from_txt(dino_val_file, is_dino=True, mode=mode)
    dino_test_ids = extract_ids_from_txt(dino_test_file, is_dino=True, mode=mode)
    
    # 각 split별 비교
    train_result = compare_ids(yolo_train_ids, dino_train_ids, 'train')
    val_result = compare_ids(yolo_val_ids, dino_val_ids, 'val')
    test_result = compare_ids(yolo_test_ids, dino_test_ids, 'test')
    
    # 결과 출력
    print_comparison_result(train_result)
    print_comparison_result(val_result)
    print_comparison_result(test_result)
    
    # 전체 요약
    all_match = train_result['is_match'] and val_result['is_match'] and test_result['is_match']
    
    print(f"\n{'='*60}")
    print("전체 요약")
    print(f"{'='*60}")
    print(f"Train: {'✅ 일치' if train_result['is_match'] else '❌ 불일치'}")
    print(f"Val:   {'✅ 일치' if val_result['is_match'] else '❌ 불일치'}")
    print(f"Test:  {'✅ 일치' if test_result['is_match'] else '❌ 불일치'}")
    
    if all_match:
        print(f"\n✅ 모든 split에서 YOLO와 DINO의 고유 ID가 완전히 일치합니다!")
    else:
        print(f"\n❌ 일부 split에서 고유 ID가 일치하지 않습니다.")
    
    return all_match


def main():
    parser = argparse.ArgumentParser(description='UnifiedSplit.py로 생성된 TXT 파일들의 고유 ID 일치 여부 검증')
    
    # 방법 1: 이름으로 검증
    parser.add_argument('--name', type=str, help='검증할 데이터셋 이름 (예: Bolt, Door)')
    parser.add_argument('--area', type=str, choices=['high', 'mid', 'low'],
                       help='Door 모드: 검증할 영역 (high, mid, low)')
    parser.add_argument('--all-areas', action='store_true',
                       help='Door 모드: 모든 영역 검증')
    
    # 방법 2: 파일 경로 직접 지정
    parser.add_argument('--yolo-train', type=Path, help='YOLO train 파일 경로')
    parser.add_argument('--yolo-val', type=Path, help='YOLO val 파일 경로')
    parser.add_argument('--yolo-test', type=Path, help='YOLO test 파일 경로')
    parser.add_argument('--dino-train', type=Path, help='DINO train 파일 경로')
    parser.add_argument('--dino-val', type=Path, help='DINO val 파일 경로')
    parser.add_argument('--dino-test', type=Path, help='DINO test 파일 경로')
    
    args = parser.parse_args()
    
    # 파일 경로 직접 지정한 경우
    if args.yolo_train and args.yolo_val and args.yolo_test and \
       args.dino_train and args.dino_val and args.dino_test:
        verify_files(
            args.yolo_train, args.yolo_val, args.yolo_test,
            args.dino_train, args.dino_val, args.dino_test,
            mode='door' if args.area else 'bolt'
        )
    # 이름으로 검증
    elif args.name:
        verify_by_name(args.name, args.area, args.all_areas)
    else:
        parser.error("--name 또는 파일 경로들을 모두 지정해야 합니다.")


if __name__ == '__main__':
    main()

