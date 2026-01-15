#!/usr/bin/env python3
"""
데이터셋 경로 관련 공통 유틸리티 함수
"""

import os


def get_dataset_path():
    """
    데이터셋 기본 경로를 반환합니다.
    
    우선순위:
    1. 환경 변수 DATASET_PATH
    2. 자동 감지: /workspace/datasets (도커 환경)
    3. 자동 감지: /home/cteam/work/datasets (호스트 환경)
    4. 기본값: /workspace/datasets
    
    Returns:
        str: 데이터셋 기본 경로
    """
    # 환경 변수에서 경로 가져오기
    base_path = os.environ.get('DATASET_PATH')
    if base_path:
        if os.path.exists(base_path):
            return base_path
        else:
            print(f"⚠️  경고: 환경 변수 DATASET_PATH={base_path}가 존재하지 않습니다.")
            print(f"   자동 감지로 전환합니다.")
    
    # 자동 감지: 도커 환경
    if os.path.exists('/workspace/datasets'):
        return "/workspace/datasets"
    
    # 자동 감지: 호스트 환경
    if os.path.exists('/home/cteam/work/datasets'):
        return "/home/cteam/work/datasets"
    
    # 기본값 (도커 경로)
    default_path = "/workspace/datasets"
    print(f"⚠️  경로를 자동 감지하지 못했습니다. 기본값 사용: {default_path}")
    print(f"   환경 변수 DATASET_PATH를 설정하거나 경로가 존재하는지 확인하세요.")
    return default_path


def print_dataset_path(base_path):
    """
    데이터셋 경로를 출력합니다.
    
    Args:
        base_path (str): 데이터셋 기본 경로
    """
    print(f"📁 데이터셋 경로: {base_path}")

