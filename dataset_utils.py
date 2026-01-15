#!/usr/bin/env python3
"""
데이터셋 경로를 관리하는 유틸리티 모듈
환경에 따라 적절한 경로를 반환합니다.

우선순위:
1. DATASET_PATH 환경변수
2. 도커 환경 확인 (/.dockerenv 파일 존재)
3. 기본 경로들 확인 (존재하는 경로 사용)
"""

import os
from pathlib import Path


def is_docker_environment():
    """
    도커 환경인지 확인합니다.
    
    Returns:
        bool: 도커 환경이면 True, 아니면 False
    """
    # /.dockerenv 파일 존재 여부 확인
    if os.path.exists('/.dockerenv'):
        return True
    
    # 환경변수로도 확인 (일부 도커 환경)
    if os.environ.get('DOCKER_CONTAINER') == 'true':
        return True
    
    # cgroup을 통한 확인 (선택적)
    try:
        with open('/proc/self/cgroup', 'r') as f:
            if 'docker' in f.read():
                return True
    except (FileNotFoundError, PermissionError):
        pass
    
    return False


def get_dataset_path():
    """
    데이터셋의 기본 경로를 반환합니다.
    환경변수, 도커 환경, 경로 존재 여부를 확인하여 적절한 경로를 반환합니다.
    
    우선순위:
    1. DATASET_PATH 환경변수 (설정되어 있으면 우선 사용)
    2. 도커 환경이면 /workspace/datasets
    3. 로컬 환경이면 /home/cteam/work/datasets
    4. 경로 존재 여부 확인하여 존재하는 경로 사용
    
    Returns:
        str: 데이터셋 기본 경로
    """
    # 1. 환경변수 확인 (최우선)
    env_path = os.environ.get('DATASET_PATH')
    if env_path:
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            return env_path
        else:
            print(f"⚠️  경고: DATASET_PATH 환경변수에 지정된 경로가 존재하지 않습니다: {env_path}")
    
    # 2. 도커 환경 확인
    if is_docker_environment():
        docker_path = "/workspace/datasets"
        if os.path.exists(docker_path):
            return docker_path
        else:
            print(f"⚠️  경고: 도커 환경이지만 경로가 존재하지 않습니다: {docker_path}")
    
    # 3. 로컬 환경 경로들 확인 (존재하는 경로 사용)
    local_paths = [
        "/home/cteam/work/datasets",  # 로컬 환경 우선 경로
        "/workspace/datasets",  # 도커가 아니어도 이 경로가 존재할 수 있음
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            return path
    
    # 4. 기본값 (경로가 존재하지 않아도 반환)
    # 도커 환경이면 도커 경로, 아니면 로컬 경로
    default_path = "/workspace/datasets" if is_docker_environment() else "/home/cteam/work/datasets"
    print(f"⚠️  경고: 기본 경로를 사용합니다 (경로 존재 여부 확인 안 됨): {default_path}")
    return default_path


def print_dataset_path(base_path):
    """
    사용 중인 데이터셋 경로를 출력합니다.
    환경 정보도 함께 출력합니다.
    
    Args:
        base_path (str): 데이터셋 기본 경로
    """
    env_info = []
    
    # 환경변수 확인
    if os.environ.get('DATASET_PATH'):
        env_info.append("환경변수 사용")
    
    # 도커 환경 확인
    if is_docker_environment():
        env_info.append("도커 환경")
    else:
        env_info.append("로컬 환경")
    
    # 경로 존재 여부
    if os.path.exists(base_path):
        env_info.append("경로 존재")
    else:
        env_info.append("⚠️  경로 없음")
    
    env_str = " | ".join(env_info)
    print(f"데이터셋 경로: {base_path} ({env_str})")

