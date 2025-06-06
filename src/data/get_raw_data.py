import pandas as pd
import numpy as np
import re

def sort_columns(columns):
    # 고정 prefix가 아닌 컬럼은 먼저 수집
    fixed_columns = [col for col in columns if not re.search(r'_\d+$', col)]

    # 시간 관련 컬럼들 (a_n 형태)
    variable_columns = [col for col in columns if re.search(r'_\d+$', col)]

    # prefix별로 그룹핑
    from collections import defaultdict
    grouped = defaultdict(list)
    for col in variable_columns:
        prefix, num = col.rsplit('_', 1)
        grouped[prefix].append((int(num), col))  # 숫자 기준 정렬을 위해 int 변환

    # 정렬 후 컬럼 순서 재구성
    sorted_columns = fixed_columns.copy()
    for prefix in sorted(grouped.keys()):
        sorted_columns.extend([col for _, col in sorted(grouped[prefix])])

    return sorted_columns

# def insert_next_day_avg_temp(df):
#     # 다음 날 평균 기온 계산
#     df.insert(4, 'next_day_avg_temp', df['target'] + df['climatology_temp'])
#     return df


def sundur_simple_impute(df):
    prefix = 'sunshine_duration_'
    simple_fill_hours = list(range(6)) + [22, 23]

    for i in simple_fill_hours:
        col = f'{prefix}{i}'
        df[col] = df[col].fillna(0)

    return df

def drop_visibility(df):
    visibility_columns = [col for col in df.columns if col.startswith('visibility_')]
    df = df.drop(columns=visibility_columns)
    return df

def drop_wind_direction(df):
    wind_direction_columns = [col for col in df.columns if col.startswith('wind_direction_')]
    df = df.drop(columns=wind_direction_columns)
    return df

def bin_cloud_height(df):
    cloud_height_bins = [-np.inf, 3, 10, 25, 50, np.inf]
    cloud_height_labels = [0,1,2,4,8]

    for i in range(24):
        col = f'min_cloud_height_{i}'
        if col in df.columns:
            # 원본 값 보존 (선택사항)
            # df[f'{col}_original'] = df[col]
            # 구간화 적용
            df[col] = pd.cut(df[col], bins=cloud_height_bins, labels=cloud_height_labels).astype(float)

    return df

# TODO wind direction을 사용할 수도 있음. 성능 개선 필요시..
# def bin_wind_direction(df):
#     direction_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#     direction_labels = ['북', '북동', '동', '남동', '남', '남서', '서', '북서', '북']
#
#     for i in range(24):
#         col = f'wind_direction_{i}'
#         if col in df.columns:
#             # 원본 값 보존
#             df[f'{col}_original'] = df[col]
#             # 구간화 적용
#             df[col] = pd.cut(df[col], bins=direction_bins, labels=direction_labels)
#             df[col] = df[col].astype('category')
#
#     return df

def add_terrain_features_to_station_df(station_df):
    """관측소 데이터프레임에 지형 정보(바다, 산, 도시, 강) 추가"""
    print("관측소 지형 정보 추가 중...")

    # 지형 정보 데이터 정의
    terrain_data = {
        98: {'바다': 0, '산': 1, '도시': 1, '강': 1},
        99: {'바다': 0, '산': 0, '도시': 0, '강': 1},
        201: {'바다': 1, '산': 1, '도시': 0, '강': 0},
        112: {'바다': 1, '산': 0, '도시': 1, '강': 0},
        203: {'바다': 0, '산': 0, '도시': 0, '강': 0},
        202: {'바다': 0, '산': 1, '도시': 1, '강': 1},
        108: {'바다': 0, '산': 1, '도시': 1, '강': 1},
        119: {'바다': 0, '산': 0, '도시': 1, '강': 0}
    }

    # 데이터프레임에 새 컬럼 추가
    for feature in ['바다', '산', '도시', '강']:
        station_df[feature] = station_df.index.map(lambda x: terrain_data.get(x, {}).get(feature, 0))

    return station_df

def get_raw_data(data_type='train'):
    # 데이터 로드
    train_df = pd.read_csv(fr'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\raw\{data_type}_dataset.csv')
    station_df = pd.read_csv(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\raw\station_info.csv')

    train_df = train_df[sort_columns(train_df.columns)]
    # train_df = insert_next_day_avg_temp(train_df)
    train_df = train_df.replace(-9999, np.nan)
    train_df = sundur_simple_impute(train_df)
    train_df = drop_visibility(train_df)
    train_df = drop_wind_direction(train_df)
    train_df = bin_cloud_height(train_df)

    station_df.rename(columns={'지점': 'station'}, inplace=True)
    station_df.set_index('station', inplace=True)
    station_df = add_terrain_features_to_station_df(station_df)
    station_df = station_df[station_df.종료일 != '2019-07-24']

    return train_df, station_df









