import pandas as pd
import numpy as np
import os


def add_time_features(df):
    """시간 관련 기본 특성 추가"""
    print("시간 관련 기본 특성 추가 중...")

    # 날짜 변환 (월-일 형식을 datetime으로)
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime('2024-' + df['date'], format='%Y-%m-%d')

    # 기본 시간 특성
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    return df


def add_cyclical_features(df):
    """주기적 시간 특성(Cyclical Features) 추가"""
    print("주기적 시간 특성 추가 중...")

    # 시간에 대한 주기 특성 (24시간)
    for hour in range(24):
        df[f'hour_{hour}_sin'] = np.sin(2 * np.pi * hour / 24)
        df[f'hour_{hour}_cos'] = np.cos(2 * np.pi * hour / 24)

    # 연중 주기 특성 (계절성)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    return df


def add_temperature_features(df):
    """기온 관련 파생 변수 추가"""
    print("기온 관련 파생 변수 추가 중...")

    # 시간별 컬럼 확인
    temps = [f'surface_temp_{i}' for i in range(24) if f'surface_temp_{i}' in df.columns]

    if temps:
        # 일교차: 하루 중 최고-최저 기온
        df['daily_temp_range'] = df[temps].max(axis=1) - df[temps].min(axis=1)

        # 평균 기온
        df['daily_mean_temp'] = df[temps].mean(axis=1)

        # 기온 변동성 (표준편차)
        df['daily_temp_std'] = df[temps].std(axis=1)

        # 주간/야간 기온차 (6-18시를 주간으로 가정)
        day_temps = [f'surface_temp_{i}' for i in range(6, 19) if f'surface_temp_{i}' in df.columns]
        night_temps = [f'surface_temp_{i}' for i in range(24) if f'surface_temp_{i}' in df.columns and i < 6 or i >= 19]

        if day_temps and night_temps:
            df['day_mean_temp'] = df[day_temps].mean(axis=1)
            df['night_mean_temp'] = df[night_temps].mean(axis=1)
            df['day_night_temp_diff'] = df['day_mean_temp'] - df['night_mean_temp']

    return df


def add_wind_features(df):
    """바람 관련 파생 변수 추가"""
    print("바람 관련 파생 변수 추가 중...")

    # 하루 평균 풍속
    wind_speeds = [f'wind_speed_{i}' for i in range(24) if f'wind_speed_{i}' in df.columns]

    if wind_speeds:
        df['daily_mean_wind'] = df[wind_speeds].mean(axis=1)
        df['daily_max_wind'] = df[wind_speeds].max(axis=1)
        df['daily_wind_std'] = df[wind_speeds].std(axis=1)

    # # 풍향을 사인, 코사인 성분으로 분해
    # for hour in range(24):
    #     if f'wind_direction_{hour}' in df.columns and f'wind_speed_{hour}' in df.columns:
    #         # 풍향을 라디안으로 변환
    #         wind_rad = np.radians(df[f'wind_direction_{hour}'])
    #         # 동서방향(u) 및 남북방향(v) 성분
    #         df[f'wind_u_{hour}'] = df[f'wind_speed_{hour}'] * np.sin(wind_rad)
    #         df[f'wind_v_{hour}'] = df[f'wind_speed_{hour}'] * np.cos(wind_rad)

    return df


def add_humidity_features(df):
    """습도 관련 파생 변수 추가"""
    print("습도 관련 파생 변수 추가 중...")

    humidity_cols = [f'humidity_{i}' for i in range(24) if f'humidity_{i}' in df.columns]

    if humidity_cols:
        df['mean_humidity'] = df[humidity_cols].mean(axis=1)
        df['max_humidity'] = df[humidity_cols].max(axis=1)
        df['min_humidity'] = df[humidity_cols].min(axis=1)
        df['humidity_range'] = df['max_humidity'] - df['min_humidity']

    return df


def add_precipitation_features(df):
    """강수 관련 파생 변수 추가"""
    print("강수 관련 파생 변수 추가 중...")

    precip_cols = [f'precipitation_{i}' for i in range(24) if f'precipitation_{i}' in df.columns]

    if precip_cols:
        df['total_precipitation'] = df[precip_cols].sum(axis=1)
        df['has_rain'] = (df['total_precipitation'] > 0).astype(int)
        df['rain_intensity'] = df['total_precipitation'] / (df[precip_cols] > 0).sum(axis=1).replace(0, 1)

    # 눈 관련 변수
    snow_cols = [f'snow_depth_{i}' for i in range(24) if f'snow_depth_{i}' in df.columns]
    if snow_cols and len(snow_cols) > 0:
        df['max_snow_depth'] = df[snow_cols].max(axis=1)
        df['has_snow'] = (df['max_snow_depth'] > 0).astype(int)

    return df


def add_pressure_features(df):
    """기압 관련 파생 변수 추가"""
    print("기압 관련 파생 변수 추가 중...")

    # 해면기압 변수
    sea_pressure_cols = [f'sea_level_pressure_{i}' for i in range(24) if f'sea_level_pressure_{i}' in df.columns]

    if sea_pressure_cols:
        df['mean_sea_pressure'] = df[sea_pressure_cols].mean(axis=1)
        df['sea_pressure_range'] = df[sea_pressure_cols].max(axis=1) - df[sea_pressure_cols].min(axis=1)

        # 기압 변화율 계산
        if len(sea_pressure_cols) >= 2:
            df['sea_pressure_change'] = df[sea_pressure_cols[-1]] - df[sea_pressure_cols[0]]
            df['sea_pressure_trend'] = np.sign(df['sea_pressure_change'])

    # 지역 기압 변수
    local_pressure_cols = [f'local_pressure_{i}' for i in range(24) if f'local_pressure_{i}' in df.columns]

    if local_pressure_cols:
        df['mean_local_pressure'] = df[local_pressure_cols].mean(axis=1)
        df['local_pressure_range'] = df[local_pressure_cols].max(axis=1) - df[local_pressure_cols].min(axis=1)

    return df


def add_cloud_features(df):
    """구름 관련 파생 변수 추가"""
    print("구름 관련 파생 변수 추가 중...")

    # 운량 관련 변수
    cloud_cover_cols = [f'cloud_cover_{i}' for i in range(24) if f'cloud_cover_{i}' in df.columns]

    if cloud_cover_cols:
        df['mean_cloud_cover'] = df[cloud_cover_cols].mean(axis=1)
        df['clear_hours'] = (df[cloud_cover_cols] <= 2).sum(axis=1)  # 맑은 시간 (0-2)
        df['overcast_hours'] = (df[cloud_cover_cols] >= 8).sum(axis=1)  # 흐린 시간 (8-10)

    # 일조량 관련 변수
    sunshine_cols = [f'sunshine_duration_{i}' for i in range(24) if f'sunshine_duration_{i}' in df.columns]

    if sunshine_cols:
        df['total_sunshine'] = df[sunshine_cols].sum(axis=1)

    return df


def add_station_specific_features(df, station_df):
    """관측소별 특화 특성 추가"""
    print("관측소별 특화 특성 추가 중...")

    # 기존 특성 유지
    df['elevation_effect'] = df['station'].map(station_df['노장해발고도(m)']) * df['daily_temp_range']
    df['pressure_elevation_ratio'] = df['mean_sea_pressure'] / df['station'].map(
        station_df['노장해발고도(m)'].replace(0, 0.1))
    df['near_sea'] = df['station'].map(station_df['바다'])
    df['near_mountain'] = df['station'].map(station_df['산'])
    df['urban_effect'] = df['station'].map(station_df['도시']) * df['daily_temp_range']
    df['river_effect'] = df['station'].map(station_df['강']) * df['humidity_range']

    # 지리적 특성 추가
    df['latitude'] = df['station'].map(station_df['위도'])
    df['longitude'] = df['station'].map(station_df['경도'])

    # 위도에 따른 일사량 효과
    df['latitude_sunshine_effect'] = df['latitude'] * df['total_sunshine']

    # 관측장비 높이 관련 특성
    df['temp_sensor_height'] = df['station'].map(station_df['기온계(관측장비지상높이(m))'])
    df['wind_sensor_height'] = df['station'].map(station_df['풍속계(관측장비지상높이(m))'])

    # 풍속 보정 (센서 높이가 높을수록 측정값 증가)
    df['wind_height_corrected'] = df['daily_mean_wind'] * (10 / df['wind_sensor_height']).replace(np.inf, 1)

    # 복합 지형 효과
    # 1. 해안 도시 특성 (해안가 도시는 해륙풍 영향)
    df['coastal_urban'] = df['near_sea'] * df['urban_effect']

    # 2. 산지 기온 효과 (산지는 고도에 따른 기온 감소)
    df['mountain_temp_effect'] = df['near_mountain'] * df['elevation_effect']

    # 3. 계절에 따른 지형 효과 변화
    if 'month' in df.columns:
        # 겨울철(12-2월) 산지 효과 강화
        winter_mask = df['month'].isin([12, 1, 2])
        df['winter_mountain_effect'] = 0
        df.loc[winter_mask, 'winter_mountain_effect'] = df.loc[winter_mask, 'near_mountain'] * 1.5

        # 여름철(6-8월) 해안 효과 강화
        summer_mask = df['month'].isin([6, 7, 8])
        df['summer_sea_effect'] = 0
        df.loc[summer_mask, 'summer_sea_effect'] = df.loc[summer_mask, 'near_sea'] * 1.5

    # 4. 동서남북 위치에 따른 효과
    df['east_west_position'] = (df['longitude'] - 126.5) * 5  # 중앙 경도 기준
    df['north_south_position'] = (df['latitude'] - 37.5) * 5  # 중앙 위도 기준

    # 동서 방향 기단 영향 (동쪽=해양성, 서쪽=대륙성)
    df['continental_influence'] = -df['east_west_position'] * df['day_night_temp_diff']

    return df

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

def create_all_features(df, station_df):
    """모든 파생 변수를 생성하는 함수"""
    print("=== 파생 변수 생성 시작 ===")

    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_temperature_features(df)
    df = add_wind_features(df)
    df = add_humidity_features(df)
    df = add_precipitation_features(df)
    df = add_pressure_features(df)
    df = add_cloud_features(df)
    df = add_station_specific_features(df, station_df)

    print("=== 파생 변수 생성 완료 ===")
    return df

# TODO make_dataset.py에서 실행할 수 있게 수정하기..
def main():
    from src.data.get_raw_data import get_raw_data
    # 데이터 로드
    input_dir = r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed'
    input_file = os.path.join(input_dir, 'imputed_data.csv')
    df = pd.read_csv(input_file)

    _, station_df = get_raw_data()
    station_df = add_terrain_features_to_station_df(station_df)


    print(f"데이터 로드 완료: {df.shape}")

    # 파생 변수 생성
    df_featured = create_all_features(df, station_df)
    print(f"파생 변수 생성 후 크기: {df_featured.shape}") #(13132, 295) -> (13132, 399)

    # 결과 저장
    output_file = os.path.join(input_dir, 'featured_data.csv')
    df_featured.to_csv(output_file, index=False)
    print(f"결과 저장 완료: {output_file}")

if __name__ == "__main__":
    main()