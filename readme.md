https://www.kaggle.com/competitions/next-day-air-temperature-forecast-challenge-2
# Kaggle competetion info

## ❇ 제공되는 입력변수:
(n은 0부터 23까지의 정수로 시간을 의미합니다)

- id: 순서
- station: 지상관측소 번호
- station_name: 지상관측소 이름
- date: 날짜(월-일)
- cloud_cover_n: 증하층운량(10분위)
- dew_point_n: 이슬점 온도(°C)
- humidity_n: 습도(%)
- local_pressure_n: 현지기압(hPa)
- min_cloud_height_n: 최저운고(100m)
- precipitation_n: 강수량(mm)
- sea_level_pressure_n: 해면기압(hPa)
- snow_depth_n: 적설(cm)
- sunshine_duration_n: 일조(hr)
- surface_temp_n: 지면온도(°C)
- vapor_pressure_n: 증기압(hPa)
- visibility_n: 시정(10m)
- wind_speed_n: 풍속(m/s)
- wind_direction_n: 풍향(°)
- climatology_temp: 해당 날짜의 평균 기온(°C) (7년 평균 – 1월 1일인 경우 2018년부터 2024년까지 1월 1일 평균)

### 추가한 파생변수
- next_day_temp: 다음날 평균 온도. climatology_temp+target으로 계산됨.

## ❇ 타겟 변수: 
target: 다음날 평균 기온(°C)에서 climatology_temp를 뺀 값
해당 날짜의 평균적인 기온에 비해 얼마나 기온이 높고 낮았는 지를 예측하는 프로젝트 입니다.

## ❇ 결측치 처리:
- -9999: 관측소 기계에서 감지한 결측치 또는 이상치
- 해당 칸이 비어있을 경우(NaN): 변수에 따라 의미가 달라집니다. snow_depth의 경우 눈이 오지 않았거나 sunshine_duration의 경우 해가 진 시간을 의미할 수 있습니다. 하지만 대부분 수치가 입력되어 있지만 일부분 비어있는 경우 결측치일 가능성이 있습니다.

# Program Info

## ❇ 실행 순서
1. process_dataset.py에서 데이터 전처리 
2. run_base_model.py에서 베이스 모델 학습
3. run_meta_model.py에서 메타 모델 학습
4. run_model.py에서 모델 예측