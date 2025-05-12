import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.get_raw_data import get_raw_data

# 1. 데이터 로드
X_full, station_df = get_raw_data()
X = X_full.drop(columns=['id', 'station_name', 'date', 'climatology_temp', 'target'])

# 2. 저장된 임퓨팅 데이터와 마스크 불러오기
X_imputed = pd.read_csv(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed\imputed_train_data.csv')
mask = np.load(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed\mask.npy')

# 3. 컬럼별 평가
column_metrics = []

for col_idx, column in enumerate(X.columns):
    # 해당 컬럼에서 마스크가 True인 위치 찾기
    col_mask_indices = np.where(mask[:, col_idx])[0]

    # 원본 값과 임퓨팅된 값 추출
    original_values = []
    imputed_values = []

    for i in col_mask_indices:
        if i < X.shape[0] and i < X_imputed.shape[0]:
            original_value = X.iloc[i, col_idx]
            imputed_value = X_imputed.iloc[i, col_idx]

            # 원본값과 임퓨팅값이 모두 숫자인 경우만 평가에 포함
            if not pd.isna(original_value) and not pd.isna(imputed_value):
                try:
                    # 명시적 숫자 변환 시도
                    float_original = float(original_value)
                    float_imputed = float(imputed_value)
                    original_values.append(float_original)
                    imputed_values.append(float_imputed)
                except (ValueError, TypeError):
                    # 숫자로 변환 불가능한 경우 건너뜀
                    continue

    # 메트릭 계산
    count = len(original_values)
    if count > 0:
        mae = mean_absolute_error(original_values, imputed_values)
        rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
    else:
        mae = float('nan')
        rmse = float('nan')

    column_metrics.append({
        'column': column,
        'count': count,
        'mae': mae,
        'rmse': rmse
    })

# 결과를 DataFrame으로 변환
metrics_df = pd.DataFrame(column_metrics)

# 4. 결과 출력: RMSE 기준으로 정렬
sorted_metrics = metrics_df.sort_values(by='rmse', ascending=False)
valid_metrics = sorted_metrics[~sorted_metrics['mae'].isna()]

print(f"=== 컬럼별 임퓨팅 성능 평가 ===")
print(f"총 컬럼 수: {len(metrics_df)}")
print(f"평가 가능한 컬럼 수: {len(valid_metrics)}")

print("\n=== RMSE 기준 상위 10개 컬럼 (성능이 낮은 순) ===")
print(valid_metrics.head(10)[['column', 'count', 'mae', 'rmse']])

print("\n=== RMSE 기준 하위 10개 컬럼 (성능이 높은 순) ===")
print(valid_metrics.tail(10)[['column', 'count', 'mae', 'rmse']])

# 5. 전체 성능 요약
overall_count = valid_metrics['count'].sum()
weighted_mae = (valid_metrics['mae'] * valid_metrics['count']).sum() / overall_count
weighted_rmse = (valid_metrics['rmse'] * valid_metrics['count']).sum() / overall_count

print("\n=== 전체 성능 요약 ===")
print(f"평가 대상 데이터 수: {overall_count}")
print(f"가중 평균 MAE: {weighted_mae:.4f}")
print(f"가중 평균 RMSE: {weighted_rmse:.4f}")

# 6. 카테고리별 평가 (선택적)
categories = {}
for idx, row in valid_metrics.iterrows():
    col = row['column']
    category = col.split('_')[0]  # 첫 번째 '_' 앞부분을 카테고리로 사용

    if category not in categories:
        categories[category] = {
            'count': 0,
            'mae_sum': 0,
            'rmse_sum': 0
        }

    categories[category]['count'] += row['count']
    categories[category]['mae_sum'] += row['mae'] * row['count']
    categories[category]['rmse_sum'] += row['rmse'] * row['count']