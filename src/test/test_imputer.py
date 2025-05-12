import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.get_raw_data import get_raw_data, bin_cloud_height
from src.data.impute_missing import EnsembleImputer, PhysicsImputer, InterpolationImputer, CorrelationImputer, \
    ModelImputer, GPUModelImputer

# 1. 데이터 로드
X_full, station_df = get_raw_data()
X = X_full.drop(columns=['id', 'station_name','date', 'climatology_temp', 'target'])
print("=== data loaded ===")

# 2. 결측치 생성
np.random.seed(42)
mask = np.zeros_like(X.values, dtype=bool)
feature_mask = np.random.rand(X.shape[0], X.shape[1]-1) < 0.1  # station 열 제외한 나머지에 10% 결측치
mask[:, 1:] = feature_mask  # station 열(0번 인덱스)을 제외하고 마스크 적용
np.save(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed\mask.npy', mask)
X_missing = X.copy()
for i, j in zip(*np.where(mask)):
    X_missing.iloc[i, j] = np.nan
print("=== missing data created ===")

# 3. 보간 수행
print("=== imputing data ===")
ensemble_imputer = EnsembleImputer(
    imputers=[
        ("physics", PhysicsImputer(station_info=station_df)),
        ("interp", InterpolationImputer()),
        ("corr", CorrelationImputer()),
        ("model", GPUModelImputer(gpu_id=0, verbose=True))
    ],
    strategy="mean",
    verbose=True
)

X_imputed = ensemble_imputer.fit_transform(X_missing)
X_imputed = bin_cloud_height(X_imputed)

# 4. 보간 결과 저장
output_dir = r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed'

output_file = os.path.join(output_dir, 'imputed_data.csv')
X_imputed.to_csv(output_file, index=False)
print("=== saved imputed data ===")

# 5. 평가 지표 계산
print("=== evaluating imputation ===")
# 원본 데이터에 NaN이 없는 위치만 평가에 포함
original_values = []
imputed_values = []

for i, j in zip(*np.where(mask)):
    original_value = X.iloc[i, j]
    imputed_value = X_imputed.iloc[i, j]

    # 원본 값이 NaN이 아닌 경우만 평가에 포함
    if not pd.isna(original_value) and not pd.isna(imputed_value):
        original_values.append(original_value)
        imputed_values.append(imputed_value)

# 평가 대상 데이터 확인
print(f"평가 대상 데이터 수: {len(original_values)}")

# NaN이 없는지 확인
if len(original_values) > 0:
    mae = mean_absolute_error(original_values, imputed_values)
    rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
else:
    print("평가할 데이터가 없습니다.")

# 6. 문제 원인 파악을 위한 추가 검증
print("\n=== 문제 원인 분석 ===")

# 실제로 결측치가 있었는지 확인
print(f"인위적으로 생성된 결측치 수: {X_missing.isna().sum().sum()}")

# 임퓨팅 데이터에 결측치가 없는지 확인
print(f"임퓨팅된 데이터의 결측치 수: {X_imputed.isna().sum().sum()}")

# 임퓨팅이 제대로 수행되었는지 샘플 검사
missing_positions = np.where(mask)
sample_idx = 10  # 처음 10개 위치만 확인
for k in range(min(sample_idx, len(missing_positions[0]))):
    i, j = missing_positions[0][k], missing_positions[1][k]
    col_name = X.columns[j]
    print(f"위치 ({i},{j}) - 열 '{col_name}':")
    print(f"  원본 값: {X.iloc[i, j]}")
    print(f"  결측치 처리 전: {X_missing.iloc[i, j]}")  # NaN이어야 함
    print(f"  임퓨팅 후: {X_imputed.iloc[i, j]}")
    print()

# 마스크 적용 검증
print("\n=== 마스크 적용 검증 ===")
# 마스크 요약 정보
print(f"마스크 합계 (예상 결측치 수): {np.sum(mask)}")

# 마스크가 True인 위치 샘플 확인
mask_true_count = 0
for i, j in zip(*np.where(mask)):
    if mask_true_count < 5:
        print(f"마스크 True 위치 ({i},{j})의 X_missing 값: {X_missing.iloc[i, j]}")
        print(f"이 값이 NaN인가?: {pd.isna(X_missing.iloc[i, j])}")
    mask_true_count += 1
    if mask_true_count >= 5:
        break

# 가장 중요한 확인: 마스크와 결측치 위치 일치 여부
missing_in_data = X_missing.isna()
mask_match = (mask == missing_in_data.values)
print(f"마스크와 실제 결측치 위치 일치율: {np.mean(mask_match) * 100:.2f}%")