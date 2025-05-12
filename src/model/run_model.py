import pandas as pd
import numpy as np
import os

from src.data.process_dataset import process_dataset
from src.model.run_meta_model import MetaXGBoostAverager

# 데이터 처리
print("1. 데이터 처리 중...")
test_df, output_path = process_dataset(data_type='test', save_interim=True)

# 예측에 불필요한 컬럼 제거
print("2. 예측 데이터 준비 중...")
feature_cols = test_df.columns.tolist()
drop_cols = ['id', 'station', 'station_name', 'date', 'target', 'next_day_avg_temp', 'climatology_temp']
drop_cols = [col for col in drop_cols if col in feature_cols]

# ID 컬럼 보존
id_column = None
if 'id' in test_df.columns:
    id_column = test_df['id'].copy()

# 예측에 사용할 특성 데이터 준비
X_test = test_df.drop(columns=drop_cols) if drop_cols else test_df.copy()
print(f"예측에 사용할 특성 수: {X_test.shape[1]}")

# 메타 모델 로드
print("3. 메타 모델 로드 중...")
model_path = 'models/meta_model.pkl'
if os.path.exists(model_path):
    meta_model = MetaXGBoostAverager.load(model_path)
else:
    print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
    exit(1)

# 예측 수행
print("4. 예측 수행 중...")
predictions = meta_model.predict(X_test)

# 예측 결과 저장
results = pd.DataFrame({'predicted_value': predictions})
if id_column is not None:
    results['id'] = id_column

# CSV 파일로 저장
output_file = 'predictions.csv'
results.to_csv(output_file, index=False)
print(f"✅ 예측 완료: {len(predictions)}개 샘플")
print(f"✅ 결과 저장 완료: {output_file}")

# 예측값 통계 확인
print("\n예측값 통계:")
print(f"평균: {np.mean(predictions):.4f}")
print(f"최소: {np.min(predictions):.4f}")
print(f"최대: {np.max(predictions):.4f}")