import pandas as pd
import os
from src.data.get_raw_data import get_raw_data
from src.data.preprocess_data import EnsembleImputer, PhysicsImputer, InterpolationImputer, CorrelationImputer, \
    ModelImputer, GPUModelImputer

# 1. 데이터 로드
X_full, station_df = get_raw_data()
X = X_full.drop(columns=['id', 'station_name','date', 'climatology_temp', 'target','next_day_avg_temp'])
print("=== data loaded ===")

# 2. 보간 수행
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

X_imputed = ensemble_imputer.fit_transform(X)

# 3. 원상 복귀
drop_cols = ['id', 'station_name', 'date', 'climatology_temp', 'target', 'next_day_avg_temp']
drop_df = X_full[drop_cols].copy()
X_final = pd.concat([drop_df, X_imputed], axis=1)

# 4. 보간 결과 저장
output_dir = r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed'
output_file = os.path.join(output_dir, 'imputed_data.csv')
X_final.to_csv(output_file, index=False)
print("=== saved imputed data ===")
