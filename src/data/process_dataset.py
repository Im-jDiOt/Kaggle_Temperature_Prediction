import pandas as pd
import os
import argparse

from src.data.feature_engineering import create_all_features
from src.data.get_raw_data import get_raw_data
from src.data.impute_missing import EnsembleImputer, PhysicsImputer, InterpolationImputer, CorrelationImputer, \
    ModelImputer, GPUModelImputer


def load_data(data_type='train'):
    """데이터 로드 함수"""
    print("=== 데이터 로드 중... ===")
    X_full, station_df = get_raw_data(data_type)

    # 예측에 불필요한 컬럼 분리
    drop_cols = ['id', 'station_name', 'date', 'climatology_temp', 'target', 'next_day_avg_temp']
    drop_cols = [col for col in drop_cols if col in X_full.columns]

    drop_df = X_full[drop_cols].copy() if drop_cols else pd.DataFrame(index=X_full.index)
    X = X_full.drop(columns=drop_cols) if drop_cols else X_full.copy()

    print(f"원본 데이터 크기: {X_full.shape}, 특성 데이터 크기: {X.shape}")
    return X, drop_df, station_df


def impute_data(X, station_df, gpu_id=0):
    """결측치 보간 함수"""
    print("=== 결측치 보간 중... ===")
    ensemble_imputer = EnsembleImputer(
        imputers=[
            ("physics", PhysicsImputer(station_info=station_df)),
            ("interp", InterpolationImputer()),
            ("corr", CorrelationImputer()),
            ("model", GPUModelImputer(gpu_id=gpu_id, verbose=True))
        ],
        strategy="mean",
        verbose=True
    )

    X_imputed = ensemble_imputer.fit_transform(X)
    print(f"보간 완료: {X_imputed.shape}")
    return X_imputed


def engineer_features(X_imputed, drop_df, station_df):
    """피처 엔지니어링 함수"""
    print("=== 피처 엔지니어링 중... ===")

    # 원본 컬럼 복원
    X_final = pd.concat([drop_df, X_imputed], axis=1)

    # 피처 엔지니어링 적용
    df_featured = create_all_features(X_final, station_df)
    print(f"파생 변수 생성 후 크기: {df_featured.shape}")

    return df_featured


def save_data(df, filename, output_dir=None):
    """데이터 저장 함수"""
    if output_dir is None:
        output_dir = r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed'

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    df.to_csv(output_file, index=False)
    print(f"=== 저장 완료: {output_file} ===")

    return output_file


def process_dataset(data_type='train', output_dir=None, save_interim=True, gpu_id=0):
    """전체 데이터 처리 파이프라인"""
    # 1. 데이터 로드
    X, drop_df, station_df = load_data(data_type)

    # 2. 결측치 보간
    X_imputed = impute_data(X, station_df, gpu_id=gpu_id)

    # 3. 중간 결과 저장 (선택적)
    if save_interim:
        X_final = pd.concat([drop_df, X_imputed], axis=1)
        save_data(X_final, f'imputed_{data_type}_data.csv', output_dir)

    # 4. 피처 엔지니어링
    df_featured = engineer_features(X_imputed, drop_df, station_df)

    # 5. 최종 결과 저장
    final_path = save_data(df_featured, f'featured_{data_type}_data.csv', output_dir)

    return df_featured, final_path


def main():
    """명령줄에서 실행할 때 사용하는 메인 함수"""
    parser = argparse.ArgumentParser(description='데이터셋 전처리 및 피처 엔지니어링')
    parser.add_argument('--data_type', type=str, default='train', choices=['train', 'test'],
                        help='처리할 데이터 유형 (train 또는 test)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 경로')
    parser.add_argument('--no_interim', action='store_false', dest='save_interim',
                        help='중간 결과를 저장하지 않음')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='사용할 GPU ID')

    args = parser.parse_args()

    # 데이터 처리 실행
    _, final_path = process_dataset(
        data_type=args.data_type,
        output_dir=args.output_dir,
        save_interim=args.save_interim,
        gpu_id=args.gpu_id
    )

    print(f"데이터 처리 완료: {final_path}")


if __name__ == "__main__":
    main()