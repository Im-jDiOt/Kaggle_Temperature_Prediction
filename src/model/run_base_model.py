# run_base_model.py 수정
import os
import pandas as pd
import joblib
import time
import numpy as np

from src.model.base_model import (
    train_lightgbm,
    train_xgboost,
    train_randomforest,
    train_elasticnet,
    train_cnn,
    AverageEnsembleRegressor
)


def load_data():
    print(f"[데이터 로드 시작]")
    start_time = time.time()
    df = pd.read_csv(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed\featured_data.csv', index_col=0)
    df.drop(columns=['station_name', 'date'], inplace=True)
    print(f"✅ 데이터 로드 완료: {df.shape[0]}개 샘플, {df.shape[1]}개 특성 ({time.time() - start_time:.2f}초)")
    return df


def main():
    total_start = time.time()
    df = load_data()
    stations = df['station'].unique()
    stations_str = [str(int(st)) for st in stations]
    print(f"\n[처리 대상 관측소] 총 {len(stations)}개: {', '.join(stations_str)}")

    for i, st in enumerate(stations):
        st_str = str(int(st))  # 문자열로 변환
        station_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"[관측소 {i + 1}/{len(stations)}] {st} 처리 시작 ({(i + 1) / len(stations) * 100:.1f}%)")
        print(f"{'=' * 50}")

        df_st = df[df['station'] == st].drop(columns=['station'])
        X = df_st.drop(columns=['target'])
        y = df_st['target']
        print(f"관측소 {st} 데이터: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")

        # 문자열로 변환한 station 값 사용
        model_dir = os.path.join('models', st_str)
        os.makedirs(model_dir, exist_ok=True)
        print(f"모델 저장 디렉토리: {model_dir}")

        # 아래 코드에서도 모두 st_str 사용
        base_models = [
            train_lightgbm(X, y, save_path=os.path.join(model_dir, f'lightgbm_{st_str}.pkl')),
            train_xgboost(X, y, save_path=os.path.join(model_dir, f'xgboost_{st_str}.pkl')),
            train_randomforest(X, y, save_path=os.path.join(model_dir, f'random_forest_{st_str}.pkl')),
            train_elasticnet(X, y, save_path=os.path.join(model_dir, f'elasticnet_{st_str}.pkl')),
            # train_cnn(X.values, y.values, save_path=os.path.join(model_dir, f'cnn_regressor_{st_str}.pt'))
        ]
        print(f"\n✅ 5개 Base 모델 학습 완료")

        # 2) 앙상블 모델 생성 및 저장
        print(f"\n[앙상블 모델 생성]")
        ensemble = AverageEnsembleRegressor(base_models=base_models)
        ensemble_path = os.path.join(model_dir, f'ensemble_{st_str}.pkl')
        joblib.dump(ensemble, ensemble_path)
        print(f"✅ 앙상블 모델 저장: {ensemble_path}")

        station_elapsed = time.time() - station_start
        print(f"\n✅ 관측소 {st} 처리 완료 (소요시간: {station_elapsed:.1f}초)")

        # 추정 남은 시간 계산
        if i < len(stations) - 1:
            est_remaining = station_elapsed * (len(stations) - (i + 1))
            remaining_min = int(est_remaining // 60)
            remaining_sec = int(est_remaining % 60)
            print(f"⏱️ 예상 남은 시간: {remaining_min}분 {remaining_sec}초")

    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)
    print(f"\n{'=' * 50}")
    print(f"✅ 전체 학습 완료! (총 소요시간: {total_min}분 {total_sec}초)")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()