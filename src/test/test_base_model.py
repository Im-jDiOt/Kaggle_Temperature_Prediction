import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model.base_model import AverageEnsembleRegressor


def load_station_data(station_id):
    """특정 관측소의 데이터만 로드"""
    print(f"[데이터 로드] 관측소 {station_id}")
    df = pd.read_csv(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\data\processed\featured_data.csv', index_col=0)
    df.drop(columns=['station_name', 'date'], inplace=True)

    # 특정 관측소 데이터만 필터링
    station_df = df[df['station'] == float(station_id)].drop(columns=['station'])
    print(f"관측소 {station_id} 데이터: {station_df.shape[0]}개 샘플, {station_df.shape[1]}개 특성")
    return station_df


def load_models(station_id):
    """특정 관측소에 대해 훈련된 모델들을 로드"""
    model_dir = os.path.join(r'C:\Users\USER\PycharmProjects\ML_kaggle\src\model\models', str(station_id))

    models = {
        'LightGBM': joblib.load(os.path.join(model_dir, f'lightgbm_{station_id}.pkl')),
        'XGBoost': joblib.load(os.path.join(model_dir, f'xgboost_{station_id}.pkl')),
        'RandomForest': joblib.load(os.path.join(model_dir, f'random_forest_{station_id}.pkl')),
        'ElasticNet': joblib.load(os.path.join(model_dir, f'elasticnet_{station_id}.pkl')),
    }

    # # CNN 모델 로드
    # cnn_path = os.path.join(model_dir, f'cnn_regressor_{station_id}.pt')
    # if os.path.exists(cnn_path):
    #     # 임포트는 이 함수 내에서만 필요하므로 지연 임포트
    #     from src.model.base_model import CNNRegressor
    #     cnn_model = CNNRegressor(input_dim=394)  # 입력 차원은 데이터에서 자동으로 가져옴
    #     cnn_model.load_state_dict(torch.load(cnn_path))
    #     cnn_model.eval()
    #     models['CNN'] = cnn_model

    # 앙상블 모델 로드
    ensemble_path = os.path.join(model_dir, f'ensemble_{station_id}.pkl')
    if os.path.exists(ensemble_path):
        models['Ensemble'] = joblib.load(ensemble_path)

    print(f"로드된 모델: {', '.join(models.keys())}")
    return models


def evaluate_model(model_name, model, X_test, y_test):
    """모델 평가 및 결과 반환"""
    # if 'CNN' in model_name:
    #     # CNN 모델 특별 처리
    #     model.eval()
    #     with torch.no_grad():
    #         X_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test,
    #                                dtype=torch.float32)
    #         y_pred = model(X_tensor).numpy().flatten()
    # else:
    #     y_pred = model.predict(X_test)
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'model': model_name,
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_predictions(y_test, results, station_id):
    """각 모델의 예측 결과 시각화"""
    plt.figure(figsize=(12, 8))

    # 실제값 플롯
    plt.scatter(range(len(y_test)), y_test, label='실제값', color='black', alpha=0.5)

    # 각 모델의 예측값 플롯
    for result in results:
        plt.plot(range(len(y_test)), result['y_pred'],
                 label=f"{result['model']} (RMSE: {result['rmse']:.4f})", alpha=0.7)

    plt.title(f'관측소 {station_id} - 모델별 예측 결과 비교')
    plt.xlabel('샘플 인덱스')
    plt.ylabel('타겟값')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 저장 디렉토리 확인
    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)

    # 그래프 저장
    plt.savefig(os.path.join(save_dir, f'predictions_station_{station_id}.png'))
    plt.close()


def main():
    # 테스트할 관측소 선택
    station_id = 108  # 변경 가능

    # 관측소 데이터 로드
    station_df = load_station_data(station_id)

    # 학습/테스트 분리 (테스트에 20% 사용)
    X = station_df.drop(columns=['target'])
    y = station_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"학습 데이터: {X_train.shape[0]}개 샘플")
    print(f"테스트 데이터: {X_test.shape[0]}개 샘플")

    # 모델 로드
    models = load_models(station_id)

    # 각 모델 평가
    results = []
    print("\n[모델 평가 결과]")
    print("-" * 50)
    print(f"{'모델':<15} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
    print("-" * 50)

    for model_name, model in models.items():
        result = evaluate_model(model_name, model, X_test, y_test)
        results.append(result)
        print(f"{model_name:<15} {result['mse']:<10.4f} {result['rmse']:<10.4f} "
              f"{result['mae']:<10.4f} {result['r2']:<10.4f}")

    # 결과 시각화
    plot_predictions(y_test, results, station_id)
    print(f"\n✅ 시각화 저장 완료: test_results/predictions_station_{station_id}.png")


if __name__ == "__main__":
    main()