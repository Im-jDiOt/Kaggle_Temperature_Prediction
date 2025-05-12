import os
import time
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class CNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(CNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x.squeeze()


def train_lightgbm(X, y, save_path='models/lightgbm.pkl'):
    print(f"\n[모델 학습 시작] LightGBM")
    start = time.time()

    # GPU 사용 가능 여부 확인
    gpu_available = False
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_available = len(gpus) > 0
    except (ImportError, ModuleNotFoundError):
        print("  GPU 확인 모듈 없음, CPU 사용")

    # GPU 파라미터 설정
    device_params = {}
    if gpu_available:
        print("  GPU 사용 가능, LightGBM에서 GPU 활성화")
        device_params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        **device_params
    )

    model.fit(X, y)
    elapsed = time.time() - start
    print(f"✅ LightGBM 학습 완료 ({elapsed:.1f}초)")
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장: {save_path}")
    return model


def train_xgboost(X, y, save_path='models/xgboost.pkl'):
    print(f"\n[모델 학습 시작] XGBoost")
    start = time.time()

    # GPU 사용 가능 여부 확인
    gpu_available = False
    try:
        import cupy
        gpu_available = True
    except (ImportError, ModuleNotFoundError):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_available = len(gpus) > 0
        except (ImportError, ModuleNotFoundError):
            print("  GPU 확인 모듈 없음, CPU 사용")

    # GPU 파라미터 설정
    tree_method = 'hist'
    if gpu_available:
        print("  GPU 사용 가능, XGBoost에서 GPU 활성화")
        tree_method = 'gpu_hist'

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        tree_method=tree_method
    )

    model.fit(X, y)
    elapsed = time.time() - start
    print(f"✅ XGBoost 학습 완료 ({elapsed:.1f}초)")
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장: {save_path}")
    return model


def train_randomforest(X, y, save_path='models/random_forest.pkl'):
    print(f"\n[모델 학습 시작] RandomForest")
    start = time.time()
    model = RandomForestRegressor(n_estimators=300, max_depth=10, n_jobs=-1)  # 병렬 처리 활성화
    model.fit(X, y)
    elapsed = time.time() - start
    print(f"✅ RandomForest 학습 완료 ({elapsed:.1f}초)")
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장: {save_path}")
    return model


def train_elasticnet(X, y, save_path='models/elasticnet.pkl'):
    print(f"\n[모델 학습 시작] ElasticNet")
    start = time.time()
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    elapsed = time.time() - start
    print(f"✅ ElasticNet 학습 완료 ({elapsed:.1f}초)")
    joblib.dump(model, save_path)
    print(f"✅ 모델 저장: {save_path}")
    return model


def train_cnn(X_array, y_array, num_epochs=30, batch_size=32, lr=1e-3,
              save_path='models/cnn_regressor.pt'):
    print(f"\n[모델 학습 시작] CNN")

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  학습 장치: {device}")

    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                        batch_size=batch_size, shuffle=True)

    model = CNNRegressor(X_array.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start = time.time()
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  에포크 {epoch + 1}/{num_epochs} - 손실: {running_loss / len(loader):.4f} "
                  f"({(epoch + 1) / num_epochs * 100:.1f}%)")

    elapsed = time.time() - start
    print(f"✅ CNN 학습 완료 ({elapsed:.1f}초)")

    # 모델 저장 시 CPU로 이동하여 저장
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), save_path)
    print(f"✅ 모델 저장: {save_path}")

    # 원래 장치로 모델 반환
    return model.to(device)


class AverageEnsembleRegressor:
    def __init__(self, base_models, weights=None):
        """
        base_models : list of fitted estimators
        weights     : list or array of 가중치 (길이 == len(base_models)), 없으면 단순 평균
        """
        self.base_models = base_models
        self.weights = None if weights is None else np.array(weights)
        if self.weights is not None and len(self.weights) != len(self.base_models):
            raise ValueError("weights 길이는 base_models 개수와 같아야 합니다.")

    def predict(self, X):
        """
        X : array-like or DataFrame
        반환 : ensemble 예측값 (1차원 array)
        """
        # 각 모델 예측값을 (n_models, n_samples) 형태로 쌓기
        preds = []
        for model in self.base_models:
            # CNN 모델인 경우 특별 처리
            if isinstance(model, CNNRegressor):
                model.eval()
                with torch.no_grad():
                    device = next(model.parameters()).device
                    X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
                    pred = model(X_tensor).cpu().numpy()
                preds.append(pred)
            else:
                preds.append(model.predict(X))

        preds = np.stack(preds, axis=0)

        if self.weights is None:
            return preds.mean(axis=0)
        # weights 적용된 평균 계산
        return np.average(preds, axis=0, weights=self.weights)