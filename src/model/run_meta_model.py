import os
import pandas as pd
import numpy as np
import joblib
import glob
from typing import List, Optional

class MetaXGBoostAverager:
    """
    각 관측소의 XGBoost 모델의 예측값을 평균내어 최종 예측을 생성하는 메타 모델
    """
    def __init__(self, models_dir: str = 'models'):
        """
        초기화 함수

        Args:
            models_dir: 모델이 저장된 상위 디렉토리 경로
        """
        self.models_dir = models_dir
        self.xgboost_models = {}  # 관측소별 XGBoost 모델을 저장할 딕셔너리
        self._load_models()

    def _load_models(self) -> None:
        """모든 관측소의 XGBoost 모델을 로드하는 내부 함수"""
        # 모델 디렉토리의 모든 서브 디렉토리(관측소별) 탐색
        station_dirs = [d for d in os.listdir(self.models_dir)
                       if os.path.isdir(os.path.join(self.models_dir, d))]

        print(f"발견된 관측소 디렉토리: {len(station_dirs)}개")

        for station in station_dirs:
            # 해당 관측소의 XGBoost 모델 파일 찾기
            xgb_pattern = os.path.join(self.models_dir, station, f'xgboost_{station}.pkl')
            xgb_files = glob.glob(xgb_pattern)

            if xgb_files:
                # 모델 로드
                model_path = xgb_files[0]
                try:
                    model = joblib.load(model_path)
                    self.xgboost_models[station] = model
                    print(f"✅ 관측소 {station}의 XGBoost 모델 로드 완료: {model_path}")
                except Exception as e:
                    print(f"⚠️ 관측소 {station}의 XGBoost 모델 로드 실패: {e}")
            else:
                print(f"⚠️ 관측소 {station}의 XGBoost 모델을 찾을 수 없음")

        print(f"총 {len(self.xgboost_models)}개 관측소의 XGBoost 모델 로드 완료")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        모든 관측소의 XGBoost 모델 예측값의 평균을 계산

        Args:
            X: 입력 특성 데이터프레임

        Returns:
            모든 관측소 모델의 예측 평균값
        """
        if not self.xgboost_models:
            raise ValueError("로드된 모델이 없습니다.")

        # 각 관측소 모델의 예측값 저장
        predictions = []

        for station, model in self.xgboost_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                print(f"관측소 {station} 모델 예측 완료")
            except Exception as e:
                print(f"⚠️ 관측소 {station} 모델 예측 실패: {e}")

        if not predictions:
            raise ValueError("예측에 실패했습니다.")

        # 모든 예측값의 평균 계산
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions

    def save(self, filepath: str) -> None:
        """
        메타 모델 저장

        Args:
            filepath: 저장할 파일 경로
        """
        joblib.dump(self, filepath)
        print(f"✅ 메타 모델 저장 완료: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MetaXGBoostAverager':
        """
        저장된 메타 모델 로드

        Args:
            filepath: 로드할 파일 경로

        Returns:
            로드된 메타 모델 객체
        """
        model = joblib.load(filepath)
        print(f"✅ 메타 모델 로드 완료: {filepath}")
        return model

# 메타 모델 생성 및 저장 함수
def create_meta_model(models_dir: str = 'models', save_path: str = 'models/meta_model.pkl') -> MetaXGBoostAverager:
    """
    메타 모델 생성 및 저장 함수

    Args:
        models_dir: 모델이 저장된 상위 디렉토리 경로
        save_path: 메타 모델을 저장할 경로

    Returns:
        생성된 메타 모델 객체
    """
    meta_model = MetaXGBoostAverager(models_dir=models_dir)
    meta_model.save(save_path)
    return meta_model

# 테스트를 위한 메인 함수
if __name__ == '__main__':
    # 메타 모델 생성 및 저장
    meta_model = create_meta_model()