from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time

from xgboost import XGBRegressor


# 1) Custom Imputer
class PhysicsImputer(BaseEstimator, TransformerMixin):
    def __init__(self, station_info=None):
        self.station_info = station_info

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        for n in range(24):
            suffix = f"_{n}"

            # 필드 정의
            dew_col = f"dew_point{suffix}"
            vp_col = f"vapor_pressure{suffix}"
            lp_col = f"local_pressure{suffix}"
            slp_col = f"sea_level_pressure{suffix}"
            st_col = f"surface_temp{suffix}"
            hum_col = f"humidity{suffix}"
            cc_col = f"cloud_cover{suffix}"
            sun_col = f"sunshine_duration{suffix}"
            vis_col = f"visibility{suffix}"

            # 1. dew_point -> vapor_pressure
            if dew_col in X_.columns and vp_col in X_.columns:
                mask = X_[vp_col].isnull() & X_[dew_col].notnull()
                if mask.any():
                    dp = X_.loc[mask, dew_col]
                    X_.loc[mask, vp_col] = 6.11 * 10.0 ** ((7.5 * dp) / (237.3 + dp))

            # 2. vapor_pressure -> dew_point
            if vp_col in X_.columns and dew_col in X_.columns:
                mask = X_[dew_col].isnull() & X_[vp_col].notnull()
                if mask.any():
                    vp = X_.loc[mask, vp_col]
                    valid = vp > 0
                    if valid.any():
                        vp_valid = vp[valid]
                        log_term = np.log10(vp_valid / 6.11)
                        num = 237.3 * log_term
                        denom = 7.5 - log_term
                        safe_mask = np.abs(denom) > 1e-9
                        X_.loc[mask & valid, dew_col] = num[safe_mask] / denom[safe_mask]

            # 3. sea_level_pressure -> local_pressure
            if all(c in X_.columns for c in [slp_col, lp_col, st_col]) and self.station_info is not None:
                mask = X_[lp_col].isnull() & X_[slp_col].notnull() & X_[st_col].notnull()
                if mask.any():
                    slp = X_.loc[mask, slp_col]
                    temp = X_.loc[mask, st_col]
                    altitude = self.station_info.loc[X_.loc[mask, 'station'], '노장해발고도(m)'].values
                    T = temp + 273.15
                    g, M, R = 9.80665, 0.0289644, 8.31447
                    X_.loc[mask, lp_col] = slp * np.exp((-g * M * altitude) / (R * T))

            # 4. local_pressure -> sea_level_pressure
            if all(c in X_.columns for c in [slp_col, lp_col, st_col]) and self.station_info is not None:
                mask = X_[slp_col].isnull() & X_[lp_col].notnull() & X_[st_col].notnull()
                if mask.any():
                    lp = X_.loc[mask, lp_col]
                    temp = X_.loc[mask, st_col]
                    altitude = self.station_info.loc[X_.loc[mask, 'station'], '노장해발고도(m)'].values
                    T = temp + 273.15
                    g, M, R = 9.80665, 0.0289644, 8.31447
                    X_.loc[mask, slp_col] = lp / np.exp((-g * M * altitude) / (R * T))

            # 5. vapor_pressure, surface_temp -> humidity
            if all(c in X_.columns for c in [vp_col, st_col, hum_col]):
                mask = X_[hum_col].isnull() & X_[vp_col].notnull() & X_[st_col].notnull()
                if mask.any():
                    vp = X_.loc[mask, vp_col]
                    temp = X_.loc[mask, st_col]
                    sat_vp = 6.11 * 10.0 ** ((7.5 * temp) / (237.3 + temp))
                    rh = (vp / sat_vp) * 100
                    X_.loc[mask, hum_col] = np.clip(rh, 0, 100)

            # 6. sunshine_duration -> cloud_cover
            if all(c in X_.columns for c in [sun_col, cc_col]):
                mask = X_[cc_col].isnull() & X_[sun_col].notnull()
                if mask.any():
                    sunshine = X_.loc[mask, sun_col]
                    max_sun = 1.0
                    est_cloud = (1 - sunshine / max_sun) * 10
                    X_.loc[mask, cc_col] = np.clip(est_cloud, 0, 10)

            # 7. humidity -> visibility
            if all(c in X_.columns for c in [hum_col, vis_col]):
                mask = X_[vis_col].isnull() & X_[hum_col].notnull()
                if mask.any():
                    hum = X_.loc[mask, hum_col]
                    fog = hum >= 95
                    X_.loc[mask & fog, vis_col] = 100    # 1 km
                    X_.loc[mask & ~fog, vis_col] = 1000  # 10 km

        return X_

class InterpolationImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.interpolate(method='linear', limit_direction='both')

class CorrelationImputer(BaseEstimator, TransformerMixin):
            def __init__(self, n_neighbors=5):
                self.n_neighbors = n_neighbors
                self.imputer = KNNImputer(n_neighbors=n_neighbors)

            def fit(self, X, y=None):
                self.imputer.fit(X)
                return self

            def transform(self, X):
                    X_ = X.copy()

                    # KNNImputer로 결측치 채우기
                    imputed_array = self.imputer.transform(X_) #(13132, 329) <- 전체가 NaN인 열은 학습 대상에서 자동 제외되어 컬럼 수가 줄어듦. 을 그냥 NaN인 열이 차피 밤일 때 일조량이라 0으로 보간함.

                    imputed_df = pd.DataFrame(
                        imputed_array,
                        index=X_.index,
                        columns= X_.columns
                    )

                    return X_

class ModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=True):
        self.models = {}
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.verbose:
            print(f"ModelImputer 학습 시작 - 총 {len(X.columns)}개 컬럼")
            start_time = time.time()
            total_cols = len(X.columns)

        for i, col in enumerate(X.columns):
            if self.verbose:
                col_start = time.time()
                print(f"  [{i+1}/{total_cols}] '{col}' 컬럼 모델 학습 중...", end="", flush=True)

            mask = X[col].notnull()
            non_null_count = mask.sum()

            if non_null_count > 0:  # 데이터가 존재하는 경우에만 학습
                try:
                    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
                    X_train = X.loc[mask].drop(col, axis=1)
                    y_train = X.loc[mask, col]

                    # 학습 데이터가 충분한지 확인
                    if len(X_train) > 5:  # 최소 5개 이상의 샘플 필요
                        rf.fit(X_train, y_train)
                        self.models[col] = rf
                        status = "완료"
                    else:
                        status = "샘플 부족으로 건너뜀"
                except Exception as e:
                    status = f"오류 발생: {str(e)}"
            else:
                status = "데이터 없음"

            if self.verbose:
                col_time = time.time() - col_start
                print(f" {status} ({col_time:.2f}초)")

                # 진행률 및 예상 남은 시간 계산
                if i > 0:  # 최소 1개 이상 처리 후에만 예상 시간 계산
                    elapsed = time.time() - start_time
                    progress = (i + 1) / total_cols
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed

                    print(f"    진행률: {progress*100:.1f}% - 예상 남은 시간: {remaining:.1f}초")

        if self.verbose:
            total_time = time.time() - start_time
            models_count = len(self.models)
            print(f"ModelImputer 학습 완료 - {models_count}/{total_cols} 컬럼 모델 생성 ({total_time:.2f}초)")

        return self

    def transform(self, X):
        if self.verbose:
            print(f"ModelImputer 변환 시작 - 결측치 채우기")
            start_time = time.time()
            total_missing_before = X.isnull().sum().sum()
            filled_cols = 0

        X_ = X.copy()

        for i, (col, rf) in enumerate(self.models.items()):
            if self.verbose:
                col_start = time.time()
                print(f"  [{i+1}/{len(self.models)}] '{col}' 컬럼 결측치 처리 중...", end="", flush=True)

            mask = X_[col].isnull()
            missing_count = mask.sum()

            if missing_count > 0:
                try:
                    X_.loc[mask, col] = rf.predict(X_.loc[mask].drop(col, axis=1))
                    status = f"{missing_count}개 결측치 처리됨"
                    filled_cols += 1
                except Exception as e:
                    status = f"오류 발생: {str(e)}"
            else:
                status = "결측치 없음"

            if self.verbose:
                col_time = time.time() - col_start
                print(f" {status} ({col_time:.2f}초)")

        if self.verbose:
            total_time = time.time() - start_time
            total_missing_after = X_.isnull().sum().sum()
            filled_count = total_missing_before - total_missing_after

            if total_missing_before > 0:
                filled_percent = 100 * filled_count / total_missing_before
            else:
                filled_percent = 0

            print(f"ModelImputer 변환 완료 - {filled_cols}개 컬럼 처리")
            print(f"  결측치: {total_missing_before}개 -> {total_missing_after}개 (처리됨: {filled_count}개, {filled_percent:.1f}%)")
            print(f"  소요 시간: {total_time:.2f}초")

        return X_

class GPUModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=True, n_estimators=100, gpu_id=0):
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.gpu_id = gpu_id
        self.models = {}

    def fit(self, X, y=None):
        if self.verbose:
            print(f"GPUModelImputer 학습 시작 - 총 {len(X.columns)}개 컬럼")
            start = time.time()

        for i, col in enumerate(X.columns):
            if self.verbose:
                print(f"  [{i+1}/{len(X.columns)}] '{col}' 모델 학습 중...", end="", flush=True)
            mask = X[col].notnull()
            if mask.sum() > 5:
                model = XGBRegressor(
                    tree_method='gpu_hist',
                    gpu_id=self.gpu_id,
                    n_estimators=self.n_estimators,
                    random_state=42,
                    verbosity=0
                )
                X_train = X.loc[mask].drop(col, axis=1)
                y_train = X.loc[mask, col]
                model.fit(X_train, y_train)
                self.models[col] = model
                status = "완료"
            else:
                status = "샘플 부족"
            if self.verbose:
                print(f" {status}")

        if self.verbose:
            print(f"GPUModelImputer 학습 완료 ({time.time()-start:.1f}초)")

        return self

    def transform(self, X):
        X_ = X.copy()
        if self.verbose:
            print("GPUModelImputer 변환 시작")

        for i, (col, model) in enumerate(self.models.items()):
            mask = X_[col].isnull()
            if mask.any():
                X_.loc[mask, col] = model.predict(X_.loc[mask].drop(col, axis=1))
        if self.verbose:
            print("GPUModelImputer 변환 완료")
        return X_

# 2) Ensemble Imputer
class EnsembleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputers, weights=None, strategy="mean", verbose=True):
        """
        imputers: list of (str, transformer) 형태
        weights: 각 imputer에 대한 가중치 (strategy가 'weighted'일 때 필요)
        strategy: 'mean', 'median', 'weighted' 중 선택
        verbose: 진행 상황 출력 여부
        """
        self.imputers = imputers
        self.weights = weights
        self.strategy = strategy
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.verbose:
            print(f"EnsembleImputer 학습 시작 - 총 {len(self.imputers)}개 임퓨터")
            start_time = time.time()

        self.fitted_imputers_ = []

        for i, (name, imputer) in enumerate(self.imputers):
            if self.verbose:
                print(f"  [{i + 1}/{len(self.imputers)}] {name} 임퓨터 학습 중...")
                imputer_start = time.time()

            fitted = clone(imputer).fit(X)
            self.fitted_imputers_.append((name, fitted))

            if self.verbose:
                print(f"  [{i + 1}/{len(self.imputers)}] {name} 임퓨터 학습 완료 "
                      f"({time.time() - imputer_start:.2f}초)")

        if self.verbose:
            print(f"EnsembleImputer 학습 완료 (총 {time.time() - start_time:.2f}초)")

        return self

    def transform(self, X):
        if self.verbose:
            print(f"EnsembleImputer 변환 시작 - 데이터 크기: {X.shape}")
            start_time = time.time()

        imputations = []

        for i, (name, imputer) in enumerate(self.fitted_imputers_):
            if self.verbose:
                print(f"  [{i + 1}/{len(self.fitted_imputers_)}] {name} 임퓨터로 결측치 대체 중...")
                imputer_start = time.time()

                # 변환 전 결측치 수 계산
                missing_before = X.isnull().sum().sum()

            X_trans = imputer.transform(X)
            imputations.append(X_trans)

            if self.verbose:
                # 변환 후 결측치 수 계산
                missing_after = X_trans.isnull().sum().sum()
                filled_count = missing_before - missing_after

                print(f"  [{i + 1}/{len(self.fitted_imputers_)}] {name} 임퓨터 변환 완료 "
                      f"(결측치 {filled_count}개 처리, {time.time() - imputer_start:.2f}초)")

        if self.verbose:
            print(f"개별 임퓨터 변환 완료, 결과 집계 중...")
            agg_start = time.time()

        imputations = np.array([df.values for df in imputations])  # shape: (n_imputers, n_samples, n_features)

        if self.strategy == "mean":
            final = np.nanmean(imputations, axis=0)
        elif self.strategy == "median":
            final = np.nanmedian(imputations, axis=0)
        elif self.strategy == "weighted":
            if self.weights is None:
                raise ValueError("Weights must be provided for weighted strategy.")
            weights = np.array(self.weights)[:, np.newaxis, np.newaxis]
            final = np.nansum(imputations * weights, axis=0) / np.nansum(weights * ~np.isnan(imputations), axis=0)
        else:
            raise ValueError("Unsupported strategy")

        result = pd.DataFrame(final, columns=X.columns, index=X.index)

        if self.verbose:
            # 최종 결측치 수 계산
            missing_final = result.isnull().sum().sum()
            total_missing = X.isnull().sum().sum()
            filled_percent = 100 * (1 - missing_final / total_missing) if total_missing > 0 else 100

            print(f"결과 집계 완료 ({time.time() - agg_start:.2f}초)")
            print(f"총 결측치: {total_missing}개, 채워진 결측치: {total_missing - missing_final}개 "
                  f"({filled_percent:.2f}%)")
            print(f"EnsembleImputer 변환 완료 (총 {time.time() - start_time:.2f}초)")

        return result


# 3) 사용 예시
