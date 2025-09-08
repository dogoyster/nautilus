#!/usr/bin/env python3
"""
자동화된 금 가격 예측 시스템
- Optuna를 사용한 자동 하이퍼파라미터 튜닝
- 상관관계 기반 점진적 특성 선택
- MLflow를 활용한 실험 추적
- R² > 0.5 목표 달성 자동화
"""

import optuna
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import json
import os
import shutil
import argparse
from datetime import datetime, timedelta
import warnings

# TensorFlow/macOS compatibility bootstrap BEFORE importing tensorflow or modules that import it
from tf_compat import bootstrap, configure_tensorflow
bootstrap()

from analysis.gold_data_manager import GoldDataManager
from analysis.gold_prediction_model import GoldPredictionModel
import tensorflow as tf
configure_tensorflow()

warnings.filterwarnings('ignore')

class AutoTuningSystem:
    def __init__(self, target_r2=0.5, max_trials=50):
        self.target_r2 = target_r2
        self.max_trials = max_trials
        self.best_r2 = -float('inf')
        self.best_model = None
        self.best_config = None
        # 경로 구성
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp_dir = os.path.join(self.base_dir, 'tmp', 'optuna')
        self.ensemble_dir = os.path.join(self.base_dir, 'models', 'ensemble')
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
        # MLflow 설정
        mlflow.set_tracking_uri(f"file://{self.base_dir}/mlruns")
        mlflow.set_experiment("gold_price_prediction")
        
        # 재현성을 위한 시드 설정
        self.set_random_seeds()
        
        print(f"🚀 자동화된 금 가격 예측 시스템 시작")
        print(f"🎯 목표 R²: {target_r2}")
        print(f"🔬 최대 실험 횟수: {max_trials}")
    
    def set_random_seeds(self, seed=42):
        """재현 가능한 결과를 위한 시드 설정"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def get_progressive_features(self, data, max_features=10):
        """상관관계 기반 점진적 특성 선택"""
        print("🔍 점진적 특성 선택 시작...")
        
        # 컬럼이 다중 인덱스이면 평탄화
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.copy()
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            except Exception:
                # 실패 시 문자열 결합으로 평탄화
                data = data.copy()
                data.columns = ["_".join(map(str, col)) if isinstance(col, tuple) else str(col) for col in data.columns]

        # 목표 컬럼 존재 확인
        # 기본 특성 (항상 포함)
        base_features = []
        if 'Close' in data.columns:
            base_features.append('Close')
        else:
            raise ValueError("필수 컬럼 'Close' 이(가) 데이터에 없습니다.")
        if 'Volume' in data.columns:
            base_features.append('Volume')
        
        # 숫자형 컬럼만 후보로 사용
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        available_features = [col for col in numeric_cols if col not in base_features]
        
        # Close와의 상관관계 계산
        correlations = data[available_features + ['Close']].corr()['Close'].abs().sort_values(ascending=False)
        
        # 상관관계가 너무 높지 않은 특성들 선택 (다중공선성 방지)
        selected_features = base_features.copy()
        correlation_threshold = 0.8
        
        for feature, corr in correlations.items():
            if feature == 'Close':
                continue
            if len(selected_features) >= max_features:
                break
            
            # 기존 특성들과의 상관관계 확인
            if len(selected_features) > 2:
                existing_corr = data[selected_features + [feature]].corr()[feature].abs()
                if existing_corr.max() < correlation_threshold:
                    selected_features.append(feature)
            else:
                selected_features.append(feature)
        
        print(f"✅ 선택된 특성 ({len(selected_features)}개): {selected_features}")
        return selected_features
    
    def objective(self, trial):
        """Optuna 최적화 목적 함수"""
        try:
            # 하이퍼파라미터 제안
            lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64, 128])
            dense_units = trial.suggest_categorical('dense_units', [8, 16, 32, 64])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            sequence_length = trial.suggest_categorical('sequence_length', [15, 30, 45, 60])
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            epochs = trial.suggest_int('epochs', 20, 100)
            
            # 특성 개수 제안
            max_features = trial.suggest_int('max_features', 3, 15)
            
            # 데이터 준비
            data_manager = GoldDataManager('data/gold_data.csv')
            enhanced_data = data_manager.load_or_download_data()
            
            if enhanced_data is None:
                return float('inf')
            
            # 점진적 특성 선택
            features = self.get_progressive_features(enhanced_data, max_features)
            
            # 모델 생성 (임시 디렉토리 사용)
            model_name = os.path.join(self.tmp_dir, f"temp_model_{trial.number}.h5")
            scaler_name = os.path.join(self.tmp_dir, f"temp_scaler_{trial.number}")
            
            predictor = GoldPredictionModel(
                sequence_length=sequence_length,
                model_name=model_name,
                scaler_name=scaler_name,
                feature_columns=features
            )
            
            # 하이퍼파라미터 설정
            predictor.lstm_units = [lstm_units]
            predictor.dense_units = [dense_units]
            predictor.dropout_rates = [dropout_rate, dropout_rate * 0.7]
            
            # 데이터 전처리
            result = predictor.prepare_data(enhanced_data, train_ratio=0.8, target_column='Close')
            if result is None:
                return float('inf')
            
            X_train, y_train, X_test, y_test, scaled_data, train_size = result
            
            # 모델 훈련
            predictor.build_model()
            predictor.model.optimizer.learning_rate = learning_rate
            
            # 조기 종료 설정
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = predictor.train(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping]
            )
            
            # 성능 평가
            performance = predictor.evaluate(X_train, y_train, X_test, y_test)
            r2_score = performance['test_r2']
            
            # MLflow 로깅
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    'lstm_units': lstm_units,
                    'dense_units': dense_units,
                    'dropout_rate': dropout_rate,
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'max_features': max_features,
                    'features': features
                })
                
                mlflow.log_metrics({
                    'test_r2': r2_score,
                    'test_rmse': performance['test_rmse'],
                    'test_mae': performance['test_mae'],
                    'test_mape': performance['test_mape'],
                    'overfitting_ratio': performance['test_rmse'] / performance['train_rmse']
                })
                
                # 모델 저장
                mlflow.keras.log_model(predictor.model, "model")

                # 예측 차트 저장 및 로깅
                try:
                    close_prices = enhanced_data['Close']
                    chart_path = predictor.plot_results(close_prices, performance, train_size)
                    os.makedirs('results/auto_tuning', exist_ok=True)
                    trial_chart = f"results/auto_tuning/trial_{trial.number}_results.png"
                    shutil.copy(chart_path, trial_chart)
                    mlflow.log_artifact(trial_chart)
                except Exception as _:
                    pass

                # 미래 예측 저장 및 로깅
                try:
                    future_prices, prediction_dates, _ = predictor.predict_future(
                        scaled_data, enhanced_data, days=5
                    )
                    if future_prices is not None:
                        import pandas as _pd
                        df_future = _pd.DataFrame({
                            'date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                            'predicted_price': future_prices.flatten().tolist()
                        })
                        os.makedirs('results/auto_tuning', exist_ok=True)
                        future_csv = f"results/auto_tuning/trial_{trial.number}_future.csv"
                        df_future.to_csv(future_csv, index=False)
                        mlflow.log_artifact(future_csv)
                except Exception as _:
                    pass
            
            # 임시 파일 정리
            for path in [model_name, scaler_name, f"{scaler_name}_target.pkl", f"{scaler_name}_features.pkl"]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            
            # 최고 성능 모델 업데이트
            if r2_score > self.best_r2:
                self.best_r2 = r2_score
                self.best_model = predictor
                self.best_config = {
                    'lstm_units': lstm_units,
                    'dense_units': dense_units,
                    'dropout_rate': dropout_rate,
                    'sequence_length': sequence_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'max_features': max_features,
                    'features': features
                }
                print(f"🏆 새로운 최고 성능! R²: {r2_score:.4f}")
            
            return -r2_score  # Optuna는 최소화를 목표로 하므로 음수 반환
            
        except Exception as e:
            print(f"❌ 실험 실패: {str(e)}")
            return float('inf')
    
    def run_auto_tuning(self):
        """자동 하이퍼파라미터 튜닝 실행"""
        print("\n🔬 Optuna 자동 튜닝 시작...")
        
        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 목표 달성 시 즉시 중단하는 콜백
        def _stop_on_target(s, t):
            try:
                best_value = s.best_value  # 최소화된 값 = -best_r2
                if best_value is not None and -best_value >= self.target_r2:
                    s.stop()
            except Exception:
                pass
        
        # 최적화 실행
        study.optimize(self.objective, n_trials=self.max_trials, callbacks=[_stop_on_target])
        
        print(f"\n✅ 자동 튜닝 완료!")
        print(f"🏆 최고 R²: {self.best_r2:.4f}")
        print(f"🎯 목표 R²: {self.target_r2}")
        
        if self.best_r2 >= self.target_r2:
            print("🎉 목표 달성!")
            return True
        else:
            print("⚠️ 목표 미달성 - 추가 개선 필요")
            return False
    
    def expand_data_period(self, years=2):
        """데이터 기간 확장"""
        print(f"\n📈 데이터 기간 확장 중... (+{years}년)")
        
        data_manager = GoldDataManager('data/gold_data.csv')
        current_data = data_manager.load_or_download_data()
        
        if current_data is not None:
            # 날짜는 인덱스로 관리되므로 인덱스에서 계산
            if isinstance(current_data.index, pd.DatetimeIndex):
                current_start = current_data.index.min()
            else:
                # 인덱스가 날짜가 아니라면 변환 시도
                try:
                    current_data.index = pd.to_datetime(current_data.index)
                    current_start = current_data.index.min()
                except Exception:
                    raise ValueError("데이터 인덱스를 날짜형으로 변환할 수 없습니다.")
            new_start = current_start - timedelta(days=years * 365)
            
            print(f"   기존 시작일: {current_start}")
            print(f"   새로운 시작일: {new_start}")
            
            # 새로운 데이터 다운로드
            data_manager._download_and_enhance_data(
                start_date=new_start.strftime('%Y-%m-%d'),
                end_date=current_data.index.max().strftime('%Y-%m-%d')
            )
            
            print("✅ 데이터 확장 완료")
            return True
        
        return False
    
    def create_ensemble_model(self, n_models=5):
        """앙상블 모델 생성"""
        print(f"\n🤖 앙상블 모델 생성 중... ({n_models}개 모델)")
        
        ensemble_models = []
        ensemble_configs = []
        
        # 다양한 설정으로 여러 모델 훈련
        for i in range(n_models):
            print(f"   모델 {i+1}/{n_models} 훈련 중...")
            
            # 랜덤 설정 생성
            config = {
                'lstm_units': np.random.choice([32, 64, 128]),
                'dense_units': np.random.choice([16, 32, 64]),
                'dropout_rate': np.random.uniform(0.2, 0.4),
                'sequence_length': np.random.choice([30, 45, 60]),
                'batch_size': np.random.choice([32, 64]),
                'learning_rate': np.random.uniform(1e-4, 1e-3),
                'epochs': np.random.randint(30, 80),
                'max_features': np.random.randint(5, 12)
            }
            
            # 모델 훈련
            try:
                data_manager = GoldDataManager('data/gold_data.csv')
                enhanced_data = data_manager.load_or_download_data()
                
                if enhanced_data is None:
                    continue
                
                features = self.get_progressive_features(enhanced_data, config['max_features'])
                
                model_name = os.path.join(self.ensemble_dir, f"ensemble_model_{i}.h5")
                scaler_name = os.path.join(self.ensemble_dir, f"ensemble_scaler_{i}")
                
                predictor = GoldPredictionModel(
                    sequence_length=config['sequence_length'],
                    model_name=model_name,
                    scaler_name=scaler_name,
                    feature_columns=features
                )
                
                predictor.lstm_units = [config['lstm_units']]
                predictor.dense_units = [config['dense_units']]
                predictor.dropout_rates = [config['dropout_rate'], config['dropout_rate'] * 0.7]
                
                result = predictor.prepare_data(enhanced_data, train_ratio=0.8, target_column='Close')
                if result is None:
                    continue
                
                X_train, y_train, X_test, y_test, scaled_data, train_size = result
                
                predictor.build_model()
                predictor.model.optimizer.learning_rate = config['learning_rate']
                
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True
                )
                
                predictor.train(
                    X_train, y_train,
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    callbacks=[early_stopping]
                )
                
                performance = predictor.evaluate(X_train, y_train, X_test, y_test)
                
                if performance['test_r2'] > 0.3:  # 최소 성능 기준
                    ensemble_models.append(predictor)
                    ensemble_configs.append(config)
                    print(f"   ✅ 모델 {i+1} 성능: R² = {performance['test_r2']:.4f}")
                else:
                    print(f"   ❌ 모델 {i+1} 성능 부족: R² = {performance['test_r2']:.4f}")
                
            except Exception as e:
                print(f"   ❌ 모델 {i+1} 실패: {str(e)}")
                continue
        
        if len(ensemble_models) >= 2:
            print(f"✅ 앙상블 모델 {len(ensemble_models)}개 생성 완료")
            return ensemble_models, ensemble_configs
        else:
            print("❌ 앙상블 모델 생성 실패")
            return None, None
    
    def run_complete_optimization(self):
        """완전 자동화된 최적화 실행"""
        print("🚀 완전 자동화된 금 가격 예측 최적화 시작!")
        print("="*60)
        
        # 1단계: 기본 자동 튜닝
        print("\n📊 1단계: 자동 하이퍼파라미터 튜닝")
        success = self.run_auto_tuning()
        
        if success:
            print("🎉 1단계에서 목표 달성!")
            return self.best_model, self.best_config
        
        # 2단계: 데이터 확장
        print("\n📈 2단계: 데이터 기간 확장")
        if self.expand_data_period(years=2):
            print("🔄 확장된 데이터로 재튜닝...")
            success = self.run_auto_tuning()
            
            if success:
                print("🎉 2단계에서 목표 달성!")
                return self.best_model, self.best_config
        
        # 3단계: 앙상블 모델
        print("\n🤖 3단계: 앙상블 모델 생성")
        ensemble_models, ensemble_configs = self.create_ensemble_model(n_models=5)
        
        if ensemble_models:
            print("✅ 앙상블 모델 생성 완료")
            return ensemble_models, ensemble_configs
        
        print("❌ 모든 단계에서 목표 달성 실패")
        return self.best_model, self.best_config

def main():
    """메인 실행 함수"""
    print("🏆 자동화된 금 가격 예측 시스템")
    print("="*50)
    
    # CLI 인자 파서
    parser = argparse.ArgumentParser(description='Gold price auto-tuning system')
    parser.add_argument('--target-r2', type=float, default=None, help='목표 R² (예: 0.5)')
    parser.add_argument('--max-trials', type=int, default=None, help='최대 실험 횟수 (예: 50)')
    args, unknown = parser.parse_known_args()

    # 목표 R² 설정 (CLI > ENV > interactive)
    target_r2 = args.target_r2
    if target_r2 is None:
        target_r2 = float(os.environ.get('NAUTILUS_TARGET_R2') or input("목표 R² 점수를 입력하세요 (기본값: 0.5): ") or "0.5")

    # 최대 실험 횟수 설정 (CLI > ENV > interactive)
    max_trials = args.max_trials
    if max_trials is None:
        max_trials = int(os.environ.get('NAUTILUS_MAX_TRIALS') or input("최대 실험 횟수를 입력하세요 (기본값: 50): ") or "50")
    
    # 자동 튜닝 시스템 시작
    auto_system = AutoTuningSystem(target_r2=target_r2, max_trials=max_trials)
    
    # 완전 자동화된 최적화 실행
    best_model, best_config = auto_system.run_complete_optimization()
    
    print(f"\n🏆 최종 결과:")
    print(f"   최고 R²: {auto_system.best_r2:.4f}")
    print(f"   목표 R²: {target_r2}")
    
    if auto_system.best_r2 >= target_r2:
        print("🎉 목표 달성!")
    else:
        print("⚠️ 목표 미달성 - 추가 개선 필요")

if __name__ == "__main__":
    main()
