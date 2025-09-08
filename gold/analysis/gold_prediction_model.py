import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import os
from datetime import datetime

class GoldPredictionModel:
    """금 가격 예측 모델 클래스 - Enhanced Multi-Feature Edition"""
    
    def __init__(self, model_name='../models/gold_prediction_model.h5', 
                 scaler_name='../models/gold_scaler.pkl', sequence_length=60,
                 feature_columns=None):
        self.model_name = model_name
        self.scaler_name = scaler_name
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # 기본 특성 컬럼 (종가만 사용하는 경우)
        if feature_columns is None:
            self.feature_columns = ['Close']
        else:
            self.feature_columns = feature_columns
            
        self.n_features = len(self.feature_columns)
        
        # 하이퍼파라미터 기본값 설정
        self.lstm_units = [32]
        self.dense_units = [16]
        self.dropout_rates = [0.4, 0.3]
        
        print(f"🤖 모델 초기화:")
        print(f"   📊 사용할 특성: {self.feature_columns}")
        print(f"   🔢 특성 개수: {self.n_features}")
        print(f"   ⏱️ 시퀀스 길이: {self.sequence_length}")
        
        # 한글 폰트 설정
        self._setup_korean_font()
    
    def _setup_korean_font(self):
        """한글 폰트 자동 설정"""
        font_candidates = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            'C:/Windows/Fonts/malgun.ttf',
            'AppleGothic', 'Nanum Gothic', 'DejaVu Sans'
        ]
        
        for font in font_candidates:
            try:
                if font.startswith('/') or font.startswith('C:'):
                    if os.path.exists(font):
                        plt.rcParams['font.family'] = fm.FontProperties(fname=font).get_name()
                        break
                else:
                    plt.rcParams['font.family'] = font
                    break
            except:
                continue
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def prepare_data(self, enhanced_data, train_ratio=0.8, target_column='Close'):
        """다중 특성 데이터 전처리 및 분할"""
        print("🔧 다중 특성 데이터 전처리 중...")
        
        # 선택된 특성들만 추출
        if isinstance(enhanced_data, pd.Series):
            # 단일 시리즈인 경우 (기존 호환성)
            feature_data = enhanced_data.to_frame(name='Close')
            self.feature_columns = ['Close']
            self.n_features = 1
        else:
            # DataFrame인 경우
            missing_features = [col for col in self.feature_columns if col not in enhanced_data.columns]
            if missing_features:
                print(f"⚠️ 누락된 특성들: {missing_features}")
                available_features = [col for col in self.feature_columns if col in enhanced_data.columns]
                print(f"✅ 사용 가능한 특성들: {available_features}")
                self.feature_columns = available_features
                self.n_features = len(self.feature_columns)
            
            feature_data = enhanced_data[self.feature_columns].copy()
        
        # 타겟 컬럼 확인
        if target_column not in enhanced_data.columns:
            print(f"❌ 타겟 컬럼 '{target_column}'이 없습니다.")
            return None
            
        target_data = enhanced_data[target_column].copy()
        
        print(f"📊 데이터 정보:")
        print(f"   🔢 특성 개수: {self.n_features}")
        print(f"   📈 데이터 포인트: {len(feature_data)}")
        print(f"   🎯 타겟: {target_column}")
        
        # 스케일러 로드 또는 생성
        self.scaler = self._load_or_create_scaler(feature_data)
        self.target_scaler = self._load_or_create_target_scaler(target_data)
        
        # 정규화
        scaled_features = self.scaler.transform(feature_data.values)
        scaled_target = self.target_scaler.transform(target_data.values.reshape(-1, 1))
        
        # 훈련/테스트 분할
        train_size = int(len(scaled_features) * train_ratio)
        train_features = scaled_features[:train_size]
        test_features = scaled_features[train_size:]
        train_target = scaled_target[:train_size]
        test_target = scaled_target[train_size:]
        
        # 시계열 시퀀스 생성
        X_train, y_train = self._create_sequences_multivariate(train_features, train_target.flatten())
        X_test, y_test = self._create_sequences_multivariate(test_features, test_target.flatten())
        
        print(f"✅ 데이터 준비 완료:")
        print(f"   훈련 데이터: {X_train.shape}")
        print(f"   테스트 데이터: {X_test.shape}")
        print(f"   특성 차원: {X_train.shape[2]}")
        
        return X_train, y_train, X_test, y_test, (scaled_features, scaled_target), train_size
    
    def _load_or_create_scaler(self, data):
        """특성용 스케일러 로드 또는 생성"""
        scaler_path = self.scaler_name.replace('.pkl', '_features.pkl')
        
        if os.path.exists(scaler_path):
            print("📁 기존 특성 스케일러 불러오는 중...")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print("🔧 새 특성 스케일러 생성 중...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            if isinstance(data, pd.DataFrame):
                scaler.fit(data.values)
            else:
                scaler.fit(data.values.reshape(-1, 1))
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        return scaler
    
    def _load_or_create_target_scaler(self, target_data):
        """타겟용 스케일러 로드 또는 생성"""
        # 스케일러 이름이 .pkl로 끝나지 않으면 추가
        if not self.scaler_name.endswith('.pkl'):
            target_scaler_path = self.scaler_name + '_target.pkl'
        else:
            target_scaler_path = self.scaler_name.replace('.pkl', '_target.pkl')
        
        
        if os.path.isfile(target_scaler_path):
            print("📁 기존 타겟 스케일러 불러오는 중...")
            with open(target_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print("🔧 새 타겟 스케일러 생성 중...")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(target_data.values.reshape(-1, 1))
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(target_scaler_path), exist_ok=True)
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        return scaler
    
    def _create_sequences(self, data):
        """단변량 시계열 시퀀스 생성 (기존 호환성)"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def _create_sequences_multivariate(self, features, target):
        """다변량 시계열 시퀀스 생성"""
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            # 특성들의 시퀀스
            X.append(features[i-self.sequence_length:i])
            # 타겟 값
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self):
        """동적 하이퍼파라미터 기반 LSTM 모델 구성"""
        print(f"🏗️ LSTM 모델 구성 중... (특성 차원: {self.n_features})")
        print(f"   🔧 LSTM 유닛: {self.lstm_units}")
        print(f"   🔧 Dense 유닛: {self.dense_units}")
        print(f"   🔧 드롭아웃: {self.dropout_rates}")
        
        layers = []
        
        # LSTM 레이어들 구성
        for i, units in enumerate(self.lstm_units):
            # Keras는 Python int를 기대하므로 안전 캐스팅
            units = int(units)
            if i == 0:
                # 첫 번째 LSTM 레이어
                return_sequences = len(self.lstm_units) > 1
                layers.append(LSTM(
                    units, 
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, self.n_features),
                    name=f'lstm_{i+1}'
                ))
            else:
                # 나머지 LSTM 레이어들
                return_sequences = i < len(self.lstm_units) - 1
                layers.append(LSTM(
                    int(units),
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}'
                ))
            
            # 드롭아웃 추가 (있는 경우)
            if i < len(self.dropout_rates):
                layers.append(Dropout(self.dropout_rates[i], name=f'dropout_lstm_{i+1}'))
        
        # Dense 레이어들 구성
        for i, units in enumerate(self.dense_units):
            layers.append(Dense(int(units), activation='relu', name=f'dense_{i+1}'))
            
            # 드롭아웃 추가 (있는 경우)
            dropout_idx = len(self.lstm_units) + i
            if dropout_idx < len(self.dropout_rates):
                layers.append(Dropout(self.dropout_rates[dropout_idx], name=f'dropout_dense_{i+1}'))
        
        # 출력 레이어
        layers.append(Dense(1, name='output'))
        
        # 모델 생성
        self.model = Sequential(layers)
        
        # 컴파일 - 다중 특성에 맞는 최적화
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print("✅ 모델 구성 완료!")
        print(f"   📊 모델 파라미터 수: {self.model.count_params():,}")
        
        return self.model
    
    def load_or_create_model(self, retrain=False):
        """모델 로드 또는 생성"""
        if os.path.exists(self.model_name) and not retrain:
            print("📁 기존 모델 불러오는 중...")
            self.model = load_model(self.model_name)
            self.is_trained = True
            print("✅ 모델 로드 완료!")
            return False  # 훈련 불필요
        else:
            print("🔧 새 모델 생성 중...")
            self.build_model()
            return True  # 훈련 필요
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=None):
        """모델 훈련"""
        print(f"🎯 모델 훈련 시작... ({datetime.now().strftime('%H:%M:%S')})")
        
        default_callbacks = [
            ModelCheckpoint(
                self.model_name, 
                save_best_only=True, 
                monitor='val_loss',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        # 외부에서 전달된 콜백을 기본 콜백 뒤에 병합
        if callbacks is None:
            callbacks_to_use = default_callbacks
        else:
            # 중복 방지: 타입 기반으로 간단 병합
            existing_types = {type(cb) for cb in default_callbacks}
            extra = [cb for cb in callbacks if type(cb) not in existing_types]
            callbacks_to_use = default_callbacks + extra
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_to_use,
            verbose=1
        )
        
        self.is_trained = True
        print(f"✅ 훈련 완료! ({datetime.now().strftime('%H:%M:%S')})")
        return history
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """다중 특성 모델 성능 평가"""
        if not self.is_trained:
            print("❌ 모델이 훈련되지 않았습니다.")
            return None
        
        print("📊 모델 성능 평가 중...")
        
        # 예측
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        # 타겟 스케일러로 역정규화
        train_pred = self.target_scaler.inverse_transform(train_pred)
        test_pred = self.target_scaler.inverse_transform(test_pred)
        y_train_actual = self.target_scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # 성능 지표 계산
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        
        # 추가 성능 지표
        train_mape = np.mean(np.abs((y_train_actual - train_pred) / y_train_actual)) * 100
        test_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100
        
        # R² 스코어
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_train_actual, train_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        
        results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_train_actual': y_train_actual,
            'y_test_actual': y_test_actual
        }
        
        print(f"\n📊 성능 평가 결과:")
        print(f"   🎯 훈련 RMSE: ${train_rmse:.2f}")
        print(f"   🎯 테스트 RMSE: ${test_rmse:.2f}")
        print(f"   📏 테스트 MAE: ${test_mae:.2f}")
        print(f"   📊 테스트 MAPE: {test_mape:.2f}%")
        print(f"   📈 테스트 R²: {test_r2:.4f}")
        print(f"   🔍 사용된 특성: {self.feature_columns}")
        
        return results
    
    def predict_future(self, scaled_data_tuple, price_data, days=5):
        """다중 특성 미래 가격 예측"""
        if not self.is_trained:
            print("❌ 모델이 훈련되지 않았습니다.")
            return None, None, None
        
        scaled_features, scaled_target = scaled_data_tuple
        
        # 훈련 데이터 기간 정보
        train_start_date = price_data.index[0]
        train_end_date = price_data.index[-1]
        
        # 예측 시작 날짜 (마지막 데이터 다음 날)
        last_date = price_data.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=days, 
            freq='D'
        )
        
        print(f"\n🔮 다중 특성 예측 정보:")
        print(f"   📅 훈련 데이터 기간: {train_start_date.strftime('%Y-%m-%d')} ~ {train_end_date.strftime('%Y-%m-%d')}")
        print(f"   📊 총 훈련 일수: {len(price_data)}일")
        print(f"   🎯 예측 기간: {prediction_dates[0].strftime('%Y-%m-%d')} ~ {prediction_dates[-1].strftime('%Y-%m-%d')}")
        print(f"   🔍 사용 특성: {self.feature_columns}")
        
        future_predictions = []
        # 마지막 시퀀스 (모든 특성 포함)
        last_sequence = scaled_features[-self.sequence_length:].copy()
        
        for day in range(days):
            # 현재 시퀀스로 예측
            next_pred = self.model.predict(
                last_sequence.reshape(1, self.sequence_length, self.n_features), 
                verbose=0
            )
            future_predictions.append(next_pred[0, 0])
            
            # 다음 시퀀스를 위해 업데이트
            # 실제 구현에서는 다른 특성들의 미래값도 예측하거나 추정해야 함
            # 여기서는 단순화하여 마지막 값들을 사용
            new_row = last_sequence[-1].copy()
            
            # 타겟 특성 (Close)의 인덱스 찾기
            if 'Close' in self.feature_columns:
                close_idx = self.feature_columns.index('Close')
                new_row[close_idx] = next_pred[0, 0]
            
            # 시퀀스 업데이트 (한 칸씩 이동)
            last_sequence = np.vstack([last_sequence[1:], new_row])
        
        # 타겟 스케일러로 역정규화
        predictions = self.target_scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )
        
        return predictions, prediction_dates, (train_start_date, train_end_date)
    
    def plot_results(self, price_data, results, train_size, save_plots=True):
        """결과 시각화"""
        plt.figure(figsize=(15, 8))
        
        # 전체 데이터
        plt.subplot(2, 1, 1)
        plt.plot(price_data.index, price_data.values, 
                label='Actual Gold Price', alpha=0.7, linewidth=1)
        plt.axvline(x=price_data.index[train_size], color='red', 
                   linestyle='--', label='Train/Test Split', alpha=0.7)
        plt.title('Gold Price - Full Data', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 테스트 구간 예측 결과
        plt.subplot(2, 1, 2)
        test_dates = price_data.index[train_size + self.sequence_length:]
        plt.plot(test_dates, results['y_test_actual'], 
                label='Actual Price', marker='o', markersize=1.5, linewidth=1)
        plt.plot(test_dates, results['test_pred'], 
                label='Predicted Price', marker='x', markersize=1.5, linewidth=1)
        plt.title(f'Test Results (RMSE: ${results["test_rmse"]:.2f})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            import os
            # 현재 작업 디렉토리에 따라 경로 조정
            if os.path.basename(os.getcwd()) == 'analysis':
                save_path = '../results/prediction_results.png'
            else:
                save_path = 'results/prediction_results.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 예측 결과 차트 저장됨: {save_path}")
            return save_path
        else:
            plt.show()

# 사용 예시
if __name__ == "__main__":
    print("🧪 Gold Prediction Model 테스트 시작...")
    
    # 간단한 테스트용 더미 데이터 생성
    from datetime import datetime, timedelta
    
    # 테스트용 금 가격 데이터 생성 (실제로는 gold_data_manager에서 가져옴)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    base_price = 1800
    price_trend = np.cumsum(np.random.randn(len(dates)) * 5) + base_price
    test_data = pd.Series(price_trend, index=dates, name='gold_price')
    
    print(f"📊 테스트 데이터: {len(test_data)}일 치 금 가격")
    
    # 모델 초기화
    model = GoldPredictionModel()
    
    try:
        # 데이터 준비
        X_train, y_train, X_test, y_test, scaled_data, train_size = model.prepare_data(test_data)
        
        # 모델 로드 또는 생성
        needs_training = model.load_or_create_model()
        
        if needs_training:
            print("🎯 모델 훈련이 필요합니다...")
            history = model.train(X_train, y_train, epochs=50, batch_size=16)
        
        # 모델 평가
        results = model.evaluate(X_train, y_train, X_test, y_test)
        
        if results:
            # 결과 시각화
            model.plot_results(test_data, results, train_size)
            
            # 미래 예측
            future_pred, pred_dates, train_info = model.predict_future(scaled_data, test_data, days=7)
            if future_pred is not None:
                print(f"\n🔮 향후 7일 금 가격 예측:")
                for i, (price, date) in enumerate(zip(future_pred.flatten(), pred_dates), 1):
                    print(f"   {i}일 후 ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
            else:
                print("❌ 예측 실패")
        
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()