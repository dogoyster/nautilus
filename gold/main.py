#!/usr/bin/env python3
"""
금 가격 예측 시스템 - 단순하고 명확한 버전
사용법: python main.py
"""

import warnings
import os

# TensorFlow/macOS compatibility bootstrap BEFORE importing modules that import TensorFlow
from tf_compat import bootstrap, configure_tensorflow
bootstrap()

# Now safe to import modules that transitively import TensorFlow
from analysis.gold_data_manager import GoldDataManager
from analysis.gold_prediction_model import GoldPredictionModel

# Apply TensorFlow runtime configuration (threads, optional GPU disable)
configure_tensorflow()
warnings.filterwarnings('ignore')

# 설정
CONFIG = {
    # 데이터 설정
    'data_file': 'data/gold_data.csv',
    
    # 모델 설정
    'model_file': 'models/gold_model.h5',
    'scaler_file': 'models/gold_scaler',
    
    # 특성 설정 (하이퍼파라미터 튜닝 결과 기반)
    'features': ['Close', 'MA_20', 'RSI', 'Volatility'],  # 4개 특성 (튜닝 결과 최적)
    # 'features': ['Close', 'MA_20', 'RSI'],  # 3개 특성 (단순함)
    # 'features': ['Close', 'MA_20', 'RSI', 'MACD', 'BB_Position'],  # 5개 특성
    
    # 하이퍼파라미터
    'sequence_length': 30,  # 30일 데이터로 예측
    'epochs': 20,          # 빠른 훈련
    'batch_size': 32,
    'predict_days': 5,     # 5일 예측
    
    # 모델 구조
    'lstm_units': [16],    # 단순한 구조 (과적합 방지)
    'dense_units': [8],
    'dropout_rates': [0.3, 0.2],
    
    # 기타
    'retrain': True,       # True: 새로 훈련, False: 기존 모델 사용
    'save_results': True   # 결과 저장 여부
}

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = ['models', 'results', 'analysis/charts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """메인 실행 함수"""
    print("🏆 금 가격 예측 시스템")
    print("="*50)
    
    # 디렉토리 설정
    setup_directories()
    
    # 1. 데이터 준비
    print("\n📈 1단계: 데이터 준비")
    data_manager = GoldDataManager(data_file=CONFIG['data_file'])
    enhanced_data = data_manager.load_or_download_data()
    
    if enhanced_data is None:
        print("❌ 데이터 로드 실패")
        return
    
    data_manager.get_data_info()
    
    # 2. 모델 준비
    print(f"\n🤖 2단계: 모델 준비")
    predictor = GoldPredictionModel(
        sequence_length=CONFIG['sequence_length'],
        model_name=CONFIG['model_file'],
        scaler_name=CONFIG['scaler_file'],
        feature_columns=CONFIG['features']
    )
    
    # 하이퍼파라미터 설정
    predictor.lstm_units = CONFIG['lstm_units']
    predictor.dense_units = CONFIG['dense_units']
    predictor.dropout_rates = CONFIG['dropout_rates']
    
    print(f"   📊 사용할 특성: {CONFIG['features']}")
    print(f"   🔢 특성 개수: {len(CONFIG['features'])}개")
    
    # 3. 데이터 전처리
    print(f"\n🔧 3단계: 데이터 전처리")
    result = predictor.prepare_data(enhanced_data, train_ratio=0.8, target_column='Close')
    
    if result is None:
        print("❌ 데이터 전처리 실패")
        return
        
    X_train, y_train, X_test, y_test, scaled_data, train_size = result
    
    # 4. 모델 훈련/로드
    print(f"\n🎯 4단계: 모델 훈련/로드")
    
    if CONFIG['retrain'] or not os.path.exists(CONFIG['model_file']):
        print("🔧 새 모델 훈련 중...")
        predictor.build_model()
        history = predictor.train(
            X_train, y_train, 
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size']
        )
        print("✅ 훈련 완료!")
    else:
        print("📁 기존 모델 로드 중...")
        predictor.load_or_create_model()
        print("✅ 모델 로드 완료!")
    
    # 5. 성능 평가
    print(f"\n📊 5단계: 성능 평가")
    results = predictor.evaluate(X_train, y_train, X_test, y_test)
    
    print(f"\n📊 성능 결과:")
    print(f"   🎯 훈련 RMSE: ${results['train_rmse']:.2f}")
    print(f"   🎯 테스트 RMSE: ${results['test_rmse']:.2f}")
    print(f"   📏 테스트 MAE: ${results['test_mae']:.2f}")
    print(f"   📊 테스트 MAPE: {results['test_mape']:.2f}%")
    print(f"   📈 테스트 R²: {results['test_r2']:.4f}")
    print(f"   🔍 과적합 비율: {results['test_rmse']/results['train_rmse']:.1f}x")
    
    # 6. 결과 시각화
    print(f"\n📈 6단계: 결과 시각화")
    close_prices = enhanced_data['Close']
    chart_path = predictor.plot_results(close_prices, results, train_size)
    print(f"📊 예측 차트 저장: {chart_path}")
    
    # 7. 미래 예측
    print(f"\n🔮 7단계: 미래 예측")
    future_prices, prediction_dates, train_period = predictor.predict_future(
        scaled_data, enhanced_data, days=CONFIG['predict_days']
    )
    
    if future_prices is not None:
        current_price = float(close_prices.iloc[-1])
        last_date = close_prices.index[-1].strftime('%Y-%m-%d')
        
        print(f"\n📊 예측 결과:")
        print(f"   💰 현재 가격: ${current_price:.2f} ({last_date})")
        print(f"   📅 훈련 기간: {train_period}")
        print(f"\n🔮 향후 {CONFIG['predict_days']}일 예측:")
        
        for i, (price, date) in enumerate(zip(future_prices.flatten(), prediction_dates), 1):
            change = price - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   {i}일 후 ({date.strftime('%Y-%m-%d')}): ${price:.2f} ({direction} {change:+.2f}, {change_pct:+.1f}%)")
    
    # 8. 결과 저장
    if CONFIG['save_results']:
        print(f"\n💾 8단계: 결과 저장")
        
        # 결과 텍스트 파일 저장
        result_file = f"results/prediction_results.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("금 가격 예측 결과\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"설정 정보:\n")
            f.write(f"  특성: {CONFIG['features']}\n")
            f.write(f"  특성 개수: {len(CONFIG['features'])}개\n")
            f.write(f"  시퀀스 길이: {CONFIG['sequence_length']}일\n")
            f.write(f"  훈련 에포크: {CONFIG['epochs']}\n\n")
            
            f.write(f"성능 결과:\n")
            f.write(f"  훈련 RMSE: ${results['train_rmse']:.2f}\n")
            f.write(f"  테스트 RMSE: ${results['test_rmse']:.2f}\n")
            f.write(f"  테스트 MAE: ${results['test_mae']:.2f}\n")
            f.write(f"  테스트 MAPE: {results['test_mape']:.2f}%\n")
            f.write(f"  테스트 R²: {results['test_r2']:.4f}\n")
            f.write(f"  과적합 비율: {results['test_rmse']/results['train_rmse']:.1f}x\n\n")
            
            f.write(f"예측 결과:\n")
            f.write(f"  현재 가격: ${current_price:.2f} ({last_date})\n")
            f.write(f"  훈련 기간: {train_period}\n\n")
            
            if future_prices is not None:
                f.write(f"향후 {CONFIG['predict_days']}일 예측:\n")
                for i, (price, date) in enumerate(zip(future_prices.flatten(), prediction_dates), 1):
                    change = price - current_price
                    change_pct = (change / current_price) * 100
                    f.write(f"  {i}일 후 ({date.strftime('%Y-%m-%d')}): ${price:.2f} ({change:+.2f}, {change_pct:+.1f}%)\n")
        
        print(f"📄 결과 파일 저장: {result_file}")
        print(f"📊 차트 파일 저장: {chart_path}")
    
    print(f"\n✅ 모든 작업 완료!")
    print(f"📁 생성된 파일:")
    print(f"   - 모델: {CONFIG['model_file']}")
    print(f"   - 스케일러: {CONFIG['scaler_file']}_*.pkl")
    if CONFIG['save_results']:
        print(f"   - 결과: results/prediction_results.txt")
        print(f"   - 차트: {chart_path}")

if __name__ == "__main__":
    main()