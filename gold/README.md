# 🏆 Gold Price Prediction System v5.0

AI 기반 금 가격 예측 시스템 - **완전 자동화된 하이퍼파라미터 튜닝** 및 **MLflow 실험 추적**

## 🚀 주요 특징

- **🤖 LSTM 딥러닝 모델**: 다중 기술적 지표를 활용한 시계열 예측
- **🔬 Optuna 자동 튜닝**: 하이퍼파라미터 자동 최적화
- **📊 MLflow 실험 추적**: 모든 실험 자동 로깅 및 버전 관리
- **🎯 목표 기반 최적화**: R² > 0.5 자동 달성 시스템
- **🔍 상관관계 기반 특성 선택**: 점진적 특성 추가로 최적 조합 탐색
- **🤖 앙상블 모델**: 자동 앙상블 생성 및 성능 향상

## 📁 프로젝트 구조

```
gold/
├── main.py                     # 기본 실행 파일 (단순하고 명확한 버전)
├── auto_tuning_system.py      # 🚀 완전 자동화 시스템 (추천)
├── analysis/                   # 분석 모듈들
│   ├── gold_data_manager.py    # 데이터 수집 및 전처리
│   ├── gold_prediction_model.py # LSTM 모델
│   ├── correlation_analysis.py # 상관관계 분석
│   └── charts/                # 상관관계 분석 차트
├── data/                       # 데이터 저장소
│   └── gold_data.csv          # 향상된 금 가격 데이터
├── models/                     # 훈련된 모델 저장소
├── tmp/optuna                  # Optuna 임시 모델/스케일러 저장 (자동 정리)
└── results/                    # 예측 결과 및 차트
```

## 🛠️ 설치 및 설정

### 필수 라이브러리 설치

```bash
# 기본 라이브러리
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn yfinance ta

# 자동화 시스템
pip install optuna mlflow
```

## 🎯 사용법

### 🚀 1. 완전 자동화 시스템 (추천)

CLI 인자 방식(권장):
```bash
cd gold
python auto_tuning_system.py --target-r2 0.6 --max-trials 100
```

환경변수(선택):
```bash
export NAUTILUS_TARGET_R2=0.6
export NAUTILUS_MAX_TRIALS=100
python auto_tuning_system.py
```

**특징:**
- Optuna를 사용한 자동 하이퍼파라미터 튜닝
- MLflow를 통한 실험 추적 및 버전 관리
- R² > 0.5 목표 달성 자동화
- 상관관계 기반 점진적 특성 선택
- 앙상블 모델 자동 생성

### 🎯 2. 기본 실행

```bash
python main.py
```

**특징:**
- 단순하고 명확한 버전
- 빠른 테스트용
- 기본 설정으로 즉시 실행

## 🔬 자동화 시스템 특징

### 🚀 자동화 시스템 (`auto_tuning_system.py`)

**1단계: Optuna 자동 튜닝**
- 하이퍼파라미터 자동 최적화 (LSTM units, Dense units, Dropout, Sequence length, Batch size, Learning rate)
- 상관관계 기반 점진적 특성 선택 (3-15개 특성)
- MLflow를 통한 모든 실험 자동 로깅
- R² > 0.5 목표 달성 시 자동 종료

**2단계: 앙상블 모델 (목표 미달성 시)**
- 5개의 다양한 설정으로 앙상블 모델 생성
- 각 모델의 성능을 평가하여 최적 조합 선택
- 앙상블을 통한 성능 향상

## 📊 실험 결과 해석

### 성능 지표 설명

- **RMSE (Root Mean Square Error)**: 예측 오차 (낮을수록 좋음)
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **MAPE (Mean Absolute Percentage Error)**: 평균 절대 백분율 오차
- **R² Score**: 결정계수 (1에 가까울수록 좋음, 음수면 평균보다 나쁨)
- **과적합 비율**: 테스트 RMSE ÷ 훈련 RMSE (낮을수록 좋음)

## 🎯 권장 사용 시나리오

### 🔰 초보자
1. **`auto_tuning_system.py`** 실행으로 완전 자동화된 최적화
2. MLflow UI로 실험 결과 시각화
3. 목표 달성 여부 확인

### 🔬 연구자/개발자
1. **`auto_tuning_system.py`**로 완전 자동화된 최적화
2. MLflow를 통한 실험 추적 및 모델 버전 관리
3. 앙상블 모델을 통한 성능 극대화

### 🏢 실무자
1. **`auto_tuning_system.py`**로 프로덕션 레벨 모델 구축
2. R² > 0.5 목표 달성으로 신뢰할 수 있는 예측
3. 정기적으로 재실행하여 성능 모니터링

### 🚀 고급 사용자
1. 특정 비즈니스 요구사항에 맞는 모델 개발
2. 다양한 시나리오별 모델 비교 및 선택

## 🔍 MLflow 실험 추적

### MLflow UI 실행

```bash
# MLflow UI 실행 (웹 브라우저에서 확인)
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000

# 또는 절대경로 사용
mlflow ui --backend-store-uri file:/Users/frankie.gg/Documents/nautilus/gold/mlruns --port 5000
```

**MLflow UI에서 확인 가능한 정보:**
- 모든 실험의 하이퍼파라미터와 성능 지표
- 모델 버전 관리 및 비교
- 실험 실행 시간 및 리소스 사용량
- 모델 아티팩트 다운로드

## 📈 기술적 지표 목록

### 가격 지표 (5개)
- Close, High, Low, Open, Volume

### 이동평균 (4개)
- MA_5, MA_10, MA_20, MA_50

### 모멘텀 지표 (7개)
- RSI, MACD, MACD_Signal, MACD_Hist, Stoch_K, Stoch_D, Williams_R

### 변동성 지표 (5개)
- BB_Upper, BB_Lower, BB_Width, BB_Position, Volatility

### 거래량 지표 (2개)
- Volume_MA, Volume_Change

### 가격 변화 지표 (3개)
- Price_Change, Price_Change_5, HL_Spread, CO_Spread

## 🚨 주의사항

1. **과적합 문제**: 과적합 비율이 10배 이상이면 모델 단순화 필요
2. **음의 R² 점수**: 모델이 단순 평균보다 나쁜 상태, 특성 재선택 필요
3. **극단적 예측**: ±20% 이상의 극단적 예측은 신뢰도 낮음
4. **데이터 품질**: 인터넷 연결 필요 (yfinance를 통한 실시간 데이터 수집)

## 🔧 문제 해결

### macOS/Apple Silicon(TensorFlow) 이슈

- 본 프로젝트는 macOS/Apple Silicon 환경에서 TensorFlow 초기화 시 발생할 수 있는 fork/mutex 문제를 완화하기 위해 `gold/tf_compat.py`를 도입했습니다.
- 모든 진입점(`main.py`, `auto_tuning_system.py`)은 TensorFlow를 import하기 전에 부트스트랩을 수행합니다.

사용 팁:

```bash
# GPU/Metal을 기본적으로 비활성화합니다. GPU를 사용하려면 아래처럼 0으로 설정하세요.
export NAUTILUS_TF_DISABLE_GPU=0

# 스레드 수를 제어하여 잠금 경합을 줄일 수 있습니다.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

참고:
- 부트스트랩은 `spawn` start method 설정 및 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` 적용으로 포크-안전 문제를 완화합니다.
- 필요시 `NAUTILUS_TF_DISABLE_GPU=1`로 Metal/GPU를 비활성화해 드라이버 관련 mutex 이슈를 우회할 수 있습니다.

### 성능 개선 팁

1. **특성 수 최적화**: 3-8개가 적절, 너무 많으면 과적합
2. **에포크 수 조정**: 30-50이 적절, 너무 많으면 과적합
3. **드롭아웃 증가**: 과적합 발생 시 0.4-0.5로 증가
4. **시퀀스 길이 조정**: 30-60일이 적절

## 📞 지원

- 버그 리포트: GitHub Issues
- 기능 요청: GitHub Discussions
- 문서 개선: Pull Request 환영

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

🎯 **추천**: 처음 사용하시는 분은 **`python auto_tuning_system.py --target-r2 0.6 --max-trials 100`**으로 시작하세요!