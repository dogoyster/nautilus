# nautilus
AI 기반 트레이드 시스템

## Binance CLI (테스트넷 기본)

이 저장소에는 Binance 거래를 위한 간단한 CLI가 포함되어 있으며, 기본값은 안전을 위해 Binance Testnet입니다.

### 사전 요구사항

- Python 3.10 이상 권장

### 설치 (가상환경 포함)

1) 가상환경 생성 및 활성화 (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) 패키지 설치 (편집 가능 모드)

```bash
pip install -e .
```

3) 환경변수 설정 (둘 중 하나 선택)

- .env 파일 사용:

```bash
cat > .env <<'EOF'
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
BINANCE_USE_TESTNET=1
EOF
```

- 명령 실행 시 인라인 설정:

```bash
BINANCE_API_KEY=YOUR_KEY BINANCE_API_SECRET=YOUR_SECRET BINANCE_USE_TESTNET=1 \
  python -m binance.cli price BTCUSDT
```

### 사용법

- 도움말 보기

```bash
python -m binance.cli --help
```

- 시세 조회

```bash
python -m binance.cli price BTCUSDT
```

- 잔고 조회 (0이 아닌 잔고)

```bash
python -m binance.cli balances
```

- 시장가 매수/매도 (수량은 기초자산 수량)

```bash
python -m binance.cli buy BTCUSDT 0.001
python -m binance.cli sell BTCUSDT 0.001
```

- 지정가 매수/매도 (특정 가격에 주문)

```bash
python -m binance.cli limit-buy BTCUSDT 0.001 50000
python -m binance.cli limit-sell BTCUSDT 0.001 60000
```

- 손절매/익절매 주문

```bash
python -m binance.cli stop-loss BTCUSDT 0.001 45000
python -m binance.cli take-profit BTCUSDT 0.001 65000
```

- 미체결 주문 조회 / 주문 취소

```bash
python -m binance.cli orders
python -m binance.cli cancel BTCUSDT <ORDER_ID>
```

참고: 안정적으로 사용하려면 `python -m binance.cli ...` 형태를 권장합니다. 패키지 스크립트 엔트리(`binance-cli`)도 함께 설치됩니다.

### 추가 정보

- 기본 엔드포인트는 Binance Testnet(`https://testnet.binance.vision`)입니다. 실거래를 원한다면 `.env` 또는 인라인 환경변수에서 `BINANCE_USE_TESTNET=0`으로 변경하세요.
- 거래에는 위험이 있습니다. 반드시 테스트넷에서 충분히 검증 후 실거래로 전환하세요.
- 시장가 주문은 최소 주문 수량/금액, 자산 정밀도 등의 제약을 받습니다. 조건을 충족하지 못하면 거래소에서 거절될 수 있습니다.
