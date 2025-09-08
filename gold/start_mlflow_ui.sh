#!/bin/bash
# MLflow UI 시작 스크립트

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

# MLflow UI 시작 (기본 포트 5000)
echo "🚀 MLflow UI 시작 중..."
echo "📊 데이터 저장소: $(pwd)/mlruns"
echo "🌐 접속 URL: http://localhost:5000"
echo ""

mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host localhost --port 5000
