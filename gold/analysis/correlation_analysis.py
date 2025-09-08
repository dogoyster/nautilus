#!/usr/bin/env python3
"""
금 데이터 상관행렬 분석 스크립트
Gold Data Correlation Matrix Analysis

이 스크립트는 금 가격 데이터의 여러 feature들 간의 상관관계를 분석합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib)
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data(file_path):
    """
    데이터를 로드하고 전처리합니다.
    """
    print("데이터 로딩 중...")
    df = pd.read_csv(file_path)
    
    # 헤더 행들 제거 (Ticker, Date 행)
    df = df.iloc[2:].copy()
    
    # Date 컬럼 추가 (Price 컬럼에서 날짜 정보 추출)
    df['Date'] = pd.to_datetime(df['Price'], errors='coerce')
    
    # 모든 숫자 컬럼들을 float로 변환
    numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'MA_5', 'MA_20', 'MA_60', 
                      'RSI', 'Volatility', 'Volume_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                      'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
                      'Stoch_K', 'Stoch_D', 'Williams_R', 'Price_Change', 'Price_Change_5', 
                      'Price_Change_20', 'Volume_Change', 'HL_Spread', 'CO_Spread']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Price 컬럼을 Close 가격으로 설정
    df['Price'] = df['Close']
    
    # NaN이 있는 행 제거
    df = df.dropna()
    
    print(f"전처리 후 데이터 크기: {df.shape}")
    return df

def get_feature_groups():
    """
    Feature들을 그룹별로 분류합니다.
    """
    feature_groups = {
        'Price_Features': ['Price', 'Close', 'High', 'Low', 'Open'],
        'Volume_Features': ['Volume', 'Volume_MA', 'Volume_Change'],
        'Moving_Averages': ['MA_5', 'MA_20', 'MA_60'],
        'Technical_Indicators': ['RSI', 'Volatility', 'MACD', 'MACD_Signal', 'MACD_Histogram'],
        'Bollinger_Bands': ['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'],
        'Stochastic': ['Stoch_K', 'Stoch_D', 'Williams_R'],
        'Price_Changes': ['Price_Change', 'Price_Change_5', 'Price_Change_20'],
        'Spreads': ['HL_Spread', 'CO_Spread']
    }
    return feature_groups

def calculate_correlation_matrix(df, features=None):
    """
    상관행렬을 계산합니다.
    """
    if features is None:
        # 숫자형 컬럼만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        features = numeric_df.columns.tolist()
    
    correlation_matrix = df[features].corr()
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix, title="Correlation Matrix", figsize=(15, 12)):
    """
    상관행렬 히트맵을 그립니다.
    """
    plt.figure(figsize=figsize)
    
    # 마스크 생성 (상삼각형 숨기기)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # 히트맵 그리기
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()

def analyze_high_correlations(correlation_matrix, threshold=0.8):
    """
    높은 상관관계를 가진 feature 쌍들을 분석합니다.
    """
    print(f"\n=== 높은 상관관계 분석 (임계값: {threshold}) ===")
    
    # 상삼각형만 고려 (중복 제거)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))
    
    # 상관계수 절댓값으로 정렬
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if high_corr_pairs:
        print(f"발견된 높은 상관관계 쌍: {len(high_corr_pairs)}개")
        print("\nFeature 1\t\t\tFeature 2\t\t\tCorrelation")
        print("-" * 70)
        for feature1, feature2, corr in high_corr_pairs:
            print(f"{feature1:<25}\t{feature2:<25}\t{corr:.4f}")
    else:
        print("높은 상관관계를 가진 feature 쌍이 없습니다.")
    
    return high_corr_pairs

def plot_feature_group_correlations(df, feature_groups):
    """
    Feature 그룹별 상관관계를 시각화합니다.
    """
    n_groups = len(feature_groups)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (group_name, features) in enumerate(feature_groups.items()):
        if idx >= len(axes):
            break
            
        # 해당 그룹의 feature들만 선택
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            axes[idx].text(0.5, 0.5, f'{group_name}\n(Not enough features)', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(group_name)
            continue
        
        group_corr = df[available_features].corr()
        
        # 작은 히트맵 그리기
        sns.heatmap(group_corr, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar=False,
                   ax=axes[idx],
                   annot_kws={'size': 6})
        
        axes[idx].set_title(group_name, fontsize=10, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', labelsize=6)
    
    # 빈 subplot 숨기기
    for idx in range(len(feature_groups), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def generate_correlation_report(df, correlation_matrix, high_corr_pairs):
    """
    상관관계 분석 보고서를 생성합니다.
    """
    report = []
    report.append("=" * 60)
    report.append("금 데이터 상관행렬 분석 보고서")
    report.append("=" * 60)
    report.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"데이터 기간: {df['Date'].min()} ~ {df['Date'].max()}")
    report.append(f"총 데이터 포인트: {len(df)}")
    report.append(f"분석된 Feature 수: {len(correlation_matrix.columns)}")
    report.append("")
    
    # Feature 목록
    report.append("분석된 Features:")
    for i, feature in enumerate(correlation_matrix.columns, 1):
        report.append(f"  {i:2d}. {feature}")
    report.append("")
    
    # 높은 상관관계 요약
    report.append("높은 상관관계 요약 (|r| >= 0.8):")
    if high_corr_pairs:
        for feature1, feature2, corr in high_corr_pairs[:10]:  # 상위 10개만
            report.append(f"  • {feature1} ↔ {feature2}: {corr:.4f}")
    else:
        report.append("  • 높은 상관관계를 가진 feature 쌍이 없습니다.")
    report.append("")
    
    # 통계 요약
    corr_values = correlation_matrix.values
    corr_values = corr_values[~np.eye(corr_values.shape[0], dtype=bool)]  # 대각선 제거
    
    report.append("상관계수 통계:")
    report.append(f"  • 평균: {np.mean(corr_values):.4f}")
    report.append(f"  • 표준편차: {np.std(corr_values):.4f}")
    report.append(f"  • 최대값: {np.max(corr_values):.4f}")
    report.append(f"  • 최소값: {np.min(corr_values):.4f}")
    report.append("")
    
    return "\n".join(report)

def main():
    """
    메인 실행 함수
    """
    print("금 데이터 상관행렬 분석을 시작합니다...")
    
    # 1. 데이터 로드
    df = load_and_clean_data('../data/gold_data.csv')
    
    # 2. Feature 그룹 정의
    feature_groups = get_feature_groups()
    
    # 3. 전체 상관행렬 계산
    print("\n상관행렬 계산 중...")
    correlation_matrix = calculate_correlation_matrix(df)
    
    # 4. 전체 상관행렬 히트맵
    print("전체 상관행렬 히트맵 생성 중...")
    fig1 = plot_correlation_heatmap(correlation_matrix, "Gold Data - Full Correlation Matrix")
    fig1.savefig('../visualizations/gold_correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    print("저장됨: ../visualizations/gold_correlation_matrix_full.png")
    
    # 5. 높은 상관관계 분석
    high_corr_pairs = analyze_high_correlations(correlation_matrix, threshold=0.8)
    
    # 6. Feature 그룹별 상관관계 시각화
    print("\nFeature 그룹별 상관관계 분석 중...")
    fig2 = plot_feature_group_correlations(df, feature_groups)
    fig2.savefig('../visualizations/gold_correlation_by_groups.png', dpi=300, bbox_inches='tight')
    print("저장됨: ../visualizations/gold_correlation_by_groups.png")
    
    # 7. 주요 feature들만 선별하여 상관행렬 생성
    key_features = ['Price', 'Volume', 'MA_20', 'RSI', 'Volatility', 'MACD', 
                   'BB_Position', 'Stoch_K', 'Price_Change', 'Volume_Change']
    available_key_features = [f for f in key_features if f in df.columns]
    
    if len(available_key_features) >= 2:
        print("\n주요 Feature 상관행렬 생성 중...")
        key_correlation_matrix = calculate_correlation_matrix(df, available_key_features)
        fig3 = plot_correlation_heatmap(key_correlation_matrix, 
                                       "Gold Data - Key Features Correlation Matrix",
                                       figsize=(10, 8))
        fig3.savefig('../visualizations/gold_correlation_key_features.png', dpi=300, bbox_inches='tight')
        print("저장됨: ../visualizations/gold_correlation_key_features.png")
    
    # 8. 보고서 생성
    print("\n분석 보고서 생성 중...")
    report = generate_correlation_report(df, correlation_matrix, high_corr_pairs)
    
    with open('../reports/correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("저장됨: ../reports/correlation_analysis_report.txt")
    
    # 9. 상관행렬을 CSV로 저장
    correlation_matrix.to_csv('../data/gold_correlation_matrix.csv')
    print("저장됨: ../data/gold_correlation_matrix.csv")
    
    print("\n" + "="*60)
    print("상관행렬 분석 완료!")
    print("="*60)
    print("생성된 파일들:")
    print("  • ../visualizations/gold_correlation_matrix_full.png - 전체 상관행렬 히트맵")
    print("  • ../visualizations/gold_correlation_by_groups.png - 그룹별 상관행렬")
    print("  • ../visualizations/gold_correlation_key_features.png - 주요 feature 상관행렬")
    print("  • ../reports/correlation_analysis_report.txt - 분석 보고서")
    print("  • ../data/gold_correlation_matrix.csv - 상관행렬 데이터")
    print("\n분석 보고서 미리보기:")
    print("-" * 40)
    print(report)

if __name__ == "__main__":
    main()
