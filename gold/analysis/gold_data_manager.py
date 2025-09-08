import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GoldDataManager:
    """금 가격 데이터 수집 및 관리 클래스"""
    
    def __init__(self, data_file='../data/gold_data.csv', symbol='GC=F'):
        self.data_file = data_file
        self.symbol = symbol
        self.data = None
        
    def load_or_download_data(self, start_date='2020-01-01', end_date=None, 
                             force_update=False, update_recent_days=7):
        """
        향상된 데이터를 CSV에서 불러오거나 새로 생성
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜 (None이면 오늘)
            force_update: True면 전체 재다운로드
            update_recent_days: 최근 며칠 데이터만 업데이트할지
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # CSV 파일이 존재하고 강제 업데이트가 아닌 경우
        if os.path.exists(self.data_file) and not force_update:
            print(f"📁 기존 향상된 데이터 파일 발견: {self.data_file}")
            try:
                # 멀티헤더 CSV 호환 로딩 시도
                try:
                    df = pd.read_csv(self.data_file, header=[0, 1], index_col=0)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [lvl0 for (lvl0, *rest) in df.columns.values]
                except Exception:
                    df = pd.read_csv(self.data_file, index_col=0)
                
                # 인덱스를 날짜형으로 보정하고 잘못된 헤더 행 제거
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.isna()]
                
                # 컬럼을 문자열로 통일
                df.columns = [str(c) for c in df.columns]
                self.data = df
                
                # 데이터가 비어있는지 확인
                if self.data.empty:
                    print("⚠️ CSV 파일이 비어있습니다. 새로 생성합니다.")
                    self._download_and_enhance_data(start_date, end_date)
                    return self.data
                
                # 기술적 지표가 포함되어 있는지 확인
                if 'MA_5' not in self.data.columns:
                    print("⚠️ 기술적 지표가 없습니다. 향상된 데이터를 새로 생성합니다.")
                    self._download_and_enhance_data(start_date, end_date)
                    return self.data
                
                # 최신 데이터 확인
                last_date = self.data.index[-1].strftime('%Y-%m-%d')
                print(f"📅 마지막 데이터: {last_date}")
                
                # 최근 데이터가 오래되었으면 업데이트
                if self._need_update(last_date, update_recent_days):
                    print(f"🔄 최근 {update_recent_days}일 데이터 업데이트 중...")
                    self._update_enhanced_data(update_recent_days)
                else:
                    print("✅ 향상된 데이터가 최신입니다!")
                    
            except Exception as e:
                print(f"⚠️ CSV 파일 읽기 실패: {e}")
                print("새로 생성합니다.")
                self._download_and_enhance_data(start_date, end_date)
                
        else:
            # 새로 다운로드 및 향상된 데이터 생성
            print(f"⬇️ 새 데이터 다운로드 및 기술적 지표 생성 중: {start_date} ~ {end_date}")
            self._download_and_enhance_data(start_date, end_date)
            
        return self.data
    
    def _need_update(self, last_date, update_days):
        """데이터 업데이트가 필요한지 확인"""
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        cutoff_dt = datetime.now() - timedelta(days=update_days)
        return last_dt < cutoff_dt
    
    def _download_and_enhance_data(self, start_date, end_date):
        """데이터 다운로드 후 기술적 지표 추가하여 저장"""
        try:
            print("📊 기본 데이터 다운로드 중...")
            raw_data = yf.download(self.symbol, start=start_date, end=end_date)
            basic_data = raw_data.dropna()
            
            if basic_data.empty:
                print("❌ 다운로드된 데이터가 없습니다.")
                return None
            
            print(f"✅ 기본 데이터 다운로드 완료: {len(basic_data)}개 포인트")
            
            # 기술적 지표 추가
            print("🔧 기술적 지표 계산 중...")
            enhanced_data = self._add_technical_indicators(basic_data)
            
            if enhanced_data is not None:
                self.data = enhanced_data
                # 저장 전 정규화: 단일 헤더, 날짜 인덱스 이름 지정
                if isinstance(self.data.columns, pd.MultiIndex):
                    self.data.columns = [c[0] if isinstance(c, tuple) else c for c in self.data.columns]
                self.data.index = pd.to_datetime(self.data.index)
                self.data.index.name = 'Date'
                # CSV로 저장
                self.data.to_csv(self.data_file, index=True)
                print(f"💾 향상된 데이터 저장됨: {self.data_file}")
                print(f"📊 총 {len(self.data)}개 데이터 포인트, {len(self.data.columns)}개 컬럼")
            else:
                print("❌ 기술적 지표 생성 실패")
                
        except Exception as e:
            print(f"❌ 데이터 처리 실패: {e}")
            return None
    
    def _update_enhanced_data(self, days):
        """최근 며칠 향상된 데이터 업데이트"""
        try:
            # 최근 며칠 전부터 오늘까지 다운로드
            start_update = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_update = datetime.now().strftime('%Y-%m-%d')
            
            print("📊 최신 기본 데이터 다운로드 중...")
            new_basic_data = yf.download(self.symbol, start=start_update, end=end_update)
            new_basic_data = new_basic_data.dropna()
            
            if not new_basic_data.empty:
                # 기존 기본 데이터 부분 추출 (OHLCV만)
                basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                existing_basic = self.data[basic_columns].copy()
                
                # 새 데이터와 병합
                combined_basic = pd.concat([existing_basic, new_basic_data]).drop_duplicates()
                combined_basic = combined_basic.sort_index()
                
                # 전체 데이터에 기술적 지표 다시 계산
                print("🔧 기술적 지표 재계산 중...")
                enhanced_data = self._add_technical_indicators(combined_basic)
                
                if enhanced_data is not None and not enhanced_data.empty:
                    self.data = enhanced_data
                    if isinstance(self.data.columns, pd.MultiIndex):
                        self.data.columns = [c[0] if isinstance(c, tuple) else c for c in self.data.columns]
                    self.data.index = pd.to_datetime(self.data.index)
                    self.data.index.name = 'Date'
                    self.data.to_csv(self.data_file, index=True)
                    print(f"✅ 향상된 데이터 업데이트 완료: +{len(new_basic_data)}개 새 데이터")
                else:
                    print("ℹ️ 업데이트 결과가 비어있어 기존 데이터를 유지합니다.")
            else:
                print("ℹ️ 새로운 데이터가 없습니다.")
                
        except Exception as e:
            print(f"❌ 데이터 업데이트 실패: {e}")
    
    def get_close_price(self):
        """종가 데이터만 반환"""
        if self.data is None:
            print("❌ 데이터가 없습니다. load_or_download_data()를 먼저 실행하세요.")
            return None
            
        if self.data.empty:
            print("❌ 데이터가 비어있습니다.")
            return None
            
        if 'Close' not in self.data.columns:
            print(f"❌ 'Close' 컬럼이 없습니다. 사용 가능한 컬럼: {list(self.data.columns)}")
            return None
            
        return self.data['Close'].dropna()
    
    def get_full_data(self):
        """전체 OHLCV 데이터 반환"""
        return self.data
    
    def _add_technical_indicators(self, basic_data):
        """기본 OHLCV 데이터에 모든 기술적 지표 추가"""
        print("🔧 기술적 지표 계산 중...")
        df = basic_data.copy()
        
        # 기본 가격 정보
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        # 1. 이동평균 (Moving Averages)
        print("   📊 이동평균 계산 중...")
        df['MA_5'] = close.rolling(window=5).mean()
        df['MA_20'] = close.rolling(window=20).mean()
        df['MA_60'] = close.rolling(window=60).mean()
        
        # 2. RSI (Relative Strength Index)
        print("   📊 RSI 계산 중...")
        df['RSI'] = self._calculate_rsi(close, window=14)
        
        # 3. 변동성 (Volatility)
        print("   📊 변동성 계산 중...")
        df['Volatility'] = close.rolling(window=20).std()
        
        # 4. 거래량 이동평균 (Volume Moving Average)
        print("   📊 거래량 이동평균 계산 중...")
        df['Volume_MA'] = volume.rolling(window=20).mean()
        
        # 5. MACD (Moving Average Convergence Divergence)
        print("   📊 MACD 계산 중...")
        macd_data = self._calculate_macd(close)
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Histogram'] = macd_data['Histogram']
        
        # 6. 볼린저 밴드 (Bollinger Bands)
        print("   📊 볼린저 밴드 계산 중...")
        bb_data = self._calculate_bollinger_bands(close)
        df['BB_Upper'] = bb_data['Upper']
        df['BB_Middle'] = bb_data['Middle']
        df['BB_Lower'] = bb_data['Lower']
        df['BB_Width'] = bb_data['Width']
        df['BB_Position'] = bb_data['Position']
        
        # 7. 스토캐스틱 (Stochastic Oscillator)
        print("   📊 스토캐스틱 계산 중...")
        stoch_data = self._calculate_stochastic(high, low, close)
        df['Stoch_K'] = stoch_data['%K']
        df['Stoch_D'] = stoch_data['%D']
        
        # 8. 윌리엄 %R (Williams %R)
        print("   📊 윌리엄 %R 계산 중...")
        df['Williams_R'] = self._calculate_williams_r(high, low, close)
        
        # 9. 추가 유용한 지표들
        print("   📊 추가 지표 계산 중...")
        
        # 가격 변화율
        df['Price_Change'] = close.pct_change()
        df['Price_Change_5'] = close.pct_change(periods=5)
        df['Price_Change_20'] = close.pct_change(periods=20)
        
        # 거래량 변화율
        df['Volume_Change'] = volume.pct_change()
        
        # 고가-저가 스프레드
        df['HL_Spread'] = (high - low) / close
        
        # 종가-시가 스프레드
        df['CO_Spread'] = (close - df['Open']) / df['Open']
        
        # 데이터 정리
        print("🧹 데이터 정리 중...")
        
        # 1. 무한대 값을 NaN으로 변경
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 모든 컬럼을 숫자 타입으로 변환
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. NaN 값 제거
        enhanced_data = df.dropna()
        
        print(f"✅ 기술적 지표 계산 완료! 총 {len(enhanced_data.columns)}개 컬럼")
        print(f"   📊 추가된 지표 수: {len(enhanced_data.columns) - len(basic_data.columns)}개")
        print(f"   🧹 정리 후 데이터 포인트: {len(enhanced_data)}개")
        
        return enhanced_data
    
    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'MACD': macd,
            'Signal': macd_signal,
            'Histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """볼린저 밴드 계산"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_width = (upper_band - lower_band) / rolling_mean
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return {
            'Upper': upper_band,
            'Middle': rolling_mean,
            'Lower': lower_band,
            'Width': bb_width,
            'Position': bb_position
        }
    
    def _calculate_stochastic(self, high, low, close, k_window=14, d_window=3):
        """스토캐스틱 오실레이터 계산"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    def _calculate_williams_r(self, high, low, close, window=14):
        """윌리엄 %R 계산"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def get_feature_list(self):
        """사용 가능한 모든 특성 목록 반환"""
        if self.data is None:
            print("❌ 데이터가 없습니다. load_or_download_data()를 먼저 실행하세요.")
            return None
            
        features = self.data.columns.tolist()
        
        # 카테고리별로 분류
        basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        ma_features = [f for f in features if f.startswith('MA_')]
        technical_features = ['RSI', 'Volatility', 'Volume_MA']
        macd_features = [f for f in features if f.startswith('MACD')]
        bb_features = [f for f in features if f.startswith('BB_')]
        stoch_features = [f for f in features if f.startswith('Stoch_')]
        other_features = [f for f in features if f not in basic_features + ma_features + technical_features + macd_features + bb_features + stoch_features]
        
        print(f"\n📊 사용 가능한 특성들 (총 {len(features)}개):")
        print(f"   🔹 기본 OHLCV: {basic_features}")
        print(f"   🔹 이동평균: {ma_features}")
        print(f"   🔹 기본 기술지표: {technical_features}")
        print(f"   🔹 MACD 관련: {macd_features}")
        print(f"   🔹 볼린저 밴드: {bb_features}")
        print(f"   🔹 스토캐스틱: {stoch_features}")
        print(f"   🔹 기타: {other_features}")
        
        return {
            'all': features,
            'basic': basic_features,
            'ma': ma_features,
            'technical': technical_features,
            'macd': macd_features,
            'bb': bb_features,
            'stoch': stoch_features,
            'other': other_features
        }
    
    def get_data_info(self):
        """데이터 정보 출력"""
        if self.data is None:
            print("❌ 데이터가 없습니다.")
            return
            
        # 데이터가 비어있는지 확인
        if self.data.empty:
            print("❌ 데이터가 비어있습니다.")
            return
            
        # Close 컬럼이 있는지 확인
        if 'Close' not in self.data.columns:
            print(f"❌ 'Close' 컬럼이 없습니다. 사용 가능한 컬럼: {list(self.data.columns)}")
            return
            
        print(f"\n📊 {self.symbol} 향상된 데이터 정보:")
        print(f"📅 기간: {self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"📈 총 일수: {len(self.data)}일")
        print(f"💰 최신 가격: ${float(self.data['Close'].iloc[-1]):.2f}")
        print(f"📁 데이터 파일: {self.data_file}")
        print(f"🔢 총 특성 수: {len(self.data.columns)}개")
        
        # 파일 크기 확인
        if os.path.exists(self.data_file):
            print(f"💾 파일 크기: {os.path.getsize(self.data_file) / 1024:.1f}KB")
        
        # 기술적 지표 포함 여부 확인
        technical_indicators = ['MA_5', 'RSI', 'MACD', 'BB_Position', 'Stoch_K']
        has_indicators = any(col in self.data.columns for col in technical_indicators)
        print(f"🔧 기술적 지표: {'포함됨' if has_indicators else '없음'}")

# 사용 예시
if __name__ == "__main__":
    print("🏆 Gold Data Manager v3.0 - Simplified Edition")
    print("="*60)
    
    # 1. 데이터 관리자 초기화
    gold_manager = GoldDataManager()
    
    # 2. 향상된 데이터 로드 (기술적 지표 포함)
    print("\n📈 1단계: 향상된 데이터 로드")
    data = gold_manager.load_or_download_data()
    
    # 3. 정보 확인
    gold_manager.get_data_info()
    
    # 4. 특성 목록 확인
    print("\n📊 2단계: 특성 목록 확인")
    features = gold_manager.get_feature_list()
    
    print("\n✅ 모든 작업 완료!")
    print("="*60)