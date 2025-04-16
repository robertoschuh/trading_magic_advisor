import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import ta
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class XAUUSDAnalyzer:
    def __init__(self):
        self.symbol = "GC=F"  # Símbolo de Yahoo Finance para el oro
        self.timeframe = "1h"  # Timeframe para day trading
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ema_short = 9
        self.ema_medium = 21
        self.ema_long = 50
        self.atr_period = 14
        self.risk_reward_ratio = 2  # Relación riesgo/beneficio
        self.risk_percentage = 1    # Porcentaje de riesgo por operación
        

    def get_data(self):
        """Obtiene datos históricos del XAUUSD"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 días de datos históricos
        
        print(f"Descargando datos XAUUSD desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}...")
        data = yf.download(self.symbol, start=start_date, end=end_date, interval=self.timeframe)
        
        if data.empty:
            raise Exception("No se pudieron obtener datos. Verifica tu conexión a internet o el símbolo.")
        
        # Eliminar filas con valores nulos
        data = data.dropna()
        
        # Verificar que las columnas necesarias estén presentes
        required_columns = ['Close', 'High', 'Low']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Faltan columnas necesarias en los datos descargados.")
        
        print(f"Datos descargados: {len(data)} períodos.")
        return data

    def add_indicators(self, data):
        df = data.copy()
        
        # Aseguramos que los datos sean pandas.Series directamente desde el DataFrame
        close_series = df['Close'].astype(float)  # Esto ya es una pandas.Series
        high_series = df['High'].astype(float)
        low_series = df['Low'].astype(float)
        
        # Verificar que no haya valores nulos en las series
        # close_has_nulls = close_series.isna().any()  # Esto devuelve un booleano
        # high_has_nulls = high_series.isna().any()
        # low_has_nulls = low_series.isna().any()
        close_has_nulls = close_series.isna().any().item()
        high_has_nulls = high_series.isna().any().item()
        low_has_nulls = low_series.isna().any().item()
        
        # Depuración: Imprimir los valores para confirmar que son booleanos
        print("Depuración de valores nulos:")
        print(f"close_has_nulls: {close_has_nulls} (tipo: {type(close_has_nulls)})")
        print(f"high_has_nulls: {high_has_nulls} (tipo: {type(high_has_nulls)})")
        print(f"low_has_nulls: {low_has_nulls} (tipo: {type(low_has_nulls)})")
        
        if close_has_nulls or high_has_nulls or low_has_nulls:
            raise ValueError("Las series contienen valores nulos. Por favor, limpia los datos antes de calcular indicadores.")
        
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close_series, window=self.rsi_period)
        df['RSI'] = rsi_indicator.rsi()
        
        # MACD
        macd = ta.trend.MACD(close_series, 
                    window_fast=self.macd_fast, 
                    window_slow=self.macd_slow, 
                    window_sign=self.macd_signal)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # EMAs
        df['EMA_short'] = ta.trend.EMAIndicator(close_series, window=self.ema_short).ema_indicator()
        df['EMA_medium'] = ta.trend.EMAIndicator(close_series, window=self.ema_medium).ema_indicator()
        df['EMA_long'] = ta.trend.EMAIndicator(close_series, window=self.ema_long).ema_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # ATR para cálculo de stop loss
        df['ATR'] = ta.volatility.AverageTrueRange(high_series, low_series, close_series, window=self.atr_period).average_true_range()
        
        # Soporte y resistencia (usando mínimos y máximos recientes)
        df['support'] = df['Low'].rolling(10).min()
        df['resistance'] = df['High'].rolling(10).max()
        
        # Tendencia (basada en EMAs)
        df['trend'] = np.where((df['EMA_short'] > df['EMA_medium']) & (df['EMA_medium'] > df['EMA_long']), 'ALCISTA',
                            np.where((df['EMA_short'] < df['EMA_medium']) & (df['EMA_medium'] < df['EMA_long']), 'BAJISTA', 'LATERAL'))
        
        return df
  
    def generate_signals(self, df):
        """Genera señales de trading basadas en los indicadores"""
        # Inicializar columnas de señales
        df['signal'] = 'NEUTRAL'
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Asegurarse de que tenemos suficientes datos para el análisis
        if len(df) < 2:
            return df
            
        # Últimas filas para análisis actual
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Señal basada en cruces de MACD y confirmada por RSI y tendencia EMA
        if (previous['MACD'] < previous['MACD_Signal'] and current['MACD'] > current['MACD_Signal'] and
            current['RSI'] < 70 and
            current['EMA_short'] > current['EMA_medium']):
            df.loc[df.index[-1], 'signal'] = 'LONG'
            # Stop loss: mínimo reciente o un múltiplo de ATR por debajo del precio
            sl_price = max(current['Low'] - 1.5 * current['ATR'], df['Low'].tail(5).min())
            df.loc[df.index[-1], 'stop_loss'] = round(sl_price, 2)
            # Take profit basado en risk/reward ratio
            risk = current['Close'] - sl_price
            tp_price = current['Close'] + (risk * self.risk_reward_ratio)
            df.loc[df.index[-1], 'take_profit'] = round(tp_price, 2)
            
        elif (previous['MACD'] > previous['MACD_Signal'] and current['MACD'] < current['MACD_Signal'] and
              current['RSI'] > 30 and
              current['EMA_short'] < current['EMA_medium']):
            df.loc[df.index[-1], 'signal'] = 'SHORT'
            # Stop loss: máximo reciente o un múltiplo de ATR por encima del precio
            sl_price = min(current['High'] + 1.5 * current['ATR'], df['High'].tail(5).max())
            df.loc[df.index[-1], 'stop_loss'] = round(sl_price, 2)
            # Take profit basado en risk/reward ratio
            risk = sl_price - current['Close']
            tp_price = current['Close'] - (risk * self.risk_reward_ratio)
            df.loc[df.index[-1], 'take_profit'] = round(tp_price, 2)
        
        return df
    
    def calculate_position_size(self, current_price, stop_loss, account_balance=10000):
        """Calcula el tamaño de la posición basado en la gestión de riesgo"""
        if pd.isna(stop_loss):
            return 0.0
            
        risk_amount = account_balance * (self.risk_percentage / 100)
        pip_risk = abs(current_price - stop_loss)
        
        # Para oro, 1 pip suele ser 0.01
        if pip_risk == 0:
            return 0.0
        position_size = risk_amount / (pip_risk * 100)
        return round(position_size, 2)
    
    def plot_chart(self, df):
        """Visualiza el gráfico con indicadores y señales"""
        plt.figure(figsize=(14, 10))
        
        # Crear subplots
        price_ax = plt.subplot2grid((8, 1), (0, 0), rowspan=4, colspan=1)
        macd_ax = plt.subplot2grid((8, 1), (4, 0), rowspan=2, colspan=1)
        rsi_ax = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1)
        
        # Gráfico de precios
        price_ax.plot(df.index, df['Close'], label='Precio', color='black', linewidth=1.5)
        price_ax.plot(df.index, df['EMA_short'], label=f'EMA {self.ema_short}', color='blue', alpha=0.7)
        price_ax.plot(df.index, df['EMA_medium'], label=f'EMA {self.ema_medium}', color='green', alpha=0.7)
        price_ax.plot(df.index, df['EMA_long'], label=f'EMA {self.ema_long}', color='red', alpha=0.7)
        price_ax.plot(df.index, df['BB_upper'], label='BB Superior', color='gray', linestyle='--', alpha=0.5)
        price_ax.plot(df.index, df['BB_middle'], label='BB Media', color='gray', linestyle='-', alpha=0.5)
        price_ax.plot(df.index, df['BB_lower'], label='BB Inferior', color='gray', linestyle='--', alpha=0.5)
        
        # Marcar señales en el gráfico
        for i, row in df.iterrows():
            if row['signal'] == 'LONG':
                price_ax.scatter(i, row['Close'], color='green', marker='^', s=100)
                if not pd.isna(row['stop_loss']):
                    price_ax.axhline(y=row['stop_loss'], color='red', linestyle='--', alpha=0.5)
                if not pd.isna(row['take_profit']):
                    price_ax.axhline(y=row['take_profit'], color='green', linestyle='--', alpha=0.5)
            elif row['signal'] == 'SHORT':
                price_ax.scatter(i, row['Close'], color='red', marker='v', s=100)
                if not pd.isna(row['stop_loss']):
                    price_ax.axhline(y=row['stop_loss'], color='red', linestyle='--', alpha=0.5)
                if not pd.isna(row['take_profit']):
                    price_ax.axhline(y=row['take_profit'], color='green', linestyle='--', alpha=0.5)
        
        # MACD
        macd_ax.plot(df.index, df['MACD'], label='MACD', color='blue')
        macd_ax.plot(df.index, df['MACD_Signal'], label='Señal', color='red')
        macd_ax.bar(df.index, df['MACD_Hist'], label='Histograma', color='green', alpha=0.5)
        macd_ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # RSI
        rsi_ax.plot(df.index, df['RSI'], label='RSI', color='purple')
        rsi_ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        rsi_ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        rsi_ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        # Formato de ejes
        price_ax.set_title('Análisis Técnico XAUUSD')
        price_ax.set_ylabel('Precio (USD)')
        price_ax.legend()
        price_ax.grid(True, alpha=0.3)
        
        macd_ax.set_ylabel('MACD')
        macd_ax.legend()
        macd_ax.grid(True, alpha=0.3)
        
        rsi_ax.set_ylabel('RSI')
        rsi_ax.set_xlabel('Fecha')
        rsi_ax.legend()
        rsi_ax.grid(True, alpha=0.3)
        
        # Formato de fechas
        for ax in [price_ax, macd_ax, rsi_ax]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def perform_analysis(self):
        """Realiza análisis completo y muestra resultados"""
        try:
            # Obtener datos
            data = self.get_data()
            
            # Añadir indicadores
            df = self.add_indicators(data)
            
            # Generar señales
            df = self.generate_signals(df)
            
            # Verificar que tenemos datos válidos
            if len(df) == 0:
                print("No hay datos suficientes para el análisis.")
                return None
                
            # Obtener datos de la última vela (situación actual)
            current = df.iloc[-1]
            
            # Análisis técnico completo
            print("\n======= ANÁLISIS TÉCNICO XAUUSD =======")
            print(f"Fecha/Hora: {df.index[-1]}")
            print(f"Precio actual: ${current['Close']:.2f}")
            print(f"Tendencia: {current['trend']}")
            print(f"RSI ({self.rsi_period}): {current['RSI']:.2f}")
            print(f"MACD: {current['MACD']:.4f} | Señal: {current['MACD_Signal']:.4f} | Hist: {current['MACD_Hist']:.4f}")
            print(f"EMA {self.ema_short}: ${current['EMA_short']:.2f}")
            print(f"EMA {self.ema_medium}: ${current['EMA_medium']:.2f}")
            print(f"EMA {self.ema_long}: ${current['EMA_long']:.2f}")
            print(f"Bandas de Bollinger: Superior=${current['BB_upper']:.2f} | Media=${current['BB_middle']:.2f} | Inferior=${current['BB_lower']:.2f}")
            print(f"ATR: ${current['ATR']:.2f}")
            
            # Niveles de soporte y resistencia
            print(f"\nSoporte reciente: ${current['support']:.2f}")
            print(f"Resistencia reciente: ${current['resistance']:.2f}")
            
            # Señal de trading
            print("\n======= SEÑAL DE TRADING =======")
            print(f"Señal: {current['signal']}")
            
            if current['signal'] != 'NEUTRAL':
                print(f"Entrada: ${current['Close']:.2f}")
                if not pd.isna(current['stop_loss']):
                    print(f"Stop Loss: ${current['stop_loss']:.2f} (${abs(current['Close'] - current['stop_loss']):.2f} de riesgo)")
                if not pd.isna(current['take_profit']):
                    print(f"Take Profit: ${current['take_profit']:.2f} (${abs(current['take_profit'] - current['Close']):.2f} de beneficio potencial)")
                
                # Calcular tamaño de posición para diferentes balances
                print("\n======= GESTIÓN DE RIESGO =======")
                for balance in [1000, 5000, 10000, 50000]:
                    position = self.calculate_position_size(current['Close'], current['stop_loss'], balance)
                    print(f"Para balance de ${balance}: Tamaño de posición = {position} lotes")
                
                # Análisis de zonas y niveles clave
                print("\n======= ZONAS CLAVE =======")
                if current['signal'] == 'LONG':
                    print(f"Próxima resistencia: ${current['resistance']:.2f}")
                    print(f"Soporte actual: ${current['support']:.2f}")
                elif current['signal'] == 'SHORT':
                    print(f"Próximo soporte: ${current['support']:.2f}")
                    print(f"Resistencia actual: ${current['resistance']:.2f}")
            else:
                print("Sin señal clara de trading en este momento. Esperar mejor configuración.")
                
                # Condiciones para posibles entradas
                print("\n======= POSIBLES CONFIGURACIONES =======")
                if current['RSI'] < 30:
                    print("RSI en zona de sobreventa. Posible entrada LONG si el precio rebota y el MACD cruza al alza.")
                elif current['RSI'] > 70:
                    print("RSI en zona de sobrecompra. Posible entrada SHORT si el precio retrocede y el MACD cruza a la baja.")
                
                if current['Close'] < current['BB_lower']:
                    print("Precio por debajo de la Banda de Bollinger inferior. Posible rebote alcista.")
                elif current['Close'] > current['BB_upper']:
                    print("Precio por encima de la Banda de Bollinger superior. Posible retroceso bajista.")
            
            # Visualizar gráfico
            self.plot_chart(df)
            
            return df
            
        except Exception as e:
            print(f"Error durante el análisis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_real_time_monitoring(self, interval_minutes=5):
        """Inicia monitoreo en tiempo real con actualizaciones periódicas"""
        print(f"Iniciando monitoreo en tiempo real del XAUUSD cada {interval_minutes} minutos...")
        
        try:
            while True:
                print("\n" + "="*50)
                print(f"Actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.perform_analysis()
                print(f"Próxima actualización en {interval_minutes} minutos...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\nMonitoreo detenido por el usuario.")
        except Exception as e:
            print(f"Error en el monitoreo: {e}")
            import traceback
            traceback.print_exc()

# Ejecutar el analizador
if __name__ == "__main__":
    analyzer = XAUUSDAnalyzer()
    
    # Para análisis único
    # analyzer.perform_analysis()
    
    # Para monitoreo en tiempo real (cada 15 minutos)
    analyzer.start_real_time_monitoring(interval_minutes=5)