import json
import os
from backtester.vector_engine import VectorEngine
from analytics.llm_analyzer import client # Using our modernized Gemini client
from google.genai import types

class Laboratory:
    def __init__(self):
        self.engine = VectorEngine()
        
    def generate_blueprint(self, symbols, timeframe='1h'):
        """
        Scans symbols and finds the best 'Blueprint' by testing multiple parameters.
        If symbols is 'all', it scans all symbols currently in the database.
        """
        if symbols == 'all':
            cursor = self.engine.conn.execute("SELECT DISTINCT symbol FROM ohlcv")
            symbols = [row[0] for row in cursor.fetchall()]
            print(f"Laboratory is analyzing ALL available data in DB ({len(symbols)} coins)")
            
        best_overall_blueprint = None
        max_win_rate = 0
        
        # Simple parameter sweep
        rsi_ranges = [20, 25, 30, 35]
        
        all_results = []
        
        for rsi_thresh in rsi_ranges:
            print(f"Testing RSI Threshold: {rsi_thresh}...")
            
            # Aggregate performance across all symbols
            symbol_metrics = []
            for symbol in symbols:
                df = self.engine.load_data(symbol, timeframe)
                if df.empty: continue
                
                df = self.engine.add_features(df)
                metrics = self.engine.run_simulation(df, rsi_threshold=rsi_thresh)
                symbol_metrics.append(metrics)
            
            if not symbol_metrics: continue
            
            # Average results
            avg_win_rate = sum(m['win_rate_24h'] for m in symbol_metrics) / len(symbol_metrics)
            avg_signals = sum(m['total_signals'] for m in symbol_metrics) / len(symbol_metrics)
            
            res = {
                'rsi_threshold': rsi_thresh,
                'avg_win_rate': avg_win_rate,
                'avg_signals': avg_signals,
                'metrics': symbol_metrics
            }
            all_results.append(res)
            
            if avg_win_rate > max_win_rate and avg_signals > 1:
                max_win_rate = avg_win_rate
                best_overall_blueprint = res

        if best_overall_blueprint:
            print(f"\nWinning Laboratory Pattern found!")
            print(f"RSI Threshold: {best_overall_blueprint['rsi_threshold']}")
            print(f"Avg Win Rate: {best_overall_blueprint['avg_win_rate']:.2%}")
            
            # AI Refinement Phase
            refined_blueprint = self.ai_refine_blueprint(best_overall_blueprint)
            
            # Save Blueprint
            with open("blueprint.json", "w") as f:
                json.dump(refined_blueprint, f, indent=4)
            print("Blueprint saved to blueprint.json")
            return refined_blueprint
        
        return None

    def ai_refine_blueprint(self, stats):
        """
        Sends the winning math to Gemini to refine the strategy logic and explain why it works.
        """
        prompt = f"""
        You are the 'Alpha Architect'. We have run a massive backtest on historical crypto data.
        The top-performing statistical pattern is:
        - RSI Threshold: {stats['rsi_threshold']}
        - Win Rate (24h): {stats['avg_win_rate']:.2%}
        - Avg Signals per asset: {stats['avg_signals']}
        
        Task:
        1. Explain the market psychology behind why this combination (e.g. RSI below {stats['rsi_threshold']} while in an uptrend) works.
        2. Suggest one 'Safety Filter' to add to this blueprint to avoid false signals.
        3. Write a 'Strategy Thesis' for this blueprint that we can use in our Telegram bot.
        
        Output your logic and the final refined JSON configuration.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )
            # For now, we combine the math with the AI thesis
            stats['ai_thesis'] = response.text
            return stats
        except Exception as e:
            print(f"AI Refinement failed: {e}")
            return stats

if __name__ == "__main__":
    lab = Laboratory()
    # Testing with BTC
    lab.generate_blueprint(['BTC/USDT'], '1d')
