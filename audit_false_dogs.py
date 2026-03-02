import pandas as pd
import numpy as np

def audit_false_dogs(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Logic: 
    # 1. Market Underdog: Open Odds > 0 (Positive)
    # 2. Model Favorite: Model Prob > 0.50
    
    # Note: 'A_open_odds' refers to Fighter A. 'B_open_odds' to Fighter B.
    # We need to check BOTH sides.
    
    bets_list = []
    
    for idx, row in df.iterrows():
        # Check if Fighter A is a False Underdog
        if (row['A_open_odds'] > 0) and (row['model_prob'] > 0.50):
            payout = row['A_open_odds']
            profit = payout if row['outcome'] == 1 else -100
            bets_list.append({
                'date': row['event_date'],
                'fighter': row['fighter_a_name'],
                'opponent': row['fighter_b_name'],
                'odds': row['A_open_odds'],
                'prob': row['model_prob'],
                'outcome': 'WON' if row['outcome'] == 1 else 'LOST',
                'profit': profit
            })
            
        # Check if Fighter B is a False Underdog
        # Model Prob for B is (1 - model_prob)
        if (row['B_open_odds'] > 0) and ((1 - row['model_prob']) > 0.50):
            payout = row['B_open_odds']
            profit = payout if row['outcome'] == 0 else -100
            bets_list.append({
                'date': row['event_date'],
                'fighter': row['fighter_b_name'],
                'opponent': row['fighter_a_name'],
                'odds': row['B_open_odds'],
                'prob': 1 - row['model_prob'],
                'outcome': 'WON' if row['outcome'] == 0 else 'LOST',
                'profit': profit
            })
            
    res = pd.DataFrame(bets_list)
    
    if res.empty:
        print("No False Underdogs found.")
    else:
        print(f"\n--- FALSE UNDERDOG AUDIT ({len(res)} Bets) ---")
        print(f"Total Profit: {res['profit'].sum():.2f}")
        print(f"ROI: {res['profit'].sum() / (len(res)*100) * 100:.2f}%")
        print("\nTHE BETS:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        # Format columns
        res['prob'] = (res['prob'] * 100).astype(int).astype(str) + '%'
        print(res.sort_values('date').to_string(index=False))
        res.sort_values('date').to_csv('false_dogs.csv', index=False)
        print("Saved to false_dogs.csv")

if __name__ == "__main__":
    audit_false_dogs(r"d:\Python\ufc-scraper\results 2\predictions_2025.csv")
