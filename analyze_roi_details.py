import pandas as pd
import numpy as np

def analyze_details(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # 1. Feature Check (Heuristic)
    # We can't see the feature columns here, but we can verify inputs if we had the code. 
    # For now, just analyze the bets.

    # 2. FULL EV AUDIT (All Bets where Model > Market)
    print("\n=== FULL EV AUDIT (Betting Everything with Edge > 0) ===")
    
    def get_implied_prob(odds):
        if odds > 0: return 100 / (odds + 100)
        else: return abs(odds) / (abs(odds) + 100)

    # Calculate Implied Probs
    df['A_imp'] = df['A_open_odds'].apply(get_implied_prob)
    df['B_imp'] = df['B_open_odds'].apply(get_implied_prob)
    
    # Identify Value Bets (Model > Implied)
    # Note: We must pick the side with the LARGER edge, or just any edge?
    # Usually you verify if edge exists.
    
    df['edge_A'] = df['model_prob'] - df['A_imp']
    df['edge_B'] = (1 - df['model_prob']) - df['B_imp']
    
    # Strategy: Bet on the side with positive edge.
    # If both (arb situation), bet max edge? Or just A? 
    # Let's assume standard logic: Bet if Edge > 0.
    
    bets_list = []
    
    for idx, row in df.iterrows():
        # Check A
        if row['edge_A'] > 0:
            payout = (10000 / abs(row['A_open_odds'])) if row['A_open_odds'] < 0 else row['A_open_odds']
            profit = payout if row['outcome'] == 1 else -100
            bets_list.append({
                'event_date': row['event_date'],
                'fighter': row['fighter_a_name'],
                'opponent': row['fighter_b_name'],
                'type': 'Favorite' if row['A_open_odds'] < 0 else 'Underdog',
                'market_odds': row['A_open_odds'],
                'edge': row['edge_A'],
                'prob': row['model_prob'],
                'profit': profit
            })
        elif row['edge_B'] > 0:
            payout = (10000 / abs(row['B_open_odds'])) if row['B_open_odds'] < 0 else row['B_open_odds']
            profit = payout if row['outcome'] == 0 else -100
             # Note: model_prob is prob of A. So prob B is 1-model_prob
            bets_list.append({
                'event_date': row['event_date'],
                'fighter': row['fighter_b_name'],
                'opponent': row['fighter_a_name'],
                'type': 'Favorite' if row['B_open_odds'] < 0 else 'Underdog',
                'market_odds': row['B_open_odds'],
                'edge': row['edge_B'],
                'prob': 1 - row['model_prob'],
                'profit': profit
            })

    ev_df = pd.DataFrame(bets_list)
    
    if ev_df.empty:
        print("No +EV bets found.")
        return

    # Analyze Subgroups
    for bucket in ['All', 'Favorite', 'Underdog']:
        if bucket == 'All':
            subset = ev_df
        else:
            subset = ev_df[ev_df['type'] == bucket]
            
        total_bets = len(subset)
        total_profit = subset['profit'].sum()
        roi = total_profit / (total_bets * 100) * 100
        print(f"\n--- {bucket} +EV Bets ---")
        print(f"Count: {total_bets}")
        print(f"Profit: {total_profit:.2f}")
        print(f"ROI: {roi:.2f}%")
        
        if bucket == 'Favorite':
             print(f"\n--- VALUE FAVORITES (Model sees more value than Market) ---")
             # pd.set_option('display.max_columns', None)
             # pd.set_option('display.width', 1000)
             view_cols = ['event_date', 'fighter', 'opponent', 'market_odds', 'prob', 'profit']
             print(subset[view_cols].sort_values('event_date').to_string(index=False))
             
    # Special Request: False Underdogs (Market > 0, Model > 0.5)
    print("\n--- FALSE UNDERDOGS (Market: Dog, Model: Favorite) ---")
    false_dogs = ev_df[(ev_df['type'] == 'Underdog') & (ev_df['prob'] > 0.50)]
    
    if not false_dogs.empty:
        fd_bets = len(false_dogs)
        fd_profit = false_dogs['profit'].sum()
        fd_roi = fd_profit / (fd_bets * 100) * 100
        print(f"Count: {fd_bets}")
        print(f"Profit: {fd_profit:.2f}")
        print(f"ROI: {fd_roi:.2f}%")
        pd.set_option('display.max_columns', None)
        view_cols = ['event_date', 'fighter', 'opponent', 'market_odds', 'prob', 'profit']
        print(false_dogs[view_cols].sort_values('event_date').to_string(index=False))
    else:
        print("No bets found where Model Fav > 0.5 and Market is Underdog.")

    print("\n--- CONCLUSION ---")
    print("Does 'Blindly Following Value' work? Or is the Sniper approach required?")

if __name__ == "__main__":
    analyze_details(r"d:\Python\ufc-scraper\results 2\predictions_2025.csv")
