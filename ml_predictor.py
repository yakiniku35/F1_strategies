# ... (保留原有的程式碼)

class PreRacePredictor:
    """
    New class for predicting FUTURE race outcomes before they happen.
    Uses historical season data instead of live telemetry.
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False

    def prepare_historical_data(self, year):
        """
        Fetch historical race results to train the model.
        We need: Qualifying Position -> Final Position correlation
        """
        import fastf1
        
        X_train = [] # Features: [QualifyingPos, TeamStrength, DriverHistory]
        y_train = [] # Target: [FinalPosition]

        # 獲取該年度所有已完成的比賽
        schedule = fastf1.get_event_schedule(year)
        completed_races = schedule[schedule['EventDate'] < pd.Timestamp.now()]

        print(f"Training on {len(completed_races)} past races...")

        for _, race_event in completed_races.iterrows():
            try:
                session = fastf1.get_session(year, race_event['RoundNumber'], 'R')
                session.load(telemetry=False, weather=False, messages=False)
                
                results = session.results
                
                # 簡單特徵工程：車隊平均實力 (這裡簡化處理)
                team_strength = results.groupby('TeamName')['Position'].mean().to_dict()

                for driver in results.index:
                    d_data = results.loc[driver]
                    
                    # 排除退賽 (DNF) 的數據，避免干擾訓練
                    if d_data['ClassifiedPosition'] == 'R': 
                        continue

                    # 特徵 1: 排位賽名次 (Grid Position) - 最重要的預測指標
                    grid_pos = d_data['GridPosition']
                    
                    # 特徵 2: 車隊實力 (Team Strength)
                    team_score = team_strength.get(d_data['TeamName'], 10)

                    # 特徵 3: 當前積分 (反映車手賽季狀態)
                    points = d_data['Points']

                    X_train.append([grid_pos, team_score, points])
                    y_train.append(d_data['Position'])
                    
            except Exception as e:
                print(f"Skipping round: {e}")
                continue

        return np.array(X_train), np.array(y_train)

    def train_for_season(self, year=2024):
        """
        Train the model based on past races of the year
        """
        print("Fetching historical data for pre-race prediction...")
        X, y = self.prepare_historical_data(year)
        
        if len(X) > 0:
            self.model.fit(X, y)
            self.is_trained = True
            print("Pre-Race Model Trained Successfully!")
            return True
        return False

    def predict_next_race(self, qualifying_results):
        """
        Predict the outcome of the NEXT race based on qualifying results.
        
        qualifying_results: list of dicts 
        [{'driver': 'VER', 'grid': 1, 'team': 'Red Bull', 'points': 200}, ...]
        """
        if not self.is_trained:
            return "Model not trained"

        predictions = []
        
        for driver in qualifying_results:
            # 根據我們訓練時的特徵順序來準備數據
            # [GridPosition, TeamStrength(estimated), CurrentPoints]
            # 注意：這裡為了演示簡化了 TeamStrength 的計算
            team_score = 1.0 if driver['team'] == 'Red Bull' else (5.0 if driver['team'] == 'Ferrari' else 10.0)
            
            features = np.array([[
                driver['grid'],
                team_score,
                driver['points']
            ]])
            
            predicted_pos = self.model.predict(features)[0]
            
            predictions.append({
                'driver': driver['driver'],
                'predicted_finish': predicted_pos
            })
            
        # 排序預測結果
        predictions.sort(key=lambda x: x['predicted_finish'])
        return predictions