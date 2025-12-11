"""
Race Simulator Module.
Generates simulated race data based on ML predictions.
"""

import numpy as np
import random
from typing import Optional

from src.simulation.future_race_data import FutureRaceDataProvider
from src.simulation.track_layouts import TrackLayoutManager
from src.f1_data import INTERPOLATION_FACTOR
from src.simulation.race_dynamics import RaceDynamicsManager, TyreDegradation, PitStopStrategy, OvertakingMechanics


class PredictedRaceSimulator:
    """
    Simulates a race based on ML predictions.
    Generates frame-by-frame data for visualization.
    """

    # Frame rate for simulation
    FPS = 25

    # Simulation constants
    FRAMES_PER_LAP = 200
    BASE_SPEED = 200
    SPEED_VARIATION_MIN = -20
    SPEED_VARIATION_MAX = 20

    # Stable speed variation for smoother animation
    STABLE_SPEED_VARIATION_MIN = -10
    STABLE_SPEED_VARIATION_MAX = 10
    FRAME_SPEED_VARIATION = 5  # Small per-frame variation

    # Pit stop strategy constants
    ONE_STOP_WINDOW_MIN = 0.30
    ONE_STOP_WINDOW_MAX = 0.50
    FIRST_STINT_PERCENTAGE = 0.25
    SECOND_STINT_PERCENTAGE = 0.55

    # Tyre compounds
    COMPOUNDS = {
        "SOFT": {"code": 0, "degradation": 0.8, "pace": 1.0},
        "MEDIUM": {"code": 1, "degradation": 0.5, "pace": 0.97},
        "HARD": {"code": 2, "degradation": 0.3, "pace": 0.94},
        "INTERMEDIATE": {"code": 3, "degradation": 0.4, "pace": 0.85},
        "WET": {"code": 4, "degradation": 0.35, "pace": 0.80},
    }

    def __init__(self, year: int, gp: str):
        """
        Initialize the race simulator.

        Args:
            year: Race year
            gp: Grand Prix name or round number
        """
        self.year = year
        self.gp = gp

        self.data_provider = FutureRaceDataProvider()
        self.track_manager = TrackLayoutManager()

        # Get race info
        if isinstance(gp, int):
            self.race_info = self.data_provider.get_race_by_round(gp)
            self.gp_name = self.race_info["gp"] if self.race_info else f"Round {gp}"
        else:
            self.race_info = self.data_provider.get_race_by_name(gp)
            self.gp_name = gp

        # Initialize simulation state
        self.qualifying_results = None
        self.predictions = {}
        self.pit_stop_events = []
        self.overtake_events = []
        
        # Initialize race dynamics manager
        location = self.race_info.get("location", "default") if self.race_info else "default"
        self.dynamics = RaceDynamicsManager(track_name=location)
        
        # Initialize strategy analyzer
        from src.strategy_analyzer import StrategyAnalyzer
        total_laps = self.race_info.get("laps", 50) if self.race_info else 50
        self.strategy_analyzer = StrategyAnalyzer(track_name=location, total_laps=total_laps)

    def get_qualifying_results(self) -> list:
        """
        Get qualifying results for the race.
        For future races, estimates based on current form.

        Returns:
            List of drivers in qualifying order
        """
        if self.qualifying_results is not None:
            return self.qualifying_results

        # Check if race is in the future
        if self.data_provider.is_future_race(self.year, self.gp_name):
            # Estimate qualifying
            self.qualifying_results = self.data_provider.estimate_qualifying(self.gp_name)
        else:
            # Try to get historical qualifying (future enhancement)
            self.qualifying_results = self.data_provider.estimate_qualifying(self.gp_name)

        return self.qualifying_results

    def predict_pit_stops(self, driver_code: str, total_laps: int) -> list:
        """
        Predict pit stop strategy for a driver.

        Args:
            driver_code: Driver abbreviation
            total_laps: Total race laps

        Returns:
            List of pit stop events with lap and compound
        """
        track_chars = self.data_provider.get_track_characteristics(self.gp_name)
        tyre_wear = track_chars.get("tyre_wear", "medium")

        # Base stint lengths based on tyre wear
        wear_factors = {"low": 1.2, "medium": 1.0, "high": 0.8}
        wear_factor = wear_factors.get(tyre_wear, 1.0)

        # One-stop or two-stop strategy
        if total_laps < 50:
            # Short race - likely one stop
            strategy = self._one_stop_strategy(total_laps, wear_factor)
        else:
            # Longer race - may be two stops
            if random.random() > 0.6:
                strategy = self._two_stop_strategy(total_laps, wear_factor)
            else:
                strategy = self._one_stop_strategy(total_laps, wear_factor)

        return strategy

    def _one_stop_strategy(self, total_laps: int, wear_factor: float) -> list:
        """Generate a one-stop strategy."""
        # Pit window based on class constants
        pit_lap = int(total_laps * random.uniform(
            self.ONE_STOP_WINDOW_MIN, self.ONE_STOP_WINDOW_MAX) * wear_factor)
        pit_lap = max(10, min(total_laps - 10, pit_lap))

        start_compound = random.choice(["SOFT", "MEDIUM"])
        end_compound = "HARD" if start_compound == "SOFT" else "MEDIUM"

        return [
            {"lap": 1, "compound": start_compound, "event": "start"},
            {"lap": pit_lap, "compound": end_compound, "event": "pit"},
        ]

    def _two_stop_strategy(self, total_laps: int, wear_factor: float) -> list:
        """Generate a two-stop strategy."""
        first_stint = int(total_laps * self.FIRST_STINT_PERCENTAGE * wear_factor)
        second_stint = int(total_laps * self.SECOND_STINT_PERCENTAGE * wear_factor)

        first_stint = max(8, min(total_laps - 20, first_stint))
        second_stint = max(first_stint + 10, min(total_laps - 8, second_stint))

        return [
            {"lap": 1, "compound": "SOFT", "event": "start"},
            {"lap": first_stint, "compound": "MEDIUM", "event": "pit"},
            {"lap": second_stint, "compound": "HARD", "event": "pit"},
        ]

    def predict_overtakes(self, positions: dict, lap: int) -> list:
        """
        Predict overtakes for a given lap.

        Args:
            positions: Current driver positions
            lap: Current lap number

        Returns:
            List of overtake events
        """
        overtakes = []
        overtake_prob = self.data_provider.get_overtaking_probability(self.gp_name)

        # Sort drivers by position
        sorted_drivers = sorted(positions.items(), key=lambda x: x[1]["position"])

        for i in range(len(sorted_drivers) - 1):
            ahead_code, ahead_data = sorted_drivers[i]
            behind_code, behind_data = sorted_drivers[i + 1]

            # Calculate overtake likelihood
            # Factors: pace difference, DRS, position gap
            pace_diff = behind_data.get("pace", 1.0) - ahead_data.get("pace", 1.0)

            # Add randomness based on track characteristics
            if pace_diff > 0.02 and random.random() < overtake_prob:
                overtakes.append({
                    "lap": lap,
                    "overtaker": behind_code,
                    "overtaken": ahead_code,
                    "position": ahead_data["position"],
                })

        return overtakes

    def predict_full_race(self, total_laps: Optional[int] = None) -> dict:
        """
        Predict the full race outcome with realistic dynamics.

        Args:
            total_laps: Total race laps (or auto-detect from schedule)

        Returns:
            Dictionary with race predictions
        """
        if total_laps is None:
            total_laps = self.data_provider.get_total_laps(self.gp_name)

        qualifying = self.get_qualifying_results()

        # Initialize driver states
        driver_states = {}
        driver_codes = []
        for quali in qualifying:
            code = quali["code"]
            driver_codes.append(code)
            driver_states[code] = {
                "position": quali["grid"],
                "grid": quali["grid"],
                "team": quali["team"],
                "name": quali["name"],
                "lap": 0,
                "tyre": "MEDIUM",
                "tyre_age": 0,
                "pace": self._calculate_base_pace(quali["team"]),
                "pit_stops": 0,
                "points": quali.get("points", 0),
                "speed": 200.0,  # Base speed
            }
        
        # Initialize race dynamics
        self.dynamics.initialize_race(driver_codes, starting_compound="MEDIUM")

        # Generate pit strategies for all drivers
        pit_strategies = {}
        for code in driver_states:
            pit_strategies[code] = self.predict_pit_stops(code, total_laps)

        # Simulate lap by lap
        lap_results = []
        all_overtakes = []

        for lap in range(1, total_laps + 1):
            # Update tyre age
            for code, state in driver_states.items():
                state["lap"] = lap

                # Check for pit stop
                pitted = False
                for pit in pit_strategies[code]:
                    if pit["lap"] == lap and pit["event"] == "pit":
                        # Execute pit stop with position loss
                        old_position = state["position"]
                        new_position = self.dynamics.execute_pit_stop(
                            code, 
                            pit["compound"],
                            old_position,
                            len(driver_codes)
                        )
                        state["position"] = new_position
                        state["tyre"] = pit["compound"]
                        state["tyre_age"] = 0
                        state["pit_stops"] += 1
                        pitted = True
                        
                        # Record pit event
                        self.pit_stop_events.append({
                            "lap": lap,
                            "driver": code,
                            "old_position": old_position,
                            "new_position": new_position,
                            "compound": pit["compound"]
                        })

                # Age tyres if no pit stop
                if not pitted:
                    state["tyre_age"] += 1

                # Calculate speed with tyre degradation
                tyre_performance = self.dynamics.tyre_degradation.get_performance_factor(code)
                compound = self.COMPOUNDS.get(state["tyre"], self.COMPOUNDS["MEDIUM"])
                
                base_speed = 200 + (state["pace"] * 50)  # Convert pace to speed
                state["speed"] = base_speed * compound["pace"] * tyre_performance
                
            # Update dynamics manager
            self.dynamics.update_lap(driver_states)

            # Attempt overtakes
            overtakes_this_lap = []
            positions_sorted = sorted(driver_states.items(), 
                                     key=lambda x: x[1]["position"])
            
            for i in range(len(positions_sorted) - 1):
                attacker_code, attacker_state = positions_sorted[i + 1]
                defender_code, defender_state = positions_sorted[i]
                
                # Only attempt if positions are adjacent
                if abs(attacker_state["position"] - defender_state["position"]) <= 1:
                    # Get tyre performance
                    attacker_tyres = self.dynamics.tyre_degradation.get_performance_factor(attacker_code)
                    defender_tyres = self.dynamics.tyre_degradation.get_performance_factor(defender_code)
                    
                    # Calculate overtake probability
                    overtake_prob = OvertakingMechanics.calculate_overtake_probability(
                        attacker_state["speed"],
                        defender_state["speed"],
                        attacker_tyres,
                        defender_tyres,
                        self.dynamics.track_name,
                        drs_available=True if i < 20 else False,  # DRS available in top 20
                        laps_behind=1
                    )
                    
                    # Attempt overtake
                    if OvertakingMechanics.attempt_overtake(overtake_prob):
                        # Swap positions
                        old_attacker_pos = attacker_state["position"]
                        old_defender_pos = defender_state["position"]
                        
                        attacker_state["position"] = old_defender_pos
                        defender_state["position"] = old_attacker_pos
                        
                        overtakes_this_lap.append({
                            "lap": lap,
                            "overtaker": attacker_code,
                            "overtaken": defender_code,
                            "probability": overtake_prob
                        })
            
            all_overtakes.extend(overtakes_this_lap)

            # Store lap result
            lap_result = {
                "lap": lap,
                "positions": {code: state["position"] for code, state in driver_states.items()},
                "overtakes": overtakes_this_lap,
            }
            lap_results.append(lap_result)

        # Final results
        final_order = sorted(driver_states.items(), key=lambda x: x[1]["position"])

        return {
            "year": self.year,
            "gp": self.gp_name,
            "total_laps": total_laps,
            "qualifying": qualifying,
            "final_results": [
                {
                    "position": state["position"],
                    "code": code,
                    "name": state["name"],
                    "team": state["team"],
                    "grid": state["grid"],
                    "pit_stops": state["pit_stops"],
                }
                for code, state in final_order
            ],
            "lap_results": lap_results,
            "overtakes": all_overtakes,
            "pit_strategies": pit_strategies,
        }

    def _calculate_base_pace(self, team: str) -> float:
        """Calculate base pace for a team."""
        team_strength = self.data_provider.get_team_strength(team)
        # Invert so lower strength score = faster pace
        return 1.0 - (team_strength - 1) * 0.02

    def generate_simulated_frames(self, total_laps: Optional[int] = None) -> dict:
        """
        Generate simulated frames for visualization.

        Args:
            total_laps: Total race laps

        Returns:
            Dictionary with frames, track layout, and metadata
        """
        if total_laps is None:
            total_laps = self.data_provider.get_total_laps(self.gp_name)

        # Get track layout
        track_layout = self.track_manager.get_track_layout(self.gp_name, self.year - 1)

        # Predict race
        race_prediction = self.predict_full_race(total_laps)

        # Generate frames
        frames = []

        qualifying = race_prediction["qualifying"]
        lap_results = race_prediction["lap_results"]

        # Calculate driver colors (team-based)
        driver_colors = self._get_team_colors()

        # Generate frames for each lap
        current_time = 0.0
        dt = 1.0 / self.FPS

        # Store base starting offset for each driver (stagger based on grid position)
        driver_base_offsets = {}
        for quali in qualifying:
            code = quali["code"]
            # Stagger starting positions - cars start at different points based on grid
            driver_base_offsets[code] = (quali["grid"] - 1) * 0.02

        # Pre-compute qualifying grid map for O(1) lookup
        qualifying_grid_map = {q["code"]: q["grid"] for q in qualifying}

        # Pre-compute stable speed values per lap per driver to avoid random jumps
        driver_lap_speeds = {}
        driver_lap_gears = {}
        for lap_result in lap_results:
            lap = lap_result["lap"]
            for code in lap_result["positions"].keys():
                key = (code, lap)
                # Set stable speed and gear for entire lap with only slight variation
                driver_lap_speeds[key] = self.BASE_SPEED + random.uniform(
                    self.STABLE_SPEED_VARIATION_MIN, self.STABLE_SPEED_VARIATION_MAX)
                driver_lap_gears[key] = random.randint(4, 7)

        # Pre-compute keyframe positions (positions at lap boundaries)
        lap_keyframes = {}
        for lap_idx, lap_result in enumerate(lap_results):
            lap = lap_result["lap"]
            positions = lap_result["positions"]
            
            # Get next lap positions for smooth interpolation
            next_positions = {}
            if lap_idx + 1 < len(lap_results):
                next_positions = lap_results[lap_idx + 1]["positions"]
            
            # Calculate positions at start and end of this lap
            lap_keyframes[lap] = {}
            for code, position in positions.items():
                base_offset = driver_base_offsets.get(code, 0)
                quali_pos = qualifying_grid_map.get(code, position)
                
                # Start of lap position
                position_adjustment_start = (quali_pos - position) * 0.005
                track_progress_start = (0.0 - base_offset + position_adjustment_start) % 1.0
                x_start, y_start = self.track_manager.interpolate_position(track_layout, track_progress_start)
                
                # End of lap position (need to consider if position changes)
                next_pos = next_positions.get(code, position) if next_positions else position
                position_adjustment_end = (quali_pos - next_pos) * 0.005
                track_progress_end = (1.0 - base_offset + position_adjustment_end) % 1.0
                x_end, y_end = self.track_manager.interpolate_position(track_layout, track_progress_end)
                
                lap_keyframes[lap][code] = {
                    'start': {'x': x_start, 'y': y_start, 'position': position, 'track_progress': track_progress_start},
                    'end': {'x': x_end, 'y': y_end, 'position': next_pos, 'track_progress': track_progress_end}
                }

        # Generate frames with constant velocity interpolation between keyframes
        for lap_idx, lap_result in enumerate(lap_results):
            lap = lap_result["lap"]
            positions = lap_result["positions"]
            
            # Simulate movement through this lap
            for frame_num in range(self.FRAMES_PER_LAP):
                # Progress within this lap (0 to 1) - linear for constant velocity
                lap_progress = frame_num / self.FRAMES_PER_LAP

                frame_drivers = {}
                for code, position in positions.items():
                    # Get keyframe data for this driver
                    if lap not in lap_keyframes or code not in lap_keyframes[lap]:
                        continue
                    
                    keyframe = lap_keyframes[lap][code]
                    start = keyframe['start']
                    end = keyframe['end']
                    
                    # Linear interpolation for constant velocity between keyframes
                    x = start['x'] + (end['x'] - start['x']) * lap_progress
                    y = start['y'] + (end['y'] - start['y']) * lap_progress
                    track_progress = start['track_progress'] + (end['track_progress'] - start['track_progress']) * lap_progress
                    
                    # Smooth position number interpolation for overtakes
                    interpolated_pos = start['position'] + (end['position'] - start['position']) * lap_progress

                    # Get tyre compound
                    pit_strategy = race_prediction["pit_strategies"].get(code, [])
                    current_tyre = "MEDIUM"
                    for pit in pit_strategy:
                        if pit["lap"] <= lap:
                            current_tyre = pit["compound"]

                    tyre_code = self.COMPOUNDS.get(current_tyre, self.COMPOUNDS["MEDIUM"])["code"]

                    # Use pre-computed stable speed and gear values
                    speed_key = (code, lap)
                    base_speed = driver_lap_speeds.get(speed_key, self.BASE_SPEED)
                    base_gear = driver_lap_gears.get(speed_key, 5)
                    
                    # Get next lap speed for smooth interpolation
                    next_speed_key = (code, lap + 1)
                    next_base_speed = driver_lap_speeds.get(next_speed_key, base_speed)
                    
                    # Interpolate speed smoothly between laps
                    interpolated_speed = base_speed + (next_base_speed - base_speed) * lap_progress
                    
                    # Add very small per-frame variation for realism (reduced for smoother animation)
                    frame_speed = interpolated_speed + random.uniform(
                        -self.FRAME_SPEED_VARIATION * 0.5, self.FRAME_SPEED_VARIATION * 0.5)
                    
                    frame_drivers[code] = {
                        "x": x,
                        "y": y,
                        "position": position,
                        "lap": lap,
                        "dist": lap * 5000 + lap_progress * 5000,  # Approximate distance
                        "rel_dist": track_progress,
                        "tyre": tyre_code,
                        "speed": frame_speed,
                        "gear": base_gear,
                        "drs": 0,
                    }

                frames.append({
                    "t": current_time,
                    "lap": lap,
                    "drivers": frame_drivers,
                })

                current_time += dt

        # Create example lap dataframe for track rendering
        example_lap = self.track_manager.create_example_lap_dataframe(track_layout)

        # Generate track status with possible incidents (yellow flags, safety cars, etc.)
        track_statuses = self._generate_race_incidents(total_laps, self.FRAMES_PER_LAP)

        return {
            "frames": frames,
            "track_layout": track_layout,
            "example_lap": example_lap,
            "track_statuses": track_statuses,
            "driver_colors": driver_colors,
            "race_prediction": race_prediction,
            "drivers": [q["code"] for q in qualifying],
        }

    def _get_team_colors(self) -> dict:
        """Get colors for each driver based on team."""
        team_colors = {
            "Red Bull": (30, 65, 255),
            "McLaren": (255, 135, 0),
            "Ferrari": (220, 0, 0),
            "Mercedes": (0, 210, 190),
            "Aston Martin": (0, 111, 98),
            "Williams": (0, 90, 255),
            "RB": (102, 146, 255),
            "Alpine": (0, 144, 255),
            "Haas": (182, 186, 189),
            "Sauber": (82, 226, 82),
        }

        colors = {}
        for driver in self.data_provider.DRIVERS_2025:
            team = driver["team"]
            colors[driver["code"]] = team_colors.get(team, (200, 200, 200))

        return colors

    def _generate_race_incidents(self, total_laps: int, frames_per_lap: int) -> list:
        """
        Generate realistic race incidents (yellow flags, safety cars, VSC, red flags).
        
        Args:
            total_laps: Total number of laps in the race
            frames_per_lap: Number of frames per lap
            
        Returns:
            List of track status dictionaries with timing
        """
        import random
        import time
        
        # Use time-based seed for variation
        random.seed(int(time.time() * 1000) % (2**32))
        
        track_statuses = []
        current_time = 0
        
        # Start with green flag
        track_statuses.append({
            "status": "1",  # Green
            "start_time": 0,
            "end_time": None  # Will be set if incident occurs
        })
        
        # Probability of incidents based on race length
        # Typical F1 race: 15-30% chance of safety car, 5-10% chance of VSC, <2% red flag
        incident_probability = {
            'safety_car': 0.20,     # 20% chance of at least one SC
            'vsc': 0.08,            # 8% chance of VSC
            'yellow': 0.12,         # 12% chance of yellow flag
            'red_flag': 0.02,       # 2% chance of red flag
        }
        
        # Decide incidents for this race
        has_safety_car = random.random() < incident_probability['safety_car']
        has_vsc = random.random() < incident_probability['vsc']
        has_yellow = random.random() < incident_probability['yellow']
        has_red_flag = random.random() < incident_probability['red_flag']
        
        incidents = []
        
        # Generate Safety Car incidents
        if has_safety_car:
            num_sc = random.randint(1, 2)  # 1-2 safety car periods
            for _ in range(num_sc):
                # SC usually happens mid-race
                sc_lap = random.randint(int(total_laps * 0.2), int(total_laps * 0.8))
                sc_duration_laps = random.randint(2, 5)  # 2-5 laps
                incidents.append({
                    'type': '4',  # Safety Car
                    'lap': sc_lap,
                    'duration_laps': sc_duration_laps
                })
        
        # Generate VSC incidents
        if has_vsc:
            num_vsc = random.randint(1, 2)
            for _ in range(num_vsc):
                vsc_lap = random.randint(int(total_laps * 0.15), int(total_laps * 0.9))
                vsc_duration_laps = random.randint(1, 3)  # 1-3 laps
                incidents.append({
                    'type': '6',  # VSC
                    'lap': vsc_lap,
                    'duration_laps': vsc_duration_laps
                })
        
        # Generate Yellow Flag incidents  
        if has_yellow:
            num_yellow = random.randint(1, 3)
            for _ in range(num_yellow):
                yellow_lap = random.randint(5, total_laps - 5)
                yellow_duration_laps = random.uniform(0.3, 1.5)  # Short yellow flags
                incidents.append({
                    'type': '2',  # Yellow Flag
                    'lap': yellow_lap,
                    'duration_laps': yellow_duration_laps
                })
        
        # Generate Red Flag incident (rare)
        if has_red_flag:
            red_lap = random.randint(int(total_laps * 0.1), int(total_laps * 0.6))
            red_duration_laps = random.randint(5, 15)  # Significant stoppage
            incidents.append({
                'type': '5',  # Red Flag
                'lap': red_lap,
                'duration_laps': red_duration_laps
            })
        
        # Sort incidents by lap
        incidents.sort(key=lambda x: x['lap'])
        
        # Convert lap-based incidents to time-based track statuses
        for incident in incidents:
            incident_lap = incident['lap']
            duration_laps = incident['duration_laps']
            incident_type = incident['type']
            
            # Calculate time in seconds (approximate)
            start_time = incident_lap * frames_per_lap / INTERPOLATION_FACTOR
            end_time = start_time + (duration_laps * frames_per_lap / INTERPOLATION_FACTOR)
            
            # Close previous green period
            if track_statuses[-1]['end_time'] is None:
                track_statuses[-1]['end_time'] = start_time
            
            # Add incident period
            track_statuses.append({
                "status": incident_type,
                "start_time": start_time,
                "end_time": end_time
            })
            
            # Add VSC ending if it was VSC
            if incident_type == '6':
                track_statuses.append({
                    "status": "7",  # VSC Ending
                    "start_time": end_time,
                    "end_time": end_time + 10  # Short VSC ending period
                })
                end_time += 10
            
            # Resume green flag after incident
            track_statuses.append({
                "status": "1",  # Green
                "start_time": end_time,
                "end_time": None
            })
        
        # Close the last status at race end
        if track_statuses[-1]['end_time'] is None:
            track_statuses[-1]['end_time'] = total_laps * frames_per_lap / INTERPOLATION_FACTOR
        
        return track_statuses

    def get_prediction_confidence(self) -> dict:
        """
        Calculate confidence scores for predictions.

        Returns:
            Dictionary with confidence metrics
        """
        import random
        import time
        
        # Use time-based seed for variation
        random.seed(int(time.time() * 1000) % (2**32))
        
        qualifying = self.get_qualifying_results()

        confidences = {}
        for i, quali in enumerate(qualifying):
            code = quali["code"]
            team_strength = self.data_provider.get_team_strength(quali["team"])

            # Higher confidence for stronger teams and front positions
            position_factor = max(0.5, 1 - i * 0.03)
            team_factor = max(0.5, 1 - (team_strength - 1) * 0.08)

            # Add random variation to confidence (±5%)
            random_variation = random.uniform(-0.05, 0.05)
            
            confidence = min(0.95, max(0.50, position_factor * team_factor + random_variation))
            confidences[code] = round(confidence, 2)

        return confidences
    
    def get_strategy_options(self, driver_code: str, current_position: int = 10) -> list:
        """
        Get strategy options for a specific driver.
        
        Args:
            driver_code: Driver abbreviation
            current_position: Current race position
            
        Returns:
            List of strategy options from the strategy analyzer
        """
        return self.strategy_analyzer.generate_strategy_options(current_position)
    
    def get_strategy_comparison(self, driver_code: str, current_position: int = 10) -> str:
        """
        Get formatted strategy comparison for a driver.
        
        Args:
            driver_code: Driver abbreviation
            current_position: Current race position
            
        Returns:
            Formatted strategy comparison string
        """
        strategies = self.get_strategy_options(driver_code, current_position)
        return self.strategy_analyzer.compare_strategies(strategies)
