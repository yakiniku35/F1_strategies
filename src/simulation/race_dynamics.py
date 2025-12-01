"""
Race Dynamics Module
====================

Handles realistic race dynamics including:
- Tyre degradation simulation
- Pit stop position loss
- Overtaking mechanics
- DRS effects
"""

import random
from typing import Dict, List, Tuple, Optional


class TyreDegradation:
    """
    Simulates tyre degradation and performance loss over laps.
    """
    
    # Tyre compound characteristics
    COMPOUNDS = {
        "SOFT": {
            "peak_performance": 1.00,      # 100% performance when fresh
            "degradation_rate": 0.015,     # 1.5% per lap
            "cliff_lap": 15,               # Performance cliff at lap 15
            "cliff_severity": 0.05,        # Extra 5% loss at cliff
            "min_performance": 0.70        # Minimum 70% performance
        },
        "MEDIUM": {
            "peak_performance": 0.97,      # 97% performance when fresh
            "degradation_rate": 0.008,     # 0.8% per lap
            "cliff_lap": 25,
            "cliff_severity": 0.03,
            "min_performance": 0.75
        },
        "HARD": {
            "peak_performance": 0.94,      # 94% performance when fresh
            "degradation_rate": 0.005,     # 0.5% per lap
            "cliff_lap": 40,
            "cliff_severity": 0.02,
            "min_performance": 0.80
        },
        "INTERMEDIATE": {
            "peak_performance": 0.85,
            "degradation_rate": 0.006,
            "cliff_lap": 20,
            "cliff_severity": 0.03,
            "min_performance": 0.65
        },
        "WET": {
            "peak_performance": 0.80,
            "degradation_rate": 0.007,
            "cliff_lap": 18,
            "cliff_severity": 0.04,
            "min_performance": 0.60
        }
    }
    
    def __init__(self):
        self.tyre_ages = {}  # driver_code -> laps on current tyres
        self.tyre_compounds = {}  # driver_code -> current compound
        
    def initialize_driver(self, driver_code: str, compound: str = "MEDIUM"):
        """Initialize tyre state for a driver."""
        self.tyre_ages[driver_code] = 0
        self.tyre_compounds[driver_code] = compound
        
    def change_tyres(self, driver_code: str, new_compound: str):
        """Change tyres to a new compound."""
        self.tyre_ages[driver_code] = 0
        self.tyre_compounds[driver_code] = new_compound
        
    def age_tyres(self, driver_code: str, laps: float = 1.0):
        """Age the tyres by a number of laps."""
        if driver_code in self.tyre_ages:
            self.tyre_ages[driver_code] += laps
            
    def get_performance_factor(self, driver_code: str) -> float:
        """
        Calculate current tyre performance factor (0.0 to 1.0).
        
        Returns:
            Performance multiplier based on tyre age and compound
        """
        if driver_code not in self.tyre_ages:
            return 1.0
            
        compound = self.tyre_compounds.get(driver_code, "MEDIUM")
        age = self.tyre_ages[driver_code]
        
        if compound not in self.COMPOUNDS:
            return 1.0
            
        spec = self.COMPOUNDS[compound]
        
        # Base degradation
        performance = spec["peak_performance"] - (age * spec["degradation_rate"])
        
        # Cliff effect
        if age >= spec["cliff_lap"]:
            laps_past_cliff = age - spec["cliff_lap"]
            performance -= spec["cliff_severity"] * (1 + laps_past_cliff * 0.1)
        
        # Clamp to minimum
        performance = max(performance, spec["min_performance"])
        
        return performance
        
    def get_tyre_condition(self, driver_code: str) -> str:
        """
        Get visual tyre condition indicator.
        
        Returns:
            "FRESH", "GOOD", "WORN", "CRITICAL"
        """
        performance = self.get_performance_factor(driver_code)
        
        if performance >= 0.95:
            return "FRESH"
        elif performance >= 0.85:
            return "GOOD"
        elif performance >= 0.75:
            return "WORN"
        else:
            return "CRITICAL"


class PitStopStrategy:
    """
    Handles pit stop timing and position loss calculations.
    """
    
    # Track-specific pit loss (positions typically lost during pit stop)
    TRACK_PIT_LOSS = {
        "Monaco": (3, 5),          # Very high pit loss (street circuit)
        "Singapore": (3, 5),
        "Hungary": (2, 4),
        "Barcelona": (2, 3),
        "Silverstone": (1, 3),     # Medium pit loss
        "Spa": (1, 2),
        "Monza": (1, 2),           # Low pit loss (long straights)
        "default": (2, 3)
    }
    
    # Pit stop time variation (seconds)
    PIT_TIME_MIN = 2.0
    PIT_TIME_MAX = 3.5
    PIT_TIME_OPTIMAL = 2.2
    
    @staticmethod
    def calculate_pit_loss(track_name: str = "default") -> int:
        """
        Calculate how many positions a driver loses during pit stop.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Number of positions lost (can be 0 if lucky)
        """
        pit_range = PitStopStrategy.TRACK_PIT_LOSS.get(
            track_name, 
            PitStopStrategy.TRACK_PIT_LOSS["default"]
        )
        
        # Random variation + skill factor
        base_loss = random.randint(pit_range[0], pit_range[1])
        
        # 20% chance of perfect pit stop (1 less position lost)
        if random.random() < 0.20:
            base_loss = max(0, base_loss - 1)
            
        # 10% chance of slow pit stop (1 more position lost)
        elif random.random() < 0.10:
            base_loss += 1
            
        return base_loss
        
    @staticmethod
    def get_pit_stop_time() -> float:
        """
        Get pit stop duration with random variation.
        
        Returns:
            Pit stop time in seconds
        """
        # Normal distribution around optimal time
        base_time = PitStopStrategy.PIT_TIME_OPTIMAL
        variation = random.uniform(-0.3, 0.8)  # More likely to be slower
        
        pit_time = base_time + variation
        pit_time = max(PitStopStrategy.PIT_TIME_MIN, 
                      min(PitStopStrategy.PIT_TIME_MAX, pit_time))
        
        return pit_time
        
    @staticmethod
    def is_optimal_pit_window(current_lap: int, total_laps: int, 
                              strategy: str = "one_stop") -> bool:
        """
        Check if current lap is in optimal pit window.
        
        Args:
            current_lap: Current lap number
            total_laps: Total race laps
            strategy: "one_stop" or "two_stop"
            
        Returns:
            True if in optimal window
        """
        if strategy == "one_stop":
            window_start = int(total_laps * 0.30)
            window_end = int(total_laps * 0.60)
            return window_start <= current_lap <= window_end
        elif strategy == "two_stop":
            # First stop
            first_start = int(total_laps * 0.20)
            first_end = int(total_laps * 0.35)
            # Second stop
            second_start = int(total_laps * 0.55)
            second_end = int(total_laps * 0.75)
            return (first_start <= current_lap <= first_end or 
                   second_start <= current_lap <= second_end)
        return False


class OvertakingMechanics:
    """
    Simulates overtaking probability and DRS effects.
    """
    
    # Base overtaking difficulty by track type
    TRACK_DIFFICULTY = {
        "Monaco": 0.05,           # Nearly impossible
        "Singapore": 0.08,
        "Hungary": 0.10,
        "Barcelona": 0.15,
        "Zandvoort": 0.12,
        "Suzuka": 0.18,
        "Silverstone": 0.25,      # Medium difficulty
        "Austin": 0.25,
        "Spa": 0.35,              # Easy to overtake
        "Monza": 0.40,
        "Jeddah": 0.30,
        "Bahrain": 0.28,
        "default": 0.20
    }
    
    # DRS effect multiplier
    DRS_BOOST = 2.5  # 2.5x easier to overtake with DRS
    
    # Speed delta needed for overtake (km/h)
    SPEED_DELTA_MIN = 5
    SPEED_DELTA_OPTIMAL = 15
    
    @staticmethod
    def calculate_overtake_probability(
        attacker_speed: float,
        defender_speed: float,
        attacker_tyre_performance: float,
        defender_tyre_performance: float,
        track_name: str = "default",
        drs_available: bool = False,
        laps_behind: int = 0
    ) -> float:
        """
        Calculate probability of successful overtake.
        
        Args:
            attacker_speed: Attacker's speed
            defender_speed: Defender's speed
            attacker_tyre_performance: Attacker's tyre performance (0-1)
            defender_tyre_performance: Defender's tyre performance (0-1)
            track_name: Track name
            drs_available: Whether DRS is available
            laps_behind: How many laps the attacker has been behind
            
        Returns:
            Probability of overtake (0.0 to 1.0)
        """
        # Speed advantage
        speed_delta = attacker_speed - defender_speed
        
        if speed_delta < OvertakingMechanics.SPEED_DELTA_MIN:
            return 0.0  # Not fast enough
            
        # Base probability from track
        base_prob = OvertakingMechanics.TRACK_DIFFICULTY.get(
            track_name,
            OvertakingMechanics.TRACK_DIFFICULTY["default"]
        )
        
        # DRS boost
        if drs_available:
            base_prob *= OvertakingMechanics.DRS_BOOST
            
        # Speed advantage factor
        speed_factor = min(speed_delta / OvertakingMechanics.SPEED_DELTA_OPTIMAL, 2.0)
        
        # Tyre performance difference
        tyre_advantage = attacker_tyre_performance - defender_tyre_performance
        tyre_factor = 1.0 + tyre_advantage
        
        # Persistence factor (more laps behind = more aggressive)
        persistence_factor = 1.0 + (laps_behind * 0.05)
        
        # Calculate final probability
        probability = base_prob * speed_factor * tyre_factor * persistence_factor
        
        # Clamp to reasonable range
        probability = max(0.0, min(0.95, probability))
        
        return probability
        
    @staticmethod
    def attempt_overtake(probability: float) -> bool:
        """
        Attempt an overtake based on probability.
        
        Args:
            probability: Overtake probability (0.0 to 1.0)
            
        Returns:
            True if overtake successful
        """
        return random.random() < probability
        
    @staticmethod
    def is_drs_available(gap_to_car_ahead: float, drs_threshold: float = 1.0) -> bool:
        """
        Check if DRS is available (within 1 second of car ahead).
        
        Args:
            gap_to_car_ahead: Gap in seconds
            drs_threshold: DRS activation threshold in seconds
            
        Returns:
            True if DRS can be activated
        """
        return 0 < gap_to_car_ahead <= drs_threshold


class RaceDynamicsManager:
    """
    Manages all race dynamics: tyres, pit stops, and overtaking.
    """
    
    def __init__(self, track_name: str = "default"):
        self.track_name = track_name
        self.tyre_degradation = TyreDegradation()
        self.pit_strategy = PitStopStrategy()
        self.overtaking = OvertakingMechanics()
        
        # Track driver states
        self.driver_positions = {}
        self.driver_speeds = {}
        self.pit_stop_count = {}
        
    def initialize_race(self, drivers: List[str], starting_compound: str = "MEDIUM"):
        """Initialize race for all drivers."""
        for driver in drivers:
            self.tyre_degradation.initialize_driver(driver, starting_compound)
            self.pit_stop_count[driver] = 0
            
    def execute_pit_stop(self, driver_code: str, new_compound: str,
                         current_position: int, total_drivers: int) -> int:
        """
        Execute pit stop and return new position.
        
        Args:
            driver_code: Driver code
            new_compound: New tyre compound
            current_position: Current position
            total_drivers: Total number of drivers
            
        Returns:
            New position after pit stop
        """
        # Change tyres
        self.tyre_degradation.change_tyres(driver_code, new_compound)
        self.pit_stop_count[driver_code] += 1
        
        # Calculate position loss
        positions_lost = self.pit_strategy.calculate_pit_loss(self.track_name)
        new_position = min(current_position + positions_lost, total_drivers)
        
        return new_position
        
    def update_lap(self, drivers_data: Dict[str, Dict]):
        """
        Update race state for one lap.
        
        Args:
            drivers_data: Dictionary of driver data with position, speed, etc.
        """
        # Age tyres
        for driver_code in drivers_data:
            self.tyre_degradation.age_tyres(driver_code, 1.0)
            
        # Update internal state
        self.driver_positions = {
            code: data.get('position', 99) 
            for code, data in drivers_data.items()
        }
        self.driver_speeds = {
            code: data.get('speed', 0) 
            for code, data in drivers_data.items()
        }
