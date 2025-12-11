"""
F1 Strategy Analyzer Module
============================

Provides comprehensive race strategy analysis including:
- Pit stop strategy optimization
- Undercut/overcut opportunity detection
- Fuel strategy simulation
- Tyre compound selection analysis
- Alternative strategy comparison
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyOption:
    """Represents a pit stop strategy option."""
    name: str
    stops: int
    pit_laps: List[int]
    compounds: List[str]
    estimated_time: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    description: str


class StrategyAnalyzer:
    """
    Comprehensive race strategy analyzer.
    """
    
    # Tyre compound characteristics (lap time delta in seconds)
    COMPOUND_PACE = {
        "SOFT": 0.0,      # Baseline (fastest)
        "MEDIUM": 0.3,    # ~0.3s slower per lap
        "HARD": 0.6,      # ~0.6s slower per lap
    }
    
    # Tyre degradation (seconds lost per lap due to wear)
    COMPOUND_DEGRADATION = {
        "SOFT": 0.05,     # Degrades quickly
        "MEDIUM": 0.02,   # Moderate degradation
        "HARD": 0.01,     # Minimal degradation
    }
    
    # Optimal stint length (laps)
    OPTIMAL_STINT_LENGTH = {
        "SOFT": (10, 20),
        "MEDIUM": (20, 30),
        "HARD": (30, 45),
    }
    
    # Pit stop time loss (seconds)
    PIT_STOP_TIME_LOSS = 22.0  # Typical pit lane + stationary time
    
    # Track characteristics impact
    TRACK_TYRE_STRESS = {
        "high": 1.3,    # Aggressive tracks (e.g., Silverstone, Suzuka)
        "medium": 1.0,  # Normal tracks
        "low": 0.7,     # Easy on tyres (e.g., Monza)
    }
    
    def __init__(self, track_name: str = "default", total_laps: int = 50):
        """
        Initialize strategy analyzer.
        
        Args:
            track_name: Name of the track
            total_laps: Total number of laps in the race
        """
        self.track_name = track_name
        self.total_laps = total_laps
        self.tyre_stress = self._get_track_tyre_stress()
        
    def _get_track_tyre_stress(self) -> float:
        """Get tyre stress factor for current track."""
        high_stress_tracks = ["Silverstone", "Suzuka", "Barcelona", "Spa"]
        low_stress_tracks = ["Monza", "Bahrain", "Austria"]
        
        if self.track_name in high_stress_tracks:
            return self.TRACK_TYRE_STRESS["high"]
        elif self.track_name in low_stress_tracks:
            return self.TRACK_TYRE_STRESS["low"]
        else:
            return self.TRACK_TYRE_STRESS["medium"]
    
    def generate_strategy_options(self, current_position: int = 10) -> List[StrategyOption]:
        """
        Generate multiple strategy options for comparison.
        
        Args:
            current_position: Current race position (affects risk tolerance)
            
        Returns:
            List of viable strategy options
        """
        strategies = []
        
        # Strategy 1: Aggressive One-Stop (Early pit)
        early_pit_lap = int(self.total_laps * 0.30)
        strategies.append(StrategyOption(
            name="Aggressive One-Stop",
            stops=1,
            pit_laps=[early_pit_lap],
            compounds=["SOFT", "HARD"],
            estimated_time=self._estimate_strategy_time(
                [early_pit_lap], ["SOFT", "HARD"]
            ),
            risk_level="MEDIUM",
            description=f"Pit early (lap {early_pit_lap}), push on softs then manage hards to the end"
        ))
        
        # Strategy 2: Conservative One-Stop (Later pit)
        late_pit_lap = int(self.total_laps * 0.45)
        strategies.append(StrategyOption(
            name="Conservative One-Stop",
            stops=1,
            pit_laps=[late_pit_lap],
            compounds=["MEDIUM", "MEDIUM"],
            estimated_time=self._estimate_strategy_time(
                [late_pit_lap], ["MEDIUM", "MEDIUM"]
            ),
            risk_level="LOW",
            description=f"Balanced approach, pit lap {late_pit_lap} for fresh mediums"
        ))
        
        # Strategy 3: Two-Stop (High performance)
        first_stop = int(self.total_laps * 0.25)
        second_stop = int(self.total_laps * 0.60)
        strategies.append(StrategyOption(
            name="Two-Stop Sprint",
            stops=2,
            pit_laps=[first_stop, second_stop],
            compounds=["SOFT", "MEDIUM", "SOFT"],
            estimated_time=self._estimate_strategy_time(
                [first_stop, second_stop], ["SOFT", "MEDIUM", "SOFT"]
            ),
            risk_level="HIGH",
            description=f"Maximum pace, pit laps {first_stop} & {second_stop}, fresh tyres advantage"
        ))
        
        # Strategy 4: Alternative One-Stop (Medium start)
        mid_pit_lap = int(self.total_laps * 0.40)
        strategies.append(StrategyOption(
            name="Medium-Hard One-Stop",
            stops=1,
            pit_laps=[mid_pit_lap],
            compounds=["MEDIUM", "HARD"],
            estimated_time=self._estimate_strategy_time(
                [mid_pit_lap], ["MEDIUM", "HARD"]
            ),
            risk_level="LOW",
            description=f"Safe strategy, start on mediums, switch to hards lap {mid_pit_lap}"
        ))
        
        # If in top 5, add aggressive undercut strategy
        if current_position <= 5:
            undercut_lap = int(self.total_laps * 0.28)
            strategies.append(StrategyOption(
                name="Undercut Special",
                stops=2,
                pit_laps=[undercut_lap, int(self.total_laps * 0.65)],
                compounds=["SOFT", "MEDIUM", "MEDIUM"],
                estimated_time=self._estimate_strategy_time(
                    [undercut_lap, int(self.total_laps * 0.65)],
                    ["SOFT", "MEDIUM", "MEDIUM"]
                ),
                risk_level="HIGH",
                description=f"Early pit lap {undercut_lap} for track position, react later"
            ))
        
        # Sort by estimated time
        strategies.sort(key=lambda x: x.estimated_time)
        
        return strategies
    
    def _estimate_strategy_time(self, pit_laps: List[int], compounds: List[str]) -> float:
        """
        Estimate total race time for a strategy.
        
        Args:
            pit_laps: List of laps to pit on
            compounds: List of compounds (one more than pit_laps)
            
        Returns:
            Estimated total race time in seconds
        """
        total_time = 0.0
        current_lap = 1
        
        # Add starting compound if not specified
        if len(compounds) == len(pit_laps):
            compounds = ["MEDIUM"] + compounds
        
        for i, pit_lap in enumerate(pit_laps + [self.total_laps + 1]):
            compound = compounds[i]
            stint_length = pit_lap - current_lap
            
            # Calculate stint time
            base_lap_time = 90.0  # Base lap time in seconds
            compound_delta = self.COMPOUND_PACE.get(compound, 0.0)
            degradation_rate = self.COMPOUND_DEGRADATION.get(compound, 0.02) * self.tyre_stress
            
            for lap in range(stint_length):
                lap_time = base_lap_time + compound_delta + (lap * degradation_rate)
                total_time += lap_time
            
            # Add pit stop time
            if pit_lap <= self.total_laps:
                total_time += self.PIT_STOP_TIME_LOSS
            
            current_lap = pit_lap
        
        return total_time
    
    def analyze_undercut_opportunity(self, current_lap: int, 
                                     gap_to_car_ahead: float,
                                     our_tyre_age: int,
                                     their_tyre_age: int) -> Dict:
        """
        Analyze if undercut strategy is viable.
        
        Args:
            current_lap: Current lap number
            gap_to_car_ahead: Gap in seconds
            our_tyre_age: Age of our tyres in laps
            their_tyre_age: Age of opponent's tyres in laps
            
        Returns:
            Dictionary with undercut analysis
        """
        # Undercut is viable if:
        # 1. Gap is close (< 5 seconds)
        # 2. Our tyres are relatively fresh compared to theirs
        # 3. We're in optimal pit window
        
        gap_score = max(0, 5.0 - gap_to_car_ahead) / 5.0  # 0-1 score
        tyre_advantage = max(0, their_tyre_age - our_tyre_age) / 10.0  # 0-1 score
        
        # Check if in optimal window for our compound
        in_window = 0.3 <= (current_lap / self.total_laps) <= 0.6
        
        undercut_score = (gap_score * 0.5 + tyre_advantage * 0.3 + (0.2 if in_window else 0))
        
        viable = undercut_score > 0.5 and gap_to_car_ahead < 5.0
        
        return {
            "viable": viable,
            "score": round(undercut_score, 2),
            "gap_advantage": gap_score > 0.5,
            "tyre_advantage": tyre_advantage > 0.3,
            "in_pit_window": in_window,
            "recommendation": self._get_undercut_recommendation(undercut_score, gap_to_car_ahead)
        }
    
    def _get_undercut_recommendation(self, score: float, gap: float) -> str:
        """Generate undercut recommendation text."""
        if score > 0.7:
            return f"🟢 STRONG UNDERCUT: Box this lap! Gap {gap:.1f}s is ideal"
        elif score > 0.5:
            return f"🟡 POSSIBLE UNDERCUT: Consider pitting, gap {gap:.1f}s"
        elif score > 0.3:
            return f"🟠 RISKY UNDERCUT: Gap {gap:.1f}s may be too large"
        else:
            return f"🔴 NO UNDERCUT: Gap {gap:.1f}s too big, no tyre advantage"
    
    def analyze_overcut_opportunity(self, current_lap: int,
                                    gap_to_car_ahead: float,
                                    our_tyre_age: int,
                                    our_compound: str) -> Dict:
        """
        Analyze if overcut (staying out longer) is viable.
        
        Args:
            current_lap: Current lap number
            gap_to_car_ahead: Gap in seconds
            our_tyre_age: Age of our tyres
            our_compound: Current compound
            
        Returns:
            Dictionary with overcut analysis
        """
        # Overcut works when:
        # 1. Opponent has pitted
        # 2. Our tyres still have good life
        # 3. We can extend and build gap
        
        compound_life = self.OPTIMAL_STINT_LENGTH.get(our_compound, (15, 25))
        tyres_healthy = our_tyre_age < compound_life[1] * 0.8
        
        can_extend = tyres_healthy and our_tyre_age < compound_life[1] - 5
        
        # Calculate how many more laps we can do
        remaining_life = compound_life[1] - our_tyre_age
        
        viable = can_extend and remaining_life > 3
        
        return {
            "viable": viable,
            "remaining_optimal_laps": max(0, remaining_life),
            "tyres_healthy": tyres_healthy,
            "recommended_extend_laps": min(5, remaining_life) if viable else 0,
            "recommendation": self._get_overcut_recommendation(viable, remaining_life)
        }
    
    def _get_overcut_recommendation(self, viable: bool, remaining: int) -> str:
        """Generate overcut recommendation text."""
        if viable and remaining > 5:
            return f"🟢 STAY OUT: Extend {remaining} more laps, build gap on fresh tyres"
        elif viable:
            return f"🟡 CONSIDER EXTENDING: {remaining} laps possible, monitor degradation"
        else:
            return f"🔴 BOX NOW: Tyres at limit, pit immediately"
    
    def simulate_fuel_strategy(self, fuel_load: float = 110.0) -> Dict:
        """
        Simulate fuel strategy impact on lap times.
        
        Args:
            fuel_load: Starting fuel in kg
            
        Returns:
            Dictionary with fuel strategy analysis
        """
        fuel_per_lap = fuel_load / self.total_laps
        fuel_effect_per_kg = 0.03  # ~0.03s per lap per kg
        
        lap_times = []
        current_fuel = fuel_load
        
        for lap in range(1, self.total_laps + 1):
            # Calculate fuel weight penalty
            fuel_penalty = current_fuel * fuel_effect_per_kg
            base_time = 90.0
            lap_time = base_time + fuel_penalty
            
            lap_times.append({
                "lap": lap,
                "fuel_kg": round(current_fuel, 1),
                "lap_time": round(lap_time, 2),
                "fuel_penalty": round(fuel_penalty, 2)
            })
            
            current_fuel -= fuel_per_lap
            current_fuel = max(0, current_fuel)
        
        # Calculate when car is lightest (best performance)
        best_performance_lap = int(self.total_laps * 0.7)  # Last 30% of race
        
        return {
            "fuel_per_lap": round(fuel_per_lap, 2),
            "initial_penalty": round(fuel_load * fuel_effect_per_kg, 2),
            "final_penalty": round((fuel_per_lap * 3) * fuel_effect_per_kg, 2),
            "best_performance_window": (best_performance_lap, self.total_laps),
            "strategy_tip": f"Car will be {round(fuel_load * 0.7, 1)}kg lighter at lap {best_performance_lap}, ideal for late-race push",
            "lap_times": lap_times
        }
    
    def compare_strategies(self, strategies: List[StrategyOption]) -> str:
        """
        Generate comparison table for multiple strategies.
        
        Args:
            strategies: List of strategy options
            
        Returns:
            Formatted comparison string
        """
        if not strategies:
            return "No strategies to compare"
        
        result = "\n╔══════════════════════════════════════════════════════════════╗\n"
        result += "║              STRATEGY COMPARISON                             ║\n"
        result += "╠══════════════════════════════════════════════════════════════╣\n"
        
        for i, strat in enumerate(strategies, 1):
            time_delta = strat.estimated_time - strategies[0].estimated_time
            result += f"║ {i}. {strat.name:<25} ║\n"
            result += f"║    Stops: {strat.stops}  |  Risk: {strat.risk_level:<8}           ║\n"
            result += f"║    Compounds: {' → '.join(strat.compounds):<36} ║\n"
            result += f"║    Pit Laps: {', '.join(map(str, strat.pit_laps)):<39} ║\n"
            result += f"║    Est. Time: {self._format_time(strat.estimated_time):<10} (+{time_delta:.1f}s)         ║\n"
            result += f"║    {strat.description:<54} ║\n"
            result += "╠══════════════════════════════════════════════════════════════╣\n"
        
        result += "╚══════════════════════════════════════════════════════════════╝\n"
        
        # Add recommendation
        fastest = strategies[0]
        result += f"\n💡 RECOMMENDATION: {fastest.name}\n"
        result += f"   └─ Fastest estimated time with {fastest.risk_level} risk\n"
        
        return result
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def get_strategy_summary(self, current_lap: int, 
                           current_position: int,
                           current_tyre: str,
                           tyre_age: int) -> Dict:
        """
        Get comprehensive strategy summary for current situation.
        
        Args:
            current_lap: Current lap number
            current_position: Current position
            current_tyre: Current tyre compound
            tyre_age: Current tyre age in laps
            
        Returns:
            Dictionary with strategy recommendations
        """
        # Generate strategy options
        strategies = self.generate_strategy_options(current_position)
        
        # Check tyre life
        compound_life = self.OPTIMAL_STINT_LENGTH.get(current_tyre, (15, 25))
        tyre_health = max(0, 1 - (tyre_age / compound_life[1]))
        
        # Determine if pit needed soon
        laps_until_pit = max(0, compound_life[1] - tyre_age)
        pit_urgency = "LOW" if laps_until_pit > 10 else ("MEDIUM" if laps_until_pit > 5 else "HIGH")
        
        return {
            "current_situation": {
                "lap": current_lap,
                "position": current_position,
                "tyre": current_tyre,
                "tyre_age": tyre_age,
                "tyre_health": round(tyre_health, 2),
                "laps_until_optimal_pit": laps_until_pit,
                "pit_urgency": pit_urgency
            },
            "recommended_strategies": strategies[:3],  # Top 3
            "next_action": self._get_next_action(current_lap, tyre_age, compound_life, pit_urgency)
        }
    
    def _get_next_action(self, current_lap: int, tyre_age: int, 
                        compound_life: Tuple[int, int], urgency: str) -> str:
        """Determine next recommended action."""
        if urgency == "HIGH":
            return f"⚠️ BOX THIS LAP: Tyres at {tyre_age} laps, optimal window closing"
        elif urgency == "MEDIUM":
            remaining = compound_life[1] - tyre_age
            return f"⏰ PREPARE TO PIT: {remaining} laps remaining on these tyres"
        else:
            return f"✅ STAY OUT: Tyres healthy, manage to lap {current_lap + (compound_life[1] - tyre_age)}"
