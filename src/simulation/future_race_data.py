"""
Future Race Data Provider Module.
Handles data for future races including schedules, standings, and track characteristics.
"""

import pandas as pd
from datetime import datetime
from typing import Optional


class FutureRaceDataProvider:
    """
    Provides data for future races.
    Handles schedule, standings, team performance, and track characteristics.
    """

    # 2025 F1 Season Schedule (hardcoded)
    SCHEDULE_2025 = [
        {"round": 1, "name": "Australian Grand Prix", "gp": "Australia", "location": "Melbourne", "date": "2025-03-16", "laps": 58},
        {"round": 2, "name": "Chinese Grand Prix", "gp": "China", "location": "Shanghai", "date": "2025-03-23", "laps": 56},
        {"round": 3, "name": "Japanese Grand Prix", "gp": "Japan", "location": "Suzuka", "date": "2025-04-06", "laps": 53},
        {"round": 4, "name": "Bahrain Grand Prix", "gp": "Bahrain", "location": "Sakhir", "date": "2025-04-13", "laps": 57},
        {"round": 5, "name": "Saudi Arabian Grand Prix", "gp": "Saudi Arabia", "location": "Jeddah", "date": "2025-04-20", "laps": 50},
        {"round": 6, "name": "Miami Grand Prix", "gp": "Miami", "location": "Miami", "date": "2025-05-04", "laps": 57},
        {"round": 7, "name": "Emilia Romagna Grand Prix", "gp": "Emilia Romagna", "location": "Imola", "date": "2025-05-18", "laps": 63},
        {"round": 8, "name": "Monaco Grand Prix", "gp": "Monaco", "location": "Monte Carlo", "date": "2025-05-25", "laps": 78},
        {"round": 9, "name": "Spanish Grand Prix", "gp": "Spain", "location": "Barcelona", "date": "2025-06-01", "laps": 66},
        {"round": 10, "name": "Canadian Grand Prix", "gp": "Canada", "location": "Montreal", "date": "2025-06-15", "laps": 70},
        {"round": 11, "name": "Austrian Grand Prix", "gp": "Austria", "location": "Spielberg", "date": "2025-06-29", "laps": 71},
        {"round": 12, "name": "British Grand Prix", "gp": "Great Britain", "location": "Silverstone", "date": "2025-07-06", "laps": 52},
        {"round": 13, "name": "Belgian Grand Prix", "gp": "Belgium", "location": "Spa", "date": "2025-07-27", "laps": 44},
        {"round": 14, "name": "Hungarian Grand Prix", "gp": "Hungary", "location": "Budapest", "date": "2025-08-03", "laps": 70},
        {"round": 15, "name": "Dutch Grand Prix", "gp": "Netherlands", "location": "Zandvoort", "date": "2025-08-31", "laps": 72},
        {"round": 16, "name": "Italian Grand Prix", "gp": "Italy", "location": "Monza", "date": "2025-09-07", "laps": 53},
        {"round": 17, "name": "Azerbaijan Grand Prix", "gp": "Azerbaijan", "location": "Baku", "date": "2025-09-21", "laps": 51},
        {"round": 18, "name": "Singapore Grand Prix", "gp": "Singapore", "location": "Singapore", "date": "2025-10-05", "laps": 62},
        {"round": 19, "name": "United States Grand Prix", "gp": "United States", "location": "Austin", "date": "2025-10-19", "laps": 56},
        {"round": 20, "name": "Mexico City Grand Prix", "gp": "Mexico", "location": "Mexico City", "date": "2025-10-26", "laps": 71},
        {"round": 21, "name": "São Paulo Grand Prix", "gp": "Brazil", "location": "São Paulo", "date": "2025-11-09", "laps": 71},
        {"round": 22, "name": "Las Vegas Grand Prix", "gp": "Las Vegas", "location": "Las Vegas", "date": "2025-11-22", "laps": 50},
        {"round": 23, "name": "Qatar Grand Prix", "gp": "Qatar", "location": "Lusail", "date": "2025-11-30", "laps": 57},
        {"round": 24, "name": "Abu Dhabi Grand Prix", "gp": "Abu Dhabi", "location": "Yas Marina", "date": "2025-12-07", "laps": 58},
    ]

    # 2025 Driver lineup (estimated based on confirmed signings)
    DRIVERS_2025 = [
        {"code": "VER", "name": "Max Verstappen", "team": "Red Bull", "number": 1},
        {"code": "LAW", "name": "Liam Lawson", "team": "Red Bull", "number": 30},
        {"code": "LEC", "name": "Charles Leclerc", "team": "Ferrari", "number": 16},
        {"code": "HAM", "name": "Lewis Hamilton", "team": "Ferrari", "number": 44},
        {"code": "RUS", "name": "George Russell", "team": "Mercedes", "number": 63},
        {"code": "ANT", "name": "Andrea Kimi Antonelli", "team": "Mercedes", "number": 12},
        {"code": "NOR", "name": "Lando Norris", "team": "McLaren", "number": 4},
        {"code": "PIA", "name": "Oscar Piastri", "team": "McLaren", "number": 81},
        {"code": "ALO", "name": "Fernando Alonso", "team": "Aston Martin", "number": 14},
        {"code": "STR", "name": "Lance Stroll", "team": "Aston Martin", "number": 18},
        {"code": "GAS", "name": "Pierre Gasly", "team": "Alpine", "number": 10},
        {"code": "DOO", "name": "Jack Doohan", "team": "Alpine", "number": 7},
        {"code": "ALB", "name": "Alexander Albon", "team": "Williams", "number": 23},
        {"code": "SAI", "name": "Carlos Sainz", "team": "Williams", "number": 55},
        {"code": "TSU", "name": "Yuki Tsunoda", "team": "RB", "number": 22},
        {"code": "HAD", "name": "Isack Hadjar", "team": "RB", "number": 6},
        {"code": "HUL", "name": "Nico Hulkenberg", "team": "Sauber", "number": 27},
        {"code": "BOR", "name": "Gabriel Bortoleto", "team": "Sauber", "number": 5},
        {"code": "OCO", "name": "Esteban Ocon", "team": "Haas", "number": 31},
        {"code": "BEA", "name": "Oliver Bearman", "team": "Haas", "number": 87},
    ]

    # Team strength rankings (1 = strongest, higher = weaker)
    TEAM_STRENGTH = {
        "Red Bull": 1.5,
        "McLaren": 1.8,
        "Ferrari": 2.0,
        "Mercedes": 2.5,
        "Aston Martin": 4.5,
        "Williams": 6.0,
        "RB": 5.5,
        "Alpine": 6.5,
        "Haas": 7.0,
        "Sauber": 7.5,
    }

    # Track characteristics
    TRACK_CHARACTERISTICS = {
        "Monaco": {
            "type": "street",
            "overtaking_difficulty": "very_hard",
            "tyre_wear": "low",
            "track_length": 3.337,
            "corners": 19,
            "drs_zones": 1,
        },
        "Monza": {
            "type": "permanent",
            "overtaking_difficulty": "easy",
            "tyre_wear": "low",
            "track_length": 5.793,
            "corners": 11,
            "drs_zones": 2,
        },
        "Silverstone": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 5.891,
            "corners": 18,
            "drs_zones": 2,
        },
        "Spa": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 7.004,
            "corners": 19,
            "drs_zones": 2,
        },
        "Suzuka": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "high",
            "track_length": 5.807,
            "corners": 18,
            "drs_zones": 1,
        },
        "Australia": {
            "type": "street",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 5.278,
            "corners": 14,
            "drs_zones": 3,
        },
        "Bahrain": {
            "type": "permanent",
            "overtaking_difficulty": "easy",
            "tyre_wear": "high",
            "track_length": 5.412,
            "corners": 15,
            "drs_zones": 3,
        },
        "Saudi Arabia": {
            "type": "street",
            "overtaking_difficulty": "medium",
            "tyre_wear": "low",
            "track_length": 6.174,
            "corners": 27,
            "drs_zones": 3,
        },
        "Miami": {
            "type": "street",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 5.412,
            "corners": 19,
            "drs_zones": 3,
        },
        "Imola": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "medium",
            "track_length": 4.909,
            "corners": 19,
            "drs_zones": 1,
        },
        "Spain": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "high",
            "track_length": 4.657,
            "corners": 16,
            "drs_zones": 2,
        },
        "Canada": {
            "type": "street",
            "overtaking_difficulty": "medium",
            "tyre_wear": "low",
            "track_length": 4.361,
            "corners": 14,
            "drs_zones": 2,
        },
        "Austria": {
            "type": "permanent",
            "overtaking_difficulty": "easy",
            "tyre_wear": "low",
            "track_length": 4.318,
            "corners": 10,
            "drs_zones": 3,
        },
        "Hungary": {
            "type": "permanent",
            "overtaking_difficulty": "very_hard",
            "tyre_wear": "medium",
            "track_length": 4.381,
            "corners": 14,
            "drs_zones": 1,
        },
        "Belgium": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 7.004,
            "corners": 19,
            "drs_zones": 2,
        },
        "Netherlands": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "high",
            "track_length": 4.259,
            "corners": 14,
            "drs_zones": 2,
        },
        "Italy": {
            "type": "permanent",
            "overtaking_difficulty": "easy",
            "tyre_wear": "low",
            "track_length": 5.793,
            "corners": 11,
            "drs_zones": 2,
        },
        "Azerbaijan": {
            "type": "street",
            "overtaking_difficulty": "medium",
            "tyre_wear": "low",
            "track_length": 6.003,
            "corners": 20,
            "drs_zones": 2,
        },
        "Singapore": {
            "type": "street",
            "overtaking_difficulty": "hard",
            "tyre_wear": "low",
            "track_length": 4.940,
            "corners": 19,
            "drs_zones": 3,
        },
        "United States": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 5.513,
            "corners": 20,
            "drs_zones": 2,
        },
        "Mexico": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 4.304,
            "corners": 17,
            "drs_zones": 3,
        },
        "Brazil": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 4.309,
            "corners": 15,
            "drs_zones": 2,
        },
        "Las Vegas": {
            "type": "street",
            "overtaking_difficulty": "easy",
            "tyre_wear": "low",
            "track_length": 6.201,
            "corners": 17,
            "drs_zones": 2,
        },
        "Qatar": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 5.419,
            "corners": 16,
            "drs_zones": 2,
        },
        "Abu Dhabi": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 5.281,
            "corners": 16,
            "drs_zones": 2,
        },
        "China": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 5.451,
            "corners": 16,
            "drs_zones": 2,
        },
        "Japan": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "high",
            "track_length": 5.807,
            "corners": 18,
            "drs_zones": 1,
        },
        "Great Britain": {
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "high",
            "track_length": 5.891,
            "corners": 18,
            "drs_zones": 2,
        },
        "Emilia Romagna": {
            "type": "permanent",
            "overtaking_difficulty": "hard",
            "tyre_wear": "medium",
            "track_length": 4.909,
            "corners": 19,
            "drs_zones": 1,
        },
    }

    def __init__(self):
        """Initialize the future race data provider."""
        self._estimated_points = self._initialize_points()

    def _initialize_points(self) -> dict:
        """Initialize estimated points for 2025 drivers."""
        # Start with estimated points based on 2024 performance
        estimated_points = {}
        base_points = {
            "VER": 400, "NOR": 320, "LEC": 280, "PIA": 260, "HAM": 220,
            "RUS": 200, "SAI": 180, "ALO": 60, "STR": 30, "HUL": 25,
            "GAS": 20, "OCO": 15, "TSU": 20, "ALB": 15, "LAW": 10,
            "BEA": 5, "ANT": 0, "HAD": 0, "DOO": 0, "BOR": 0,
        }
        for driver in self.DRIVERS_2025:
            estimated_points[driver["code"]] = base_points.get(driver["code"], 0)
        return estimated_points

    def get_2025_schedule(self) -> list:
        """
        Get the 2025 F1 season schedule.

        Returns:
            List of race dictionaries with round, name, gp, location, date, laps
        """
        return self.SCHEDULE_2025.copy()

    def get_schedule_dataframe(self) -> pd.DataFrame:
        """
        Get schedule as a pandas DataFrame.

        Returns:
            DataFrame with schedule information
        """
        df = pd.DataFrame(self.SCHEDULE_2025)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def get_race_by_round(self, round_number: int) -> Optional[dict]:
        """
        Get race information by round number.

        Args:
            round_number: The race round number (1-24)

        Returns:
            Race dictionary or None if not found
        """
        for race in self.SCHEDULE_2025:
            if race["round"] == round_number:
                return race.copy()
        return None

    def get_race_by_name(self, gp_name: str) -> Optional[dict]:
        """
        Get race information by GP name.

        Args:
            gp_name: The Grand Prix name (e.g., "Monaco", "Silverstone")

        Returns:
            Race dictionary or None if not found
        """
        gp_name_lower = gp_name.lower()
        for race in self.SCHEDULE_2025:
            if (gp_name_lower in race["gp"].lower() or
                gp_name_lower in race["name"].lower() or
                gp_name_lower in race["location"].lower()):
                return race.copy()
        return None

    def is_future_race(self, year: int, gp_name: str) -> bool:
        """
        Check if a race is in the future.

        Args:
            year: Race year
            gp_name: Grand Prix name

        Returns:
            True if the race hasn't happened yet
        """
        if year > datetime.now().year:
            return True
        if year < datetime.now().year:
            return False

        race = self.get_race_by_name(gp_name)
        if race:
            race_date = datetime.strptime(race["date"], "%Y-%m-%d")
            return race_date > datetime.now()
        return True

    def get_current_standings(self) -> list:
        """
        Get current driver standings (estimated for 2025).

        Returns:
            List of drivers with estimated points
        """
        standings = []
        for driver in self.DRIVERS_2025:
            standings.append({
                "code": driver["code"],
                "name": driver["name"],
                "team": driver["team"],
                "points": self._estimated_points.get(driver["code"], 0),
                "number": driver["number"],
            })
        standings.sort(key=lambda x: x["points"], reverse=True)
        return standings

    def get_drivers_list(self) -> list:
        """
        Get list of 2025 drivers.

        Returns:
            List of driver dictionaries
        """
        return self.DRIVERS_2025.copy()

    def get_driver_by_code(self, code: str) -> Optional[dict]:
        """
        Get driver information by abbreviation code.

        Args:
            code: Driver abbreviation (e.g., "VER", "HAM")

        Returns:
            Driver dictionary or None
        """
        for driver in self.DRIVERS_2025:
            if driver["code"] == code:
                return driver.copy()
        return None

    def get_team_performance(self) -> dict:
        """
        Get team performance indicators.

        Returns:
            Dictionary mapping team name to strength score
        """
        return self.TEAM_STRENGTH.copy()

    def get_team_strength(self, team_name: str) -> float:
        """
        Get strength score for a specific team.

        Args:
            team_name: Team name

        Returns:
            Strength score (lower is better)
        """
        return self.TEAM_STRENGTH.get(team_name, 10.0)

    def estimate_qualifying(self, gp_name: str) -> list:
        """
        Estimate qualifying results for a future race.
        Based on team strength and driver performance.

        Args:
            gp_name: Grand Prix name

        Returns:
            List of drivers in estimated qualifying order
        """
        import random
        import time
        
        # Use time-based seed for variation between runs
        random.seed(int(time.time() * 1000) % (2**32))

        qualifying = []
        for driver in self.DRIVERS_2025:
            team_strength = self.get_team_strength(driver["team"])
            driver_points = self._estimated_points.get(driver["code"], 0)

            # Base score: lower is better
            # Team strength has major impact, driver points add variation
            base_score = team_strength * 10 - (driver_points / 50)

            # Add significant random variation to create different outcomes
            # Increased from (-2, 2) to (-5, 5) for more variation
            variation = random.uniform(-5, 5)
            final_score = base_score + variation

            qualifying.append({
                "code": driver["code"],
                "name": driver["name"],
                "team": driver["team"],
                "grid": 0,  # Will be set after sorting
                "points": self._estimated_points.get(driver["code"], 0),
                "score": final_score,
            })

        # Sort by score (lower is better position)
        qualifying.sort(key=lambda x: x["score"])

        # Assign grid positions
        for i, driver in enumerate(qualifying):
            driver["grid"] = i + 1
            del driver["score"]  # Remove internal score

        return qualifying

    def get_track_characteristics(self, gp_name: str) -> dict:
        """
        Get track characteristics for a given GP.

        Args:
            gp_name: Grand Prix name

        Returns:
            Dictionary of track characteristics
        """
        # Try to find by GP name
        gp_name_lower = gp_name.lower()
        for track_name, characteristics in self.TRACK_CHARACTERISTICS.items():
            if gp_name_lower in track_name.lower():
                return {
                    "name": track_name,
                    **characteristics
                }

        # Try to find from schedule
        race = self.get_race_by_name(gp_name)
        if race:
            gp = race["gp"]
            if gp in self.TRACK_CHARACTERISTICS:
                return {
                    "name": gp,
                    **self.TRACK_CHARACTERISTICS[gp]
                }

        # Default characteristics
        return {
            "name": gp_name,
            "type": "permanent",
            "overtaking_difficulty": "medium",
            "tyre_wear": "medium",
            "track_length": 5.0,
            "corners": 15,
            "drs_zones": 2,
        }

    def get_total_laps(self, gp_name: str) -> int:
        """
        Get total laps for a race.

        Args:
            gp_name: Grand Prix name

        Returns:
            Number of laps
        """
        race = self.get_race_by_name(gp_name)
        if race:
            return race.get("laps", 50)
        return 50  # Default

    def get_overtaking_probability(self, gp_name: str) -> float:
        """
        Get probability factor for overtaking at a track.

        Args:
            gp_name: Grand Prix name

        Returns:
            Probability multiplier (0.0-1.0)
        """
        characteristics = self.get_track_characteristics(gp_name)
        difficulty_map = {
            "very_hard": 0.1,
            "hard": 0.25,
            "medium": 0.5,
            "easy": 0.75,
        }
        return difficulty_map.get(characteristics.get("overtaking_difficulty", "medium"), 0.5)
