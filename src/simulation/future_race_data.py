"""
Future Race Data Provider Module.
Handles data for future races including schedules, standings, and track characteristics.

The schedule and driver lineup are read from FastF1 for whichever season is
asked for, so the app follows the real calendar instead of being frozen to one
year. The bundled 2025 tables are kept as an offline fallback: FastF1 needs the
network on a cold cache, and a season that has not been published yet has no
data at all.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FutureRaceDataProvider:
    """
    Provides data for future races.
    Handles schedule, standings, team performance, and track characteristics.

    Args:
        year: Season to describe. Defaults to the current calendar year.

    The ``schedule`` and ``drivers`` properties load lazily, so building a
    provider never blocks on the network - the cost is only paid by whoever
    actually asks for the data.
    """

    # Season the bundled fallback tables below describe.
    FALLBACK_YEAR = 2025

    # Offline fallback schedule, used when FastF1 cannot supply the season.
    FALLBACK_SCHEDULE = [
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

    # Offline fallback driver lineup, used when FastF1 cannot supply the season.
    FALLBACK_DRIVERS = [
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

    def __init__(self, year: Optional[int] = None):
        """
        Initialize the future race data provider.

        Args:
            year: Season to describe. Defaults to the current calendar year.
        """
        self.year = int(year) if year else datetime.now().year

        # Lazily populated by the schedule/drivers properties.
        self._schedule = None
        self._drivers = None
        self._estimated_points = None
        # "fastf1" or "fallback" - what the loaded data actually came from, so
        # callers (and the user) can tell live data from the bundled tables.
        # Exposed through properties that trigger the load, so reading the
        # source before the data can never hand back a misleading None.
        self._schedule_source = None
        self._drivers_source = None

    # ------------------------------------------------------------------ data

    @property
    def schedule(self) -> list:
        """The season schedule, loaded on first use."""
        if self._schedule is None:
            self._schedule, self._schedule_source = self._load_schedule()
        return self._schedule

    @property
    def schedule_source(self) -> str:
        """Where the schedule came from: "fastf1" or "fallback"."""
        self.schedule  # force the load so the answer is never None
        return self._schedule_source

    @property
    def drivers(self) -> list:
        """The season driver lineup, loaded on first use."""
        if self._drivers is None:
            self._drivers, self._drivers_source = self._load_drivers()
        return self._drivers

    @property
    def drivers_source(self) -> str:
        """Where the driver lineup came from: "fastf1" or "fallback"."""
        self.drivers  # force the load so the answer is never None
        return self._drivers_source

    def _load_schedule(self) -> tuple:
        """Load the season schedule, preferring FastF1 over the bundled table.

        Returns:
            (schedule list, source string) where source is "fastf1" or "fallback".
        """
        if self.year == self.FALLBACK_YEAR:
            # The bundled table already describes this season exactly, and it
            # carries lap counts that FastF1's schedule does not.
            return [race.copy() for race in self.FALLBACK_SCHEDULE], "fallback"

        try:
            import fastf1

            events = fastf1.get_event_schedule(self.year, include_testing=False)
            schedule = []

            for _, event in events.iterrows():
                name = str(event.get("EventName", "")).strip()
                if not name:
                    continue

                location = str(event.get("Location", "")).strip()
                gp = name.replace("Grand Prix", "").strip() or location

                event_date = event.get("EventDate")
                try:
                    date = pd.to_datetime(event_date).strftime("%Y-%m-%d")
                except Exception:
                    date = f"{self.year}-01-01"

                schedule.append({
                    "round": int(event.get("RoundNumber", len(schedule) + 1)),
                    "name": name,
                    "gp": gp,
                    "location": location,
                    "date": date,
                    # FastF1's schedule carries no lap count, so reuse the
                    # bundled figure for the same circuit where we have one.
                    "laps": self._lookup_fallback_laps(gp, name, location),
                })

            if schedule:
                return schedule, "fastf1"
            logger.warning("FastF1 returned an empty schedule for %s", self.year)

        except Exception as exc:
            # No network, no cache, or a season FastF1 does not know about yet.
            logger.warning("Could not load the %s schedule from FastF1 (%s); "
                           "falling back to the bundled %s calendar",
                           self.year, exc, self.FALLBACK_YEAR)

        return [race.copy() for race in self.FALLBACK_SCHEDULE], "fallback"

    def _lookup_fallback_laps(self, gp: str, name: str, location: str) -> int:
        """Find a race distance for a circuit in the bundled table.

        Args:
            gp: Short Grand Prix name (e.g. "Monaco").
            name: Full event name (e.g. "Monaco Grand Prix").
            location: Circuit location (e.g. "Monte Carlo").

        Returns:
            The bundled lap count for a matching circuit, or 55 as a neutral
            default for a circuit the bundled table does not contain.
        """
        needles = [value.lower() for value in (gp, name, location) if value]

        for race in self.FALLBACK_SCHEDULE:
            haystack = (race["gp"].lower(), race["name"].lower(), race["location"].lower())
            if any(needle in haystack or needle in " ".join(haystack) for needle in needles):
                return race["laps"]

        return 55

    def _load_drivers(self) -> tuple:
        """Load the season driver lineup, preferring FastF1.

        Reads the classification of the most recent race of the season that
        actually has results. Only the results are loaded (no laps, telemetry or
        weather), which keeps this far cheaper than a full session load.

        Returns:
            (driver list, source string) where source is "fastf1" or "fallback".
        """
        if self.year == self.FALLBACK_YEAR:
            return [driver.copy() for driver in self.FALLBACK_DRIVERS], "fallback"

        try:
            import fastf1

            # Walk backwards from the last round: the most recent completed race
            # gives the lineup as it stands now.
            for race in sorted(self.schedule, key=lambda r: r["round"], reverse=True):
                try:
                    session = fastf1.get_session(self.year, race["round"], "R")
                    session.load(laps=False, telemetry=False, weather=False, messages=False)
                    results = session.results
                except Exception:
                    continue

                if results is None or len(results) == 0:
                    continue

                drivers = []
                for _, row in results.iterrows():
                    code = str(row.get("Abbreviation", "")).strip()
                    if not code:
                        continue
                    try:
                        number = int(row.get("DriverNumber", 0))
                    except (TypeError, ValueError):
                        number = 0
                    drivers.append({
                        "code": code,
                        "name": str(row.get("FullName", code)).strip() or code,
                        "team": str(row.get("TeamName", "Unknown")).strip() or "Unknown",
                        "number": number,
                    })

                if drivers:
                    return drivers, "fastf1"

            logger.warning("No %s race results available yet for a driver lineup", self.year)

        except Exception as exc:
            logger.warning("Could not load the %s driver lineup from FastF1 (%s); "
                           "falling back to the bundled %s lineup",
                           self.year, exc, self.FALLBACK_YEAR)

        return [driver.copy() for driver in self.FALLBACK_DRIVERS], "fallback"

    def _initialize_points(self) -> dict:
        """Estimate a starting championship position for each driver.

        These are rough seeds used to rank drivers before a season has run, not
        real standings. Anyone not listed starts from zero.
        """
        base_points = {
            "VER": 400, "NOR": 320, "LEC": 280, "PIA": 260, "HAM": 220,
            "RUS": 200, "SAI": 180, "ALO": 60, "STR": 30, "HUL": 25,
            "GAS": 20, "OCO": 15, "TSU": 20, "ALB": 15, "LAW": 10,
            "BEA": 5, "ANT": 0, "HAD": 0, "DOO": 0, "BOR": 0,
        }
        return {driver["code"]: base_points.get(driver["code"], 0)
                for driver in self.drivers}

    @property
    def estimated_points(self) -> dict:
        """Seed points per driver, computed on first use."""
        if self._estimated_points is None:
            self._estimated_points = self._initialize_points()
        return self._estimated_points

    def get_schedule(self) -> list:
        """
        Get the schedule for this provider's season.

        Returns:
            List of race dictionaries with round, name, gp, location, date, laps
        """
        return [race.copy() for race in self.schedule]

    def get_2025_schedule(self) -> list:
        """
        Deprecated alias for get_schedule(), kept for backward compatibility.

        Returns:
            List of race dictionaries for this provider's season (which is not
            necessarily 2025 - the name is historical).
        """
        return self.get_schedule()

    def get_schedule_dataframe(self) -> pd.DataFrame:
        """
        Get schedule as a pandas DataFrame.

        Returns:
            DataFrame with schedule information
        """
        df = pd.DataFrame(self.schedule)
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
        for race in self.schedule:
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
        for race in self.schedule:
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
        current_year = datetime.now().year
        if year > current_year:
            return True
        if year < current_year:
            return False

        race = self.get_race_by_name(gp_name)
        if race:
            try:
                race_date = datetime.strptime(race["date"], "%Y-%m-%d")
            except (ValueError, KeyError, TypeError):
                # Unparseable date: treat it as upcoming rather than crashing.
                return True
            return race_date > datetime.now()
        return True

    def get_current_standings(self) -> list:
        """
        Get current driver standings (estimated for 2025).

        Returns:
            List of drivers with estimated points
        """
        standings = []
        for driver in self.drivers:
            standings.append({
                "code": driver["code"],
                "name": driver["name"],
                "team": driver["team"],
                "points": self.estimated_points.get(driver["code"], 0),
                "number": driver["number"],
            })
        standings.sort(key=lambda x: x["points"], reverse=True)
        return standings

    def get_drivers_list(self) -> list:
        """
        Get the driver lineup for this provider's season.

        Returns:
            List of driver dictionaries
        """
        return [driver.copy() for driver in self.drivers]

    def get_driver_by_code(self, code: str) -> Optional[dict]:
        """
        Get driver information by abbreviation code.

        Args:
            code: Driver abbreviation (e.g., "VER", "HAM")

        Returns:
            Driver dictionary or None
        """
        for driver in self.drivers:
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
        for driver in self.drivers:
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
