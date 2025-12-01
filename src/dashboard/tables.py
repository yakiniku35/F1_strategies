"""
F1 Dashboard Tables Module.
Generates data tables for race predictions and strategy analysis.
Inspired by f1-dash (https://github.com/slowlydev/f1-dash) style.
"""

import json
import os
from typing import Optional

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


def generate_driver_standings_table(frame: dict, predictions: dict) -> str:
    """
    Generate driver standings prediction table.

    Args:
        frame: Current race frame with driver data
        predictions: ML predictions for all drivers

    Returns:
        Formatted table string
    """
    if not frame or 'drivers' not in frame:
        return "No driver data available"

    drivers_data = frame['drivers']
    rows = []

    # Sort drivers by current position
    sorted_drivers = sorted(
        drivers_data.items(),
        key=lambda x: x[1].get('position', 99)
    )

    for code, data in sorted_drivers:
        position = data.get('position', '-')
        lap = data.get('lap', '-')
        speed = data.get('speed', 0)

        # Get prediction data
        pred = predictions.get(code, {})
        pred_pos = pred.get('predicted_position', '-')
        trend = pred.get('trend', 'stable')
        confidence = pred.get('confidence', 0)

        # Trend indicator
        if trend == 'improving':
            trend_icon = '↑'
        elif trend == 'declining':
            trend_icon = '↓'
        else:
            trend_icon = '→'

        pos_change = pred.get('position_change', 0)
        if isinstance(pos_change, (int, float)):
            change_str = f"{'+' if pos_change > 0 else ''}{pos_change:.1f}"
        else:
            change_str = '-'

        rows.append([
            position,
            code,
            lap,
            f"{speed:.0f}",
            pred_pos if isinstance(pred_pos, str) else f"{pred_pos:.1f}",
            f"{trend_icon} {change_str}",
            f"{confidence * 100:.0f}%" if confidence else '-'
        ])

    headers = ['Pos', 'Driver', 'Lap', 'Speed', 'Pred Pos', 'Trend', 'Conf']

    if TABULATE_AVAILABLE:
        return tabulate(rows, headers=headers, tablefmt='pretty')
    else:
        # Fallback formatting without tabulate
        result = ' | '.join(headers) + '\n'
        result += '-' * 60 + '\n'
        for row in rows:
            result += ' | '.join(str(cell) for cell in row) + '\n'
        return result


def generate_strategy_table(frame: dict, predictions: dict) -> str:
    """
    Generate strategy recommendations table.

    Args:
        frame: Current race frame with driver data
        predictions: ML predictions with pit strategy

    Returns:
        Formatted strategy table string
    """
    if not frame or 'drivers' not in frame:
        return "No driver data available"

    drivers_data = frame['drivers']
    rows = []

    # Tyre compound mapping
    tyre_names = {0: 'SOFT', 1: 'MEDIUM', 2: 'HARD', 3: 'INTER', 4: 'WET'}

    sorted_drivers = sorted(
        drivers_data.items(),
        key=lambda x: x[1].get('position', 99)
    )

    for code, data in sorted_drivers:
        position = data.get('position', '-')
        current_tyre = tyre_names.get(data.get('tyre', -1), 'UNKNOWN')
        current_lap = data.get('lap', 1)

        # Get pit strategy from predictions
        pred = predictions.get(code, {})
        pit_strategy = pred.get('pit_strategy', {})

        if pit_strategy:
            pit_window = pit_strategy.get('estimated_pit_window', ('-', '-'))
            recommendation = pit_strategy.get('recommendation', 'N/A')
            if isinstance(pit_window, tuple) and len(pit_window) == 2:
                window_str = f"L{pit_window[0]}-L{pit_window[1]}"
            else:
                window_str = '-'
        else:
            window_str = '-'
            recommendation = 'Insufficient data'

        rows.append([
            position,
            code,
            current_tyre,
            current_lap,
            window_str,
            recommendation[:40] if len(recommendation) > 40 else recommendation
        ])

    headers = ['Pos', 'Driver', 'Tyre', 'Lap', 'Pit Window', 'Recommendation']

    if TABULATE_AVAILABLE:
        return tabulate(rows, headers=headers, tablefmt='pretty')
    else:
        result = ' | '.join(headers) + '\n'
        result += '-' * 80 + '\n'
        for row in rows:
            result += ' | '.join(str(cell) for cell in row) + '\n'
        return result


def generate_battle_table(frame: dict, predictions: dict) -> str:
    """
    Generate battle predictions table.

    Args:
        frame: Current race frame with driver data
        predictions: ML predictions for all drivers

    Returns:
        Formatted battle table string
    """
    if not frame or 'drivers' not in frame:
        return "No driver data available"

    drivers_data = frame['drivers']
    rows = []

    # Sort drivers by position
    sorted_drivers = sorted(
        drivers_data.items(),
        key=lambda x: x[1].get('position', 99)
    )

    # Find potential battles (drivers close in predicted position)
    for i, (code1, data1) in enumerate(sorted_drivers[:-1]):
        code2, data2 = sorted_drivers[i + 1]

        pred1 = predictions.get(code1, {})
        pred2 = predictions.get(code2, {})

        pred_pos1 = pred1.get('predicted_position', data1.get('position', 99))
        pred_pos2 = pred2.get('predicted_position', data2.get('position', 99))

        # Check if battle is predicted (positions close or swapping)
        pos_diff = abs(pred_pos1 - pred_pos2) if isinstance(pred_pos1, (int, float)) and isinstance(pred_pos2, (int, float)) else 10

        if pos_diff < 1.5:
            battle_intensity = 'HIGH' if pos_diff < 0.5 else 'MEDIUM'
            trend1 = pred1.get('trend', 'stable')
            trend2 = pred2.get('trend', 'stable')

            # Determine likely outcome
            if trend1 == 'improving' and trend2 != 'improving':
                likely_winner = code1
            elif trend2 == 'improving' and trend1 != 'improving':
                likely_winner = code2
            else:
                likely_winner = '50/50'

            rows.append([
                f"P{data1.get('position', '?')}",
                code1,
                'vs',
                code2,
                battle_intensity,
                likely_winner
            ])

    if not rows:
        return "No significant battles predicted"

    headers = ['Position', 'Driver 1', '', 'Driver 2', 'Intensity', 'Likely Outcome']

    if TABULATE_AVAILABLE:
        return tabulate(rows, headers=headers, tablefmt='pretty')
    else:
        result = ' | '.join(headers) + '\n'
        result += '-' * 70 + '\n'
        for row in rows:
            result += ' | '.join(str(cell) for cell in row) + '\n'
        return result


def export_to_csv(table_data: str, filepath: str) -> bool:
    """
    Export table data to CSV file.

    Args:
        table_data: Formatted table string
        filepath: Path to output CSV file

    Returns:
        True if export successful, False otherwise
    """
    try:
        # Parse the table string and convert to CSV format
        lines = table_data.strip().split('\n')

        # Filter out separator lines
        data_lines = [line for line in lines if not line.startswith('-') and not line.startswith('+')]

        csv_content = []
        for line in data_lines:
            # Remove table formatting characters
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                csv_content.append(','.join(cells))

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_content))

        print(f"Exported CSV to: {filepath}")
        return True

    except (IOError, OSError) as e:
        print(f"Error exporting CSV: {e}")
        return False


def export_to_json(predictions: dict, filepath: str) -> bool:
    """
    Export predictions to JSON file.

    Args:
        predictions: Predictions dictionary
        filepath: Path to output JSON file

    Returns:
        True if export successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, default=str)

        print(f"Exported JSON to: {filepath}")
        return True

    except (IOError, OSError, TypeError) as e:
        print(f"Error exporting JSON: {e}")
        return False
