"""
Training and performance functions for Garmin Connect MCP Server
"""

import json
import datetime
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

# The garmin_client will be set by the main file
garmin_client = None

# Cache for activity type mapping
_activity_type_cache: Optional[Dict[int, str]] = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client, _activity_type_cache
    garmin_client = client
    _activity_type_cache = None  # Reset cache when client changes


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return a dict for Garmin sections that may be null or another shape."""
    return value if isinstance(value, dict) else {}


# Performance Management Chart constants. CTL/ATL are exponentially weighted
# moving averages of daily load with these time constants, so a day's load
# keeps influencing CTL far longer than ATL -- that spread is what makes TSB
# meaningful.
_CTL_TAU_DAYS = 42
_ATL_TAU_DAYS = 7
# CTL needs several time constants of history before it settles. Days closer
# than one time constant to the start of history are reported as ramp-in.
_PMC_RAMP_IN_DAYS = _CTL_TAU_DAYS
# Warm-up fetched before the requested window so CTL does not restart at zero
# on the first requested day. ~4 time constants leaves <2% of the seed error.
_PMC_WARMUP_DAYS = 4 * _CTL_TAU_DAYS

# Weights of the locally computed readiness score, one entry per component.
# Kept as a single dict so they are easy to retune. When a component's data
# is missing it is dropped and the remaining weights are renormalized; a
# missing signal never contributes a neutral value.
_READINESS_WEIGHTS = {
    "hrv": 0.40,  # last night's HRV vs the 60-day personal baseline
    "sleep": 0.20,  # Garmin's own sleep score, used as-is
    "rhr": 0.15,  # resting HR elevation above the 7-day median
    "tsb": 0.25,  # freshness from the same PMC as get_training_load_trend
}
_READINESS_HRV_BASELINE_DAYS = 60
_READINESS_RHR_BASELINE_DAYS = 7


def _ewma_next(previous: float, load: float, tau_days: int) -> float:
    """Advance an exponentially weighted moving average by one day."""
    alpha = 1.0 - math.exp(-1.0 / tau_days)
    return previous + alpha * (load - previous)


def _daily_training_loads(
    start: datetime.date, end: datetime.date
) -> Tuple[Dict[datetime.date, float], Optional[datetime.date], int]:
    """Sum per-activity training load onto the local calendar day it started.

    Returns the loads by date, the earliest activity date seen (the start of
    available history), and the activity count. Activities are dated by
    startTimeLocal because a training log day is the athlete's day, not UTC.
    """
    activities = garmin_client.get_activities_by_date(
        start.isoformat(), end.isoformat()
    )
    loads: Dict[datetime.date, float] = {}
    earliest: Optional[datetime.date] = None
    counted = 0

    for activity in activities or []:
        started = activity.get("startTimeLocal") or activity.get("startTimeGMT")
        if not started:
            continue
        try:
            day = datetime.date.fromisoformat(str(started)[:10])
        except ValueError:
            continue

        counted += 1
        if earliest is None or day < earliest:
            earliest = day

        # An activity Garmin has not scored contributes nothing rather than
        # dropping the day, so its date still counts towards available history.
        load = activity.get("activityTrainingLoad")
        if isinstance(load, (int, float)):
            loads[day] = loads.get(day, 0.0) + float(load)

    return loads, earliest, counted


def _compute_pmc_rows(
    start: datetime.date, end: datetime.date
) -> Tuple[List[Dict[str, Any]], Optional[datetime.date], int]:
    """Compute the PMC (CTL/ATL/TSB) day by day over the requested window.

    Fetches _PMC_WARMUP_DAYS of activities before `start` and walks the EWMAs
    from the earliest activity seen, so CTL/ATL arrive at the window already
    warmed up. Returns one row per day in [start, end] plus the earliest
    activity date and the activity count from _daily_training_loads.
    """
    warmup_start = start - datetime.timedelta(days=_PMC_WARMUP_DAYS)
    loads, earliest, activity_count = _daily_training_loads(warmup_start, end)

    history_start = earliest or start
    atl = 0.0
    ctl = 0.0
    rows: List[Dict[str, Any]] = []
    current = min(history_start, start)
    while current <= end:
        load = loads.get(current, 0.0)
        atl = _ewma_next(atl, load, _ATL_TAU_DAYS)
        ctl = _ewma_next(ctl, load, _CTL_TAU_DAYS)
        if current >= start:
            row: Dict[str, Any] = {
                "date": current.isoformat(),
                "load": round(load, 1),
                "atl": round(atl, 1),
                "ctl": round(ctl, 1),
                "tsb": round(ctl - atl, 1),
            }
            if ctl > 0:
                row["acwr"] = round(atl / ctl, 2)
            if (current - history_start).days < _PMC_RAMP_IN_DAYS:
                row["ramp_in"] = True
            rows.append(row)
        current += datetime.timedelta(days=1)

    return rows, earliest, activity_count


def _linear_score(value: float, floor: float, ceiling: float) -> float:
    """Map value onto 0-100 linearly between floor and ceiling, clamped."""
    if value >= ceiling:
        return 100.0
    if value <= floor:
        return 0.0
    return 100.0 * (value - floor) / (ceiling - floor)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _last_night_hrv(date_str: str) -> Optional[float]:
    """Fetch one night's average HRV; None when absent or unfetchable."""
    try:
        summary = _as_dict(garmin_client.get_hrv_data(date_str)).get("hrvSummary")
    except Exception:
        return None
    value = _as_dict(summary).get("lastNightAvg")
    return float(value) if _is_number(value) else None


def _resting_heart_rate(date_str: str) -> Optional[float]:
    """Fetch one day's resting HR; None when absent or unfetchable."""
    try:
        data = garmin_client.get_rhr_day(date_str)
    except Exception:
        return None
    metrics = _as_dict(_as_dict(_as_dict(data).get("allMetrics")).get("metricsMap"))
    entries = metrics.get("WELLNESS_RESTING_HEART_RATE")
    for entry in entries if isinstance(entries, list) else []:
        value = _as_dict(entry).get("value")
        if _is_number(value):
            return float(value)
    return None


def _extract_vo2_measurements(data: Any) -> Dict[str, float]:
    """Find all VO2 max values by sport in known Garmin response shapes."""
    if isinstance(data, list):
        measurements: Dict[str, float] = {}
        for item in data:
            for sport, value in _extract_vo2_measurements(item).items():
                measurements.setdefault(sport, value)
        return measurements

    data = _as_dict(data)
    # Garmin uses "generic" for its running/non-cycling VO2 max series.
    candidate_paths = (
        (("vo2MaxRunning",), "running"),
        (("vo2MaxCycling",), "cycling"),
        (("vo2Max",), "running"),
        (("vo2MaxValue",), "running"),
        (("vo2MaxPreciseValue",), "running"),
        (("generic", "vo2MaxValue"), "running"),
        (("generic", "vo2MaxPreciseValue"), "running"),
        (("cycling", "vo2MaxValue"), "cycling"),
        (("cycling", "vo2MaxPreciseValue"), "cycling"),
        (("mostRecentVO2Max", "generic", "vo2MaxValue"), "running"),
        (("mostRecentVO2Max", "generic", "vo2MaxPreciseValue"), "running"),
        (("mostRecentVO2Max", "cycling", "vo2MaxValue"), "cycling"),
        (("mostRecentVO2Max", "cycling", "vo2MaxPreciseValue"), "cycling"),
        (("userData", "vo2MaxRunning"), "running"),
        (("userData", "vo2MaxCycling"), "cycling"),
    )

    measurements = {}
    for path, sport in candidate_paths:
        current: Any = data
        for key in path:
            current = _as_dict(current).get(key)
        if (
            sport not in measurements
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
            and math.isfinite(current)
        ):
            measurements[sport] = float(current)
    return measurements


def _extract_dated_vo2_measurements(data: Any) -> Dict[str, Dict[str, float]]:
    """Index max-metrics range values by calendar date and sport."""
    by_date: Dict[str, Dict[str, float]] = {}

    def collect(item: Any) -> None:
        if isinstance(item, list):
            for nested_item in item:
                collect(nested_item)
            return

        item = _as_dict(item)
        measurements = _extract_vo2_measurements(item)
        for sport, value in measurements.items():
            section_name = "generic" if sport == "running" else "cycling"
            section = _as_dict(item.get(section_name))
            calendar_date = section.get("calendarDate") or item.get("calendarDate")
            if isinstance(calendar_date, str):
                by_date.setdefault(calendar_date, {}).setdefault(sport, value)

    collect(data)
    return by_date


def _get_max_metrics_range(
    client: Any, start_date: str, end_date: str
) -> Tuple[bool, Any]:
    """Fetch max metrics for a range when the Garmin client supports it."""
    connectapi = getattr(client, "connectapi", None)
    metrics_url = getattr(client, "garmin_connect_metrics_url", None)
    if not callable(connectapi) or not isinstance(metrics_url, str) or not metrics_url:
        return False, None

    return True, connectapi(f"{metrics_url}/{start_date}/{end_date}")


def _get_activity_type_mapping() -> Dict[int, str]:
    """Get or build a cached mapping of activity type IDs to names"""
    global _activity_type_cache
    if _activity_type_cache is not None:
        return _activity_type_cache

    try:
        activity_types = garmin_client.get_activity_types()
        _activity_type_cache = {
            at.get("typeId"): at.get("typeKey", "unknown")
            for at in activity_types
            if at.get("typeId") is not None
        }
    except Exception:
        _activity_type_cache = {}

    return _activity_type_cache


def _map_contributor(
    contributor: Dict[str, Any], activity_type_mapping: Dict[int, str]
) -> Dict[str, Any]:
    """Map a contributor dict to include human-readable activity type"""
    activity_type_id = contributor.get("activityTypeId")
    group = contributor.get("group")
    contribution = contributor.get("contribution")

    result: Dict[str, Any] = {
        "contribution_percent": round(contribution, 2) if contribution else None
    }

    if activity_type_id is not None:
        result["activity_type"] = activity_type_mapping.get(
            activity_type_id, f"unknown_{activity_type_id}"
        )
        result["activity_type_id"] = activity_type_id
    elif group is not None:
        # Group categories when activityTypeId is None
        # TODO: Find a proper mapping for these groups
        group_names = {
            0: "running (?)",
            1: "biking (?)",
            8: "Other Activities",
        }
        result["group"] = group_names.get(group, f"group_{group}")

    return result


def register_tools(app):
    """Register all training-related tools with the MCP server app"""

    @app.tool()
    async def get_progress_summary_between_dates(
        start_date: str, end_date: str, metric: str
    ) -> str:
        """Get progress summary for a metric between dates

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            metric: Metric to get progress for (e.g., "elevationGain", "duration", "distance", "movingDuration")
        """
        try:
            summary_data = garmin_client.get_progress_summary_between_dates(
                start_date, end_date, metric
            )

            if not summary_data:
                return f"No progress summary found for {metric} between {start_date} and {end_date}."

            # The API returns a list with one item containing stats by activity type
            if isinstance(summary_data, list) and len(summary_data) > 0:
                data = summary_data[0]
            else:
                return f"Unexpected response format from API"

            # Curate to essential fields only
            curated = {
                "metric": metric,
                "start_date": start_date,
                "end_date": end_date,
                "date": data.get("date"),
                "count_of_activities": data.get("countOfActivities"),
                "stats_by_activity_type": {},
            }

            # Parse stats by activity type
            stats = data.get("stats", {})
            for activity_type, activity_stats in stats.items():
                if metric in activity_stats:
                    metric_data = activity_stats[metric]
                    if metric_data and metric_data.get("count", 0) > 0:
                        curated["stats_by_activity_type"][activity_type] = {
                            "count": metric_data.get("count"),
                            "sum": metric_data.get("sum"),
                            "avg": metric_data.get("avg"),
                            "min": metric_data.get("min"),
                            "max": metric_data.get("max"),
                        }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving progress summary: {str(e)}"

    @app.tool()
    async def get_hill_score(start_date: str, end_date: str) -> str:
        """Get hill score data between dates

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        try:
            hill_score_data = garmin_client.get_hill_score(start_date, end_date)

            if not hill_score_data:
                return f"No hill score data found between {start_date} and {end_date}."

            # Parse the period average and max score
            period_avg_score = hill_score_data.get("periodAvgScore", {})
            avg_score = (
                next(iter(period_avg_score.values())) if period_avg_score else None
            )

            # Get the most recent daily score (first in list)
            daily_scores = hill_score_data.get("hillScoreDTOList", [])
            latest_score = daily_scores[0] if daily_scores else {}

            # Curate to essential fields only
            curated = {
                "start_date": start_date,
                "end_date": end_date,
                "period_avg_score": avg_score,
                "max_score": hill_score_data.get("maxScore"),
                # Latest daily score
                "latest_date": latest_score.get("calendarDate"),
                "latest_overall_score": latest_score.get("overallScore"),
                "latest_strength_score": latest_score.get("strengthScore"),
                "latest_endurance_score": latest_score.get("enduranceScore"),
                "latest_classification_id": latest_score.get(
                    "hillScoreClassificationId"
                ),
                # Daily scores over the period
                "daily_scores": [
                    {
                        "date": score.get("calendarDate"),
                        "overall": score.get("overallScore"),
                        "strength": score.get("strengthScore"),
                        "endurance": score.get("enduranceScore"),
                    }
                    for score in daily_scores
                ],
            }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving hill score data: {str(e)}"

    @app.tool()
    async def get_endurance_score(start_date: str, end_date: str) -> str:
        """Get endurance score data between dates

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        try:
            endurance_data = garmin_client.get_endurance_score(start_date, end_date)
            if not endurance_data:
                return f"No endurance score data found between {start_date} and {end_date}."

            # Get activity type mapping for human-readable names
            activity_type_mapping = _get_activity_type_mapping()

            # Extract the current endurance score DTO
            score_dto = endurance_data.get("enduranceScoreDTO", {})

            # Map classification number to label
            classification_labels = {
                1: "recreational",
                2: "intermediate",
                3: "trained",
                4: "well_trained",
                5: "expert",
                6: "superior",
                7: "elite",
            }
            classification_id = score_dto.get("classification")
            classification_label = (
                classification_labels.get(
                    classification_id, f"level_{classification_id}"
                )
                if classification_id is not None
                else None
            )

            # Process contributors with activity type names
            raw_contributors = score_dto.get("contributors", [])
            contributors = (
                [_map_contributor(c, activity_type_mapping) for c in raw_contributors]
                if raw_contributors
                else None
            )

            # Process weekly breakdown from groupMap
            group_map = endurance_data.get("groupMap", {})
            weekly_breakdown = []
            for week_date, week_data in sorted(group_map.items()):
                week_contributors = [
                    _map_contributor(c, activity_type_mapping)
                    for c in week_data.get("enduranceContributorDTOList", [])
                ]
                weekly_breakdown.append(
                    {
                        "week_start": week_date,
                        "avg_score": week_data.get("groupAverage"),
                        "max_score": week_data.get("groupMax"),
                        "contributors": (
                            week_contributors if week_contributors else None
                        ),
                    }
                )

            # Curate to essential fields
            curated = {
                "start_date": start_date,
                "end_date": end_date,
                # Period summary
                "period_avg_score": endurance_data.get("avg"),
                "period_max_score": endurance_data.get("max"),
                # Current/latest endurance score
                "current_score": score_dto.get("overallScore"),
                "current_date": score_dto.get("calendarDate"),
                "classification": classification_label,
                "classification_id": classification_id,
                # Classification thresholds for context
                "thresholds": (
                    {
                        "intermediate": score_dto.get(
                            "classificationLowerLimitIntermediate"
                        ),
                        "trained": score_dto.get("classificationLowerLimitTrained"),
                        "well_trained": score_dto.get(
                            "classificationLowerLimitWellTrained"
                        ),
                        "expert": score_dto.get("classificationLowerLimitExpert"),
                        "superior": score_dto.get("classificationLowerLimitSuperior"),
                        "elite": score_dto.get("classificationLowerLimitElite"),
                    }
                    if score_dto.get("classificationLowerLimitTrained")
                    else None
                ),
                # Contributors breakdown with activity names
                "contributors": contributors,
                # Weekly breakdown
                "weekly_breakdown": weekly_breakdown if weekly_breakdown else None,
            }

            # Remove None values recursively
            def remove_none(obj):
                if isinstance(obj, dict):
                    return {k: remove_none(v) for k, v in obj.items() if v is not None}
                elif isinstance(obj, list):
                    return [remove_none(item) for item in obj]
                return obj

            curated = remove_none(curated)

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving endurance score data: {str(e)}"

    @app.tool()
    async def get_training_effect(activity_id: int) -> str:
        """Get training effect data for a specific activity

        Args:
            activity_id: ID of the activity to retrieve training effect for
        """
        try:
            # Training effect data is available through get_activity
            # The garminconnect library doesn't have a separate get_training_effect method
            activity = garmin_client.get_activity(activity_id)
            if not activity:
                return f"No activity found with ID {activity_id}."

            # Extract training effect data from activity summary
            summary = activity.get("summaryDTO", {})

            # Curate to essential fields only
            curated = {
                "activity_id": activity_id,
                "training_effect": summary.get("trainingEffect"),
                "aerobic_effect": summary.get("trainingEffect"),
                "anaerobic_effect": summary.get("anaerobicTrainingEffect"),
                "training_effect_label": summary.get("trainingEffectLabel"),
                # Recovery metrics
                "recovery_time_hours": (
                    round(summary.get("recoveryTime", 0) / 60, 1)
                    if summary.get("recoveryTime")
                    else None
                ),
                # Training load
                "training_load": summary.get("activityTrainingLoad"),
                # Additional metrics that may be available
                "performance_condition": summary.get("performanceCondition"),
            }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving training effect data: {str(e)}"

    @app.tool()
    async def get_hrv_data(date: str, return_timeseries: bool = False) -> str:
        """Get Heart Rate Variability (HRV) data

        Args:
            date: Date in YYYY-MM-DD format
            return_timeseries: If True, include detailed 5-minute HRV readings (can be large)
        """
        try:
            hrv_data = garmin_client.get_hrv_data(date)
            if not hrv_data:
                return f"No HRV data found for {date}."

            # Extract the summary from hrvSummary key
            summary = hrv_data.get("hrvSummary", {})
            baseline = summary.get("baseline", {})

            # Curate to essential fields only
            curated = {
                "date": summary.get("calendarDate") or date,
                # Current HRV values
                "last_night_avg_hrv_ms": summary.get("lastNightAvg"),
                "last_night_5min_high_hrv_ms": summary.get("lastNight5MinHigh"),
                # Weekly average
                "weekly_avg_hrv_ms": summary.get("weeklyAvg"),
                # Baseline thresholds
                "baseline_balanced_low_ms": baseline.get("balancedLow"),
                "baseline_balanced_upper_ms": baseline.get("balancedUpper"),
                "baseline_low_upper_ms": baseline.get("lowUpper"),
                # Status and feedback
                "status": summary.get("status"),
                "feedback": summary.get("feedbackPhrase"),
                # Sleep window timestamps
                "sleep_start": hrv_data.get("sleepStartTimestampLocal"),
                "sleep_end": hrv_data.get("sleepEndTimestampLocal"),
            }

            # Optionally include the detailed HRV readings timeseries
            if return_timeseries:
                readings = hrv_data.get("hrvReadings", [])
                curated["hrv_readings"] = [
                    {
                        "time": r.get("readingTimeLocal"),
                        "hrv_ms": r.get("hrvValue"),
                    }
                    for r in readings
                ]
                curated["readings_count"] = len(readings)

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving HRV data: {str(e)}"

    @app.tool()
    async def get_fitnessage_data(date: str, details: bool = False) -> str:
        """Get fitness age data

        Args:
            date: Date in YYYY-MM-DD format
            details: If True, include component breakdown (BMI, RHR, vigorous activity)
                     with targets and improvement suggestions
        """
        try:
            fitness_age = garmin_client.get_fitnessage_data(date)
            if not fitness_age:
                return f"No fitness age data found for {date}."

            # Calculate age difference
            chrono_age = fitness_age.get("chronologicalAge")
            fit_age = fitness_age.get("fitnessAge")
            age_diff = None
            if chrono_age is not None and fit_age is not None:
                age_diff = round(chrono_age - fit_age, 1)

            # Curate to essential fields
            curated = {
                "date": date,
                "fitness_age_years": round(fit_age, 1) if fit_age else None,
                "chronological_age_years": chrono_age,
                "age_difference_years": age_diff,
                "achievable_fitness_age_years": (
                    round(fitness_age.get("achievableFitnessAge"), 1)
                    if fitness_age.get("achievableFitnessAge")
                    else None
                ),
                "previous_fitness_age_years": (
                    round(fitness_age.get("previousFitnessAge"), 1)
                    if fitness_age.get("previousFitnessAge")
                    else None
                ),
                "last_updated": fitness_age.get("lastUpdated"),
            }

            # Optionally include component details
            if details:
                components = fitness_age.get("components", {})
                curated_components = {}

                for comp_name, comp_data in components.items():
                    if not isinstance(comp_data, dict):
                        continue

                    comp_info = {"value": comp_data.get("value")}

                    # Add target and improvement info if present
                    if comp_data.get("targetValue") is not None:
                        comp_info["target"] = comp_data.get("targetValue")
                    if comp_data.get("improvementValue") is not None:
                        comp_info["improvement_needed"] = comp_data.get("improvementValue")
                    if comp_data.get("potentialAge") is not None:
                        comp_info["potential_age_if_improved"] = round(
                            comp_data.get("potentialAge"), 1
                        )
                    if comp_data.get("priority") is not None:
                        comp_info["priority"] = comp_data.get("priority")
                    if comp_data.get("stale") is not None:
                        comp_info["stale"] = comp_data.get("stale")
                    if comp_data.get("lastMeasurementDate") is not None:
                        comp_info["last_measurement"] = comp_data.get("lastMeasurementDate")

                    curated_components[comp_name] = comp_info

                if curated_components:
                    curated["components"] = curated_components

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving fitness age data: {str(e)}"

    @app.tool()
    async def get_training_status(date: str) -> str:
        """Get training status with curated metrics

        Returns comprehensive training status including load, VO2 max, recovery,
        and training readiness indicators.

        Args:
            date: Date in YYYY-MM-DD format
        """
        try:
            status = garmin_client.get_training_status(date)
            if not status:
                return f"No training status data found for {date}."

            # Extract from nested structure
            # Use `(x.get(key) or {})` instead of `x.get(key, {})` so that
            # explicit null values in the API response are treated as missing
            # rather than causing `NoneType has no attribute 'get'` errors.
            recent_status = (status.get("mostRecentTrainingStatus") or {})
            latest_data = (recent_status.get("latestTrainingStatusData") or {})

            # Get first device data (usually the primary device)
            device_data = {}
            for device_id, data in latest_data.items():
                device_data = data
                break

            acwr_data = (device_data.get("acuteTrainingLoadDTO") or {})

            # VO2 Max data
            vo2_data = (status.get("mostRecentVO2Max") or {}).get("generic") or {}
            cycling_vo2_data = (status.get("mostRecentVO2Max") or {}).get("cycling") or {}

            # Training load balance
            load_balance = (status.get("mostRecentTrainingLoadBalance") or {})
            load_map = (load_balance.get("metricsTrainingLoadBalanceDTOMap") or {})
            load_data = {}
            for device_id, data in load_map.items():
                load_data = data
                break

            # Curate to essential fields only - remove userIds
            curated = {
                "date": device_data.get("calendarDate", date),
                # Training status
                "training_status": device_data.get("trainingStatus"),
                "training_status_feedback": device_data.get(
                    "trainingStatusFeedbackPhrase"
                ),
                "sport": device_data.get("sport"),
                "fitness_trend": device_data.get("fitnessTrend"),
                # Acute Chronic Workload Ratio
                "acute_load": acwr_data.get("dailyTrainingLoadAcute"),
                "chronic_load": acwr_data.get("dailyTrainingLoadChronic"),
                "load_ratio": acwr_data.get("dailyAcuteChronicWorkloadRatio"),
                "acwr_status": acwr_data.get("acwrStatus"),
                "acwr_percent": acwr_data.get("acwrPercent"),
                "optimal_chronic_load_min": acwr_data.get("minTrainingLoadChronic"),
                "optimal_chronic_load_max": acwr_data.get("maxTrainingLoadChronic"),
                # VO2 Max
                "vo2_max": vo2_data.get("vo2MaxValue"),
                "vo2_max_precise": vo2_data.get("vo2MaxPreciseValue"),
                "cycling_vo2_max": cycling_vo2_data.get("vo2MaxValue"),
                "cycling_vo2_max_precise": cycling_vo2_data.get("vo2MaxPreciseValue"),
                # Monthly training load
                "monthly_load_aerobic_low": load_data.get("monthlyLoadAerobicLow"),
                "monthly_load_aerobic_high": load_data.get("monthlyLoadAerobicHigh"),
                "monthly_load_anaerobic": load_data.get("monthlyLoadAnaerobic"),
                "training_balance_feedback": load_data.get(
                    "trainingBalanceFeedbackPhrase"
                ),
            }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving training status data: {str(e)}"

    @app.tool()
    async def get_cycling_ftp() -> str:
        """Get the latest cycling Functional Threshold Power (FTP) data.

        Returns the most recent cycling FTP estimate available from Garmin.
        """
        try:
            ftp_data = garmin_client.get_cycling_ftp()

            if not ftp_data:
                return "No cycling FTP data found"

            curated = {
                "sport": ftp_data.get("sport"),
                "functional_threshold_power_watts": ftp_data.get(
                    "functionalThresholdPower"
                ),
                "calendar_date": ftp_data.get("calendarDate"),
                "is_stale": ftp_data.get("isStale"),
                "biometric_source_type": ftp_data.get("biometricSourceType"),
            }

            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving cycling FTP data: {str(e)}"

    @app.tool()
    async def get_lactate_threshold(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Get lactate threshold data

        Returns lactate threshold information, which is the exercise intensity at
        which lactate starts to accumulate in the blood. This is a key metric for
        endurance training.

        Args:
            start_date: Start date in YYYY-MM-DD format (optional, omit for latest)
            end_date: End date in YYYY-MM-DD format (optional, omit for latest)
        """
        try:
            # Call API with appropriate parameters
            if start_date and end_date:
                threshold = garmin_client.get_lactate_threshold(
                    latest=False,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                threshold = garmin_client.get_lactate_threshold(latest=True)

            if not threshold:
                if start_date and end_date:
                    return f"No lactate threshold data found between {start_date} and {end_date}"
                return "No lactate threshold data found"

            # Handle different response formats based on query type
            if start_date and end_date:
                # Historical range format: {speed: [...], heartRate: [...], power: [...]}
                curated = {
                    "start_date": start_date,
                    "end_date": end_date,
                }

                # Process speed history
                speed_history = threshold.get("speed", [])
                if speed_history:
                    curated["speed_history"] = [
                        {
                            "date": entry.get("from"),
                            "speed_mps": entry.get("value"),
                            "series": entry.get("series"),
                        }
                        for entry in speed_history
                    ]

                # Process heart rate history
                hr_history = threshold.get("heartRate", [])
                if hr_history:
                    curated["heart_rate_history"] = [
                        {
                            "date": entry.get("from"),
                            "heart_rate_bpm": entry.get("value"),
                            "series": entry.get("series"),
                        }
                        for entry in hr_history
                    ]

                # Process power history
                power_history = threshold.get("power", [])
                if power_history:
                    curated["power_history"] = [
                        {
                            "date": entry.get("from"),
                            "power_watts": entry.get("value"),
                            "series": entry.get("series"),
                        }
                        for entry in power_history
                    ]
            else:
                # Latest format: {speed_and_heart_rate: {...}, power: {...}}
                speed_hr = threshold.get("speed_and_heart_rate", {})
                power = threshold.get("power", {})

                curated = {
                    # Speed and heart rate data
                    "lactate_threshold_speed_mps": speed_hr.get("speed"),
                    "lactate_threshold_heart_rate_bpm": speed_hr.get("heartRate"),
                    "heart_rate_cycling_bpm": speed_hr.get("heartRateCycling"),
                    "speed_hr_date": speed_hr.get("calendarDate"),
                    # Power data
                    "functional_threshold_power_watts": power.get(
                        "functionalThresholdPower"
                    ),
                    "weight_kg": power.get("weight"),
                    "power_to_weight": power.get("powerToWeight"),
                    "sport": power.get("sport"),
                    "power_date": power.get("calendarDate"),
                    "is_stale": power.get("isStale"),
                }

            # Remove None values
            curated = {k: v for k, v in curated.items() if v is not None}

            return json.dumps(curated, indent=2)
        except Exception as e:
            return f"Error retrieving lactate threshold data: {str(e)}"

    @app.tool()
    async def request_reload(date: str) -> str:
        """Request reload of epoch data

        Args:
            date: Date in YYYY-MM-DD format
        """
        try:
            result = garmin_client.request_reload(date)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error requesting data reload: {str(e)}"

    @app.tool()
    async def get_training_load_trend(start_date: str, end_date: str) -> str:
        """Get the Performance Management Chart (CTL/ATL/TSB) over a date range.

        Returns one row per day: the day's total training load, Chronic Training
        Load (CTL, fitness), Acute Training Load (ATL, fatigue), Training Stress
        Balance (TSB = CTL - ATL, form/freshness) and the Acute:Chronic Workload
        Ratio (ACWR). Use this to assess whether the athlete is building fitness,
        peaking, or accumulating too much fatigue.

        These numbers are computed here, not read from Garmin. They are
        exponentially weighted moving averages of each activity's
        activityTrainingLoad, with a 42-day time constant for CTL and 7 days for
        ATL. Garmin's own training-status model is deliberately not consulted,
        so the series means the same thing on every account -- but it will not
        match the CTL/ATL shown in Garmin Connect for accounts where Garmin
        does publish them.

        Rest days count as zero load rather than being skipped, so a gap in
        training shows up as decay. Days are the athlete's local calendar days.

        Rows carry "ramp_in": true while less than 42 days of history precede
        them. CTL is still filling there, which makes TSB read misleadingly
        positive. Do not present those values as settled.

        ACWR is omitted on days where CTL is still zero, because the ratio is
        undefined rather than zero.

        Recommended range: 4-8 weeks. Maximum: 90 days.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        MAX_DAYS = 90
        try:
            start = datetime.date.fromisoformat(start_date)
            end = datetime.date.fromisoformat(end_date)
        except ValueError as e:
            return f"Invalid date format: {e}. Use YYYY-MM-DD."

        days = (end - start).days + 1
        if days > MAX_DAYS:
            return f"Date range too large ({days} days). Maximum is {MAX_DAYS} days."
        if days < 1:
            return "end_date must be on or after start_date."

        try:
            rows, earliest, activity_count = _compute_pmc_rows(start, end)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error retrieving activities: {str(e)}",
            }, indent=2)

        result: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "source": "computed_locally",
            "model": {
                "ctl_time_constant_days": _CTL_TAU_DAYS,
                "atl_time_constant_days": _ATL_TAU_DAYS,
                "warmup_days_fetched": _PMC_WARMUP_DAYS,
                "load_field": "activityTrainingLoad",
            },
            "days": len(rows),
            "days_with_load": sum(1 for d in rows if d["load"] > 0),
        }
        if earliest is not None:
            result["history_starts"] = earliest.isoformat()
        if activity_count == 0:
            fetch_start = start - datetime.timedelta(days=_PMC_WARMUP_DAYS)
            result["status"] = "no_activities_in_range"
            result["message"] = (
                f"No activities between {fetch_start.isoformat()} and {end_date}, "
                "so every day carries zero load and the chart is flat at zero."
            )
        result["trend"] = rows
        return json.dumps(result, indent=2)

    @app.tool()
    async def get_readiness(date: str) -> str:
        """Get a locally computed 0-100 readiness score for one morning.

        This is a heuristic computed by this server from raw Garmin data with
        fixed, visible weights. It is NOT Garmin's Training Readiness and will
        not match it. The component subscores are the point: read them, not
        just the total, because the total hides which signal is off.

        Components and weights (see _READINESS_WEIGHTS):
          hrv   (0.40) last night's HRV vs the mean of the previous 60 nights.
                       ratio >= 1.00 scores 100, <= 0.70 scores 0, linear
                       in between.
          sleep (0.20) Garmin's own sleep score for the night ending on
                       `date`, used as-is.
          rhr   (0.15) resting HR vs the median of the previous 7 days.
                       Each bpm above the median costs 10 points; at or
                       below the median scores 100.
          tsb   (0.25) Training Stress Balance for `date` from the same
                       locally computed PMC as get_training_load_trend.
                       tsb >= 0 scores 100, <= -60 scores 0, linear in
                       between.

        A component whose data is missing for the date is dropped, the
        remaining weights are renormalized, and its name is listed in
        "missing". Absent data is never replaced with a neutral value.

        recovery_hours_remaining (Garmin's remaining recovery timer for the
        morning) is informational only. It is not weighted into the score
        because it is itself derived largely from recent training load, which
        the tsb component already covers.

        Note: the HRV baseline alone needs 60 per-day Garmin requests, so
        this tool is noticeably slower than single-date tools.

        Args:
            date: Date in YYYY-MM-DD format, the morning being scored
        """
        try:
            day = datetime.date.fromisoformat(date)
        except ValueError as e:
            return f"Invalid date format: {e}. Use YYYY-MM-DD."

        components: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        # --- hrv: last night vs the 60-night personal baseline ---
        last_night = _last_night_hrv(date)
        baseline_nights: List[float] = []
        if last_night is not None:
            for offset in range(1, _READINESS_HRV_BASELINE_DAYS + 1):
                night = (day - datetime.timedelta(days=offset)).isoformat()
                value = _last_night_hrv(night)
                if value is not None:
                    baseline_nights.append(value)
        baseline = (
            sum(baseline_nights) / len(baseline_nights) if baseline_nights else 0.0
        )
        if last_night is not None and baseline > 0:
            ratio = last_night / baseline
            components["hrv"] = {
                "score": round(_linear_score(ratio, 0.70, 1.00), 1),
                "weight": _READINESS_WEIGHTS["hrv"],
                "last_night_ms": round(last_night, 1),
                "baseline_ms": round(baseline, 1),
                "baseline_nights": len(baseline_nights),
                "ratio": round(ratio, 3),
            }
        else:
            missing.append("hrv")

        # --- sleep: Garmin's own score for the night ending on `date` ---
        sleep_score = None
        try:
            daily_sleep = _as_dict(
                _as_dict(garmin_client.get_sleep_data(date)).get("dailySleepDTO")
            )
            value = _as_dict(
                _as_dict(daily_sleep.get("sleepScores")).get("overall")
            ).get("value")
            if _is_number(value):
                sleep_score = float(value)
        except Exception:
            sleep_score = None
        if sleep_score is not None:
            components["sleep"] = {
                "score": round(min(max(sleep_score, 0.0), 100.0), 1),
                "weight": _READINESS_WEIGHTS["sleep"],
                "garmin_sleep_score": round(sleep_score, 1),
            }
        else:
            missing.append("sleep")

        # --- rhr: elevation above the previous 7 days' median ---
        today_rhr = _resting_heart_rate(date)
        previous_rhr: List[float] = []
        if today_rhr is not None:
            for offset in range(1, _READINESS_RHR_BASELINE_DAYS + 1):
                value = _resting_heart_rate(
                    (day - datetime.timedelta(days=offset)).isoformat()
                )
                if value is not None:
                    previous_rhr.append(value)
        if today_rhr is not None and previous_rhr:
            median = statistics.median(previous_rhr)
            elevation = max(0.0, today_rhr - median)
            components["rhr"] = {
                "score": round(min(max(100.0 - 10.0 * elevation, 0.0), 100.0), 1),
                "weight": _READINESS_WEIGHTS["rhr"],
                "today_bpm": round(today_rhr, 1),
                "7d_median_bpm": round(median, 1),
                "elevation": round(elevation, 1),
            }
        else:
            missing.append("rhr")

        # --- tsb: freshness from the locally computed PMC ---
        try:
            rows, _earliest, activity_count = _compute_pmc_rows(day, day)
        except Exception:
            rows, activity_count = [], 0
        if rows and activity_count > 0:
            row = rows[-1]
            components["tsb"] = {
                "score": round(_linear_score(row["tsb"], -60.0, 0.0), 1),
                "weight": _READINESS_WEIGHTS["tsb"],
                "tsb": row["tsb"],
                "ctl": row["ctl"],
                "atl": row["atl"],
            }
        else:
            missing.append("tsb")

        # --- informational: Garmin's remaining recovery timer ---
        recovery_hours = None
        try:
            readiness_data = garmin_client.get_morning_training_readiness(date)
            minutes = _as_dict(readiness_data).get("recoveryTime")
            if _is_number(minutes):
                recovery_hours = round(minutes / 60.0, 1)
        except Exception:
            recovery_hours = None
        if recovery_hours is None:
            missing.append("recovery_hours_remaining")

        if not components:
            return json.dumps({
                "date": date,
                "status": "no_data",
                "missing": missing,
                "message": (
                    "None of the readiness components have data for this "
                    "date, so no score can be computed."
                ),
            }, indent=2)

        total_weight = sum(c["weight"] for c in components.values())
        readiness = round(
            sum(c["weight"] * c["score"] for c in components.values()) / total_weight
        )
        if readiness >= 80:
            band = "prime"
        elif readiness >= 60:
            band = "ready"
        elif readiness >= 40:
            band = "compromised"
        else:
            band = "rest"

        # The component pulling the weighted total down the hardest, not
        # merely the lowest raw subscore.
        weakest = max(
            components,
            key=lambda name: components[name]["weight"]
            * (100.0 - components[name]["score"]),
        )
        if components[weakest]["score"] >= 80:
            reason = "All available components are healthy."
        else:
            reason = (
                f"{weakest} is dragging readiness down the most "
                f"(score {round(components[weakest]['score'])}/100)."
            )
        if recovery_hours is not None and recovery_hours > 24 and readiness >= 60:
            reason += (
                f" Note: Garmin's recovery timer still shows {recovery_hours}h "
                "remaining, so yesterday's session may not be fully paid off "
                "even though today's markers look good."
            )

        return json.dumps({
            "date": date,
            "readiness": readiness,
            "band": band,
            "recovery_hours_remaining": recovery_hours,
            "components": components,
            "missing": missing,
            "reason": reason,
        }, indent=2)

    @app.tool()
    async def get_training_load_balance(date: str) -> str:
        """Get Garmin's Load Focus — the distribution of the trailing-month
        training load across Aerobic Low, Aerobic High, and Anaerobic intensity
        bands, plus the system's feedback phrase (e.g. AEROBIC_HIGH_SHORTAGE,
        BALANCED, ANAEROBIC_SHORTAGE).

        Use this to assess whether the athlete's training mix is balanced or
        deficient in a particular intensity band. Each band reports its load
        alongside Garmin's target range; a `status` of "below", "within", or
        "above" is computed from the load relative to that range.

        Args:
            date: Date in YYYY-MM-DD format
        """
        try:
            data = garmin_client.get_training_status(date)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "date": date,
                "message": f"Error retrieving training load balance: {str(e)}",
            }, indent=2)

        if not data:
            return json.dumps({
                "status": "api_returned_no_data",
                "date": date,
                "message": (
                    "Garmin's training-status endpoint returned an empty "
                    f"response for {date}. This is a transport-level empty, "
                    "not a statement about the device."
                ),
            }, indent=2)

        # Same `or {}` pattern as the rest of this module: Garmin returns
        # explicit None for sections the user has no data in, which breaks
        # chained .get with a default arg.
        load_balance = data.get("mostRecentTrainingLoadBalance") or {}
        load_map = load_balance.get("metricsTrainingLoadBalanceDTOMap") or {}

        # device-keyed dict; prefer the primary training device, fall back to
        # the first device with data.
        load_data: Dict[str, Any] = {}
        for dev_data in load_map.values():
            if not isinstance(dev_data, dict):
                continue
            if dev_data.get("primaryTrainingDevice"):
                load_data = dev_data
                break
            if not load_data:
                load_data = dev_data

        if not load_data:
            return json.dumps({
                "status": "not_available_for_this_account",
                "date": date,
                "message": (
                    "Garmin answered but published no Load Focus for any "
                    "device. Load Focus comes from Garmin's training-status "
                    "model, which only populates for accounts and devices "
                    "that record qualifying activities. Retrying, or trying "
                    "another date, will not change this. Per-activity "
                    "aerobic/anaerobic training effect is still available "
                    "from get_training_effect, and get_training_load_trend "
                    "computes CTL/ATL locally."
                ),
            }, indent=2)

        def _band(load_key: str, min_key: str, max_key: str) -> Optional[Dict[str, Any]]:
            load = load_data.get(load_key)
            tmin = load_data.get(min_key)
            tmax = load_data.get(max_key)
            if load is None and tmin is None and tmax is None:
                return None
            band: Dict[str, Any] = {}
            if load is not None:
                band["load"] = round(load, 1)
            if tmin is not None:
                band["target_min"] = tmin
            if tmax is not None:
                band["target_max"] = tmax
            if load is not None and tmin is not None and tmax is not None:
                if load < tmin:
                    band["status"] = "below"
                elif load > tmax:
                    band["status"] = "above"
                else:
                    band["status"] = "within"
            return band

        result: Dict[str, Any] = {
            "date": load_data.get("calendarDate", date),
        }
        feedback = load_data.get("trainingBalanceFeedbackPhrase")
        if feedback:
            result["feedback"] = feedback

        aerobic_low = _band(
            "monthlyLoadAerobicLow",
            "monthlyLoadAerobicLowTargetMin",
            "monthlyLoadAerobicLowTargetMax",
        )
        if aerobic_low:
            result["aerobic_low"] = aerobic_low

        aerobic_high = _band(
            "monthlyLoadAerobicHigh",
            "monthlyLoadAerobicHighTargetMin",
            "monthlyLoadAerobicHighTargetMax",
        )
        if aerobic_high:
            result["aerobic_high"] = aerobic_high

        anaerobic = _band(
            "monthlyLoadAnaerobic",
            "monthlyLoadAnaerobicTargetMin",
            "monthlyLoadAnaerobicTargetMax",
        )
        if anaerobic:
            result["anaerobic"] = anaerobic

        return json.dumps(result, indent=2)

    @app.tool()
    async def get_hrv_trend(start_date: str, end_date: str) -> str:
        """Get HRV (Heart Rate Variability) trend over a date range.

        Returns daily HRV values and weekly rolling averages. Single-day HRV is too noisy
        to act on — use this tool to identify baseline shifts that signal accumulated fatigue
        or recovery. A drop of >10ms from the 7-day baseline warrants reducing training load.

        Recommended range: 7-21 days. Maximum: 30 days.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        MAX_DAYS = 30
        try:
            start = datetime.date.fromisoformat(start_date)
            end = datetime.date.fromisoformat(end_date)
        except ValueError as e:
            return f"Invalid date format: {e}. Use YYYY-MM-DD."

        days = (end - start).days + 1
        if days > MAX_DAYS:
            return f"Date range too large ({days} days). Maximum is {MAX_DAYS} days."
        if days < 1:
            return "end_date must be on or after start_date."

        trend = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            try:
                data = garmin_client.get_hrv_data(date_str)
                if data:
                    hrv_summary = data.get("hrvSummary", {})
                    entry: Dict[str, Any] = {"date": date_str}
                    last_night = hrv_summary.get("lastNight")
                    weekly_avg = hrv_summary.get("weeklyAvg")
                    status = hrv_summary.get("status")
                    feedback = hrv_summary.get("feedbackPhrase")
                    high_hrv = hrv_summary.get("lastNight5MinHigh")
                    if last_night is not None:
                        entry["last_night_avg_hrv_ms"] = round(last_night, 1)
                    if weekly_avg is not None:
                        entry["weekly_avg_hrv_ms"] = round(weekly_avg, 1)
                    if high_hrv is not None:
                        entry["last_night_5min_high_hrv_ms"] = round(high_hrv, 1)
                    if status:
                        entry["status"] = status
                    if feedback:
                        entry["feedback"] = feedback
                    if len(entry) > 1:
                        trend.append(entry)
            except Exception:
                pass
            current += datetime.timedelta(days=1)

        if not trend:
            return f"No HRV data found between {start_date} and {end_date}."

        # Compute 7-day rolling average from the collected data
        hrv_values = [e.get("last_night_avg_hrv_ms") for e in trend if e.get("last_night_avg_hrv_ms") is not None]
        rolling_avg = None
        if hrv_values:
            rolling_avg = round(sum(hrv_values) / len(hrv_values), 1)

        return json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "days_with_data": len(trend),
            "period_avg_hrv_ms": rolling_avg,
            "trend": trend,
        }, indent=2)

    @app.tool()
    async def get_vo2max_trend(start_date: str, end_date: str) -> str:
        """Get VO2 max trend over a date range.

        Returns daily VO2 max estimates from Garmin's FirstBeat algorithm. Use this to track
        whether training is producing fitness gains over weeks or months. Flat or declining
        VO2 max over 4+ weeks suggests insufficient training stimulus or overreaching.

        Note: VO2 max estimates are smoothed and update gradually — daily changes of <0.5
        are within normal noise. Focus on the 4-6 week trend direction.

        If historical values are unavailable, the current profile estimate is returned
        separately and is not represented as a historical trend point.

        Recommended range: 4-12 weeks. Maximum: 90 days.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        MAX_DAYS = 90
        try:
            start = datetime.date.fromisoformat(start_date)
            end = datetime.date.fromisoformat(end_date)
        except ValueError as e:
            return f"Invalid date format: {e}. Use YYYY-MM-DD."

        days = (end - start).days + 1
        if days > MAX_DAYS:
            return f"Date range too large ({days} days). Maximum is {MAX_DAYS} days."
        if days < 1:
            return "end_date must be on or after start_date."

        histories: Dict[str, List[Dict[str, Any]]] = {
            "running": [],
            "cycling": [],
        }
        range_measurements: Optional[Dict[str, Dict[str, float]]] = None
        try:
            range_supported, range_data = _get_max_metrics_range(
                garmin_client, start_date, end_date
            )
            if range_supported:
                range_measurements = _extract_dated_vo2_measurements(range_data)
        except Exception:
            # The range endpoint exists but failed. Continue with training status
            # without retrying the same max-metrics endpoint once per day.
            range_measurements = {}

        current = start
        while current <= end:
            date_str = current.isoformat()
            measurements: Dict[str, float] = {}
            source = None

            if range_measurements is not None:
                measurements = range_measurements.get(date_str, {})
                if measurements:
                    source = "get_max_metrics"
                method_names = ("get_training_status",) if not measurements else ()
            else:
                method_names = ("get_training_status", "get_max_metrics")

            for method_name in method_names:
                method = getattr(garmin_client, method_name, None)
                if not callable(method):
                    continue
                try:
                    measurements = _extract_vo2_measurements(method(date_str))
                except Exception:
                    continue
                if not measurements:
                    continue
                source = method_name
                break

            for sport, vo2 in measurements.items():
                histories[sport].append(
                    {
                        "date": date_str,
                        "vo2_max": round(vo2, 1),
                        "source": source,
                    }
                )
            current += datetime.timedelta(days=1)

        selected_sport = None
        if any(histories.values()):
            # Prefer the sport with the best coverage; make ties deterministic.
            selected_sport = max(
                histories,
                key=lambda sport: (
                    len(histories[sport]),
                    sport == "running",
                ),
            )

        trend = []
        last_vo2 = None
        if selected_sport is not None:
            for entry in histories[selected_sport]:
                if entry["vo2_max"] != last_vo2:
                    trend.append(entry)
                    last_vo2 = entry["vo2_max"]

        current_estimate = None
        current_sport = None
        if not trend:
            try:
                profile = garmin_client.get_user_profile()
            except Exception:
                profile = None

            profile_measurements = _extract_vo2_measurements(profile)
            if profile_measurements:
                current_sport = (
                    "running" if "running" in profile_measurements else "cycling"
                )
                current_estimate = profile_measurements[current_sport]
            else:
                return f"No VO2 max data found between {start_date} and {end_date}."

        first_vo2 = trend[0]["vo2_max"] if trend else None
        latest_vo2 = trend[-1]["vo2_max"] if trend else None
        change = (
            round(latest_vo2 - first_vo2, 1)
            if (first_vo2 is not None and latest_vo2 is not None)
            else None
        )

        response = {
            "start_date": start_date,
            "end_date": end_date,
            "data_points": len(trend),
            "first_vo2_max": first_vo2,
            "latest_vo2_max": latest_vo2,
            "change": change,
            "trend": trend,
        }
        if selected_sport is not None:
            response["sport"] = selected_sport

        if current_estimate is not None:
            response["current_vo2_max_estimate"] = {
                "vo2_max": round(current_estimate, 1),
                "sport": current_sport,
                "source": "get_user_profile",
            }
            response["note"] = (
                "Historical VO2 max values were not available from Garmin; "
                "returning the current profile estimate separately."
            )

        return json.dumps(response, indent=2)

    @app.tool()
    async def get_respiration_trend(start_date: str, end_date: str) -> str:
        """Get overnight respiration rate trend over a date range.

        Elevated resting respiration rate (compared to personal baseline) is an early
        warning sign for overreaching, illness, or poor recovery. Use this alongside HRV
        trend for a complete recovery picture.

        Recommended range: 7-21 days. Maximum: 30 days.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        MAX_DAYS = 30
        try:
            start = datetime.date.fromisoformat(start_date)
            end = datetime.date.fromisoformat(end_date)
        except ValueError as e:
            return f"Invalid date format: {e}. Use YYYY-MM-DD."

        days = (end - start).days + 1
        if days > MAX_DAYS:
            return f"Date range too large ({days} days). Maximum is {MAX_DAYS} days."
        if days < 1:
            return "end_date must be on or after start_date."

        trend = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            try:
                data = garmin_client.get_respiration_data(date_str)
                if data:
                    entry: Dict[str, Any] = {"date": date_str}
                    avg_waking = data.get("avgWakingRespirationValue")
                    avg_sleep = data.get("avgSleepRespirationValue")
                    high_sleep = data.get("highestRespirationValue")
                    low_sleep = data.get("lowestRespirationValue")
                    if avg_waking is not None:
                        entry["avg_waking_breaths_per_min"] = round(avg_waking, 1)
                    if avg_sleep is not None:
                        entry["avg_sleep_breaths_per_min"] = round(avg_sleep, 1)
                    if high_sleep is not None:
                        entry["highest_breaths_per_min"] = round(high_sleep, 1)
                    if low_sleep is not None:
                        entry["lowest_breaths_per_min"] = round(low_sleep, 1)
                    if len(entry) > 1:
                        trend.append(entry)
            except Exception:
                pass
            current += datetime.timedelta(days=1)

        if not trend:
            return f"No respiration data found between {start_date} and {end_date}."

        sleep_values = [e["avg_sleep_breaths_per_min"] for e in trend if "avg_sleep_breaths_per_min" in e]
        avg_sleep_overall = round(sum(sleep_values) / len(sleep_values), 1) if sleep_values else None

        return json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "days_with_data": len(trend),
            "period_avg_sleep_breaths_per_min": avg_sleep_overall,
            "trend": trend,
        }, indent=2)

    return app
