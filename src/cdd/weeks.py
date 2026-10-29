"""Calendar helpers. Periods are ISO week numbers within one calendar year."""

from __future__ import annotations

import calendar
from datetime import date

MONTHS: dict[str, int] = {
    name.lower(): number for number, name in enumerate(calendar.month_abbr) if name
}


def weeks_in_year(year: int) -> int:
    return date(year, 12, 28).isocalendar().week


def week_of(year: int, month: int, day: int = 1) -> int:
    """ISO week of a date, clamped into ``year`` at both ends."""
    iso = date(year, month, day).isocalendar()
    if iso.year < year:
        return 1
    if iso.year > year:
        return weeks_in_year(year)
    return iso.week


def last_week_of_month(year: int, month: int) -> int:
    return week_of(year, month, calendar.monthrange(year, month)[1])


def month_of_week(year: int, week: int) -> int:
    """Month that contains the Thursday of an ISO week."""
    return date.fromisocalendar(year, week, 4).month


def month_number(token: str) -> int:
    return MONTHS[token[:3].lower()]


def month_abbr(month: int) -> str:
    return calendar.month_abbr[month]
