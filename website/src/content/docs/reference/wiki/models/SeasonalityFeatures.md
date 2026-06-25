---
title: "SeasonalityFeatures"
description: "Flags for selecting which seasonality and calendar features to generate."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Flags for selecting which seasonality and calendar features to generate.

## For Beginners

Seasonality features capture time-based patterns in your data.
Different features are useful for different scenarios:

- Fourier features: Best for smooth, cyclical patterns
- Time features: Useful when patterns vary by hour, day, month
- Holiday features: Important for retail, energy, travel data
- Trading day features: Specific to financial data

## Fields

| Field | Summary |
|:-----|:--------|
| `All` | All available seasonality features. |
| `CalendarEvents` | All calendar event features. |
| `DayOfMonth` | Day of month feature (1-31). |
| `DayOfWeek` | Day of week feature (0-6, Sunday=0). |
| `DayOfYear` | Day of year feature (1-366). |
| `FourierFeatures` | Fourier features (sin/cos at seasonal frequencies). |
| `HolidayFeatures` | Holiday indicator features. |
| `HourOfDay` | Hour of day feature (0-23). |
| `IsWeekend` | Is weekend binary feature. |
| `MonthOfYear` | Month of year feature (1-12). |
| `MonthStartEnd` | Is month start/end features. |
| `None` | No seasonality features. |
| `QuarterOfYear` | Quarter of year feature (1-4). |
| `QuarterStartEnd` | Is quarter start/end features. |
| `TimeFeatures` | All time-based features. |
| `TradingDayOfMonth` | Trading day of month (skips weekends/holidays). |
| `TradingDayOfWeek` | Trading day of week (1-5). |
| `TradingFeatures` | All trading-specific features. |
| `WeekOfYear` | Week of year feature (1-53). |
| `Year` | Year feature (actual year number). |

