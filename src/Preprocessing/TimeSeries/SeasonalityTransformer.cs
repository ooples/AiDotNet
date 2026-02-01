using AiDotNet.Models.Options;
using System.Globalization;

namespace AiDotNet.Preprocessing.TimeSeries;

/// <summary>
/// Generates seasonality and calendar features for time series data.
/// </summary>
/// <remarks>
/// <para>
/// This transformer creates features that capture time-based patterns including:
/// - Fourier features for smooth cyclical patterns
/// - Calendar features (hour, day, week, month, quarter, year)
/// - Holiday and event features
/// - Trading day features for financial data
/// </para>
/// <para><b>For Beginners:</b> Many real-world patterns repeat over time:
///
/// - <b>Daily patterns</b>: Energy usage peaks in morning/evening
/// - <b>Weekly patterns</b>: Retail sales higher on weekends
/// - <b>Monthly patterns</b>: Bill payments cluster at month start/end
/// - <b>Yearly patterns</b>: Tourism peaks in summer, retail peaks in December
///
/// This transformer creates numerical features that help ML models learn these patterns.
/// For example, instead of just knowing "it's Monday", the model gets:
/// - Day of week = 1 (Monday)
/// - Is weekend = 0 (no)
/// - Sin/Cos waves that smoothly encode the position in the week
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SeasonalityTransformer<T> : TimeSeriesTransformerBase<T>
{
    #region Fields

    /// <summary>
    /// The enabled seasonality features.
    /// </summary>
    private readonly SeasonalityFeatures _enabledFeatures;

    /// <summary>
    /// Seasonal periods for Fourier features.
    /// </summary>
    private readonly int[] _seasonalPeriods;

    /// <summary>
    /// Number of Fourier terms per period.
    /// </summary>
    private readonly int _fourierTerms;

    /// <summary>
    /// Start date of the time series.
    /// </summary>
    private readonly DateTime? _startDate;

    /// <summary>
    /// Time interval between data points.
    /// </summary>
    private readonly TimeSpan? _interval;

    /// <summary>
    /// Holiday dates for holiday features.
    /// </summary>
    private readonly HashSet<DateTime>? _holidayDates;

    /// <summary>
    /// Window days around holidays.
    /// </summary>
    private readonly int _holidayWindowDays;

    /// <summary>
    /// Whether data represents trading days only.
    /// </summary>
    private readonly bool _isTradingDayData;

    /// <summary>
    /// Cached feature names.
    /// </summary>
    private readonly string[] _featureNames;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new seasonality transformer with the specified options.
    /// </summary>
    /// <param name="options">Configuration options, or null for defaults.</param>
    public SeasonalityTransformer(TimeSeriesFeatureOptions? options = null)
        : base(options)
    {
        _enabledFeatures = Options.EnabledSeasonalityFeatures;
        _seasonalPeriods = Options.SeasonalPeriods ?? [7, 30, 365];
        _fourierTerms = Options.FourierTerms;
        _startDate = Options.TimeSeriesStartDate;
        _interval = Options.TimeSeriesInterval;
        _holidayWindowDays = Options.HolidayWindowDays;
        _isTradingDayData = Options.IsTradingDayData;

        if (Options.HolidayDates != null)
        {
            _holidayDates = new HashSet<DateTime>(
                Options.HolidayDates.Select(d => d.Date));
        }

        _featureNames = GenerateFeatureNames();
    }

    #endregion

    #region Properties

    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Gets the number of output features per time step.
    /// </summary>
    public int SeasonalityFeatureCount => _featureNames.Length;

    #endregion

    #region Core Implementation

    /// <inheritdoc />
    protected override void FitCore(Tensor<T> data)
    {
        // Seasonality features don't need to learn parameters from data
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformCore(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int numFeatures = _featureNames.Length;

        var output = new Tensor<T>(new[] { timeSteps, numFeatures });

        for (int t = 0; t < timeSteps; t++)
        {
            DateTime? date = GetDateForTimeStep(t);
            int outputIdx = 0;

            // Fourier features
            if ((_enabledFeatures & SeasonalityFeatures.FourierFeatures) != 0)
            {
                ComputeFourierFeatures(t, timeSteps, output, t, ref outputIdx);
            }

            // Time features (require date information)
            if (date.HasValue)
            {
                ComputeTimeFeatures(date.Value, output, t, ref outputIdx);
                ComputeCalendarEventFeatures(date.Value, output, t, ref outputIdx);
                ComputeTradingFeatures(date.Value, t, output, t, ref outputIdx);
            }
            else
            {
                // Use index-based features when no date is available
                ComputeIndexBasedFeatures(t, timeSteps, output, t, ref outputIdx);
            }
        }

        return output;
    }

    /// <inheritdoc />
    protected override Tensor<T> TransformParallel(Tensor<T> data)
    {
        int timeSteps = GetTimeSteps(data);
        int numFeatures = _featureNames.Length;

        var output = new Tensor<T>(new[] { timeSteps, numFeatures });

        Parallel.For(0, timeSteps, t =>
        {
            DateTime? date = GetDateForTimeStep(t);
            int outputIdx = 0;

            if ((_enabledFeatures & SeasonalityFeatures.FourierFeatures) != 0)
            {
                ComputeFourierFeatures(t, timeSteps, output, t, ref outputIdx);
            }

            if (date.HasValue)
            {
                ComputeTimeFeatures(date.Value, output, t, ref outputIdx);
                ComputeCalendarEventFeatures(date.Value, output, t, ref outputIdx);
                ComputeTradingFeatures(date.Value, t, output, t, ref outputIdx);
            }
            else
            {
                ComputeIndexBasedFeatures(t, timeSteps, output, t, ref outputIdx);
            }
        });

        return output;
    }

    #endregion

    #region Feature Computation

    /// <summary>
    /// Computes Fourier features (sin/cos at seasonal frequencies).
    /// </summary>
    private void ComputeFourierFeatures(int timeIndex, int totalTimeSteps, Tensor<T> output, int outputRow, ref int outputIdx)
    {
        foreach (int period in _seasonalPeriods)
        {
            for (int k = 1; k <= _fourierTerms; k++)
            {
                double angle = 2 * Math.PI * k * timeIndex / period;

                // Sin feature
                output[outputRow, outputIdx++] = NumOps.FromDouble(Math.Sin(angle));
                // Cos feature
                output[outputRow, outputIdx++] = NumOps.FromDouble(Math.Cos(angle));
            }
        }
    }

    /// <summary>
    /// Computes time-based features from the date.
    /// </summary>
    private void ComputeTimeFeatures(DateTime date, Tensor<T> output, int outputRow, ref int outputIdx)
    {
        // Hour of day
        if ((_enabledFeatures & SeasonalityFeatures.HourOfDay) != 0)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble(date.Hour / 23.0); // Normalized 0-1
        }

        // Day of week
        if ((_enabledFeatures & SeasonalityFeatures.DayOfWeek) != 0)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble((int)date.DayOfWeek / 6.0); // Normalized 0-1
        }

        // Day of month
        if ((_enabledFeatures & SeasonalityFeatures.DayOfMonth) != 0)
        {
            int daysInMonth = DateTime.DaysInMonth(date.Year, date.Month);
            output[outputRow, outputIdx++] = NumOps.FromDouble((date.Day - 1.0) / (daysInMonth - 1)); // Normalized 0-1
        }

        // Day of year
        if ((_enabledFeatures & SeasonalityFeatures.DayOfYear) != 0)
        {
            int daysInYear = DateTime.IsLeapYear(date.Year) ? 366 : 365;
            output[outputRow, outputIdx++] = NumOps.FromDouble((date.DayOfYear - 1.0) / (daysInYear - 1)); // Normalized 0-1
        }

        // Week of year
        if ((_enabledFeatures & SeasonalityFeatures.WeekOfYear) != 0)
        {
            int weekOfYear = CultureInfo.InvariantCulture.Calendar.GetWeekOfYear(
                date, CalendarWeekRule.FirstFourDayWeek, DayOfWeek.Monday);
            output[outputRow, outputIdx++] = NumOps.FromDouble((weekOfYear - 1.0) / 52.0); // Normalized 0-1
        }

        // Month of year
        if ((_enabledFeatures & SeasonalityFeatures.MonthOfYear) != 0)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble((date.Month - 1.0) / 11.0); // Normalized 0-1
        }

        // Quarter of year
        if ((_enabledFeatures & SeasonalityFeatures.QuarterOfYear) != 0)
        {
            int quarter = (date.Month - 1) / 3 + 1;
            output[outputRow, outputIdx++] = NumOps.FromDouble((quarter - 1.0) / 3.0); // Normalized 0-1
        }

        // Year (normalize relative to a base year)
        if ((_enabledFeatures & SeasonalityFeatures.Year) != 0)
        {
            // Normalize year relative to 2000, scaled so most years fall in -1 to 1 range
            output[outputRow, outputIdx++] = NumOps.FromDouble((date.Year - 2000.0) / 50.0);
        }
    }

    /// <summary>
    /// Computes calendar event features.
    /// </summary>
    private void ComputeCalendarEventFeatures(DateTime date, Tensor<T> output, int outputRow, ref int outputIdx)
    {
        // Is weekend
        if ((_enabledFeatures & SeasonalityFeatures.IsWeekend) != 0)
        {
            bool isWeekend = date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday;
            output[outputRow, outputIdx++] = NumOps.FromDouble(isWeekend ? 1.0 : 0.0);
        }

        // Month start/end
        if ((_enabledFeatures & SeasonalityFeatures.MonthStartEnd) != 0)
        {
            int daysInMonth = DateTime.DaysInMonth(date.Year, date.Month);
            bool isMonthStart = date.Day <= 3;
            bool isMonthEnd = date.Day >= daysInMonth - 2;
            output[outputRow, outputIdx++] = NumOps.FromDouble(isMonthStart ? 1.0 : 0.0);
            output[outputRow, outputIdx++] = NumOps.FromDouble(isMonthEnd ? 1.0 : 0.0);
        }

        // Quarter start/end
        if ((_enabledFeatures & SeasonalityFeatures.QuarterStartEnd) != 0)
        {
            int[] quarterStartMonths = [1, 4, 7, 10];
            int[] quarterEndMonths = [3, 6, 9, 12];
            bool isQuarterStart = quarterStartMonths.Contains(date.Month) && date.Day <= 5;
            bool isQuarterEnd = quarterEndMonths.Contains(date.Month) &&
                               date.Day >= DateTime.DaysInMonth(date.Year, date.Month) - 4;
            output[outputRow, outputIdx++] = NumOps.FromDouble(isQuarterStart ? 1.0 : 0.0);
            output[outputRow, outputIdx++] = NumOps.FromDouble(isQuarterEnd ? 1.0 : 0.0);
        }

        // Holiday features
        if ((_enabledFeatures & SeasonalityFeatures.HolidayFeatures) != 0)
        {
            bool isHoliday = _holidayDates?.Contains(date.Date) ?? false;
            bool isNearHoliday = false;

            if (_holidayDates != null)
            {
                for (int d = -_holidayWindowDays; d <= _holidayWindowDays; d++)
                {
                    if (_holidayDates.Contains(date.AddDays(d).Date))
                    {
                        isNearHoliday = true;
                        break;
                    }
                }
            }

            output[outputRow, outputIdx++] = NumOps.FromDouble(isHoliday ? 1.0 : 0.0);
            output[outputRow, outputIdx++] = NumOps.FromDouble(isNearHoliday && !isHoliday ? 1.0 : 0.0);
        }
    }

    /// <summary>
    /// Computes trading-specific features.
    /// </summary>
    private void ComputeTradingFeatures(DateTime date, int timeIndex, Tensor<T> output, int outputRow, ref int outputIdx)
    {
        // Trading day of month
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfMonth) != 0)
        {
            int tradingDayOfMonth = GetTradingDayOfMonth(date);
            output[outputRow, outputIdx++] = NumOps.FromDouble(Math.Max(0, tradingDayOfMonth) / 22.0); // ~22 trading days/month
        }

        // Trading day of week
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfWeek) != 0)
        {
            int tradingDayOfWeek = GetTradingDayOfWeek(date);
            // Weekend days (tradingDayOfWeek=0) should be 0; Mon=0.0, Tue=0.25, Wed=0.5, Thu=0.75, Fri=1.0
            double normalized = tradingDayOfWeek == 0 ? 0.0 : (tradingDayOfWeek - 1.0) / 4.0;
            output[outputRow, outputIdx++] = NumOps.FromDouble(normalized);
        }
    }

    /// <summary>
    /// Computes index-based features when no date is available.
    /// </summary>
    private void ComputeIndexBasedFeatures(int timeIndex, int totalTimeSteps, Tensor<T> output, int outputRow, ref int outputIdx)
    {
        int timeFeatureCount = CountTimeFeatures();
        int calendarFeatureCount = CountCalendarEventFeatures();
        int tradingFeatureCount = CountTradingFeatures();

        // Fill time features with NaN or normalized index
        for (int i = 0; i < timeFeatureCount; i++)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble((double)timeIndex / Math.Max(1, totalTimeSteps - 1));
        }

        // Fill calendar features with 0 (no date info)
        for (int i = 0; i < calendarFeatureCount; i++)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble(0.0);
        }

        // Fill trading features with normalized index
        for (int i = 0; i < tradingFeatureCount; i++)
        {
            output[outputRow, outputIdx++] = NumOps.FromDouble((double)timeIndex / Math.Max(1, totalTimeSteps - 1));
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the date for a given time step.
    /// </summary>
    private DateTime? GetDateForTimeStep(int timeIndex)
    {
        if (!_startDate.HasValue)
            return null;

        if (_interval.HasValue)
            return _startDate.Value.Add(TimeSpan.FromTicks(_interval.Value.Ticks * timeIndex));

        // Default to daily if no interval specified
        return _startDate.Value.AddDays(timeIndex);
    }

    /// <summary>
    /// Calculates the trading day of the month (skips weekends).
    /// </summary>
    private int GetTradingDayOfMonth(DateTime date)
    {
        if (_isTradingDayData)
        {
            // If data is already trading days, estimate from day of month
            // Roughly: trading_day ≈ day * 5/7 (since 5 trading days per 7 calendar days)
            return Math.Max(1, (int)Math.Round(date.Day * 5.0 / 7.0));
        }

        // Count weekdays from start of month
        DateTime monthStart = new DateTime(date.Year, date.Month, 1);
        int tradingDays = 0;
        for (DateTime d = monthStart; d <= date; d = d.AddDays(1))
        {
            if (d.DayOfWeek != DayOfWeek.Saturday && d.DayOfWeek != DayOfWeek.Sunday)
            {
                if (!(_holidayDates?.Contains(d.Date) ?? false))
                    tradingDays++;
            }
        }
        return tradingDays;
    }

    /// <summary>
    /// Gets the trading day of the week (1-5, Monday-Friday).
    /// </summary>
    private static int GetTradingDayOfWeek(DateTime date)
    {
        return date.DayOfWeek switch
        {
            DayOfWeek.Monday => 1,
            DayOfWeek.Tuesday => 2,
            DayOfWeek.Wednesday => 3,
            DayOfWeek.Thursday => 4,
            DayOfWeek.Friday => 5,
            _ => 0 // Weekend - shouldn't happen for trading data
        };
    }

    #endregion

    #region Incremental Computation

    /// <summary>
    /// Computes seasonality features incrementally based on time index.
    /// </summary>
    protected override T[] ComputeIncrementalFeatures(IncrementalState<T> state, T[] newDataPoint)
    {
        var features = new T[OutputFeatureCount];
        int outputIdx = 0;

        // Time index is tracked in state.PointsProcessed
        // PointsProcessed is the count BEFORE this point is processed (incremented after ComputeIncrementalFeatures)
        // So PointsProcessed is the 0-based index of the current point
        int timeIndex = (int)state.PointsProcessed;
        int estimatedTotalTimeSteps = timeIndex + 1000; // Estimate for normalization
        DateTime? date = GetDateForTimeStep(timeIndex);

        // Fourier features
        if ((_enabledFeatures & SeasonalityFeatures.FourierFeatures) != 0)
        {
            foreach (int period in _seasonalPeriods)
            {
                for (int k = 1; k <= _fourierTerms; k++)
                {
                    double angle = 2 * Math.PI * k * timeIndex / period;
                    features[outputIdx++] = NumOps.FromDouble(Math.Sin(angle));
                    features[outputIdx++] = NumOps.FromDouble(Math.Cos(angle));
                }
            }
        }

        // Time features
        if (date.HasValue)
        {
            ComputeTimeFeaturesIncremental(date.Value, features, ref outputIdx);
            ComputeCalendarEventFeaturesIncremental(date.Value, features, ref outputIdx);
            ComputeTradingFeaturesIncremental(date.Value, features, ref outputIdx);
        }
        else
        {
            // Fill with index-based values
            int timeFeatureCount = CountTimeFeatures();
            int calendarFeatureCount = CountCalendarEventFeatures();
            int tradingFeatureCount = CountTradingFeatures();

            for (int i = 0; i < timeFeatureCount + calendarFeatureCount + tradingFeatureCount; i++)
            {
                features[outputIdx++] = NumOps.FromDouble((double)timeIndex / Math.Max(1, estimatedTotalTimeSteps - 1));
            }
        }

        return features;
    }

    private void ComputeTimeFeaturesIncremental(DateTime date, T[] features, ref int outputIdx)
    {
        if ((_enabledFeatures & SeasonalityFeatures.HourOfDay) != 0)
            features[outputIdx++] = NumOps.FromDouble(date.Hour / 23.0);
        if ((_enabledFeatures & SeasonalityFeatures.DayOfWeek) != 0)
            features[outputIdx++] = NumOps.FromDouble((int)date.DayOfWeek / 6.0);
        if ((_enabledFeatures & SeasonalityFeatures.DayOfMonth) != 0)
            features[outputIdx++] = NumOps.FromDouble((date.Day - 1.0) / (DateTime.DaysInMonth(date.Year, date.Month) - 1));
        if ((_enabledFeatures & SeasonalityFeatures.DayOfYear) != 0)
            features[outputIdx++] = NumOps.FromDouble((date.DayOfYear - 1.0) / ((DateTime.IsLeapYear(date.Year) ? 366 : 365) - 1));
        if ((_enabledFeatures & SeasonalityFeatures.WeekOfYear) != 0)
        {
            int weekOfYear = CultureInfo.InvariantCulture.Calendar.GetWeekOfYear(date, CalendarWeekRule.FirstFourDayWeek, DayOfWeek.Monday);
            features[outputIdx++] = NumOps.FromDouble((weekOfYear - 1.0) / 52.0);
        }
        if ((_enabledFeatures & SeasonalityFeatures.MonthOfYear) != 0)
            features[outputIdx++] = NumOps.FromDouble((date.Month - 1.0) / 11.0);
        if ((_enabledFeatures & SeasonalityFeatures.QuarterOfYear) != 0)
            features[outputIdx++] = NumOps.FromDouble(((date.Month - 1) / 3) / 3.0);
        if ((_enabledFeatures & SeasonalityFeatures.Year) != 0)
            features[outputIdx++] = NumOps.FromDouble((date.Year - 2000.0) / 50.0);
    }

    private void ComputeCalendarEventFeaturesIncremental(DateTime date, T[] features, ref int outputIdx)
    {
        if ((_enabledFeatures & SeasonalityFeatures.IsWeekend) != 0)
            features[outputIdx++] = NumOps.FromDouble((date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday) ? 1.0 : 0.0);
        if ((_enabledFeatures & SeasonalityFeatures.MonthStartEnd) != 0)
        {
            features[outputIdx++] = NumOps.FromDouble(date.Day <= 3 ? 1.0 : 0.0);
            features[outputIdx++] = NumOps.FromDouble(date.Day >= DateTime.DaysInMonth(date.Year, date.Month) - 2 ? 1.0 : 0.0);
        }
        if ((_enabledFeatures & SeasonalityFeatures.QuarterStartEnd) != 0)
        {
            int[] quarterStartMonths = [1, 4, 7, 10];
            int[] quarterEndMonths = [3, 6, 9, 12];
            features[outputIdx++] = NumOps.FromDouble(quarterStartMonths.Contains(date.Month) && date.Day <= 5 ? 1.0 : 0.0);
            features[outputIdx++] = NumOps.FromDouble(quarterEndMonths.Contains(date.Month) && date.Day >= DateTime.DaysInMonth(date.Year, date.Month) - 4 ? 1.0 : 0.0);
        }
        if ((_enabledFeatures & SeasonalityFeatures.HolidayFeatures) != 0)
        {
            bool isHoliday = _holidayDates?.Contains(date.Date) ?? false;
            bool isNearHoliday = false;
            if (_holidayDates != null)
            {
                for (int d = -_holidayWindowDays; d <= _holidayWindowDays; d++)
                {
                    if (_holidayDates.Contains(date.AddDays(d).Date)) { isNearHoliday = true; break; }
                }
            }
            features[outputIdx++] = NumOps.FromDouble(isHoliday ? 1.0 : 0.0);
            features[outputIdx++] = NumOps.FromDouble(isNearHoliday && !isHoliday ? 1.0 : 0.0);
        }
    }

    private void ComputeTradingFeaturesIncremental(DateTime date, T[] features, ref int outputIdx)
    {
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfMonth) != 0)
            features[outputIdx++] = NumOps.FromDouble(GetTradingDayOfMonth(date) / 22.0);
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfWeek) != 0)
        {
            int tradingDayOfWeek = GetTradingDayOfWeek(date);
            double normalized = tradingDayOfWeek == 0 ? 0.0 : (tradingDayOfWeek - 1.0) / 4.0;
            features[outputIdx++] = NumOps.FromDouble(normalized);
        }
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Exports transformer-specific parameters for serialization.
    /// </summary>
    protected override Dictionary<string, object> ExportParameters()
    {
        return new Dictionary<string, object>
        {
            ["EnabledFeatures"] = (int)_enabledFeatures,
            ["SeasonalPeriods"] = _seasonalPeriods,
            ["FourierTerms"] = _fourierTerms,
            ["HolidayWindowDays"] = _holidayWindowDays,
            ["IsTradingDayData"] = _isTradingDayData
        };
    }

    /// <summary>
    /// Imports transformer-specific parameters for validation.
    /// </summary>
    protected override void ImportParameters(Dictionary<string, object> parameters)
    {
        if (parameters.TryGetValue("EnabledFeatures", out var featuresObj))
        {
            int savedFeatures = Convert.ToInt32(featuresObj);
            if (savedFeatures != (int)_enabledFeatures)
            {
                throw new ArgumentException(
                    $"Saved EnabledFeatures ({savedFeatures}) does not match current configuration ({(int)_enabledFeatures}).");
            }
        }
    }

    #endregion

    #region Feature Naming

    /// <inheritdoc />
    protected override string[] GenerateFeatureNames()
    {
        var names = new List<string>();
        var sep = GetSeparator();

        // Fourier features
        if ((_enabledFeatures & SeasonalityFeatures.FourierFeatures) != 0)
        {
            foreach (int period in _seasonalPeriods)
            {
                for (int k = 1; k <= _fourierTerms; k++)
                {
                    names.Add($"fourier{sep}sin{sep}p{period}{sep}k{k}");
                    names.Add($"fourier{sep}cos{sep}p{period}{sep}k{k}");
                }
            }
        }

        // Time features
        if ((_enabledFeatures & SeasonalityFeatures.HourOfDay) != 0)
            names.Add($"hour{sep}of{sep}day");
        if ((_enabledFeatures & SeasonalityFeatures.DayOfWeek) != 0)
            names.Add($"day{sep}of{sep}week");
        if ((_enabledFeatures & SeasonalityFeatures.DayOfMonth) != 0)
            names.Add($"day{sep}of{sep}month");
        if ((_enabledFeatures & SeasonalityFeatures.DayOfYear) != 0)
            names.Add($"day{sep}of{sep}year");
        if ((_enabledFeatures & SeasonalityFeatures.WeekOfYear) != 0)
            names.Add($"week{sep}of{sep}year");
        if ((_enabledFeatures & SeasonalityFeatures.MonthOfYear) != 0)
            names.Add($"month{sep}of{sep}year");
        if ((_enabledFeatures & SeasonalityFeatures.QuarterOfYear) != 0)
            names.Add($"quarter{sep}of{sep}year");
        if ((_enabledFeatures & SeasonalityFeatures.Year) != 0)
            names.Add("year");

        // Calendar event features
        if ((_enabledFeatures & SeasonalityFeatures.IsWeekend) != 0)
            names.Add($"is{sep}weekend");
        if ((_enabledFeatures & SeasonalityFeatures.MonthStartEnd) != 0)
        {
            names.Add($"is{sep}month{sep}start");
            names.Add($"is{sep}month{sep}end");
        }
        if ((_enabledFeatures & SeasonalityFeatures.QuarterStartEnd) != 0)
        {
            names.Add($"is{sep}quarter{sep}start");
            names.Add($"is{sep}quarter{sep}end");
        }
        if ((_enabledFeatures & SeasonalityFeatures.HolidayFeatures) != 0)
        {
            names.Add($"is{sep}holiday");
            names.Add($"is{sep}near{sep}holiday");
        }

        // Trading features
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfMonth) != 0)
            names.Add($"trading{sep}day{sep}of{sep}month");
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfWeek) != 0)
            names.Add($"trading{sep}day{sep}of{sep}week");

        return [.. names];
    }

    /// <inheritdoc />
    protected override string[] GetOperationNames()
    {
        return ["seasonality"];
    }

    private int CountTimeFeatures()
    {
        int count = 0;
        if ((_enabledFeatures & SeasonalityFeatures.HourOfDay) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.DayOfWeek) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.DayOfMonth) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.DayOfYear) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.WeekOfYear) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.MonthOfYear) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.QuarterOfYear) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.Year) != 0) count++;
        return count;
    }

    private int CountCalendarEventFeatures()
    {
        int count = 0;
        if ((_enabledFeatures & SeasonalityFeatures.IsWeekend) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.MonthStartEnd) != 0) count += 2;
        if ((_enabledFeatures & SeasonalityFeatures.QuarterStartEnd) != 0) count += 2;
        if ((_enabledFeatures & SeasonalityFeatures.HolidayFeatures) != 0) count += 2;
        return count;
    }

    private int CountTradingFeatures()
    {
        int count = 0;
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfMonth) != 0) count++;
        if ((_enabledFeatures & SeasonalityFeatures.TradingDayOfWeek) != 0) count++;
        return count;
    }

    #endregion
}
