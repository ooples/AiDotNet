using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.Models.Options;

public class STLDecompositionOptions<T> : TimeSeriesRegressionOptions<T>
{
    public new int SeasonalPeriod { get; set; } = 12; // Default to monthly data
    public int SeasonalDegree { get; set; } = 1;
    public int TrendDegree { get; set; } = 1;
    public int SeasonalJump { get; set; } = 1;
    public int TrendJump { get; set; } = 1;
    public int InnerLoopPasses { get; set; } = 2;
    public int RobustIterations { get; set; } = 1;
    public double RobustWeightThreshold { get; set; } = 0.001;
    public double SeasonalBandwidth { get; set; } = 0.75;
    public double TrendBandwidth { get; set; } = 0.75;
    public double LowPassBandwidth { get; set; } = 0.75;
    public int TrendWindowSize { get; set; } = 18; // 1.5 * default SeasonalPeriod
    public int SeasonalLoessWindow { get; set; } = 121; // 10 * default SeasonalPeriod + 1
    public int LowPassFilterWindowSize { get; set; } = 12; // Same as default SeasonalPeriod
    public int TrendLoessWindow { get; set; } = 12; // Same as default SeasonalPeriod
    public STLAlgorithmType AlgorithmType { get; set; } = STLAlgorithmType.Standard;
    public DateTime[] Dates { get; set; } = [];
    public DateTime? StartDate { get; set; }
    public TimeSpan? Interval { get; set; }
    public bool AdjustForDayOfWeek { get; set; } = false;
    public bool AdjustForMonthOfYear { get; set; } = false;
    public bool AdjustForHolidays { get; set; } = false;
    public Vector<T> DayOfWeekFactors { get; set; } = new Vector<T>(7);
    public Vector<T> MonthOfYearFactors { get; set; } = new Vector<T>(12);
    public Dictionary<DateTime, T> Holidays { get; set; } = [];
    public OutlierDetectionMethod OutlierDetectionMethod { get; set; } = OutlierDetectionMethod.ZScore;
    public double ZScoreThreshold { get; set; } = 3.0; // Default value: 3 standard deviations
    public double IQRMultiplier { get; set; } = 1.5; // Default value: 1.5 times the IQR
}