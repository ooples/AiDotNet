namespace AiDotNet.Models.Options;

public class ProphetOptions<T> : TimeSeriesRegressionOptions<T>
{
    public double InitialTrendValue { get; set; } = 0.0;
    public List<int> SeasonalPeriods { get; set; } = [];
    public List<DateTime> Holidays { get; set; } = [];
    public double InitialChangepointValue { get; set; } = 0.0;
    public int RegressorCount { get; set; } = 0;
    public int ForecastHorizon { get; set; } = 30;
    public double ChangePointPriorScale { get; set; } = 0.05;
    public double SeasonalityPriorScale { get; set; } = 10.0;
    public double HolidayPriorScale { get; set; } = 10.0;
    public bool YearlySeasonality { get; set; } = true;
    public bool WeeklySeasonality { get; set; } = true;
    public bool DailySeasonality { get; set; } = false;
    public IOptimizer<T>? Optimizer { get; set; } = null;
    public int FourierOrder { get; set; } = 3;
    public List<T> Changepoints { get; set; } = new List<T>();
}