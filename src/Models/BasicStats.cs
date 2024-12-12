namespace AiDotNet.Models;

public class BasicStats<T>
{
    private readonly INumericOperations<T> NumOps;

    public T Mean { get; private set; }
    public T Variance { get; private set; }
    public T StandardDeviation { get; private set; }
    public T Skewness { get; private set; }
    public T Kurtosis { get; private set; }
    public T Min { get; private set; }
    public T Max { get; private set; }
    public int N { get; private set; }
    public T Median { get; private set; }
    public T FirstQuartile { get; private set; }
    public T ThirdQuartile { get; private set; }
    public T InterquartileRange { get; private set; }
    public T MAD { get; private set; }

    public BasicStats(Vector<T> values)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        
        // Initialize all class variables
        Mean = NumOps.Zero;
        Variance = NumOps.Zero;
        StandardDeviation = NumOps.Zero;
        Skewness = NumOps.Zero;
        Kurtosis = NumOps.Zero;
        Min = NumOps.Zero;
        Max = NumOps.Zero;
        N = 0;
        Median = NumOps.Zero;
        FirstQuartile = NumOps.Zero;
        ThirdQuartile = NumOps.Zero;
        InterquartileRange = NumOps.Zero;
        MAD = NumOps.Zero;

        CalculateStats(values);
    }

    public static BasicStats<T> Empty()
    {
        return new BasicStats<T>(Vector<T>.Empty());
    }

    private void CalculateStats(Vector<T> values)
    {
        N = values.Length;

        if (N == 0) return;

        Mean = values.Average();
        Variance = values.Variance();
        StandardDeviation = NumOps.Sqrt(Variance);
        (Skewness, Kurtosis) = StatisticsHelper<T>.CalculateSkewnessAndKurtosis(values, Mean, StandardDeviation, N);
        Min = values.Min();
        Max = values.Max();
        Median = StatisticsHelper<T>.CalculateMedian(values);
        (FirstQuartile, ThirdQuartile) = StatisticsHelper<T>.CalculateQuantiles(values);
        InterquartileRange = NumOps.Subtract(ThirdQuartile, FirstQuartile);
        MAD = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, Median);
    }
}