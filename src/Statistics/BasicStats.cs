using AiDotNet.Models.Inputs;

namespace AiDotNet.Statistics;

public class BasicStats<T>
{
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


    private readonly INumericOperations<T> _numOps;

    internal BasicStats(BasicStatsInputs<T> inputs)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize all class variables
        Mean = _numOps.Zero;
        Variance = _numOps.Zero;
        StandardDeviation = _numOps.Zero;
        Skewness = _numOps.Zero;
        Kurtosis = _numOps.Zero;
        Min = _numOps.Zero;
        Max = _numOps.Zero;
        N = 0;
        Median = _numOps.Zero;
        FirstQuartile = _numOps.Zero;
        ThirdQuartile = _numOps.Zero;
        InterquartileRange = _numOps.Zero;
        MAD = _numOps.Zero;

        CalculateStats(inputs.Values);
    }

    public static BasicStats<T> Empty()
    {
        return new BasicStats<T>(new());
    }

    private void CalculateStats(Vector<T> values)
    {
        N = values.Length;

        if (N == 0) return;

        Mean = values.Average();
        Variance = values.Variance();
        StandardDeviation = _numOps.Sqrt(Variance);
        (Skewness, Kurtosis) = StatisticsHelper<T>.CalculateSkewnessAndKurtosis(values, Mean, StandardDeviation, N);
        Min = values.Min();
        Max = values.Max();
        Median = StatisticsHelper<T>.CalculateMedian(values);
        (FirstQuartile, ThirdQuartile) = StatisticsHelper<T>.CalculateQuantiles(values);
        InterquartileRange = _numOps.Subtract(ThirdQuartile, FirstQuartile);
        MAD = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, Median);
    }
}