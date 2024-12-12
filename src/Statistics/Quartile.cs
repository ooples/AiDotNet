namespace AiDotNet.Statistics;

public class Quartile<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Vector<T> _sortedData;

    public T Q1 { get; }
    public T Q2 { get; }
    public T Q3 { get; }

    public Quartile(Vector<T> data, INumericOperations<T> numOps)
    {
        _numOps = numOps;
        _sortedData = new Vector<T>([.. data.OrderBy(x => x)], _numOps);

        Q1 = StatisticsHelper<T>.CalculateQuantile(_sortedData, _numOps.FromDouble(0.25));
        Q2 = StatisticsHelper<T>.CalculateQuantile(_sortedData, _numOps.FromDouble(0.5));
        Q3 = StatisticsHelper<T>.CalculateQuantile(_sortedData, _numOps.FromDouble(0.75));
    }
}