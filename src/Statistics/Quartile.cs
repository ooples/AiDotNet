namespace AiDotNet.Statistics;

public class Quartile<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Vector<T> _sortedData;

    public T Q1 { get; }
    public T Q2 { get; }
    public T Q3 { get; }

    public Quartile(Vector<T> data)
    {
        _sortedData = new Vector<T>([.. data.OrderBy(x => x)]);

        Q1 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.25));
        Q2 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.5));
        Q3 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.75));
    }
}