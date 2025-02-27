namespace AiDotNet.Models.Results;

public class TTestResult<T>
{
    public T TStatistic { get; set; }
    public int DegreesOfFreedom { get; set; }
    public T PValue { get; set; }
    public bool IsSignificant { get; set; }
    public T SignificanceLevel { get; set; }

    public TTestResult(T tStatistic, int degreesOfFreedom, T pValue, T significanceLevel)
    {
        TStatistic = tStatistic;
        DegreesOfFreedom = degreesOfFreedom;
        PValue = pValue;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}