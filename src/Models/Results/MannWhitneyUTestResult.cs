namespace AiDotNet.Models.Results;

public class MannWhitneyUTestResult<T>
{
    public T UStatistic { get; set; }
    public T ZScore { get; set; }
    public T PValue { get; set; }
    public bool IsSignificant { get; set; }
    public T SignificanceLevel { get; set; }

    public MannWhitneyUTestResult(T uStatistic, T zScore, T pValue, T significanceLevel)
    {
        UStatistic = uStatistic;
        ZScore = zScore;
        PValue = pValue;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}