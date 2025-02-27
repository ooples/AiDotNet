namespace AiDotNet.Models.Results;

public class FTestResult<T>
{
    public T FStatistic { get; set; }
    public T PValue { get; set; }
    public int NumeratorDegreesOfFreedom { get; set; }
    public int DenominatorDegreesOfFreedom { get; set; }
    public T LeftVariance { get; set; }
    public T RightVariance { get; set; }
    public T LowerConfidenceInterval { get; set; }
    public T UpperConfidenceInterval { get; set; }
    public bool IsSignificant { get; set; }
    public T SignificanceLevel { get; set; }

    public FTestResult(T fStatistic, T pValue, int numeratorDf, int denominatorDf, T leftVariance, T rightVariance, T lowerCI, T upperCI, T significanceLevel)
    {
        FStatistic = fStatistic;
        PValue = pValue;
        NumeratorDegreesOfFreedom = numeratorDf;
        DenominatorDegreesOfFreedom = denominatorDf;
        LeftVariance = leftVariance;
        RightVariance = rightVariance;
        LowerConfidenceInterval = lowerCI;
        UpperConfidenceInterval = upperCI;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}