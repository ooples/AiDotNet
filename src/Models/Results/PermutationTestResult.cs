namespace AiDotNet.Models.Results;

public class PermutationTestResult<T>
{
    public T ObservedDifference { get; set; }
    public T PValue { get; set; }
    public int Permutations { get; set; }
    public int CountExtremeValues { get; set; }
    public bool IsSignificant { get; set; }
    public T SignificanceLevel { get; set; }

    public PermutationTestResult(T observedDifference, T pValue, int permutations, int countExtremeValues, T significanceLevel)
    {
        ObservedDifference = observedDifference;
        PValue = pValue;
        Permutations = permutations;
        CountExtremeValues = countExtremeValues;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}