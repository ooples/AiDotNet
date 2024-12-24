namespace AiDotNet.Models.Results;

public class ChiSquareTestResult<T>
{
    public T ChiSquareStatistic { get; set; }
    public T PValue { get; set; }
    public int DegreesOfFreedom { get; set; }
    public Vector<T> LeftObserved { get; set; }
    public Vector<T> RightObserved { get; set; }
    public Vector<T> LeftExpected { get; set; }
    public Vector<T> RightExpected { get; set; }
    public T CriticalValue { get; set; }
    public bool IsSignificant { get; set; }

    public ChiSquareTestResult()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        ChiSquareStatistic = numOps.Zero;
        PValue = numOps.Zero;
        CriticalValue = numOps.Zero;
        LeftObserved = Vector<T>.Empty();
        RightObserved = Vector<T>.Empty();
        LeftExpected = Vector<T>.Empty();
        RightExpected = Vector<T>.Empty();
    }
}