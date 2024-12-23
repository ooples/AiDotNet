using AiDotNet.Models.Results;

namespace AiDotNet.Models;

public class OptimizationIterationInfo<T>
{
    public int Iteration { get; set; }
    private T _fitness;
    public T Fitness
    {
        get { return _fitness; }
        set { _fitness = value; }
    }

    public FitDetectorResult<T> FitDetectionResult { get; set; }

    public OptimizationIterationInfo()
    {
        FitDetectionResult = new FitDetectorResult<T>();
        _fitness = MathHelper.GetNumericOperations<T>().Zero;
    }

    public OptimizationIterationInfo(T fitness) : this()
    {
        Fitness = fitness;
    }
}