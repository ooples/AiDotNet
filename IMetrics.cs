namespace AiDotNet;

public abstract class IMetrics
{
    internal abstract double CalculateMeanSquaredError();

    internal abstract double CalculateRootMeanSquaredError();
}