namespace AiDotNet.Interfaces;

public abstract class IMetrics
{
    internal abstract double CalculateMeanSquaredError();

    internal abstract double CalculateRootMeanSquaredError();

    internal abstract double CalculateAdjustedR2(double r2);

    internal abstract double CalculateAverageStandardError();

    internal abstract double CalculatePredictionStandardError();

    internal abstract double CalculateAverageStandardDeviation();

    internal abstract double CalculatePredictionStandardDeviation();

    internal abstract int CalculateDegreesOfFreedom();
}