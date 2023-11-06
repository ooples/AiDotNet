namespace AiDotNet.Interfaces;

/// <summary>
/// Metrics data to help evaluate the performance of a model by comparing the predicted values to the actual values.
/// Predicted values are taken from the out of sample (oos) data only.
/// </summary>
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