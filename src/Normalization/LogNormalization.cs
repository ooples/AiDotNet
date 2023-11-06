namespace AiDotNet.Normalization;

/// <summary>
/// Normalizes the data by taking the natural log of each value.
/// </summary>
public class LogNormalization : INormalization
{
    internal override double[] Normalize(double[] rawValues)
    {
        var normalizedValues = new double[rawValues.Length];

        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues[i] = Math.Log(rawValues[i]);
        }

        return normalizedValues;
    }

    internal override double[][] Normalize(double[][] rawValues)
    {
        var normalizedValues = Array.Empty<double[]>();
        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues[i] = Normalize(rawValues[i]);
        }

        return normalizedValues;
    }

    internal override (double[], double[], double[], double[]) PrepareData(double[] inputs, double[] outputs, int trainingSize)
    {
        var preparedInputs = Normalize(inputs);
        var preparedOutputs = Normalize(outputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedInputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedOutputs);

        return NormalizationHelper.SplitData(preparedInputs, preparedOutputs, trainingSize);
    }

    internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) PrepareData(
        double[][] inputs, double[] outputs, int trainingSize)
    {
        var preparedInputs = Normalize(inputs);
        var preparedOutputs = Normalize(outputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedInputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedOutputs);

        return NormalizationHelper.SplitData(preparedInputs, preparedOutputs, trainingSize);
    }

    internal override (double[][] trainingInputs, double[][] trainingOutputs, double[][] oosInputs, double[][] oosOutputs) 
        PrepareData(double[][] inputs, double[][] outputs, int trainingSize)
    {
        var preparedInputs = Normalize(inputs);
        var preparedOutputs = Normalize(outputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedInputs);
        ValidationHelper.CheckForNaNOrInfinity(preparedOutputs);

        return NormalizationHelper.SplitData(preparedInputs, preparedOutputs, trainingSize);
    }
}