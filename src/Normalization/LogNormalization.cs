namespace AiDotNet.Normalization;

public class LogNormalization : INormalization
{
    internal override double[] Normalize(double[] rawValues)
    {
        var normalizedValues = new double[rawValues.Length];

        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues.SetValue(Math.Log(rawValues[i]), i);
        }

        return normalizedValues;
    }

    internal override (double[], double[], double[], double[]) PrepareData(double[] inputs, double[] outputs, int trainingSize)
    {
        var preparedInputs = Normalize(inputs);
        var preparedOutputs = Normalize(outputs);

        if (preparedInputs.Contains(double.NaN))
        {
            throw new ArgumentException("Normalized Inputs can't contain NaN values. " +
                                        "Log Normalization creates NaN values when a raw input value is negative.", nameof(inputs));
        }

        if (preparedOutputs.Contains(double.NaN))
        {
            throw new ArgumentException("Normalized Outputs can't contain NaN values. " +
                                        "Log Normalization creates NaN values when a raw output value is negative.", nameof(outputs));
        }

        if (preparedInputs.Contains(double.PositiveInfinity) || preparedInputs.Contains(double.NegativeInfinity))
        {
            throw new ArgumentException("Normalized Inputs can't contain Infinity values. " +
                                        "Log Normalization creates Infinity values when a raw input value is 0 or infinity.", nameof(inputs));
        }

        if (preparedOutputs.Contains(double.PositiveInfinity) || preparedOutputs.Contains(double.NegativeInfinity))
        {
            throw new ArgumentException("Normalized Outputs can't contain Infinity values. " +
                                        "Log Normalization creates Infinity values when a raw output value is 0 or infinity.", nameof(outputs));
        }

        return NormalizationHelper.SplitData(preparedInputs, preparedOutputs, trainingSize);
    }
}