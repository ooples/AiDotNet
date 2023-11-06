namespace AiDotNet.Normalization;

/// <summary>
/// Normalizes the data by
/// 1) Subtracting the minimum value from each value
/// 2) Dividing each value from step #1 by the absolute difference between the maximum and minimum values
/// </summary>
public class MinMaxNormalization : INormalization
{
    private double InputsMin { get; set; }
    private double InputsMax { get; set; }

    internal override double[] Normalize(double[] rawValues)
    {
        var normalizedValues = new double[rawValues.Length];
        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues[i] = Math.Abs(InputsMax - InputsMin) > 0
                ? (rawValues[i] - InputsMin) / Math.Abs(InputsMax - InputsMin)
                : double.NaN;
        }

        if (normalizedValues.Contains(double.NaN))
        {
            throw new ArgumentException("Normalized values can't contain NaN values. " +
                                        "MinMax Normalization creates NaN values when the training data has all of the same values or invalid data.", nameof(rawValues));
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

    internal override (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) PrepareData(
        double[] inputs, double[] outputs, int trainingSize)
    {
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) = NormalizationHelper.SplitData(inputs, outputs, trainingSize);
        InputsMin = trainingInputs.Min();
        InputsMax = trainingInputs.Max();

        return (Normalize(trainingInputs), Normalize(trainingOutputs), Normalize(oosInputs), Normalize(oosOutputs));
    }

    internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) 
        PrepareData(double[][] inputs, double[] outputs, int trainingSize)
    {
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) = NormalizationHelper.SplitData(inputs, outputs, trainingSize);
        InputsMin = trainingInputs.Min();
        InputsMax = trainingInputs.Max();

        return (Normalize(trainingInputs), Normalize(trainingOutputs), Normalize(oosInputs), Normalize(oosOutputs));
    }

    internal override (double[][] trainingInputs, double[][] trainingOutputs, double[][] oosInputs, double[][] oosOutputs) 
        PrepareData(double[][] inputs, double[][] outputs, int trainingSize)
    {
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) = NormalizationHelper.SplitData(inputs, outputs, trainingSize);
        InputsMin = trainingInputs.Min();
        InputsMax = trainingInputs.Max();

        return (Normalize(trainingInputs), Normalize(trainingOutputs), Normalize(oosInputs), Normalize(oosOutputs));
    }
}