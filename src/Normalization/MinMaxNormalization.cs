namespace AiDotNet.Normalization;

internal class MinMaxNormalization : INormalization
{
    private double InputsMin { get; set; }
    private double InputsMax { get; set; }

    internal override double[] Normalize(double[] rawValues)
    {
        var normalizedValues = new double[rawValues.Length];
        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues.SetValue(Math.Abs(InputsMax - InputsMin) > 0 ? (rawValues[i] - InputsMin) / (InputsMax - InputsMin) : double.NaN, i);
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