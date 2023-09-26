namespace AiDotNet.Normalization;

internal class ZScoreNormalization : INormalization
{
    internal override double[] Normalize(double[] rawValues)
    {
        var normalizedValues = new double[rawValues.Length];

        var rawValuesAvg = rawValues.Average();
        var rawValuesStdDev = Math.Sqrt(rawValues.Select(x => Math.Pow(x - rawValuesAvg, 2)).Sum() / rawValues.Length);

        for (var i = 0; i < rawValues.Length; i++)
        {
            normalizedValues.SetValue(rawValuesStdDev != 0 ? (rawValues[i] - rawValuesAvg) / rawValuesStdDev : double.NaN, i);
        }

        if (normalizedValues.Contains(double.NaN))
        {
            throw new ArgumentException("Normalized values can't contain NaN values. " +
                                        "Z-Score Normalization creates NaN values when the training data has all of the same values or invalid data.", nameof(rawValues));
        }

        return normalizedValues;
    }

    internal override (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) PrepareData(double[] inputs, double[] outputs, int trainingSize)
    {
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) = NormalizationHelper.SplitData(inputs, outputs, trainingSize);

        return (Normalize(trainingInputs), Normalize(trainingOutputs), Normalize(oosInputs), Normalize(oosOutputs));
    }
}