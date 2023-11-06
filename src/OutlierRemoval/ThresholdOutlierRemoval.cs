namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Removes outliers from the data using the threshold method. This method is not recommended for data sets with less than 15 data points.
/// </summary>
public class ThresholdOutlierRemoval : IOutlierRemoval
{
    internal override (double[] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[] rawInputs, double[] rawOutputs)
    {
        var sortedValues = new List<double>(rawInputs.OrderBy(v => v));
        var median = sortedValues[sortedValues.Count / 2];
        var deviations = new List<double>(sortedValues.Select(v => Math.Abs(v - median)));
        var medianDeviation = deviations[deviations.Count / 2];
        var threshold = 3 * medianDeviation;

        var cleanedInputs = new List<double>();
        var cleanedOutputs = new List<double>();
        for (var i = 0; i < rawInputs.Length; i++)
        {
            if (Math.Abs(rawInputs[i] - median) > threshold) continue;

            cleanedInputs.Add(rawInputs[i]);
            cleanedOutputs.Add(rawOutputs[i]);
        }

        return (cleanedInputs.ToArray(), cleanedOutputs.ToArray());
    }

    internal override (double[][] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[][] rawInputs, double[] rawOutputs)
    {
        throw new NotImplementedException();
    }
}