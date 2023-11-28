namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Removes outliers from the data using the threshold method. This method is not recommended for data sets with less than 15 data points.
/// </summary>
public class ThresholdOutlierRemoval : IOutlierRemoval
{
    internal override (double[], double[]) RemoveOutliers(double[] rawInputs, double[] rawOutputs)
    {
        var sortedInputs = new List<double>(rawInputs);
        var sortedOutputs = rawOutputs.ToArray();
        var median = sortedInputs[sortedInputs.Count / 2];
        var deviations = new List<double>(sortedInputs.Select(v => Math.Abs(v - median)).OrderBy(x => x));
        var medianDeviation = deviations[deviations.Count / 2];
        var threshold = 3 * medianDeviation;

        var ignoredIndices = new List<int>();
        for (var i = 0; i < rawInputs.Length; i++)
        {
            if (Math.Abs(rawInputs[i] - median) > threshold)
            {
                ignoredIndices.Add(i);
            }
        }

        return QuartileHelper.FilterArraysWithIndices(sortedInputs, sortedOutputs, ignoredIndices);
    }

    internal override (double[][], double[]) RemoveOutliers(double[][] rawInputs, double[] rawOutputs)
    {
        var length = rawInputs[0].Length;

        var finalInputs = Array.Empty<double[]>();
        var finalOutputs = Array.Empty<double>();
        for (var i = 0; i < length; i++)
        {
            var (cleanedInputs, cleanedOutputs) = RemoveOutliers(rawInputs[i], rawOutputs);
            finalInputs[i] = cleanedInputs;
            finalOutputs = cleanedOutputs;
        }

        return (finalInputs, finalOutputs);
    }

    internal override (double[][], double[][]) RemoveOutliers(double[][] rawInputs, double[][] rawOutputs)
    {
        var finalInputs = Array.Empty<double[]>();
        var finalOutputs = Array.Empty<double[]>();
        for (var i = 0; i < rawInputs.Length; i++)
        {
            var (cleanedInputs, cleanedOutputs) = RemoveOutliers(rawInputs[i], rawOutputs[i]);
            finalInputs[i] = cleanedInputs;
            finalOutputs[i] = cleanedOutputs;
        }

        return (finalInputs, finalOutputs);
    }
}