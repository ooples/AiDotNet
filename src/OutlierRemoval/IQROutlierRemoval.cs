namespace AiDotNet.OutlierRemoval;

public class IQROutlierRemoval : IOutlierRemoval
{
    public IQROutlierRemoval(IQuartile? quartile = null)
    {
        Quartile = quartile;
    }

    internal int[] DetermineIQR(double[] unfiltered)
    {
        //initializes new type StandardQuartile of class IQuartile only if type is null else leaves the same.
        var (q1Value, _, q3Value) = QuartileHelper.FindQuartiles(unfiltered, Quartile ?? new StandardQuartile());
        var iQR = q3Value - q1Value;
        var factor = 1.5 * iQR;
        var minLimit = q1Value - factor;
        var maxLimit = q3Value + factor;

        return QuartileHelper.FindIndicesToRemove(unfiltered, minLimit, maxLimit);
    }

    internal override (double[], double[]) RemoveOutliers(double[] rawInputs, double[] rawOutputs)
    {
        //Create Deep Copy
        var sortedInputs = rawInputs.ToArray();
        var sortedOutputs = rawOutputs.ToArray();
        //Sort Both Arrays according to Input's ascending order
        Array.Sort(sortedInputs, sortedOutputs);
        var ignoredIndices = DetermineIQR(sortedInputs);

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