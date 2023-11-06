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

    internal override (double[] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[] rawInputs, double[] rawOutputs)
    {
        //Create Deep Copy
        var sortedInputs = rawInputs.ToArray();
        var sortedOutputs = rawOutputs.ToArray();
        //Sort Both Arrays according to Input's ascending order
        Array.Sort(sortedInputs, sortedOutputs);

        var ignoredIndices = DetermineIQR(sortedInputs);
        var cleanedInputs  = QuartileHelper.FilterArrayWithIndices(sortedInputs, ignoredIndices);
        var cleanedOutputs = QuartileHelper.FilterArrayWithIndices(sortedOutputs, ignoredIndices);

        return (cleanedInputs, cleanedOutputs);
    }

    internal override (double[][] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[][] rawInputs, double[] rawOutputs)
    {
        throw new NotImplementedException();
    }
}
