namespace AiDotNet.Interfaces;

public abstract class IOutlierRemoval
{
    internal abstract (double[] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[] rawInputs, double[] rawOutputs);

    internal abstract (double[][] cleanedInputs, double[] cleanedOutputs) RemoveOutliers(double[][] rawInputs, double[] rawOutputs);

    internal IQuartile? Quartile { get; set; }
}