namespace AiDotNet.Interfaces;

public abstract class IOutlierRemoval
{
    internal abstract (double[], double[]) RemoveOutliers(double[] rawInputs, double[] rawOutputs);

    internal abstract (double[][], double[]) RemoveOutliers(double[][] rawInputs, double[] rawOutputs);

    internal abstract (double[][], double[][]) RemoveOutliers(double[][] rawInputs, double[][] rawOutputs);

    internal IQuartile? Quartile { get; set; }
}