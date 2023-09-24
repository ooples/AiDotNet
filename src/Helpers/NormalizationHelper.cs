namespace AiDotNet.Helpers;

internal static class NormalizationHelper
{
    public static (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) SplitData(double[] inputs, double[] outputs,
        int trainingSize)
    {
        return (inputs.Take(trainingSize).ToArray(), outputs.Take(trainingSize).ToArray(),
            inputs.Skip(trainingSize).ToArray(), outputs.Skip(trainingSize).ToArray());
    }
}