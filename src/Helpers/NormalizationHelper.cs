namespace AiDotNet.Helpers;

internal static class NormalizationHelper
{
    public static (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) SplitData(double[] inputs, double[] outputs,
        int trainingSize)
    {
        return (inputs.Take(trainingSize).ToArray(), outputs.Take(trainingSize).ToArray(),
            inputs.Skip(trainingSize).ToArray(), outputs.Skip(trainingSize).ToArray());
    }

    public static (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) SplitData(double[][] inputs, double[] outputs,
        int trainingSize)
    {
        var oosInputs = new List<double[]>();
        var trainingInputs = new List<double[]>();
        for (var i = 0; i < inputs.Length; i++)
        {
            trainingInputs.Add(inputs[i].Take(trainingSize).ToArray());
            oosInputs.Add(inputs[i].Skip(trainingSize).ToArray());
        }

        return (trainingInputs.ToArray(), outputs.Take(trainingSize).ToArray(),
            oosInputs.ToArray(), outputs.Skip(trainingSize).ToArray());
    }
}