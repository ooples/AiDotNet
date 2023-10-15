namespace AiDotNet.Helpers;

internal static class NormalizationHelper
{
    internal static (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) SplitData(double[] inputs, double[] outputs,
        int trainingSize)
    {
        return (inputs.Take(trainingSize).ToArray(), outputs.Take(trainingSize).ToArray(),
            inputs.Skip(trainingSize).ToArray(), outputs.Skip(trainingSize).ToArray());
    }

    internal static (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) SplitData(double[][] inputs, double[] outputs,
        int trainingSize)
    {
        var oosInputs = new List<double[]>();
        var trainingInputs = new List<double[]>();
        for (var i = 0; i < inputs.Length; i++)
        {
            trainingInputs.Add(inputs[i].Take(trainingSize).ToArray());
            oosInputs.Add(inputs[i].Skip(trainingSize).ToArray());
        }

        return (trainingInputs.ToArray(), outputs.Take(trainingSize).ToArray(), oosInputs.ToArray(), outputs.Skip(trainingSize).ToArray());
    }

    internal static (double[][] trainingInputs, double[][] trainingOutputs, double[][] oosInputs, double[][] oosOutputs) SplitData(double[][] inputs, double[][] outputs,
        int trainingSize)
    {
        var oosInputs = new List<double[]>();
        var trainingInputs = new List<double[]>();
        for (var i = 0; i < inputs.Length; i++)
        {
            trainingInputs.Add(inputs[i].Take(trainingSize).ToArray());
            oosInputs.Add(inputs[i].Skip(trainingSize).ToArray());
        }

        var oosOutputs = new List<double[]>();
        var trainingOutputs = new List<double[]>();
        for (var i = 0; i < outputs.Length; i++)
        {
            trainingOutputs.Add(outputs[i].Take(trainingSize).ToArray());
            oosOutputs.Add(outputs[i].Skip(trainingSize).ToArray());
        }

        return (trainingInputs.ToArray(), trainingOutputs.ToArray(), oosInputs.ToArray(), oosOutputs.ToArray());
    }
}