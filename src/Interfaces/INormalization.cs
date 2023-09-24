namespace AiDotNet.Interfaces;

public abstract class INormalization
{
    internal abstract double[] Normalize(double[] rawValues);

    internal abstract (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) 
        PrepareData(double[] inputs, double[] outputs, int trainingSize);
}