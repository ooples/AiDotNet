namespace AiDotNet.Interfaces;

public abstract class IRegression
{
    internal abstract (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) 
        PrepareData(double[] inputs, double[] outputs, int trainingSize, INormalization normalization);

    internal abstract void Fit(double[] inputs, double[] outputs);

    internal abstract double[] Transform(double[] inputs);
}