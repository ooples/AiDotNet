namespace AiDotNet;

public abstract class IRegression
{
    internal abstract void Fit(double[] inputs, double[] outputs);

    internal abstract double[] Transform(double[] inputs);
}