namespace AiDotNet;

public abstract class IRegression
{
    internal abstract (double yIntercept, double slope) Fit(double[] inputs, double[] outputs);

    internal abstract double[] Transform(double[] inputs);
}