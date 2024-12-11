namespace AiDotNet.Extensions;

public static class VectorExtensions
{
    public static double Variance(this Vector<double> vector)
    {
        double mean = vector.Mean();
        return vector.Select(x => Math.Pow(x - mean, 2)).Mean();
    }
}