using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Simple mock loss function for testing.
/// </summary>
public class SimpleMockLossFunction : ILossFunction<double>
{
    public double CalculateLoss(Vector<double> predicted, Vector<double> actual)
    {
        // Simple mean squared error
        double sum = 0;
        for (int i = 0; i < predicted.Length && i < actual.Length; i++)
        {
            double diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / predicted.Length;
    }

    public Vector<double> CalculateDerivative(Vector<double> predicted, Vector<double> actual)
    {
        var gradient = new Vector<double>(predicted.Length);
        for (int i = 0; i < predicted.Length && i < actual.Length; i++)
        {
            gradient[i] = 2.0 * (predicted[i] - actual[i]) / predicted.Length;
        }
        return gradient;
    }
}
