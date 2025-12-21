using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Helpers;

/// <summary>
/// Mock loss function for testing meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MockLossFunction<T> : ILossFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the name of the loss function.
    /// </summary>
    public string Name => "MockLossFunction";

    /// <summary>
    /// Calculates the loss between predicted and actual values.
    /// </summary>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        // Simple MSE-like loss for testing
        T sum = NumOps.Zero;
        int length = Math.Min(predicted.Length, actual.Length);

        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        if (length > 0)
        {
            return NumOps.Divide(sum, NumOps.FromDouble(length));
        }

        return NumOps.Zero;
    }

    /// <summary>
    /// Calculates the derivative of the loss function.
    /// </summary>
    public Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        int length = Math.Min(predicted.Length, actual.Length);
        var derivative = new Vector<T>(length);

        if (length == 0)
        {
            return derivative;
        }

        T twoOverN = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.FromDouble(length));

        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            derivative[i] = NumOps.Multiply(twoOverN, diff);
        }

        return derivative;
    }
}
