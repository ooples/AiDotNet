using AiDotNet.Interfaces;
using System;
using AiDotNet.Tensors.Engines;
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
    /// GPU loss and gradient calculation - not supported in mock.
    /// </summary>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        // Both the value and the gradient come from one differentiated forward.
        return this.ComputeLossAndGradient(predicted, actual);
    }

    /// <summary>
    /// Computes the loss as a tape-differentiable scalar tensor.
    /// </summary>
    /// <param name="predicted">The predicted tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <returns>A rank-0 scalar tensor holding the mean squared error.</returns>
    /// <remarks>
    /// Built from engine operations rather than an element-wise loop so the gradient tape can
    /// differentiate it. The mock supplies no derivative of its own, exactly like a real loss.
    /// </remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(predicted, target);
        var squared = engine.TensorMultiply(diff, diff);

        var axes = new int[squared.Shape.Length];
        for (int i = 0; i < axes.Length; i++) axes[i] = i;

        var summed = engine.ReduceSum(squared, axes, keepDims: false);
        return engine.TensorDivideScalar(summed, NumOps.FromDouble(Math.Max(1, squared.Length)));
    }
}
