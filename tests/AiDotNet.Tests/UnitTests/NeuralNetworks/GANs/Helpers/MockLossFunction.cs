using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Interfaces;

namespace AiDotNetTests.UnitTests.NeuralNetworks.GANs.Helpers;

/// <summary>
/// Mock loss function for unit testing GAN classes.
/// Tracks calls and provides controllable return values for deterministic testing.
/// </summary>
public class MockLossFunction<T> : ILossFunction<T>
{
    private readonly Func<Vector<T>, Vector<T>, T> _lossFunc;
    private readonly Func<Vector<T>, Vector<T>, Vector<T>> _derivativeFunc;

    public int CalculateLossCallCount { get; private set; }
    public int CalculateDerivativeCallCount { get; private set; }
    public Vector<T>? LastPredicted { get; private set; }
    public Vector<T>? LastActual { get; private set; }

    public MockLossFunction(T defaultLoss, Vector<T>? defaultDerivative = null)
    {
        _lossFunc = (_, _) => defaultLoss;
        _derivativeFunc = (predicted, _) => defaultDerivative ?? new Vector<T>(predicted.Length);
    }

    public MockLossFunction(
        Func<Vector<T>, Vector<T>, T> lossFunc,
        Func<Vector<T>, Vector<T>, Vector<T>> derivativeFunc)
    {
        _lossFunc = lossFunc ?? throw new ArgumentNullException(nameof(lossFunc));
        _derivativeFunc = derivativeFunc ?? throw new ArgumentNullException(nameof(derivativeFunc));
    }

    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        CalculateLossCallCount++;
        LastPredicted = predicted;
        LastActual = actual;
        return _lossFunc(predicted, actual);
    }


    public void Reset()
    {
        CalculateLossCallCount = 0;
        CalculateDerivativeCallCount = 0;
        LastPredicted = null;
        LastActual = null;
    }

    /// <summary>
    /// GPU loss and gradient calculation - not supported in mock.
    /// </summary>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        throw new NotSupportedException("GPU operations are not supported in MockLossFunction.");
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
        return engine.TensorDivideScalar(summed, MathHelper.GetNumericOperations<T>().FromDouble(Math.Max(1, squared.Length)));
    }
}
