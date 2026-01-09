using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Gpu;

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

    public Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        CalculateDerivativeCallCount++;
        LastPredicted = predicted;
        LastActual = actual;
        return _derivativeFunc(predicted, actual);
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
    public (T Loss, IGpuTensor<T> Gradient) CalculateLossAndGradientGpu(IGpuTensor<T> predicted, IGpuTensor<T> actual)
    {
        throw new NotSupportedException("GPU operations are not supported in MockLossFunction.");
    }
}
