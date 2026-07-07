using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Concrete <see cref="ICreditLayer{T}"/> describing one trainable layer for a single training step.
/// Built by <see cref="CreditAssignmentGradientComputer{T}"/>.
/// </summary>
internal sealed class CreditLayer<T> : ICreditLayer<T>
{
    internal CreditLayer(int index, bool isOutputLayer, Tensor<T> output, Matrix<T>? weights)
    {
        Index = index;
        IsOutputLayer = isOutputLayer;
        Output = output;
        OutputShape = output.Shape.ToArray();
        Weights = weights;

        int flat = 1;
        for (int i = 1; i < output.Shape.Length; i++)
            flat *= output.Shape[i];
        FlatFeatureSize = flat;
    }

    public int Index { get; }
    public bool IsOutputLayer { get; }
    public int[] OutputShape { get; }
    public int FlatFeatureSize { get; }
    public Tensor<T> Output { get; }
    public Matrix<T>? Weights { get; }
    public Tensor<T>? TeachingSignal { get; set; }
}

/// <summary>
/// Concrete <see cref="ICreditAssignmentContext{T}"/> for a single training step.
/// </summary>
internal sealed class CreditAssignmentContext<T> : ICreditAssignmentContext<T>
{
    internal CreditAssignmentContext(
        IReadOnlyList<ICreditLayer<T>> layers,
        Tensor<T> outputError,
        INumericOperations<T> numOps,
        Random random)
    {
        Layers = layers;
        OutputError = outputError;
        NumOps = numOps;
        Random = random;
    }

    public IReadOnlyList<ICreditLayer<T>> Layers { get; }
    public Tensor<T> OutputError { get; }
    public int BatchSize => OutputError.Shape[0];
    public INumericOperations<T> NumOps { get; }
    public Random Random { get; }
}
