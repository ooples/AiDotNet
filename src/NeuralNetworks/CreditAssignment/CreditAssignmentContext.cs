using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Concrete <see cref="ICreditLayer{T}"/> holding one dense layer's current weights and the forward
/// activations captured for a single training step. Built by <see cref="CreditAssignmentGradientComputer{T}"/>.
/// </summary>
internal sealed class CreditLayer<T> : ICreditLayer<T>
{
    private readonly IActivationFunction<T>? _scalarActivation;
    private readonly INumericOperations<T> _numOps;

    internal CreditLayer(
        int index,
        bool isOutputLayer,
        Matrix<T> weights,
        Matrix<T> input,
        Matrix<T> preActivation,
        Matrix<T> output,
        IActivationFunction<T>? scalarActivation,
        INumericOperations<T> numOps)
    {
        Index = index;
        IsOutputLayer = isOutputLayer;
        Weights = weights;
        Input = input;
        PreActivation = preActivation;
        Output = output;
        _scalarActivation = scalarActivation;
        _numOps = numOps;
        WeightGradient = new Matrix<T>(weights.Rows, weights.Columns);
        BiasGradient = new Vector<T>(weights.Rows);
    }

    public int Index { get; }
    public bool IsOutputLayer { get; }
    public int InputDim => Weights.Columns;
    public int OutputDim => Weights.Rows;
    public Matrix<T> Weights { get; }
    public Matrix<T> Input { get; }
    public Matrix<T> PreActivation { get; }
    public Matrix<T> Output { get; }
    public Matrix<T> WeightGradient { get; set; }
    public Vector<T> BiasGradient { get; set; }

    public Matrix<T> ActivationDerivative()
    {
        var result = new Matrix<T>(PreActivation.Rows, PreActivation.Columns);
        if (_scalarActivation is null)
        {
            // No element-wise activation (identity / linear) → derivative is 1.
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Columns; j++)
                    result[i, j] = _numOps.One;
            return result;
        }

        for (int i = 0; i < result.Rows; i++)
            for (int j = 0; j < result.Columns; j++)
                result[i, j] = _scalarActivation.Derivative(PreActivation[i, j]);
        return result;
    }
}

/// <summary>
/// Concrete <see cref="ICreditAssignmentContext{T}"/> for a single training step.
/// </summary>
internal sealed class CreditAssignmentContext<T> : ICreditAssignmentContext<T>
{
    internal CreditAssignmentContext(
        IReadOnlyList<ICreditLayer<T>> layers,
        Matrix<T> input,
        Matrix<T> prediction,
        Matrix<T> target,
        Matrix<T> outputError,
        INumericOperations<T> numOps,
        Random random)
    {
        Layers = layers;
        Input = input;
        Prediction = prediction;
        Target = target;
        OutputError = outputError;
        NumOps = numOps;
        Random = random;
    }

    public IReadOnlyList<ICreditLayer<T>> Layers { get; }
    public Matrix<T> Input { get; }
    public Matrix<T> Prediction { get; }
    public Matrix<T> Target { get; }
    public Matrix<T> OutputError { get; }
    public int BatchSize => Input.Rows;
    public INumericOperations<T> NumOps { get; }
    public Random Random { get; }
}
