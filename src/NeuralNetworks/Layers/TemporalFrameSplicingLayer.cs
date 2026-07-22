using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Concatenates adjacent temporal frames along the feature axis.
/// </summary>
/// <remarks>
/// For an input shaped <c>[batch, time, features]</c> and a factor <c>S</c>,
/// the output is <c>[batch, floor(time / S), features * S]</c>. As in the
/// released FireRedASR adapter, an incomplete trailing group is discarded.
/// Slicing and reshaping remain connected to the gradient tape through the
/// tensor engine.
/// </remarks>
internal sealed class TemporalFrameSplicingLayer<T> : LayerBase<T>
{
    private readonly int _factor;

    public TemporalFrameSplicingLayer()
        : this(2)
    {
    }

    public TemporalFrameSplicingLayer(int factor)
        : base(new[] { -1, -1 }, new[] { -1, -1 })
    {
        if (factor <= 0)
            throw new ArgumentOutOfRangeException(nameof(factor));
        _factor = factor;
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    public override long ParameterCount => 0;

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        if (input.Rank < 2)
        {
            throw new ArgumentException(
                $"TemporalFrameSplicingLayer expects at least [time, features], got rank {input.Rank}.",
                nameof(input));
        }

        int time = input.Shape[^2];
        int features = input.Shape[^1];
        int retainedTime = time - (time % _factor);
        if (retainedTime == 0)
        {
            throw new ArgumentException(
                $"The time dimension ({time}) must contain at least one complete group of {_factor} frames.",
                nameof(input));
        }

        ResolveShapes(
            new[] { time, features },
            new[] { retainedTime / _factor, checked(features * _factor) });
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        int time = input.Shape[^2];
        int retainedTime = time - (time % _factor);
        if (retainedTime == 0)
        {
            throw new ArgumentException(
                $"The time dimension ({time}) must contain at least one complete group of {_factor} frames.",
                nameof(input));
        }

        Tensor<T> groupedInput = input;
        if (retainedTime != time)
        {
            int[] start = new int[input.Rank];
            int[] length = input.Shape.ToArray();
            length[^2] = retainedTime;
            groupedInput = Engine.TensorSlice(input, start, length);
        }

        int[] outputShape = groupedInput.Shape.ToArray();
        outputShape[^2] /= _factor;
        outputShape[^1] = checked(outputShape[^1] * _factor);
        return Engine.Reshape(groupedInput, outputShape);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
            throw new ArgumentException($"TemporalFrameSplicingLayer has no parameters; got {parameters.Length}.", nameof(parameters));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void ResetState() { }
}
