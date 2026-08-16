// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
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
// Rank 2 [Time, Features] - the shape ResolveShapes declares. OutputAxesFor below is hand-written
// because both relations depend on the splicing factor, a constructor argument.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class TemporalFrameSplicingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _factor;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Splices <c>_factor</c> consecutive frames into one, so time shrinks by that factor and features
    /// grow by it. Resolved output is
    /// <c>[retainedTime / _factor, features * _factor]</c> where
    /// <c>retainedTime = time - (time % _factor)</c>.
    /// </para>
    /// <para>
    /// The time relation LOOKS like it needs modulo arithmetic, and I first assumed it was outside the
    /// relation vocabulary. It is not: <c>(time - time % f) / f</c> is exactly <c>floor(time / f)</c>,
    /// which is what <c>Window(kernel: f, stride: f, padding: 0)</c> evaluates to -
    /// <c>floor((time - (f-1) - 1) / f) + 1</c>. Reading the resolved shape instead of the guard
    /// condition is what showed that; the vocabulary needed no extension.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _factor <= 0) return null;

        return new[]
        {
            new OutputAxisContract(
                TensorAxis.Time,
                AxisRelation.Window(TensorAxis.Time, kernel: _factor, stride: _factor, padding: 0)),
            new OutputAxisContract(
                TensorAxis.Features, AxisRelation.Scaled(TensorAxis.Features, _factor, 1)),
        };
    }

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void ResetState() { }
}
