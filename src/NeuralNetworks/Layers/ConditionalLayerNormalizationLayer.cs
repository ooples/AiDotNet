using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Layer normalization whose scale and bias are computed from a conditioning vector instead of being free parameters:
/// <c>CLN(x, e) = (e W_γ) ⊙ (x − μ) / σ + e W_β</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AdaSpeech (Chen et al. 2021, §2.2, Eq. 1) replaces every layer normalization in the mel decoder with this layer,
/// conditioned on the speaker embedding <c>E_s</c>: "the conditional network consists of two simple linear layers
/// W_γ and W_β that take speaker embedding E_s as input and output the scale and bias vector respectively",
/// <c>γ = E_s W_γ</c>, <c>β = E_s W_β</c>. Adapting a voice then fine-tunes only these two matrices (and the speaker
/// embedding), and a deployed voice needs only the resulting <c>γ</c> and <c>β</c>.
/// </para>
/// <para>
/// Inputs: <c>x</c> is <c>[time, hidden]</c> with a condition <c>[condition]</c> (or <c>[1, condition]</c>), or
/// <c>[batch, time, hidden]</c> with a condition <c>[batch, condition]</c>. Normalization is over the last axis.
/// </para>
/// <para><b>For Beginners:</b> Ordinary layer normalization rescales each feature vector with a learned scale and
/// shift. Here the scale and shift depend on who is speaking, so changing the speaker re-tunes every normalization
/// in the decoder without touching any other weight.</para>
/// </remarks>
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(IsTrainable = true, HasTrainingMode = false, ApiShape = LayerApiShape.DualTensor,
    TestInputShape = "1, 4", TestConstructorArgs = "4, 4")]
[ElementWiseShape(Note = "Normalises over the feature axis of the first input; its shape is carried through.")]
[AutoParameters]
public partial class ConditionalLayerNormalizationLayer<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _conditionSize;
    private readonly double _epsilon;

    [SubLayerInput("_conditionSize")]
    private readonly BiasFreeLinearLayer<T> _scale;
    [SubLayerInput("_conditionSize")]
    private readonly BiasFreeLinearLayer<T> _bias;

    private readonly Tensor<T> _unitScale;
    private readonly Tensor<T> _zeroShift;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool RequiresMultipleInputs => true;

    /// <summary>Creates a conditional layer normalization.</summary>
    /// <param name="hiddenSize">Width of the normalized features.</param>
    /// <param name="conditionSize">Width of the conditioning vector (the speaker embedding in AdaSpeech).</param>
    /// <param name="epsilon">Added to the variance before the square root.</param>
    public ConditionalLayerNormalizationLayer(
        [LayerState] int hiddenSize,
        [LayerState] int conditionSize,
        [LayerState] double epsilon = 1e-5)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (conditionSize <= 0) throw new ArgumentOutOfRangeException(nameof(conditionSize));
        if (!(epsilon > 0)) throw new ArgumentOutOfRangeException(nameof(epsilon));
        _hiddenSize = hiddenSize;
        _conditionSize = conditionSize;
        _epsilon = epsilon;
        _scale = new BiasFreeLinearLayer<T>(conditionSize, hiddenSize);
        _bias = new BiasFreeLinearLayer<T>(conditionSize, hiddenSize);
        _unitScale = Tensor<T>.CreateDefault(new[] { hiddenSize }, NumOps.One);
        _zeroShift = Tensor<T>.CreateDefault(new[] { hiddenSize }, NumOps.Zero);
        RegisterSubLayer(_scale);
        RegisterSubLayer(_bias);
    }

    /// <summary>Width of the normalized features.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Width of the conditioning vector.</summary>
    public int ConditionSize => _conditionSize;

    /// <summary>The scale projection W_γ.</summary>
    public BiasFreeLinearLayer<T> ScaleProjection => _scale;

    /// <summary>The bias projection W_β.</summary>
    public BiasFreeLinearLayer<T> BiasProjection => _bias;

    /// <summary>Normalizes <paramref name="input"/> with the scale and bias computed from <paramref name="condition"/>.</summary>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> condition) => Forward(new[] { input, condition });

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
        => throw new InvalidOperationException(
            $"{nameof(ConditionalLayerNormalizationLayer<T>)} needs a conditioning vector; call Forward(input, condition).");

    /// <inheritdoc />
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length != 2)
            throw new ArgumentException("Expected two inputs: the features and the conditioning vector.", nameof(inputs));
        var input = inputs[0];
        var condition = inputs[1];
        if (input.Rank < 2 || input.Shape[input.Rank - 1] != _hiddenSize)
            throw new ArgumentException(
                $"Expected features [time, {_hiddenSize}] or [batch, time, {_hiddenSize}], got [{string.Join(", ", input.Shape)}].",
                nameof(inputs));
        if (condition.Shape[condition.Rank - 1] != _conditionSize)
            throw new ArgumentException(
                $"Expected a condition of width {_conditionSize}, got [{string.Join(", ", condition.Shape)}].", nameof(inputs));

        bool unbatched = input.Rank == 2;
        int batch = unbatched ? 1 : input.Shape[0];
        int time = input.Shape[input.Rank - 2];
        if (condition.Length != batch * _conditionSize)
            throw new ArgumentException(
                $"Expected one condition per batch row ({batch}), got [{string.Join(", ", condition.Shape)}].", nameof(inputs));

        var x = unbatched ? Engine.Reshape(input, new[] { 1, time, _hiddenSize }) : input;
        var normalized = Engine.LayerNorm(x, _unitScale, _zeroShift, _epsilon, out _, out _);

        var rows = Engine.Reshape(condition, new[] { batch, _conditionSize });
        var gamma = Engine.TensorTile(Engine.Reshape(_scale.Forward(rows), new[] { batch, 1, _hiddenSize }), new[] { 1, time, 1 });
        var beta = Engine.TensorTile(Engine.Reshape(_bias.Forward(rows), new[] { batch, 1, _hiddenSize }), new[] { 1, time, 1 });
        var output = Engine.TensorAdd(Engine.TensorMultiply(normalized, gamma), beta);
        return unbatched ? Engine.Reshape(output, input._shape) : output;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments so deserialization can rebuild the projections before loading weights.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(inv);
        metadata["ConditionSize"] = _conditionSize.ToString(inv);
        metadata["Epsilon"] = _epsilon.ToString("R", inv);
        return metadata;
    }
}
