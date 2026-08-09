using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Gated fusion of two equally-wide feature streams, as used by ABINet's fusion stage.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements ABINet's gated combination (Fang et al., CVPR 2021, arXiv:2103.06495 §3.4):
/// </para>
/// <code>
/// G   = sigmoid([F_v, F_l] W_f)
/// F_f = G * F_v + (1 - G) * F_l
/// </code>
/// <para>
/// The gate is learned from BOTH streams jointly, then used to interpolate between them
/// per position and per channel — so the model can lean on the visual reading where the image
/// is clear and on the language model's correction where it is not.
/// </para>
/// <para>
/// Input is the two streams concatenated along the last axis, <c>[..., 2 * width]</c>, which
/// keeps the layer inside the single-input <see cref="ILayer{T}"/> contract. Output is
/// <c>[..., width]</c>.
/// </para>
/// <para><b>For Beginners:</b> Two components each produce a guess. Rather than averaging them
/// or always trusting one, this learns a per-value dial: where the dial is near 1 it takes the
/// first component's answer, near 0 the second's, and in between it blends them.</para>
/// </remarks>
public partial class GatedFusionLayer<T> : LayerBase<T>
{
    /// <summary>Width of each input stream (half the concatenated input width).</summary>
    private readonly int _width;

    /// <summary>Produces the gate from the concatenated pair.</summary>
    private readonly DenseLayer<T> _gate;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount => _gate.ParameterCount;

    /// <summary>
    /// Creates a gated fusion layer.
    /// </summary>
    /// <param name="width">
    /// Width of each stream. The layer consumes <c>2 * width</c> channels and emits
    /// <c>width</c>.
    /// </param>
    public GatedFusionLayer(
        [LayerState] int width)
        : base(new[] { -1, -1, 2 * width }, new[] { -1, -1, width })
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));

        _width = width;

        // Sigmoid gate over the concatenated pair, matching sigmoid([F_v, F_l] W_f).
        _gate = new DenseLayer<T>(width, (IActivationFunction<T>)new SigmoidActivation<T>());
        RegisterSubLayer(_gate);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        int lastAxis = input.Shape.Length - 1;
        if (input.Shape[lastAxis] != 2 * _width)
            throw new ArgumentException(
                $"GatedFusionLayer expects the two streams concatenated on the last axis, so {2 * _width} channels, but got {input.Shape[lastAxis]}.",
                nameof(input));

        var gate = _gate.Forward(input);

        var visionStream = Engine.TensorNarrow(input, dim: lastAxis, start: 0, length: _width);
        var languageStream = Engine.TensorNarrow(input, dim: lastAxis, start: _width, length: _width);

        // (1 - G), built as a tensor so the whole expression stays on the tape.
        var ones = new Tensor<T>(gate.Shape.ToArray());
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        var inverseGate = Engine.TensorSubtract(ones, gate);

        return Engine.TensorAdd(
            Engine.TensorMultiply(gate, visionStream),
            Engine.TensorMultiply(inverseGate, languageStream));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The base implementation does not recurse into registered sub-layers, so the gate's
    /// tensors are surfaced explicitly.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() => _gate.GetTrainableParameters();

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
        => _gate.SetTrainableParameters(parameters);

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _gate.GetParameters();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters) => _gate.SetParameters(parameters);

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters) => _gate.UpdateParameters(parameters);

    /// <inheritdoc/>
    public override void ResetState() => _gate.ResetState();

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) => _gate.UpdateParameters(learningRate);

    /// <inheritdoc/>
    /// <remarks>
    /// Publishes the stream width so deserialization can reconstruct the layer at the right
    /// size; without it a restored layer defaults to a different width and the parameter
    /// vector no longer lines up.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Width"] = _width.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
