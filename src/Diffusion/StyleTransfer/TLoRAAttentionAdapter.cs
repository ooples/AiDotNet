using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// Wraps a UNet attention block and adds T-LoRA's timestep-masked low-rank update to its output.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This is the injection site for Soboleva et al., "T-LoRA: Single Image Diffusion Model
/// Customization Without Overfitting" (arXiv:2507.05964). The paper adapts the self- and
/// cross-attention projections of the UNet; this decorator adapts the block's OUTPUT projection
/// position, which is the standard <c>to_out</c> LoRA target, and leaves the wrapped block's own
/// weights untouched and frozen.
/// </para>
/// <para>
/// <b>Why a decorator rather than editing the attention layers.</b>
/// <see cref="AiDotNet.Diffusion.Attention.DiffusionAttention{T}"/> holds no q/k/v matrices of its
/// own — it delegates to <c>FlashAttentionLayer</c> or <c>MultiHeadAttentionLayer</c>, which every
/// other diffusion model in the library also uses. Reaching into those to add per-timestep masking
/// would put T-LoRA's mechanism on the hot path of models that do not want it. Wrapping keeps the
/// change local to the one model that asked for it, and keeps the base weights frozen by
/// construction rather than by convention: this layer reports only the adapter's parameters, so an
/// optimizer walking it cannot move W.
/// </para>
/// <para>
/// <b>The timestep.</b> <see cref="ILayer{T}.Forward"/> carries no timestep, and the rank mask is a
/// function of it, so the value is supplied out-of-band via <see cref="CurrentTimestep"/> — set once
/// per denoising step by the owning model before it walks the network. That is deliberate rather than
/// convenient: threading a timestep through every layer signature would change a library-wide
/// interface for the benefit of one model.
/// </para>
/// <para><b>For Beginners:</b> This sits on top of an existing attention block and adds a small,
/// learnable correction to what it produces. How much freedom that correction has depends on how
/// noisy the current generation step is — that is the whole idea of T-LoRA.</para>
/// </remarks>
public sealed class TLoRAAttentionAdapter<T> : LayerBase<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly ILayer<T> _inner;
    private readonly TimestepDependentLora<T> _adapter;
    private readonly int _channels;

    /// <summary>
    /// Gets the wrapped attention block, whose own weights this adapter never modifies.
    /// </summary>
    public ILayer<T> Inner => _inner;

    /// <summary>
    /// Gets the timestep-dependent low-rank adapter applied to the wrapped block's output.
    /// </summary>
    public TimestepDependentLora<T> Adapter => _adapter;

    /// <summary>
    /// Gets or sets the diffusion timestep the rank mask is evaluated at.
    /// </summary>
    /// <remarks>
    /// Set by the owning model once per denoising step. Defaults to zero, which is the FULL-rank end
    /// of the schedule — the permissive choice, so a caller that forgets to set it gets ordinary
    /// LoRA behaviour rather than a silently over-constrained adapter.
    /// </remarks>
    public int CurrentTimestep { get; set; }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Wraps <paramref name="inner"/> with a T-LoRA adapter over <paramref name="channels"/> width.
    /// </summary>
    /// <param name="inner">The attention block to wrap. Its weights stay frozen.</param>
    /// <param name="channels">The block's channel width; the adapter is square over this.</param>
    /// <param name="rank">
    /// Full adapter rank R. Clamped down to <paramref name="channels"/> when a wider rank is asked
    /// for, because a rank above the ambient width cannot have independent directions — the paper's
    /// r = 64 is larger than the narrow channel counts used by reduced test fixtures.
    /// </param>
    /// <param name="totalTimesteps">The diffusion horizon T, the schedule's denominator.</param>
    /// <param name="random">RNG for the orthogonal initialization.</param>
    public TLoRAAttentionAdapter(
        ILayer<T> inner, int channels, int rank, int totalTimesteps, Random random)
        : base(inner?.GetInputShape() ?? throw new ArgumentNullException(nameof(inner)),
               inner.GetOutputShape())
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel width must be positive.");

        _inner = inner;
        _channels = channels;
        _adapter = new TimestepDependentLora<T>(
            rank: Math.Max(1, Math.Min(rank, channels)),
            inputDim: channels, outputDim: channels,
            totalTimesteps: totalTimesteps, random: random);
    }

    /// <summary>
    /// Runs the wrapped block, then adds the adapter's timestep-masked update to every token.
    /// </summary>
    /// <remarks>
    /// The update is a residual: at initialization the adapter contributes exactly zero (see
    /// <see cref="TimestepDependentLora{T}"/>), so wrapping a block does not change what the network
    /// computes until training moves A, B or S.
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var output = _inner.Forward(input);

        // The adapter acts on the CHANNEL axis. Attention output is either sequence format
        // [B, S, C] or image format [B, C, H, W]; in both cases the channel axis is the one whose
        // length matches the adapter width, so locate it rather than assuming a layout.
        var shape = output.Shape;
        int channelAxis = -1;
        for (int d = 0; d < shape.Length; d++)
        {
            if (shape[d] == _channels) { channelAxis = d; break; }
        }

        if (channelAxis < 0)
        {
            // Nothing to adapt against — the wrapped block does not expose this width. Returning the
            // unmodified output would silently disable the paper's mechanism, so refuse instead.
            var dims = new int[shape.Length];
            for (int d = 0; d < shape.Length; d++) dims[d] = shape[d];
            throw new InvalidOperationException(
                $"T-LoRA adapter is sized for {_channels} channels but the wrapped block produced " +
                $"shape [{string.Join(", ", dims)}], which has no axis of that width. The adapter " +
                "would have no effect, which would silently disable the timestep-dependent rank schedule.");
        }

        int timestep = CurrentTimestep;
        int tokens = 1;
        for (int d = 0; d < shape.Length; d++)
        {
            if (d != channelAxis) tokens *= shape[d];
        }

        // Walk the flat buffer, gathering each token's channel vector, adapting it, and adding the
        // result back. Strides are computed from the located axis so both layouts work unchanged.
        int innerStride = 1;
        for (int d = channelAxis + 1; d < shape.Length; d++) innerStride *= shape[d];
        int channelStride = innerStride;
        int outerStride = channelStride * _channels;

        var result = output.Clone();
        var slice = new Vector<T>(_channels);

        int outerCount = tokens / Math.Max(1, innerStride);
        for (int outer = 0; outer < outerCount; outer++)
        {
            for (int inner = 0; inner < innerStride; inner++)
            {
                int baseIndex = (outer * outerStride) + inner;
                for (int c = 0; c < _channels; c++) slice[c] = output[baseIndex + (c * channelStride)];

                var delta = _adapter.Apply(slice, timestep);
                for (int c = 0; c < _channels; c++)
                {
                    int flat = baseIndex + (c * channelStride);
                    result[flat] = Ops.Add(result[flat], delta[c]);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the number of adapter parameters (A, B and S), excluding the wrapped block's.
    /// </summary>
    public int AdapterParameterCount =>
        (_adapter.DownProjection.Rows * _adapter.DownProjection.Columns)
        + (_adapter.UpProjection.Rows * _adapter.UpProjection.Columns)
        + _adapter.SingularValues.Length;

    /// <summary>
    /// Returns the wrapped block's parameters followed by the adapter's A, B and S.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The inner block's weights are included, and must be. In this library
    /// <c>GetParameters</c>/<c>SetParameters</c> is the FULL-STATE contract that clone, save and load
    /// are built on, not merely the optimizer's view — the flat concatenation has to be
    /// index-identical across the pair. Reporting only the adapter would silently drop the base
    /// attention weights from the model's state, so a round-trip or a <c>Clone</c> would return a
    /// network with re-initialized attention.
    /// </para>
    /// <para>
    /// The paper's "freeze W" is therefore expressed where it belongs — in what training updates, not
    /// in what serialization can see. <see cref="TrainableParameterOffset"/> gives callers the index
    /// where the adapter's block begins so an optimizer can restrict itself to it.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        var innerParameters = _inner.GetParameters();
        var down = _adapter.DownProjection;
        var up = _adapter.UpProjection;
        var singular = _adapter.SingularValues;

        var parameters = new Vector<T>(innerParameters.Length + AdapterParameterCount);
        int index = 0;
        for (int i = 0; i < innerParameters.Length; i++) parameters[index++] = innerParameters[i];
        for (int r = 0; r < down.Rows; r++)
        {
            for (int c = 0; c < down.Columns; c++) parameters[index++] = down[r, c];
        }
        for (int r = 0; r < up.Rows; r++)
        {
            for (int c = 0; c < up.Columns; c++) parameters[index++] = up[r, c];
        }
        for (int s = 0; s < singular.Length; s++) parameters[index++] = singular[s];
        return parameters;
    }

    /// <summary>
    /// Gets the index in <see cref="GetParameters"/> at which the adapter's parameters begin.
    /// </summary>
    /// <remarks>
    /// Everything before this offset is the frozen base block; everything from it on is A, B then S.
    /// This is how a caller honours the paper's freeze without the state vector having to lie about
    /// what the layer contains.
    /// </remarks>
    public int TrainableParameterOffset => _inner.GetParameters().Length;

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var innerParameters = _inner.GetParameters();
        int expected = innerParameters.Length + AdapterParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException(
                $"Expected {expected} parameters ({innerParameters.Length} for the wrapped attention " +
                $"block plus {AdapterParameterCount} for the adapter), got {parameters.Length}.",
                nameof(parameters));
        }

        var forInner = new Vector<T>(innerParameters.Length);
        for (int i = 0; i < innerParameters.Length; i++) forInner[i] = parameters[i];
        _inner.SetParameters(forInner);

        var down = _adapter.DownProjection;
        var up = _adapter.UpProjection;
        var singular = _adapter.SingularValues;

        int index = innerParameters.Length;
        for (int r = 0; r < down.Rows; r++)
        {
            for (int c = 0; c < down.Columns; c++) down[r, c] = parameters[index++];
        }
        for (int r = 0; r < up.Rows; r++)
        {
            for (int c = 0; c < up.Columns; c++) up[r, c] = parameters[index++];
        }
        for (int s = 0; s < singular.Length; s++) singular[s] = parameters[index++];
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // No gradients are accumulated on the adapter by this layer's own backward path, so there is
        // nothing to step here. Training flows through the model's parameter vector
        // (GetParameters/SetParameters), which is how LatentDiffusionModelBase drives updates.
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inner.ResetState();
    }

    // No Backward override: LayerBase removed manual backward in favour of tape-based autodiff via
    // ITrainableLayer<T>. The adapter's contribution is a residual add over ordinary tensor ops, so
    // the tape differentiates it without help.
}
