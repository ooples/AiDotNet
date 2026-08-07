using AiDotNet.Attributes;
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
[AutoParameters]
public sealed partial class TLoRAAttentionAdapter<T> : LayerBase<T>, IAttentionBlockDecorator<T>
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        return PostProcess(_inner.Forward(input));
    }

    /// <summary>
    /// Adds the timestep-masked low-rank update to whatever the wrapped block produced.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="Forward"/> so a caller that must invoke the inner block through a
    /// different signature — cross-attention needs its conditioning passed through
    /// <c>ForwardWithContext</c> — can still get the adaptation applied. Overriding only
    /// <c>Forward(input)</c> made this wrapper fall through the UNet's cross-attention dispatch to the
    /// single-argument path, which discards the conditioning entirely.
    /// </remarks>
    public Tensor<T> PostProcess(Tensor<T> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));

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

        int tokens = 1;
        for (int d = 0; d < shape.Length; d++)
        {
            if (d != channelAxis) tokens *= shape[d];
        }

        // ONE matmul for the whole block, not one per token. The delta depends only on the timestep
        // and the current weights, so it is formed once and applied as [tokens, C] x [C, C]. Every
        // reshape/permute goes through Engine so the gradient tape records it — direct Tensor
        // Reshape/Transpose would bypass the tape and silently break gradient flow, which is the
        // convention DiffusionAttention itself documents.
        var delta = _adapter.EffectiveDeltaTransposed(CurrentTimestep);

        // Move the channel axis last so the flattened view is [tokens, C].
        bool channelIsLast = channelAxis == shape.Length - 1;
        Tensor<T> channelLast = output;
        int[]? forwardPermutation = null;

        if (!channelIsLast)
        {
            forwardPermutation = new int[shape.Length];
            int next = 0;
            for (int d = 0; d < shape.Length; d++)
            {
                if (d != channelAxis) forwardPermutation[next++] = d;
            }
            forwardPermutation[shape.Length - 1] = channelAxis;
            channelLast = Engine.TensorPermute(output, forwardPermutation);
        }

        var flattened = Engine.Reshape(channelLast, new[] { tokens, _channels });
        var adapted = Engine.TensorAdd(flattened, Engine.TensorMatMul(flattened, delta));

        if (channelIsLast)
        {
            return Engine.Reshape(adapted, shape.ToArray());
        }

        // Undo the permutation: axis j of the permuted tensor came from forwardPermutation[j].
        var permutedShape = new int[shape.Length];
        for (int d = 0; d < shape.Length; d++) permutedShape[d] = shape[forwardPermutation![d]];

        var inverse = new int[shape.Length];
        for (int d = 0; d < shape.Length; d++) inverse[forwardPermutation![d]] = d;

        return Engine.TensorPermute(Engine.Reshape(adapted, permutedShape), inverse);
    }

    /// <summary>
    /// Gets the number of adapter parameters (A, B and S), excluding the wrapped block's.
    /// </summary>
    public int AdapterParameterCount =>
        (_adapter.DownProjection.Rows * _adapter.DownProjection.Columns)
        + (_adapter.UpProjection.Rows * _adapter.UpProjection.Columns)
        + _adapter.SingularValues.Length;

    /// <summary>
    /// The adapter's own trainable state — A, then B, then S — kept OUT of
    /// <see cref="GetParameters"/> on purpose.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Wrapping a layer must not change its parameter COUNT. Every parameter-copy path in this library
    /// pairs a source layer with a target layer positionally and checks the lengths match:
    /// <c>UNetNoisePredictor.Clone</c> builds a bare predictor and immediately pushes the source's
    /// chunks into it, so a decorated source and a not-yet-decorated clone mismatched with
    /// "chunk length 24704 does not match layer parameter length 16448" — and that copy happens INSIDE
    /// the predictor's clone, before the owning model can inject anything.
    /// </para>
    /// <para>
    /// Rather than trying to order decoration around that copy, the decorator is now structurally
    /// transparent: it reports exactly what the block it wraps reports, so every existing clone,
    /// chunk, share and round-trip path behaves identically whether or not the block is decorated.
    /// The adapter's state travels through this pair instead, copied explicitly by the owning model.
    /// </para>
    /// </remarks>
    public Vector<T> GetAdapterState()
    {
        var down = _adapter.DownProjection;
        var up = _adapter.UpProjection;
        var singular = _adapter.SingularValues;

        var state = new Vector<T>(AdapterParameterCount);
        int index = 0;
        for (int r = 0; r < down.Rows; r++)
        {
            for (int c = 0; c < down.Columns; c++) state[index++] = down[r, c];
        }
        for (int r = 0; r < up.Rows; r++)
        {
            for (int c = 0; c < up.Columns; c++) state[index++] = up[r, c];
        }
        for (int s = 0; s < singular.Length; s++) state[index++] = singular[s];
        return state;
    }

    /// <summary>
    /// Restores state produced by <see cref="GetAdapterState"/>.
    /// </summary>
    public void SetAdapterState(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != AdapterParameterCount)
        {
            throw new ArgumentException(
                $"Expected {AdapterParameterCount} adapter parameters, got {state.Length}. A mismatch " +
                "means the source and target adapters were built with different ranks or widths.",
                nameof(state));
        }

        var down = _adapter.DownProjection;
        var up = _adapter.UpProjection;
        var singular = _adapter.SingularValues;

        int index = 0;
        for (int r = 0; r < down.Rows; r++)
        {
            for (int c = 0; c < down.Columns; c++) down[r, c] = state[index++];
        }
        for (int r = 0; r < up.Rows; r++)
        {
            for (int c = 0; c < up.Columns; c++) up[r, c] = state[index++];
        }
        for (int s = 0; s < singular.Length; s++) singular[s] = state[index++];

        _adapter.InvalidateCache();
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
