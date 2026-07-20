using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Pre-Layer-Normalization transformer block with RMSNorm and a caller-supplied
/// self-attention sublayer. Matches the decoder-style architecture used by T5,
/// LLaMA, Gemma, Qwen2, and ChatGLM3 text encoders / language models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Block structure (Raffel 2020 §2.1, Touvron 2023 §2.2):
/// </para>
/// <list type="number">
/// <item>y = x + Attn(RMSNorm(x))</item>
/// <item>z = y + FFN(RMSNorm(y))</item>
/// </list>
/// <para>
/// The attention sublayer is injected through the constructor so the same
/// block class supports T5's relative-bias attention, Gemma's RoPE
/// multi-head attention, Qwen2's RoPE grouped-query attention, and
/// ChatGLM3's RoPE multi-query attention. The FFN is a paper-canonical
/// two-matrix linear → activation → linear stack with no biases (matching
/// the T5 / LLaMA / Gemma / Qwen2 / ChatGLM3 convention).
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class PreLNTransformerBlock<T> : LayerBase<T>
{
    private readonly RMSNormalizationLayer<T> _norm1;
    private readonly LayerBase<T> _attention;
    private readonly RMSNormalizationLayer<T> _norm2;
    // FFN "up" projection. When <see cref="_useGatedSwiGLU"/> is false this is a plain
    // DenseLayer (linear -> activation) matching the T5/Gemma/Qwen2 convention. When true it
    // is a gated SwiGLU projection (silu(W1 x) * W3(x)) — the LLaMA / MetaVoice-1B FFN
    // (Shazeer 2020, "GLU Variants Improve Transformer"): F.silu(w1(x)) * w3(x), followed by
    // the w2/c_proj down-projection in <see cref="_ffnDown"/>. Typed as the polymorphic
    // LayerBase so both variants flow through the same GetParameters/SetParameters/Forward path.
    private readonly LayerBase<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;
    private readonly IActivationFunction<T> _ffnActivation;
    private readonly bool _useGatedSwiGLU;
    private readonly int _hiddenSize;
    private readonly int _ffnDim;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initialises a pre-LN transformer block.
    /// </summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="ffnDim">Hidden dimension of the FFN (typically 4× hiddenSize for T5,
    /// 8/3 × hiddenSize for SwiGLU-style FFNs in LLaMA/Gemma).</param>
    /// <param name="attention">Pre-constructed self-attention sublayer. The caller
    /// chooses the attention variant: <see cref="T5RelativeBiasAttentionLayer{T}"/>
    /// for T5/DistilledT5, <see cref="MultiHeadAttentionLayer{T}"/> with
    /// <c>ConfigurePositionalEncoding(Rotary)</c> for Gemma, or
    /// <see cref="GroupedQueryAttentionLayer{T}"/> for Qwen2/ChatGLM3.</param>
    /// <param name="ffnActivation">FFN activation function. Paper canonical:
    /// <see cref="GELUActivation{T}"/> for T5/DistilledT5,
    /// <see cref="SiLUActivation{T}"/> for Gemma/Qwen2/ChatGLM3. Ignored when
    /// <paramref name="useGatedSwiGLU"/> is true (SwiGLU uses its own internal SiLU gate).</param>
    /// <param name="useGatedSwiGLU">When true the FFN "up" projection is a gated SwiGLU unit
    /// (<c>silu(W1·x) ⊙ W3·x</c>, Shazeer 2020) instead of the plain <c>activation(W·x)</c>
    /// projection. This is the LLaMA / MetaVoice-1B first- and second-stage transformer FFN.
    /// The down-projection (<c>c_proj</c>) is unchanged. Defaults to false so the existing
    /// T5 / Gemma / Qwen2 / ChatGLM3 encoders keep their plain FFN, byte-for-byte.</param>
    public PreLNTransformerBlock(
        int hiddenSize,
        int ffnDim,
        LayerBase<T> attention,
        IActivationFunction<T>? ffnActivation = null,
        bool useGatedSwiGLU = false)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (ffnDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (attention is null)
            throw new ArgumentNullException(nameof(attention));

        _hiddenSize = hiddenSize;
        _ffnDim = ffnDim;
        _attention = attention;
        _ffnActivation = ffnActivation ?? new GELUActivation<T>();
        _useGatedSwiGLU = useGatedSwiGLU;

        _norm1 = new RMSNormalizationLayer<T>();
        _norm2 = new RMSNormalizationLayer<T>();

        // FFN up-projection:
        //  • Plain (default): T5 / Gemma / Qwen2 / ChatGLM3 — linear → activation, expanding
        //    hidden → ffnDim. DenseLayer lazy-resolves its input dim on the first forward.
        //  • Gated SwiGLU (useGatedSwiGLU): LLaMA / MetaVoice-1B — SwiGLUFeedForwardLayer computes
        //    silu(W1·x) ⊙ (W3·x), also expanding hidden → ffnDim, then the shared _ffnDown
        //    projects ffnDim → hidden (the w2 / c_proj matrix). Both share the identical
        //    GetParameters / SetParameters / Forward code path (typed as LayerBase<T>).
        _ffnUp = useGatedSwiGLU
            ? new SwiGLUFeedForwardLayer<T>(ffnDim)
            : new DenseLayer<T>(outputSize: ffnDim, activationFunction: _ffnActivation);
        _ffnDown = new DenseLayer<T>(outputSize: hiddenSize, activationFunction: new IdentityActivation<T>());

        // Register every sublayer so TapeTrainingStep<T>.CollectParameters
        // recursively discovers their trainable tensors. Without this the
        // gradient tape only sees the parameters of layers in the top-level
        // Layers list — every weight inside this block (RMSNorm γ, Q/K/V/O
        // projections, relative-position bias table, FFN matrices) would
        // remain frozen during training. Caught by the
        // T5Conditioner_Training_ChangesParameters test.
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_attention);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_ffnUp);
        RegisterSubLayer(_ffnDown);
    }

    /// <summary>
    /// Forward pass. Routes every shape op through <see cref="LayerBase{T}.Engine"/>
    /// so the gradient tape records the residual additions and sublayer outputs.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Self-attention sublayer expects [B, S, H] (or [S, H]).
        var normed1 = _norm1.Forward(input);
        var attnOut = _attention.Forward(normed1);
        var afterAttn = Engine.TensorAdd(input, attnOut);

        var normed2 = _norm2.Forward(afterAttn);

        // FFN expects 2D input [N, hiddenSize] per the existing DenseLayer
        // contract. Flatten the leading dims, run FFN, reshape back.
        int rank = afterAttn.Shape.Length;
        int featureDim = afterAttn.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= afterAttn.Shape[i];

        var normed2Flat = Engine.Reshape(normed2, new[] { flatN, featureDim });
        var ffnUpOut = _ffnUp.Forward(normed2Flat);
        var ffnDownOut = _ffnDown.Forward(ffnUpOut);
        var afterAttnShape = afterAttn._shape;
        var ffnReshaped = Engine.Reshape(ffnDownOut, afterAttnShape);

        var output = Engine.TensorAdd(afterAttn, ffnReshaped);
        return output;
    }

    /// <inheritdoc/>
    public override long ParameterCount =>
        _norm1.ParameterCount + _attention.ParameterCount + _norm2.ParameterCount +
        _ffnUp.ParameterCount + _ffnDown.ParameterCount;

    /// <inheritdoc/>
    public override Vector<T> GetParameters() =>
        Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(_norm1.GetParameters(), _attention.GetParameters()),
                _norm2.GetParameters()),
            Vector<T>.Concatenate(_ffnUp.GetParameters(), _ffnDown.GetParameters()));

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;
        SetSubParams(_norm1, parameters, ref offset);
        SetSubParams(_attention, parameters, ref offset);
        SetSubParams(_norm2, parameters, ref offset);
        SetSubParams(_ffnUp, parameters, ref offset);
        SetSubParams(_ffnDown, parameters, ref offset);
    }

    private static void SetSubParams(LayerBase<T> layer, Vector<T> source, ref int offset)
    {
        int count = (int)layer.ParameterCount;
        if (count == 0) return;
        var slice = source.Slice(offset, count);
        layer.SetParameters(slice);
        offset += count;
    }

    /// <summary>
    /// Persists the constructor configuration (block widths, FFN variant/activation, and the
    /// injected attention sublayer's descriptor) so the block can be rebuilt during
    /// deserialization / <c>Clone</c>. The sublayer WEIGHTS travel separately in the flat
    /// parameter vector (see <see cref="GetParameters"/>); this metadata only carries the
    /// STRUCTURE needed to recreate an identically-shaped block before <see cref="SetParameters"/>
    /// runs. Reconstruction lives in <c>DeserializationHelper.CreatePreLNTransformerBlock</c>.
    /// </summary>
    internal override System.Collections.Generic.Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(ci);
        metadata["FfnDim"] = _ffnDim.ToString(ci);
        metadata["UseGatedSwiGLU"] = _useGatedSwiGLU ? "true" : "false";
        // Store the assembly-qualified name (mirrors LayerBase's ScalarActivationType) so the
        // reader can resolve it via Type.GetType through TryCreateActivationInstance.
        metadata["FfnActivationType"] =
            _ffnActivation.GetType().AssemblyQualifiedName ?? _ffnActivation.GetType().FullName ?? string.Empty;
        metadata["AttentionType"] = _attention.GetType().Name;
        // MultiHeadAttentionLayer is the only attention type wired through this block by the
        // built-in model factories (LLaMA/Gemma/MetaVoice-1B use RoPE MHA). Persist its full
        // descriptor — head count, causal masking, and the RoPE frequency/length — so the
        // reconstructed attention computes bit-identical outputs. Head dimension is derived
        // as HiddenSize / AttHeadCount on the read side.
        if (_attention is MultiHeadAttentionLayer<T> mha)
        {
            metadata["AttHeadCount"] = mha.HeadCount.ToString(ci);
            metadata["AttUseCausalMask"] = mha.UseCausalMask ? "true" : "false";
            metadata["AttPositionalEncoding"] = mha.PositionalEncoding.ToString();
            metadata["AttRopeTheta"] = mha.RopeTheta.ToString(ci);
            metadata["AttRopeMaxSeqLen"] = mha.PositionalMaxSequenceLength.ToString(ci);
        }
        return metadata;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() =>
        Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(_norm1.GetParameterGradients(), _attention.GetParameterGradients()),
                _norm2.GetParameterGradients()),
            Vector<T>.Concatenate(_ffnUp.GetParameterGradients(), _ffnDown.GetParameterGradients()));

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _norm1.ClearGradients();
        _attention.ClearGradients();
        _norm2.ClearGradients();
        _ffnUp.ClearGradients();
        _ffnDown.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _attention.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _ffnUp.UpdateParameters(learningRate);
        _ffnDown.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm1.ResetState();
        _attention.ResetState();
        _norm2.ResetState();
        _ffnUp.ResetState();
        _ffnDown.ResetState();
    }
}
