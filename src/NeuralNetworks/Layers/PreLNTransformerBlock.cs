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
    // Non-readonly so the inference optimizer can swap the attention sublayer in place (e.g.
    // GroupedQueryAttentionLayer -> CachedGroupedQueryAttention for KV-cached decode) via ReplaceAttention,
    // the same contract TransformerEncoderBlock uses.
    private LayerBase<T> _attention;
    private readonly RMSNormalizationLayer<T> _norm2;
    private readonly DenseLayer<T>? _ffnGate;
    private readonly DenseLayer<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;
    private readonly IActivationFunction<T> _ffnActivation;
    private readonly bool _gated;
    private readonly int _hiddenSize;
    private readonly int _ffnDim;

    public override bool SupportsTraining => true;

    /// <summary>The pre-attention RMSNorm sublayer (exposed for tensor-parallel serving partitioning).</summary>
    public RMSNormalizationLayer<T> Norm1 => _norm1;

    /// <summary>The pre-FFN RMSNorm sublayer (exposed for tensor-parallel serving partitioning).</summary>
    public RMSNormalizationLayer<T> Norm2 => _norm2;

    /// <summary>The self-attention sublayer (exposed for tensor-parallel serving partitioning).</summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>
    /// Swaps the attention sublayer in place (e.g. the inference optimizer replacing
    /// <see cref="GroupedQueryAttentionLayer{T}"/> with a KV-cached
    /// <see cref="AiDotNet.Inference.PagedAttention.CachedGroupedQueryAttention{T}"/> for incremental decode).
    /// Keeps the registered-sublayer set consistent so parameter enumeration still discovers the block's weights.
    /// </summary>
    public void ReplaceAttention(LayerBase<T> replacement)
    {
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));
        UnregisterSubLayer(_attention);
        _attention = replacement;
        RegisterSubLayer(_attention);
    }

    /// <summary>
    /// The FFN gate-projection DenseLayer (hidden -&gt; ffnDim, activation), present only in
    /// gated SwiGLU mode; <c>null</c> for the classic two-matrix FFN. Exposed for
    /// tensor-parallel serving partitioning and pretrained weight loading.
    /// </summary>
    public DenseLayer<T>? FfnGate => _ffnGate;

    /// <summary>
    /// Whether the FFN is a gated SwiGLU (LLaMA/Mistral/Qwen2: <c>down(act(gate(x)) * up(x))</c>)
    /// rather than the classic two-matrix <c>down(act(up(x)))</c>.
    /// </summary>
    public bool IsGated => _gated;

    /// <summary>The FFN up-projection DenseLayer (hidden -&gt; ffnDim). In gated mode this is the
    /// linear value path (no activation); in classic mode it carries the FFN activation.</summary>
    public DenseLayer<T> FfnUp => _ffnUp;

    /// <summary>The FFN down-projection DenseLayer (ffnDim -&gt; hidden, identity).</summary>
    public DenseLayer<T> FfnDown => _ffnDown;

    /// <summary>The FFN activation function applied after the up-projection.</summary>
    public IActivationFunction<T> FfnActivation => _ffnActivation;

    /// <summary>The FFN hidden dimension.</summary>
    public int FfnDim => _ffnDim;

    /// <summary>The model (input/output) feature dimension.</summary>
    public int HiddenSize => _hiddenSize;

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
    /// <see cref="SiLUActivation{T}"/> for Gemma/Qwen2/ChatGLM3.</param>
    public PreLNTransformerBlock(
        int hiddenSize,
        int ffnDim,
        LayerBase<T> attention,
        IActivationFunction<T>? ffnActivation = null,
        bool gated = false)
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
        _gated = gated;

        // hiddenSize is known here, so these parameters must exist before the
        // first GetParameters()/serialization call. Leaving them lazy made a
        // pre-forward parameter vector shorter than the post-forward layout.
        _norm1 = new RMSNormalizationLayer<T>(hiddenSize);
        _norm2 = new RMSNormalizationLayer<T>(hiddenSize);

        // DenseLayer(outputSize, activation): lazy-resolves input dim on first forward
        // and (with no init strategy) zero-inits biases, matching the bias-free FFN
        // convention of T5 / LLaMA / Gemma / Qwen2 / ChatGLM3.
        if (_gated)
        {
            // Gated SwiGLU (LLaMA/Mistral/Qwen2, Shazeer 2020): down( act(gate(x)) * up(x) ).
            // The activation sits on the gate path; the up (value) path stays linear.
            _ffnGate = new DenseLayer<T>(outputSize: ffnDim, activationFunction: _ffnActivation);
            _ffnUp = new DenseLayer<T>(outputSize: ffnDim, activationFunction: new IdentityActivation<T>());
        }
        else
        {
            // Classic two-matrix FFN (T5/Gemma-style): linear → activation → linear.
            _ffnUp = new DenseLayer<T>(outputSize: ffnDim, activationFunction: _ffnActivation);
        }

        _ffnDown = new DenseLayer<T>(outputSize: hiddenSize, activationFunction: new IdentityActivation<T>());

        // The FFN input widths are also constructor-known. Resolve only the
        // sublayers—not this block's sequence dimension—so sequence length stays
        // dynamic while parameter enumeration is complete and stable from birth.
        if (_ffnGate is not null)
            _ffnGate.ResolveFromShape(new[] { hiddenSize });
        _ffnUp.ResolveFromShape(new[] { hiddenSize });
        _ffnDown.ResolveFromShape(new[] { ffnDim });

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
        if (_ffnGate is not null)
            RegisterSubLayer(_ffnGate);
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

        Tensor<T> ffnHidden;
        if (_ffnGate is not null)
        {
            // Gated SwiGLU: act(gate(x)) * up(x). The gate carries the activation
            // (applied inside _ffnGate); the up path is linear.
            var gateOut = _ffnGate.Forward(normed2Flat);
            var upOut = _ffnUp.Forward(normed2Flat);
            ffnHidden = Engine.TensorMultiply(gateOut, upOut);
        }
        else
        {
            // Classic FFN: activation is applied inside _ffnUp.
            ffnHidden = _ffnUp.Forward(normed2Flat);
        }

        var ffnDownOut = _ffnDown.Forward(ffnHidden);
        var afterAttnShape = afterAttn._shape;
        var ffnReshaped = Engine.Reshape(ffnDownOut, afterAttnShape);

        var output = Engine.TensorAdd(afterAttn, ffnReshaped);
        return output;
    }

    /// <summary>
    /// Resolves every sublayer's shape without a data-carrying forward. This block overrides
    /// <see cref="Forward"/> directly (bypassing the base lazy-init hook), so this runs ONLY via
    /// <see cref="LayerBase{T}.ResolveFromShape"/> — the deserialization / shape-oracle path. The norms and
    /// FFN DenseLayers are lazy (input dim resolved on first use), so without this the reconstructed block
    /// reported a too-small <see cref="ParameterCount"/> and <c>SetParameters</c> rejected the saved vector.
    /// Sublayers resolve in forward order so any RNG-based weight init consumes the stream exactly as a
    /// natural forward would.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int hidden = _hiddenSize;
        var hiddenShape = new[] { 1, hidden };
        if (!_norm1.IsShapeResolved) _norm1.ResolveFromShape(hiddenShape);
        if (!_attention.IsShapeResolved) _attention.ResolveFromShape(hiddenShape);
        if (!_norm2.IsShapeResolved) _norm2.ResolveFromShape(hiddenShape);
        if (_ffnGate is not null && !_ffnGate.IsShapeResolved) _ffnGate.ResolveFromShape(hiddenShape);
        if (!_ffnUp.IsShapeResolved) _ffnUp.ResolveFromShape(hiddenShape);
        if (!_ffnDown.IsShapeResolved) _ffnDown.ResolveFromShape(new[] { 1, _ffnDim });

        int seq = input.Shape.Length >= 2 ? input.Shape[input.Shape.Length - 2] : 1;
        ResolveShapes(new[] { seq, hidden }, new[] { seq, hidden });
    }

    /// <summary>
    /// The block's sublayers in flat-parameter order. The gate (when present) sits
    /// immediately before the up-projection, so a non-gated block keeps its original
    /// layout and a gated block appends the gate deterministically.
    /// </summary>
    private IEnumerable<LayerBase<T>> OrderedSubLayers()
    {
        yield return _norm1;
        yield return _attention;
        yield return _norm2;
        if (_ffnGate is not null)
            yield return _ffnGate;
        yield return _ffnUp;
        yield return _ffnDown;
    }

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in OrderedSubLayers())
                total += layer.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var layer in OrderedSubLayers())
            acc = Vector<T>.Concatenate(acc, layer.GetParameters());
        return acc;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;
        foreach (var layer in OrderedSubLayers())
            SetSubParams(layer, parameters, ref offset);
    }

    private static void SetSubParams(LayerBase<T> layer, Vector<T> source, ref int offset)
    {
        int count = (int)layer.ParameterCount;
        if (count == 0) return;
        var slice = source.Slice(offset, count);
        layer.SetParameters(slice);
        offset += count;
    }

    /// <inheritdoc/>
    internal override System.Collections.Generic.Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["HiddenSize"] = _hiddenSize.ToString(ci);
        metadata["FfnDim"] = _ffnDim.ToString(ci);
        metadata["Gated"] = _gated.ToString();
        metadata["FfnActivationType"] = _ffnActivation.GetType().AssemblyQualifiedName
            ?? _ffnActivation.GetType().FullName ?? string.Empty;

        // Persist the injected (polymorphic T5 / MHA / GQA) attention sublayer as a self-contained sub-blob
        // under an "Attn." prefix: its concrete type, its own metadata, and its shapes. The block's ctor
        // rebuilds the norms + FFN from HiddenSize/FfnDim/Gated/FfnActivation, so only the attention needs to
        // round-trip; the deserializer reconstructs it recursively via CreateLayerFromType (which restores its
        // full config, e.g. GQA RoPE / causal mask). Without this the block had no known deserialization
        // constructor and cloning it threw — leaving the paged incremental-serving clone of a decoder inert.
        foreach (var kv in _attention.GetMetadata())
            metadata["Attn." + kv.Key] = kv.Value;
        metadata["Attn.LayerType"] = _attention.GetType().Name;
        metadata["Attn.InputShape"] = string.Join(",", _attention.GetInputShape());
        metadata["Attn.OutputShape"] = string.Join(",", _attention.GetOutputShape());
        return metadata;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var layer in OrderedSubLayers())
            acc = Vector<T>.Concatenate(acc, layer.GetParameterGradients());
        return acc;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in OrderedSubLayers())
            layer.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in OrderedSubLayers())
            layer.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in OrderedSubLayers())
            layer.ResetState();
    }
}
