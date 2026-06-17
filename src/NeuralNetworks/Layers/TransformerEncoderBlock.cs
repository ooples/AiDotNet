using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Pre-Layer-Normalization transformer encoder block — multi-head self-attention
/// and a position-wise feed-forward network, each wrapped in a residual (skip)
/// connection with layer normalization applied BEFORE the sublayer (Pre-LN).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Block structure (Pre-LN, Xiong et al. 2020 "On Layer Normalization in the
/// Transformer Architecture"; the residual/FFN design is Vaswani 2017 §3.1):
/// </para>
/// <list type="number">
/// <item>y = x + Dropout(SelfAttention(LayerNorm(x)))</item>
/// <item>z = y + Dropout(FFN(LayerNorm(y)))</item>
/// </list>
/// <para>
/// The <b>residual connections</b> (the <c>x +</c> / <c>y +</c> terms) are the
/// defining feature of the transformer: they let the input signal flow
/// unattenuated through arbitrarily deep stacks. Without them the attention/FFN
/// output REPLACES the hidden state each layer, the token-identity signal is
/// washed out (empirically ~60× per layer), and the network mode-collapses to
/// an input-independent constant output — the root cause of issue #1380.
/// </para>
/// <para>
/// <b>Pre-LN vs Post-LN:</b> normalizing the sublayer INPUT (Pre-LN) rather than
/// the residual SUM (Post-LN, the original 2017 ordering) keeps the residual
/// path un-normalized, so gradients flow cleanly through depth and the model
/// trains stably WITHOUT learning-rate warmup. Post-LN converges far slower
/// without warmup; Pre-LN is the ordering used by every modern transformer
/// (GPT-2 onward, LLaMA, etc.) and trains measurably faster here.
/// </para>
/// <para><b>For Beginners:</b> A residual connection means "add the layer's
/// input back to its output." It's like keeping a copy of the original so
/// nothing important gets lost as the data passes through. Transformers
/// literally cannot learn without them.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 2, 16, 0.0")]
public partial class TransformerEncoderBlock<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly double _dropoutRate;
    // The FFN inner activation, retained so it round-trips through serialization
    // (the FFN Dense layer is reconstructed from the block's metadata, not its own).
    private readonly IActivationFunction<T> _ffnActivation;

    // Widened from MultiHeadAttentionLayer<T> so InferenceOptimizer can swap in a
    // rewritten attention implementation (FlashAttentionLayer, CachedMultiHeadAttention,
    // PagedCachedMultiHeadAttention) via ReplaceAttention. All members used below
    // (Forward/ParameterCount/Get-SetParameters/gradients/state) are LayerBase<T> surface.
    private LayerBase<T> _attention;
    private readonly LayerNormalizationLayer<T> _norm1;
    // Widened from DenseLayer<T> (same rationale as _attention) so the inference
    // optimizer / LoRA configuration can swap in wrapped implementations
    // (QuantizedDenseLayer, StandardLoRAAdapter) via ReplaceFfnUp/ReplaceFfnDown.
    private LayerBase<T> _ffnUp;
    private LayerBase<T> _ffnDown;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DropoutLayer<T>? _attnDropout;
    private readonly DropoutLayer<T>? _ffnDropout;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initialises a Post-LN transformer encoder block.
    /// </summary>
    /// <param name="hiddenSize">Model (input/output) feature dimension.</param>
    /// <param name="numHeads">Number of self-attention heads. Must divide <paramref name="hiddenSize"/>.</param>
    /// <param name="ffnDim">Inner dimension of the feed-forward network (typically 4× hiddenSize).</param>
    /// <param name="dropoutRate">Dropout probability applied to each sublayer's output before the
    /// residual add (Vaswani §5.4). 0 disables dropout.</param>
    public TransformerEncoderBlock(int hiddenSize, int numHeads, int ffnDim, double dropoutRate = 0.0,
        IActivationFunction<T>? ffnActivation = null)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (ffnDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(ffnDim));

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _dropoutRate = dropoutRate;

        _attention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: new IdentityActivation<T>());
        // Size the norms eagerly (their featureSize is known = hiddenSize) so they are
        // not lazy; the Dense FFN layers still resolve their input dim on first forward.
        _norm1 = new LayerNormalizationLayer<T>(hiddenSize);
        // FFN inner activation defaults to ReLU (original Transformer, Vaswani 2017);
        // BERT-class encoders pass GELU (Devlin 2018) to stay paper-faithful.
        _ffnActivation = ffnActivation ?? new ReLUActivation<T>();
        _ffnUp = new DenseLayer<T>(ffnDim, _ffnActivation);
        _ffnDown = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _norm2 = new LayerNormalizationLayer<T>(hiddenSize);
        if (dropoutRate > 0)
        {
            _attnDropout = new DropoutLayer<T>(dropoutRate);
            _ffnDropout = new DropoutLayer<T>(dropoutRate);
        }

        // Register every sublayer so the gradient tape (TapeTrainingStep.CollectParameters)
        // recursively discovers their trainable tensors. Without this every weight inside
        // this block would stay frozen during training.
        RegisterSubLayer(_attention);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_ffnUp);
        RegisterSubLayer(_ffnDown);
        RegisterSubLayer(_norm2);
        if (_attnDropout is not null) RegisterSubLayer(_attnDropout);
        if (_ffnDropout is not null) RegisterSubLayer(_ffnDropout);
    }

    /// <summary>
    /// The block's current self-attention sublayer. A freshly constructed block hosts a
    /// <see cref="MultiHeadAttentionLayer{T}"/>; <see cref="ReplaceAttention"/> (used by
    /// the inference optimizer's attention rewrites) may swap in a
    /// <c>FlashAttentionLayer</c> / <c>CachedMultiHeadAttention</c> /
    /// <c>PagedCachedMultiHeadAttention</c>.
    /// </summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>
    /// Swaps the block's self-attention sublayer for <paramref name="replacement"/> —
    /// the composite-layer counterpart of the inference optimizer assigning a rewritten
    /// attention layer into <c>model.Layers[i]</c> for discrete layouts. Keeps the
    /// registered-sublayer list (the gradient tape's recursive parameter discovery) and
    /// the cached parameter count consistent.
    /// </summary>
    /// <param name="replacement">The attention layer to host. Must consume and produce
    /// the same <c>[..., seq, hiddenSize]</c> shapes as the layer it replaces.</param>
    public void ReplaceAttention(LayerBase<T> replacement)
    {
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));
        UnregisterSubLayer(_attention);
        _attention = replacement;
        RegisterSubLayer(_attention);
    }

    /// <summary>The block's current FFN up-projection (hiddenSize → ffnDim) sublayer.</summary>
    public LayerBase<T> FfnUpLayer => _ffnUp;

    /// <summary>The block's current FFN down-projection (ffnDim → hiddenSize) sublayer.</summary>
    public LayerBase<T> FfnDownLayer => _ffnDown;

    /// <summary>
    /// Swaps the FFN up-projection sublayer (e.g. for a <c>QuantizedDenseLayer</c> or a
    /// LoRA adapter). Same registered-sublayer/parameter-count consistency contract as
    /// <see cref="ReplaceAttention"/>. The replacement must map
    /// <c>[N, hiddenSize] → [N, ffnDim]</c>.
    /// </summary>
    public void ReplaceFfnUp(LayerBase<T> replacement)
    {
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));
        UnregisterSubLayer(_ffnUp);
        _ffnUp = replacement;
        RegisterSubLayer(_ffnUp);
    }

    /// <summary>
    /// Swaps the FFN down-projection sublayer (e.g. for a <c>QuantizedDenseLayer</c> or a
    /// LoRA adapter). Same registered-sublayer/parameter-count consistency contract as
    /// <see cref="ReplaceAttention"/>. The replacement must map
    /// <c>[N, ffnDim] → [N, hiddenSize]</c>.
    /// </summary>
    public void ReplaceFfnDown(LayerBase<T> replacement)
    {
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));
        UnregisterSubLayer(_ffnDown);
        _ffnDown = replacement;
        RegisterSubLayer(_ffnDown);
    }

    /// <summary>Model (feature) dimension — persisted for deserialization.</summary>
    public int HiddenSize => _hiddenSize;
    /// <summary>Number of attention heads — persisted for deserialization.</summary>
    public int NumHeads => _numHeads;
    /// <summary>Feed-forward inner dimension — persisted for deserialization.</summary>
    public int FfnDim => _ffnDim;
    /// <summary>Dropout probability — persisted for deserialization.</summary>
    public double DropoutRate => _dropoutRate;

    /// <summary>
    /// Forward pass. All shape ops route through <see cref="LayerBase{T}.Engine"/> and every
    /// transformation goes through a registered sublayer's Forward, so the gradient tape records
    /// the residual additions and sublayer outputs (no custom Backward is needed).
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Sublayer 1: self-attention with a PRE-norm residual — afterAttn = x + Attn(Norm(x)).
        // Pre-LN (Xiong et al. 2020) keeps the residual path un-normalized so gradients flow
        // cleanly through depth and the model trains stably WITHOUT learning-rate warmup. The
        // original Post-LN ordering — Norm(x + Attn(x)) — converges far slower without warmup,
        // which is why every modern transformer uses Pre-LN.
        var attnOut = _attention.Forward(_norm1.Forward(input));
        if (_attnDropout is not null) attnOut = _attnDropout.Forward(attnOut);
        var afterAttn = Engine.TensorAdd(input, attnOut);

        // Sublayer 2: position-wise FFN with a PRE-norm residual — out = afterAttn + FFN(Norm(afterAttn)).
        // DenseLayer expects 2D [N, hiddenSize]; flatten the leading dims, run the FFN, reshape back.
        int rank = afterAttn.Shape.Length;
        int featureDim = afterAttn.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= afterAttn.Shape[i];

        var normed2 = _norm2.Forward(afterAttn);
        var normed2Flat = Engine.Reshape(normed2, new[] { flatN, featureDim });
        var ffnUpOut = _ffnUp.Forward(normed2Flat);
        var ffnDownOut = _ffnDown.Forward(ffnUpOut);
        var ffnReshaped = Engine.Reshape(ffnDownOut, afterAttn._shape);
        if (_ffnDropout is not null) ffnReshaped = _ffnDropout.Forward(ffnReshaped);

        var output = Engine.TensorAdd(afterAttn, ffnReshaped);
        return output;
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _attention.SetTrainingMode(isTraining);
        _norm1.SetTrainingMode(isTraining);
        _ffnUp.SetTrainingMode(isTraining);
        _ffnDown.SetTrainingMode(isTraining);
        _norm2.SetTrainingMode(isTraining);
        _attnDropout?.SetTrainingMode(isTraining);
        _ffnDropout?.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override long ParameterCount =>
        _attention.ParameterCount + _norm1.ParameterCount +
        _ffnUp.ParameterCount + _ffnDown.ParameterCount + _norm2.ParameterCount;

    /// <inheritdoc/>
    public override Vector<T> GetParameters() =>
        Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(_attention.GetParameters(), _norm1.GetParameters()),
                Vector<T>.Concatenate(_ffnUp.GetParameters(), _ffnDown.GetParameters())),
            _norm2.GetParameters());

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // The Dense FFN sublayers resolve their input dimension lazily (on first
        // Forward), so a freshly-constructed/deserialized block reports a partial
        // ParameterCount. Materialize them at the known hidden size before slicing
        // so deserialization (SetParameters with the full trained vector) succeeds.
        if (parameters.Length != ParameterCount)
            MaterializeLazySublayers();

        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;
        SetSubParams(_attention, parameters, ref offset);
        SetSubParams(_norm1, parameters, ref offset);
        SetSubParams(_ffnUp, parameters, ref offset);
        SetSubParams(_ffnDown, parameters, ref offset);
        SetSubParams(_norm2, parameters, ref offset);
    }

    /// <summary>
    /// Runs a dummy forward at the known hidden size to force the lazy Dense FFN
    /// sublayers to allocate their weights, so <see cref="ParameterCount"/> reflects
    /// the full block. Used by <see cref="SetParameters"/> during deserialization.
    /// </summary>
    private void MaterializeLazySublayers()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var dummy = new Tensor<T>(new[] { 1, 2, _hiddenSize });
            _ = Forward(dummy);
            ResetState();
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
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
    public override Vector<T> GetParameterGradients() =>
        Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(_attention.GetParameterGradients(), _norm1.GetParameterGradients()),
                Vector<T>.Concatenate(_ffnUp.GetParameterGradients(), _ffnDown.GetParameterGradients())),
            _norm2.GetParameterGradients());

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _attention.ClearGradients();
        _norm1.ClearGradients();
        _ffnUp.ClearGradients();
        _ffnDown.ClearGradients();
        _norm2.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _attention.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _ffnUp.UpdateParameters(learningRate);
        _ffnDown.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _attention.ResetState();
        _norm1.ResetState();
        _ffnUp.ResetState();
        _ffnDown.ResetState();
        _norm2.ResetState();
        _attnDropout?.ResetState();
        _ffnDropout?.ResetState();
    }

    /// <summary>
    /// Persists the constructor's full parameter set so
    /// <c>DeserializationHelper.CreateLayerFromType</c> can reconstruct the block
    /// (it has no <c>(int[] inputShape)</c> constructor). Without these the deser
    /// path cannot rebuild the sublayers before loading their weights.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenSize"] = _hiddenSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["DropoutRate"] = _dropoutRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        // Persist the FFN inner activation so the deserialized block rebuilds the
        // same FFN (BERT GELU vs Vaswani ReLU); without this a cloned block always
        // fell back to the constructor default and diverged from the original.
        metadata["FfnActivationType"] = _ffnActivation.GetType().AssemblyQualifiedName
            ?? _ffnActivation.GetType().FullName ?? string.Empty;
        return metadata;
    }
}
