using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Pre-Layer-Normalization transformer decoder block — self-attention, a second
/// ("cross") attention, and a position-wise feed-forward network, each wrapped in a
/// residual (skip) connection with layer normalization applied BEFORE the sublayer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Block structure (Pre-LN; residual/FFN design Vaswani 2017 §3.1):
/// </para>
/// <list type="number">
/// <item>a = x + Dropout(SelfAttention(LayerNorm(x)))</item>
/// <item>b = a + Dropout(CrossAttention(LayerNorm(a)))</item>
/// <item>z = b + Dropout(FFN(LayerNorm(b)))</item>
/// </list>
/// <para>
/// As with <see cref="TransformerEncoderBlock{T}"/>, the <b>residual connections</b>
/// are what let the input signal flow through depth — their absence was the root
/// cause of issue #1380's mode-collapse. This block restores them for the decoder
/// stack. NOTE: in this sequential model the "cross" attention attends over the
/// decoder's own hidden stream (there is no separate encoder-memory input threaded
/// through the layer list); that pre-existing limitation is unchanged here — this
/// block only adds the missing residual connections and Pre-LN ordering.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 2, 16, 0.0")]
public partial class TransformerDecoderBlock<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly double _dropoutRate;

    private readonly MultiHeadAttentionLayer<T> _selfAttention;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly MultiHeadAttentionLayer<T> _crossAttention;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DenseLayer<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;
    private readonly LayerNormalizationLayer<T> _norm3;
    private readonly DropoutLayer<T>? _selfDropout;
    private readonly DropoutLayer<T>? _crossDropout;
    private readonly DropoutLayer<T>? _ffnDropout;

    public override bool SupportsTraining => true;

    /// <summary>Initialises a Pre-LN transformer decoder block.</summary>
    /// <param name="hiddenSize">Model (input/output) feature dimension.</param>
    /// <param name="numHeads">Number of attention heads. Must divide <paramref name="hiddenSize"/>.</param>
    /// <param name="ffnDim">Inner dimension of the feed-forward network.</param>
    /// <param name="dropoutRate">Dropout probability on each sublayer output (0 disables).</param>
    public TransformerDecoderBlock(int hiddenSize, int numHeads, int ffnDim, double dropoutRate = 0.0)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (ffnDim <= 0) throw new ArgumentOutOfRangeException(nameof(ffnDim));

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _dropoutRate = dropoutRate;

        _selfAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: new IdentityActivation<T>());
        _norm1 = new LayerNormalizationLayer<T>(hiddenSize);
        _crossAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: new IdentityActivation<T>());
        _norm2 = new LayerNormalizationLayer<T>(hiddenSize);
        _ffnUp = new DenseLayer<T>(ffnDim, new ReLUActivation<T>() as IActivationFunction<T>);
        _ffnDown = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _norm3 = new LayerNormalizationLayer<T>(hiddenSize);
        if (dropoutRate > 0)
        {
            _selfDropout = new DropoutLayer<T>(dropoutRate);
            _crossDropout = new DropoutLayer<T>(dropoutRate);
            _ffnDropout = new DropoutLayer<T>(dropoutRate);
        }

        RegisterSubLayer(_selfAttention);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_crossAttention);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_ffnUp);
        RegisterSubLayer(_ffnDown);
        RegisterSubLayer(_norm3);
        if (_selfDropout is not null) RegisterSubLayer(_selfDropout);
        if (_crossDropout is not null) RegisterSubLayer(_crossDropout);
        if (_ffnDropout is not null) RegisterSubLayer(_ffnDropout);
    }

    /// <summary>Model (feature) dimension — persisted for deserialization.</summary>
    public int HiddenSize => _hiddenSize;
    /// <summary>Number of attention heads — persisted for deserialization.</summary>
    public int NumHeads => _numHeads;
    /// <summary>Feed-forward inner dimension — persisted for deserialization.</summary>
    public int FfnDim => _ffnDim;
    /// <summary>Dropout probability — persisted for deserialization.</summary>
    public double DropoutRate => _dropoutRate;

    /// <summary>Pre-LN forward pass; all ops route through Engine/sublayers so the tape records them.</summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Sublayer 1: self-attention + Pre-norm residual.
        var s = _selfAttention.Forward(_norm1.Forward(input));
        if (_selfDropout is not null) s = _selfDropout.Forward(s);
        var afterSelf = Engine.TensorAdd(input, s);

        // Sublayer 2: cross-attention + Pre-norm residual.
        var c = _crossAttention.Forward(_norm2.Forward(afterSelf));
        if (_crossDropout is not null) c = _crossDropout.Forward(c);
        var afterCross = Engine.TensorAdd(afterSelf, c);

        // Sublayer 3: FFN + Pre-norm residual. DenseLayer needs 2D [N, hiddenSize].
        int rank = afterCross.Shape.Length;
        int featureDim = afterCross.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= afterCross.Shape[i];
        var normed3 = _norm3.Forward(afterCross);
        var normed3Flat = Engine.Reshape(normed3, new[] { flatN, featureDim });
        var ffnUpOut = _ffnUp.Forward(normed3Flat);
        var ffnDownOut = _ffnDown.Forward(ffnUpOut);
        var ffnReshaped = Engine.Reshape(ffnDownOut, afterCross._shape);
        if (_ffnDropout is not null) ffnReshaped = _ffnDropout.Forward(ffnReshaped);
        return Engine.TensorAdd(afterCross, ffnReshaped);
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _selfAttention.SetTrainingMode(isTraining);
        _norm1.SetTrainingMode(isTraining);
        _crossAttention.SetTrainingMode(isTraining);
        _norm2.SetTrainingMode(isTraining);
        _ffnUp.SetTrainingMode(isTraining);
        _ffnDown.SetTrainingMode(isTraining);
        _norm3.SetTrainingMode(isTraining);
        _selfDropout?.SetTrainingMode(isTraining);
        _crossDropout?.SetTrainingMode(isTraining);
        _ffnDropout?.SetTrainingMode(isTraining);
    }

    private LayerBase<T>[] Subs => new LayerBase<T>[]
        { _selfAttention, _norm1, _crossAttention, _norm2, _ffnUp, _ffnDown, _norm3 };

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get { long n = 0; foreach (var l in Subs) n += l.ParameterCount; return n; }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var result = Vector<T>.Empty();
        foreach (var l in Subs) result = Vector<T>.Concatenate(result, l.GetParameters());
        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            MaterializeLazySublayers();
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");
        int offset = 0;
        foreach (var l in Subs)
        {
            int count = (int)l.ParameterCount;
            if (count == 0) continue;
            l.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    private void MaterializeLazySublayers()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try { _ = Forward(new Tensor<T>(new[] { 1, 2, _hiddenSize })); ResetState(); }
        finally { SetTrainingMode(wasTraining); }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var result = Vector<T>.Empty();
        foreach (var l in Subs) result = Vector<T>.Concatenate(result, l.GetParameterGradients());
        return result;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var l in Subs) l.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var l in Subs) l.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var l in Subs) l.ResetState();
        _selfDropout?.ResetState();
        _crossDropout?.ResetState();
        _ffnDropout?.ResetState();
    }

    /// <summary>Persists ctor params for deserialization (no (int[] inputShape) ctor).</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenSize"] = _hiddenSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["DropoutRate"] = _dropoutRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
