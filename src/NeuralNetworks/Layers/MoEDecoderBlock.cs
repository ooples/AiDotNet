using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Pre-LN decoder block whose feed-forward sublayer is a sparse mixture-of-experts (Mixtral-style):
/// <c>y = x + Attn(RMSNorm(x))</c>, then <c>z = y + MoE(RMSNorm(y))</c>. Mirrors
/// <see cref="PreLNTransformerBlock{T}"/> but routes each token through the top-k of many gated experts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class MoEDecoderBlock<T> : LayerBase<T>
{
    private readonly RMSNormalizationLayer<T> _norm1;
    private readonly LayerBase<T> _attention;
    private readonly RMSNormalizationLayer<T> _norm2;
    private readonly MoEFeedForwardLayer<T> _moe;
    private readonly int _hiddenSize;

    public override bool SupportsTraining => false;

    /// <summary>The pre-attention RMSNorm.</summary>
    public RMSNormalizationLayer<T> Norm1 => _norm1;

    /// <summary>The self-attention sublayer.</summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>The pre-MoE RMSNorm.</summary>
    public RMSNormalizationLayer<T> Norm2 => _norm2;

    /// <summary>The mixture-of-experts feed-forward sublayer.</summary>
    public MoEFeedForwardLayer<T> Moe => _moe;

    /// <summary>The model (input/output) feature dimension.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Creates a MoE pre-LN decoder block.</summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="attention">Pre-constructed self-attention sublayer.</param>
    /// <param name="moe">Pre-constructed mixture-of-experts feed-forward sublayer.</param>
    /// <param name="rmsNormEpsilon">RMSNorm epsilon.</param>
    public MoEDecoderBlock(int hiddenSize, LayerBase<T> attention, MoEFeedForwardLayer<T> moe, double rmsNormEpsilon = 1e-6)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        Guard.NotNull(attention);
        Guard.NotNull(moe);
        _hiddenSize = hiddenSize;
        _attention = attention;
        _moe = moe;
        _norm1 = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);
        _norm2 = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);

        RegisterSubLayer(_norm1);
        RegisterSubLayer(_attention);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_moe);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var normed1 = _norm1.Forward(input);
        var attnOut = _attention.Forward(normed1);
        var afterAttn = Engine.TensorAdd(input, attnOut);

        var normed2 = _norm2.Forward(afterAttn);
        var moeOut = _moe.Forward(normed2);
        return Engine.TensorAdd(afterAttn, moeOut);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _norm1;
        yield return _attention;
        yield return _norm2;
        yield return _moe;
    }

    /// <inheritdoc/>
    public override long ParameterCount
    {
        get { long total = 0; foreach (var l in SubLayers()) total += l.ParameterCount; return total; }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var l in SubLayers()) acc = Vector<T>.Concatenate(acc, l.GetParameters());
        return acc;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new System.ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");
        int offset = 0;
        foreach (var l in SubLayers())
        {
            int count = (int)l.ParameterCount;
            if (count == 0) continue;
            l.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        Vector<T> acc = new Vector<T>(0);
        foreach (var l in SubLayers()) acc = Vector<T>.Concatenate(acc, l.GetParameterGradients());
        return acc;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var l in SubLayers()) l.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var l in SubLayers()) l.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var l in SubLayers()) l.ResetState();
    }
}
