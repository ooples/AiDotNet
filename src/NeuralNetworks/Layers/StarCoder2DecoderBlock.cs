using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// StarCoder2 pre-LN decoder block: <c>y = x + Attn(LayerNorm(x))</c> then <c>z = y + MLP(LayerNorm(y))</c>,
/// where the norms are true (mean-centered) LayerNorms <em>with bias</em>, the attention projections carry
/// biases, and the FFN is a non-gated two-matrix GELU MLP (<c>c_proj(gelu(c_fc(x)))</c>) with biases.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class StarCoder2DecoderBlock<T> : LayerBase<T>
{
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly LayerBase<T> _attention;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DenseLayer<T> _cFc;
    private readonly DenseLayer<T> _cProj;
    private readonly int _hiddenSize;

    public override bool SupportsTraining => false;

    /// <summary>Pre-attention LayerNorm (with bias).</summary>
    public LayerNormalizationLayer<T> Norm1 => _norm1;

    /// <summary>Pre-MLP LayerNorm (with bias).</summary>
    public LayerNormalizationLayer<T> Norm2 => _norm2;

    /// <summary>The self-attention sublayer (with projection biases).</summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>The FFN up projection (<c>c_fc</c>: hidden -&gt; ffnDim, GELU, with bias).</summary>
    public DenseLayer<T> CFc => _cFc;

    /// <summary>The FFN down projection (<c>c_proj</c>: ffnDim -&gt; hidden, with bias).</summary>
    public DenseLayer<T> CProj => _cProj;

    /// <summary>The model (input/output) feature dimension.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Creates a StarCoder2 decoder block.</summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="ffnDim">FFN inner dimension.</param>
    /// <param name="attention">Pre-constructed self-attention sublayer (with projection biases).</param>
    /// <param name="layerNormEpsilon">LayerNorm epsilon.</param>
    public StarCoder2DecoderBlock(int hiddenSize, int ffnDim, LayerBase<T> attention, double layerNormEpsilon = 1e-5)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        Guard.NotNull(attention);
        _hiddenSize = hiddenSize;
        _attention = attention;
        _norm1 = new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon);
        _norm2 = new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon);

        // Non-gated MLP: c_fc (GELU, with bias) then c_proj (linear, with bias).
        _cFc = new DenseLayer<T>(ffnDim, activationFunction: new GELUActivation<T>());
        _cProj = new DenseLayer<T>(hiddenSize, activationFunction: new IdentityActivation<T>());

        foreach (var l in SubLayers()) RegisterSubLayer(l);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var normed1 = _norm1.Forward(input);
        var attnOut = _attention.Forward(normed1);
        var afterAttn = Engine.TensorAdd(input, attnOut);

        var normed2 = _norm2.Forward(afterAttn);

        int rank = afterAttn.Shape.Length;
        int featureDim = afterAttn.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= afterAttn.Shape[i];

        var flat = Engine.Reshape(normed2, new[] { flatN, featureDim });
        var up = _cFc.Forward(flat);          // gelu(c_fc(x))
        var down = _cProj.Forward(up);        // c_proj(...)
        var ffnOut = Engine.Reshape(down, afterAttn._shape);

        return Engine.TensorAdd(afterAttn, ffnOut);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _norm1;
        yield return _attention;
        yield return _norm2;
        yield return _cFc;
        yield return _cProj;
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
