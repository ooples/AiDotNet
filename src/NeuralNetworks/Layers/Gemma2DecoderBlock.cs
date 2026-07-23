using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Gemma-2 decoder block with <em>sandwiched</em> RMSNorms — a norm both before and after each sublayer:
/// <c>x = x + postAttnNorm(Attn(inputNorm(x)))</c> then <c>x = x + postFfnNorm(GeGLU(preFfnNorm(x)))</c>.
/// The FFN is a gated GeGLU (tanh-GELU gate); RMSNorms use the Gemma <c>(1 + weight)</c> convention (applied
/// at load time).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class Gemma2DecoderBlock<T> : LayerBase<T>
{
    private readonly RMSNormalizationLayer<T> _normInput;
    private readonly LayerBase<T> _attention;
    private readonly RMSNormalizationLayer<T> _normPostAttn;
    private readonly RMSNormalizationLayer<T> _normPreFfn;
    private readonly DenseLayer<T> _ffnGate;
    private readonly DenseLayer<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;
    private readonly RMSNormalizationLayer<T> _normPostFfn;
    private readonly int _hiddenSize;

    public override bool SupportsTraining => false;

    /// <summary>Pre-attention RMSNorm.</summary>
    public RMSNormalizationLayer<T> NormInput => _normInput;

    /// <summary>Post-attention RMSNorm (normalizes the attention output before the residual add).</summary>
    public RMSNormalizationLayer<T> NormPostAttn => _normPostAttn;

    /// <summary>Pre-FFN RMSNorm.</summary>
    public RMSNormalizationLayer<T> NormPreFfn => _normPreFfn;

    /// <summary>Post-FFN RMSNorm (normalizes the FFN output before the residual add).</summary>
    public RMSNormalizationLayer<T> NormPostFfn => _normPostFfn;

    /// <summary>The self-attention sublayer.</summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>The GeGLU gate projection.</summary>
    public DenseLayer<T> FfnGate => _ffnGate;

    /// <summary>The GeGLU up (value) projection.</summary>
    public DenseLayer<T> FfnUp => _ffnUp;

    /// <summary>The FFN down projection.</summary>
    public DenseLayer<T> FfnDown => _ffnDown;

    /// <summary>The model (input/output) feature dimension.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Creates a Gemma-2 decoder block.</summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="ffnDim">FFN inner dimension.</param>
    /// <param name="attention">Pre-constructed self-attention sublayer.</param>
    /// <param name="rmsNormEpsilon">RMSNorm epsilon.</param>
    public Gemma2DecoderBlock(int hiddenSize, int ffnDim, LayerBase<T> attention, double rmsNormEpsilon = 1e-6)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        Guard.NotNull(attention);
        _hiddenSize = hiddenSize;
        _attention = attention;
        _normInput = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);
        _normPostAttn = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);
        _normPreFfn = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);
        _normPostFfn = new RMSNormalizationLayer<T>(hiddenSize, rmsNormEpsilon);

        // GeGLU: activation on the gate path (tanh-GELU), linear up path, bias-free.
        _ffnGate = new DenseLayer<T>(ffnDim, activationFunction: new GELUActivation<T>());
        _ffnUp = new DenseLayer<T>(ffnDim, activationFunction: new IdentityActivation<T>());
        _ffnDown = new DenseLayer<T>(hiddenSize, activationFunction: new IdentityActivation<T>());

        foreach (var l in SubLayers()) RegisterSubLayer(l);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Attention sub-block: x = x + postAttnNorm(Attn(inputNorm(x))).
        var normed = _normInput.Forward(input);
        var attnOut = _attention.Forward(normed);
        var attnNormed = _normPostAttn.Forward(attnOut);
        var afterAttn = Engine.TensorAdd(input, attnNormed);

        // FFN sub-block: x = x + postFfnNorm(GeGLU(preFfnNorm(x))).
        var preFfn = _normPreFfn.Forward(afterAttn);

        int rank = afterAttn.Shape.Length;
        int featureDim = afterAttn.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= afterAttn.Shape[i];

        var flat = Engine.Reshape(preFfn, new[] { flatN, featureDim });
        var g = _ffnGate.Forward(flat);
        var u = _ffnUp.Forward(flat);
        var prod = Engine.TensorMultiply(g, u);
        var down = _ffnDown.Forward(prod);
        var downReshaped = Engine.Reshape(down, afterAttn._shape);
        var ffnNormed = _normPostFfn.Forward(downReshaped);

        return Engine.TensorAdd(afterAttn, ffnNormed);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _normInput;
        yield return _attention;
        yield return _normPostAttn;
        yield return _normPreFfn;
        yield return _ffnGate;
        yield return _ffnUp;
        yield return _ffnDown;
        yield return _normPostFfn;
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
