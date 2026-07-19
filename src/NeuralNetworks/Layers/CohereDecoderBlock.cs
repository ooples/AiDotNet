using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Cohere (Command-R) decoder block with a <em>parallel</em> residual: a single LayerNorm feeds both the
/// attention and the gated-SwiGLU FFN, whose outputs are added together to the residual —
/// <c>x = x + Attn(norm(x)) + FFN(norm(x))</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Uses true LayerNorm (mean-centered) rather than RMSNorm, bias-free, matching Command-R. QK-normalization
/// (present in some Command-R+ variants) is not applied.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "")]
public partial class CohereDecoderBlock<T> : LayerBase<T>
{
    private readonly LayerNormalizationLayer<T> _norm;
    private readonly LayerBase<T> _attention;
    private readonly DenseLayer<T> _ffnGate;
    private readonly DenseLayer<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;
    private readonly int _hiddenSize;

    public override bool SupportsTraining => false;

    /// <summary>The single shared LayerNorm feeding both sublayers.</summary>
    public LayerNormalizationLayer<T> Norm => _norm;

    /// <summary>The self-attention sublayer.</summary>
    public LayerBase<T> AttentionLayer => _attention;

    /// <summary>The gated SwiGLU gate projection.</summary>
    public DenseLayer<T> FfnGate => _ffnGate;

    /// <summary>The gated SwiGLU up (value) projection.</summary>
    public DenseLayer<T> FfnUp => _ffnUp;

    /// <summary>The FFN down projection.</summary>
    public DenseLayer<T> FfnDown => _ffnDown;

    /// <summary>The model (input/output) feature dimension.</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Creates a Cohere parallel-residual decoder block.</summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="ffnDim">FFN inner dimension.</param>
    /// <param name="attention">Pre-constructed self-attention sublayer.</param>
    /// <param name="layerNormEpsilon">LayerNorm epsilon.</param>
    public CohereDecoderBlock(int hiddenSize, int ffnDim, LayerBase<T> attention, double layerNormEpsilon = 1e-5)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        Guard.NotNull(attention);
        _hiddenSize = hiddenSize;
        _attention = attention;
        _norm = new LayerNormalizationLayer<T>(hiddenSize, layerNormEpsilon);

        _ffnGate = new DenseLayer<T>(ffnDim, activationFunction: new SiLUActivation<T>());
        _ffnUp = new DenseLayer<T>(ffnDim, activationFunction: new IdentityActivation<T>());
        _ffnDown = new DenseLayer<T>(hiddenSize, activationFunction: new IdentityActivation<T>());

        foreach (var l in SubLayers()) RegisterSubLayer(l);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        var normed = _norm.Forward(input);
        var attnOut = _attention.Forward(normed);

        int rank = input.Shape.Length;
        int featureDim = input.Shape[rank - 1];
        int flatN = 1;
        for (int i = 0; i < rank - 1; i++) flatN *= input.Shape[i];

        var flat = Engine.Reshape(normed, new[] { flatN, featureDim });
        var g = _ffnGate.Forward(flat);
        var u = _ffnUp.Forward(flat);
        var prod = Engine.TensorMultiply(g, u);
        var down = _ffnDown.Forward(prod);
        var ffnOut = Engine.Reshape(down, input._shape);

        // Parallel residual: both sublayers read the SAME normed input and are added to the residual.
        return Engine.TensorAdd(Engine.TensorAdd(input, attnOut), ffnOut);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _norm;
        yield return _attention;
        yield return _ffnGate;
        yield return _ffnUp;
        yield return _ffnDown;
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
