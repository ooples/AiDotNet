using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Cohere (Command-R) decoder block with a <em>parallel</em> residual: a single LayerNorm feeds both the
/// attention and the gated-SwiGLU FFN, whose outputs are added together to the residual â€”
/// <c>x = x + Attn(norm(x)) + FFN(norm(x))</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Uses true LayerNorm (mean-centered) rather than RMSNorm, bias-free, matching Command-R. QK-normalization
/// (present in some Command-R+ variants) is not applied.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = false, HasTrainingMode = false, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 16, new AiDotNet.NeuralNetworks.Layers.MultiHeadAttentionLayer<double>(2, 4)")]
// Shape-preserving by CONSTRUCTION, not by coincidence. The last statement of ForwardTraced is
// "Engine.TensorAdd(Engine.TensorAdd(input, attnOut), ffnOut)" -- a residual add against the untouched
// input -- and the FFN branch is explicitly restored to the input's shape one line earlier
// ("Engine.Reshape(down, input._shape)"). A residual block that resized anything could not add.
//
// Roles are the block's own: the trailing axis is the model width the class calls HiddenSize (the base
// ctor declares [-1, hiddenSize] and ForwardTraced reads "featureDim = input.Shape[rank - 1]"), and the
// axis before it is the decoder's sequence position. Batch is optional because the leading axis is
// absent at rank 2 -- the form the base ctor declares -- and present at rank 3, the form
// [LayerProperty(TestInputShape = "1, 4, 8")] exercises; ForwardTraced flattens every leading axis
// into one before projecting, so both run the same code.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class CohereDecoderBlock<T> : LayerBase<T>, IShapeContract
{
    // Every child reads the block input; only the down-projection reads the expanded width.
    // Chained sizing walked registration order instead and built the second projection from
    // the first's output, so a restore met a differently shaped layer than the checkpoint.
    [SubLayerInput("_hiddenSize")]
    private readonly LayerNormalizationLayer<T> _norm;
    [SubLayerInput("1, _hiddenSize")]
    private readonly LayerBase<T> _attention;
    [SubLayerInput("_hiddenSize")]
    private readonly DenseLayer<T> _ffnGate;
    [SubLayerInput("_hiddenSize")]
    private readonly DenseLayer<T> _ffnUp;
    [SubLayerInput("_ffnDim")]
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

    /// <summary>Construction state: the 'ffnDim' the layer was built with.</summary>
    private readonly int _ffnDim;

    /// <summary>Construction state: the 'layerNormEpsilon' the layer was built with.</summary>
    private readonly double _layerNormEpsilon;

    /// <summary>Creates a Cohere parallel-residual decoder block.</summary>
    /// <param name="hiddenSize">Input/output feature dimension.</param>
    /// <param name="ffnDim">FFN inner dimension.</param>
    /// <param name="attention">Pre-constructed self-attention sublayer.</param>
    /// <param name="layerNormEpsilon">LayerNorm epsilon.</param>
    public CohereDecoderBlock(int hiddenSize, int ffnDim, LayerBase<T> attention, double layerNormEpsilon = 1e-5)
        : base(new[] { -1, hiddenSize }, new[] { -1, hiddenSize })
    {
        _layerNormEpsilon = layerNormEpsilon;
        _ffnDim = ffnDim;
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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
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
