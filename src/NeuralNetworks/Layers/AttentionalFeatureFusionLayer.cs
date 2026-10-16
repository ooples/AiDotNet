using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Attentional feature fusion, AFF (Dai et al., "Attentional Feature Fusion", WACV 2021), in its 2-D form: merges
/// two feature maps with a per-position, per-channel soft selection computed from both.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// For maps <c>x</c> and <c>y</c> of shape <c>[batch, C, H, W]</c>, with <c>s = x + y</c>:
/// <list type="bullet">
/// <item>local context <c>L(s)</c> = 1×1 conv (C → C/r), BatchNorm, ReLU, 1×1 conv (C/r → C), BatchNorm;</item>
/// <item>global context <c>G(s)</c> = the same stack applied to the spatial average of <c>s</c>;</item>
/// <item><c>w = sigmoid(L(s) + G(s))</c> and the output is <c>2·x·w + 2·y·(1 − w)</c>.</item>
/// </list>
/// This is the fusion module LAION-CLAP uses to merge the global and local views of a long clip (Wu et al. 2023
/// §3.3; reference <c>feature_fusion.AFF</c> with <c>type='2D'</c>, <c>r = 4</c>). Like the reference, a batch of
/// one is duplicated while training, so BatchNorm sees two identical samples rather than refusing a single one.
/// </para>
/// <para>Call <c>Forward(x, y)</c>. A single input is fused with itself, which yields <c>2·x</c>.</para>
/// <para><b>For Beginners:</b> Two pictures of the same thing (a summary and a close-up) are blended pixel by
/// pixel, and the blend weights are learned from both pictures, so the network decides where to trust which.</para>
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureFusion)]
// Dual-input: fusing a map with itself is 2x for every weight setting, so only two distinct maps exercise the
// parameters.
[LayerProperty(IsTrainable = true, HasTrainingMode = true, ApiShape = LayerApiShape.DualTensor, TestInputShape = "2, 8, 4, 4", TestConstructorArgs = "8, 4")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class AttentionalFeatureFusionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _reduction;
    private readonly int _innerChannels;

    // Every child is declared so the whole module is sized at construction: the fusion branch runs only for long
    // clips, and a child that materialized on its first use would change the parameter count mid-training.
    [SubLayerInput("_channels, 1, 1")]
    private readonly ConvolutionalLayer<T> _localReduce;
    [SubLayerInput("_innerChannels")]
    private readonly BatchNormalizationLayer<T> _localReduceNorm;
    [SubLayerInput("_innerChannels, 1, 1")]
    private readonly ConvolutionalLayer<T> _localExpand;
    [SubLayerInput("_channels")]
    private readonly BatchNormalizationLayer<T> _localExpandNorm;
    [SubLayerInput("_channels, 1, 1")]
    private readonly ConvolutionalLayer<T> _globalReduce;
    [SubLayerInput("_innerChannels")]
    private readonly BatchNormalizationLayer<T> _globalReduceNorm;
    [SubLayerInput("_innerChannels, 1, 1")]
    private readonly ConvolutionalLayer<T> _globalExpand;
    [SubLayerInput("_channels")]
    private readonly BatchNormalizationLayer<T> _globalExpandNorm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the fusion module.</summary>
    /// <param name="channels">Channels of both inputs.</param>
    /// <param name="reduction">Bottleneck ratio <c>r</c> of the context branches (reference: 4).</param>
    public AttentionalFeatureFusionLayer([LayerState] int channels, [LayerState] int reduction = 4)
        : base(new[] { channels, -1, -1 }, new[] { channels, -1, -1 })
    {
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (reduction <= 0 || channels / reduction <= 0)
            throw new ArgumentOutOfRangeException(nameof(reduction), "channels / reduction must be at least 1.");
        _channels = channels;
        _reduction = reduction;
        _innerChannels = channels / reduction;

        // torch BatchNorm2d: eps 1e-5, running = 0.9 running + 0.1 batch.
        _localReduce = new ConvolutionalLayer<T>(_innerChannels, kernelSize: 1);
        _localReduceNorm = new BatchNormalizationLayer<T>(_innerChannels);
        _localExpand = new ConvolutionalLayer<T>(channels, kernelSize: 1);
        _localExpandNorm = new BatchNormalizationLayer<T>(channels);
        _globalReduce = new ConvolutionalLayer<T>(_innerChannels, kernelSize: 1);
        _globalReduceNorm = new BatchNormalizationLayer<T>(_innerChannels);
        _globalExpand = new ConvolutionalLayer<T>(channels, kernelSize: 1);
        _globalExpandNorm = new BatchNormalizationLayer<T>(channels);
        foreach (var layer in SubLayers()) RegisterSubLayer(layer);
    }

    private IEnumerable<LayerBase<T>> SubLayers()
    {
        yield return _localReduce; yield return _localReduceNorm; yield return _localExpand; yield return _localExpandNorm;
        yield return _globalReduce; yield return _globalReduceNorm; yield return _globalExpand; yield return _globalExpandNorm;
    }

    /// <summary>The local-context branch in reference order: conv, BatchNorm, (ReLU), conv, BatchNorm.</summary>
    internal (ConvolutionalLayer<T> Reduce, BatchNormalizationLayer<T> ReduceNorm, ConvolutionalLayer<T> Expand,
        BatchNormalizationLayer<T> ExpandNorm) LocalBranch => (_localReduce, _localReduceNorm, _localExpand, _localExpandNorm);

    /// <summary>The global-context branch in reference order (after the spatial average).</summary>
    internal (ConvolutionalLayer<T> Reduce, BatchNormalizationLayer<T> ReduceNorm, ConvolutionalLayer<T> Expand,
        BatchNormalizationLayer<T> ExpandNorm) GlobalBranch => (_globalReduce, _globalReduceNorm, _globalExpand, _globalExpandNorm);

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_channels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Fuse(input, input);

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length is < 1 or > 2)
            throw new ArgumentException("AttentionalFeatureFusionLayer takes one or two feature maps.", nameof(inputs));
        return inputs.Length == 1 ? Fuse(inputs[0], inputs[0]) : Fuse(inputs[0], inputs[1]);
    }

    private Tensor<T> Fuse(Tensor<T> x, Tensor<T> residual)
    {
        if (x.Rank != 4 || x.Shape[1] != _channels)
            throw new ArgumentException($"Expected [batch, {_channels}, H, W]; got [{string.Join(", ", x.Shape.ToArray())}].", nameof(x));
        for (int d = 0; d < 4; d++)
            if (x.Shape[d] != residual.Shape[d])
                throw new ArgumentException("Both feature maps must have the same shape.", nameof(residual));

        int batch = x.Shape[0], height = x.Shape[2], width = x.Shape[3];
        var sum = Engine.TensorAdd(x, residual);
        bool duplicated = IsTrainingMode && batch == 1;
        if (duplicated) sum = Engine.TensorConcatenate(new[] { sum, sum }, 0);

        var local = _localExpandNorm.Forward(_localExpand.Forward(
            Engine.ReLU(_localReduceNorm.Forward(_localReduce.Forward(sum)))));
        var pooled = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(sum, new[] { 2, 3 }, keepDims: true), NumOps.FromDouble(1.0 / ((double)height * width)));
        var global = _globalExpandNorm.Forward(_globalExpand.Forward(
            Engine.ReLU(_globalReduceNorm.Forward(_globalReduce.Forward(pooled)))));

        var logits = Engine.TensorAdd(local, Engine.TensorBroadcastTo(global, local._shape));
        if (duplicated)
            logits = Engine.TensorSlice(logits, new[] { 0, 0, 0, 0 }, new[] { 1, _channels, height, width });
        var weight = Engine.Sigmoid(logits);

        // 2·x·w + 2·y·(1 − w) = 2·(y + w·(x − y))
        var blended = Engine.TensorAdd(residual, Engine.TensorMultiply(weight, Engine.TensorSubtract(x, residual)));
        return Engine.TensorMultiplyScalar(blended, NumOps.FromDouble(2.0));
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Reduction"] = _reduction.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { foreach (var layer in SubLayers()) layer.ResetState(); }
}
