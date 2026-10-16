using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// HTS-AT's patch embedding with LAION-CLAP's 2-D feature fusion (Wu et al. 2023 §3.3; reference
/// <c>htsat.PatchEmbed</c> with <c>enable_fusion=True, fusion_type='aff_2d'</c>).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The input is four spectrogram images <c>[batch, 4, S, S]</c>: channel 0 is the global view (the whole clip
/// shrunk to the 10-second budget) and channels 1-3 are three 10-second local crops from the front, middle and
/// back of the clip. The global view goes through the ordinary patch projection, a <c>p × p</c> stride-<c>p</c>
/// convolution. For a clip that is longer than the budget, each local crop goes through a <c>p × 3p</c>
/// stride-<c>(p, 3p)</c> convolution, the three results are laid side by side along time, zero-padded (or cut) to
/// the global map's width, and merged into the global map by attentional feature fusion
/// (<see cref="AttentionalFeatureFusionLayer{T}"/>). The map is then flattened to tokens and LayerNorm'd.
/// </para>
/// <para>
/// Which samples are "longer" is the optional second input, a <c>[batch]</c> tensor of 0/1. Samples that are not
/// longer keep their global map untouched. As in the reference, a training batch with no longer sample marks one
/// random sample as longer so the fusion branch still receives a gradient.
/// </para>
/// <para><b>For Beginners:</b> A long recording is seen twice: once squeezed into the usual length (the big
/// picture) and once as three full-detail excerpts. The layer cuts both into patches and blends the detail into
/// the big picture where it helps.</para>
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, HasTrainingMode = true, TestInputShape = "1, 4, 16, 16", TestConstructorArgs = "4, 8")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class HtsatFusionPatchEmbeddingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _patchSize;
    private readonly int _embedDim;
    private long _forwardCounter;

    [SubLayerInput("1, _patchSize, _patchSize")]
    private readonly ConvolutionalLayer<T> _projection;
    [SubLayerInput("_patchSize * _patchSize * 3")]
    private readonly DenseLayer<T> _localProjection;
    [SubLayerInput("_embedDim, 1, 1")]
    private readonly AttentionalFeatureFusionLayer<T> _fusion;
    [SubLayerInput("_embedDim")]
    private readonly LayerNormalizationLayer<T> _norm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the fused patch embedding. HTS-AT-tiny: patch 4, embedding 96.</summary>
    /// <param name="patchSize">Patch side <c>p</c> (equal to the stride).</param>
    /// <param name="embedDim">Embedding width.</param>
    public HtsatFusionPatchEmbeddingLayer([LayerState] int patchSize, [LayerState] int embedDim)
        : base(new[] { 4, -1, -1 }, new[] { embedDim })
    {
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize));
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        _patchSize = patchSize;
        _embedDim = embedDim;
        _projection = new ConvolutionalLayer<T>(embedDim, kernelSize: patchSize, stride: patchSize, padding: 0);
        // mel_conv2d: a (p, 3p) kernel at stride (p, 3p) never overlaps, so it is exactly a dense map of each
        // flattened p x 3p patch (weight [E, 1, p, 3p] read row-major).
        _localProjection = new DenseLayer<T>(embedDim, (IActivationFunction<T>)new IdentityActivation<T>());
        _fusion = new AttentionalFeatureFusionLayer<T>(embedDim, 4);
        _norm = new LayerNormalizationLayer<T>(embedDim);
        RegisterSubLayer(_projection);
        RegisterSubLayer(_localProjection);
        RegisterSubLayer(_fusion);
        RegisterSubLayer(_norm);
    }

    /// <summary>The global view's <c>p × p</c> projection (reference <c>proj</c>).</summary>
    internal ConvolutionalLayer<T> Projection => _projection;

    /// <summary>The local crops' <c>p × 3p</c> projection (reference <c>mel_conv2d</c>).</summary>
    internal DenseLayer<T> LocalProjection => _localProjection;

    /// <summary>The fusion module (reference <c>fusion_model</c>).</summary>
    internal AttentionalFeatureFusionLayer<T> Fusion => _fusion;

    /// <summary>The token LayerNorm (reference <c>norm</c>).</summary>
    internal LayerNormalizationLayer<T> Norm => _norm;

    /// <inheritdoc/>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 4
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(
                TensorAxis.Time,
                AxisRelation.ProductOf(
                    AxisRelation.Window(TensorAxis.Height, kernel: _patchSize, stride: _patchSize, padding: 0),
                    AxisRelation.Window(TensorAxis.Width, kernel: _patchSize, stride: _patchSize, padding: 0))),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_embedDim)),
        }
        : null;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Embed(input, null);

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTracedMany(params Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length is < 1 or > 2)
            throw new ArgumentException("HtsatFusionPatchEmbeddingLayer takes the images and an optional [batch] longer mask.", nameof(inputs));
        return Embed(inputs[0], inputs.Length == 2 ? inputs[1] : null);
    }

    private Tensor<T> Embed(Tensor<T> images, Tensor<T>? longerMask)
    {
        if (images.Rank != 4 || images.Shape[1] != 4)
            throw new ArgumentException(
                $"Expected the fusion images [batch, 4, S, S]; got [{string.Join(", ", images.Shape.ToArray())}].", nameof(images));
        int batch = images.Shape[0], height = images.Shape[2], width = images.Shape[3];
        if (height % _patchSize != 0 || width % _patchSize != 0)
            throw new ArgumentException(
                $"The image height ({height}) and width ({width}) must each be divisible by the patch size {_patchSize}.", nameof(images));

        var longer = new bool[batch];
        if (longerMask is not null)
        {
            if (longerMask.Length != batch)
                throw new ArgumentException($"The longer mask has {longerMask.Length} entries for a batch of {batch}.", nameof(longerMask));
            for (int b = 0; b < batch; b++) longer[b] = NumOps.ToDouble(longerMask.GetFlat(b)) > 0.5;
        }
        if (IsTrainingMode && Array.TrueForAll(longer, l => !l))
        {
            // htsat.forward: "if self.training: x['longer'][torch.randint(0, B, (1,))] = True".
            long counter = System.Threading.Interlocked.Increment(ref _forwardCounter);
            var rng = RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(unchecked((int)((uint)RandomSeed.Value * 2654435761u ^ (uint)counter)))
                : RandomHelper.CreateSecureRandom();
            longer[rng.Next(batch)] = true;
        }

        var global = _projection.Forward(Engine.TensorSlice(images, new[] { 0, 0, 0, 0 }, new[] { batch, 1, height, width }));
        int gridH = global.Shape[2], gridW = global.Shape[3];

        var longerIndices = new List<int>();
        for (int b = 0; b < batch; b++) if (longer[b]) longerIndices.Add(b);
        if (longerIndices.Count > 0)
        {
            var globalLonger = Gather(global, longerIndices);
            var local = LocalMaps(Gather(images, longerIndices), gridW);
            var fused = _fusion.Forward(globalLonger, local);
            if (longerIndices.Count == batch)
            {
                global = fused;
            }
            else
            {
                var rows = new Tensor<T>[batch];
                int next = 0;
                for (int b = 0; b < batch; b++)
                {
                    rows[b] = longer[b]
                        ? Engine.TensorSlice(fused, new[] { next++, 0, 0, 0 }, new[] { 1, _embedDim, gridH, gridW })
                        : Engine.TensorSlice(global, new[] { b, 0, 0, 0 }, new[] { 1, _embedDim, gridH, gridW });
                }
                global = Engine.TensorConcatenate(rows, 0);
            }
        }

        var tokens = Engine.Reshape(Engine.TensorPermute(global, new[] { 0, 2, 3, 1 }), new[] { batch, gridH * gridW, _embedDim });
        return _norm.Forward(tokens);
    }

    private Tensor<T> Gather(Tensor<T> source, List<int> indices)
    {
        if (indices.Count == source.Shape[0]) return source;
        var index = new Tensor<int>(new[] { indices.Count });
        for (int i = 0; i < indices.Count; i++) index[i] = indices[i];
        return Engine.TensorGather(source, index, axis: 0);
    }

    // The three local crops -> [n, E, gridH, gridW], laid out as the reference's view/permute/flatten/pad.
    private Tensor<T> LocalMaps(Tensor<T> images, int targetWidth)
    {
        int n = images.Shape[0], height = images.Shape[2], width = images.Shape[3];
        int kh = _patchSize, kw = 3 * _patchSize;
        int rows = height / kh, cols = width / kw;
        if (cols == 0)
            throw new ArgumentException($"The image width {width} is narrower than the local kernel width {kw}.", nameof(images));
        var crops = Engine.TensorSlice(images, new[] { 0, 1, 0, 0 }, new[] { n, 3, height, cols * kw });
        var patches = Engine.Reshape(crops, new[] { n, 3, rows, kh, cols, kw });
        patches = Engine.TensorPermute(patches, new[] { 0, 1, 2, 4, 3, 5 });                      // [n, 3, rows, cols, kh, kw]
        var projected = _localProjection.Forward(Engine.Reshape(patches, new[] { n * 3 * rows * cols, kh * kw }));
        var maps = Engine.Reshape(projected, new[] { n, 3, rows, cols, _embedDim });
        maps = Engine.TensorPermute(maps, new[] { 0, 4, 2, 1, 3 });                                // [n, E, rows, 3, cols]
        maps = Engine.Reshape(maps, new[] { n, _embedDim, rows, 3 * cols });
        int laid = 3 * cols;
        if (laid > targetWidth)
            return Engine.TensorSlice(maps, new[] { 0, 0, 0, 0 }, new[] { n, _embedDim, rows, targetWidth });
        if (laid < targetWidth)
            return Engine.TensorConcatenate(new[] { maps, new Tensor<T>(new[] { n, _embedDim, rows, targetWidth - laid }) }, 3);
        return maps;
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["EmbedDim"] = _embedDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _projection.ResetState();
        _localProjection.ResetState();
        _fusion.ResetState();
        _norm.ResetState();
    }
}
