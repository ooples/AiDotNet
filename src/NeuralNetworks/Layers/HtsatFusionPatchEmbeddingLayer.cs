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

        // Rows whose local views are fused: mask > 0.5 (NaN counts as not longer), in ascending row order
        // (torch.nonzero is row-major). How many there are fixes the fusion branch's batch size, so the count is a
        // property of the graph and is read here; WHICH rows they are is computed from the mask by recorded
        // operations (Select), so a compiled plan follows a mask that changes from batch to batch.
        // The count is an ordinary host read of the mask (ToArray), not one hidden from the trace: a mask computed
        // inside the trace has no values until it is read, and a plan that refuses host reads of traced values must
        // see this one.
        Tensor<T>? longer = null;
        int longerCount = 0;
        if (longerMask is not null)
        {
            if (longerMask.Length != batch)
                throw new ArgumentException($"The longer mask has {longerMask.Length} entries for a batch of {batch}.", nameof(longerMask));
            longer = Engine.TensorGreaterThan(Engine.Reshape(longerMask, new[] { batch }), NumOps.FromDouble(0.5));
            var read = new Tensor<T>(longerMask.ToArray(), new[] { batch });
            using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
                longerCount = Engine.TensorCountNonzero(Engine.TensorGreaterThan(read, NumOps.FromDouble(0.5)));
        }

        var global = _projection.Forward(Engine.TensorSlice(images, new[] { 0, 0, 0, 0 }, new[] { batch, 1, height, width }));
        int gridW = global.Shape[3];

        // A forward run only to resolve lazy shapes ahead of a training trace runs in evaluation mode with draws
        // suppressed. It takes the fusion branch too (on row 0, drawing nothing), or the branch's weights would first
        // be created inside the trace, after the plan's parameters were collected.
        if (longerCount == 0 && batch > 0
            && (IsTrainingMode || AiDotNet.Training.CompiledStepRandom<T>.InNonStepForward))
        {
            // htsat.forward: "if self.training: x['longer'][torch.randint(0, B, (1,))] = True". The row is a per-step
            // draw, so a compiled training plan picks a new one on every replay instead of keeping the traced step's.
            var row = AiDotNet.Training.CompiledStepRandom<T>.DrawData<int>(new[] { 1 },
                drawn => drawn.AsWritableSpan()[0] = NextRandomLongerRow(batch));
            var fused = _fusion.Forward(Gather(global, row), LocalMaps(Gather(images, row), gridW));
            global = batch == 1 ? fused : Engine.TensorIndexCopy(global, 0, row, fused);
        }
        else if (longerCount > 0 && longer is not null)
        {
            if (longerCount == batch)
            {
                global = _fusion.Forward(global, LocalMaps(images, gridW));
            }
            else
            {
                var selection = Selection(longer, longerCount, batch);                                      // [k, B]
                var fused = _fusion.Forward(
                    Select(selection, longer, global, longerCount),
                    LocalMaps(Select(selection, longer, images, longerCount), gridW));
                // Fused rows replace their global rows; the rest keep the global map.
                var returned = Engine.Reshape(
                    Engine.TensorMatMul(Engine.TensorTranspose(selection), Engine.Reshape(fused, new[] { longerCount, fused.Length / longerCount })),
                    global._shape);
                global = Engine.TensorWhere(RowMask(longer, global._shape), returned, global);
            }
        }
        int gridH = global.Shape[2];

        var tokens = Engine.Reshape(Engine.TensorPermute(global, new[] { 0, 2, 3, 1 }), new[] { batch, gridH * gridW, _embedDim });
        var embedded = _norm.Forward(tokens);
        if (longer is null || !ElementTypeHasNaN) return embedded;

        // The graph above is built for longerCount fused rows. A compiled plan replays it on later masks, and a mask
        // with a different count would run the fusion branch on the wrong rows without any sign of it (its batch norm
        // sees rows of zeros, or a longer row goes unfused). Such a batch needs a different graph, so its output is
        // NaN: the step fails where it can be seen instead of training on the wrong rows.
        // As a factor of matches / matches (1, or 0 / 0), not a select against a NaN constant: a select computed as a
        // blend would let the constant through on every batch.
        var count = Engine.Reshape(Engine.ReduceSum(longer, new[] { 0 }, keepDims: false), new[] { 1, 1, 1 });
        var matches = Engine.StopGradient(Engine.TensorEquals(count, NumOps.FromDouble(longerCount)));
        return Engine.TensorMultiply(embedded, Engine.TensorBroadcastTo(Engine.TensorDivide(matches, matches), embedded._shape));
    }

    // float, double and Half can carry the NaN that marks a batch the compiled graph was not built for.
    private static readonly bool ElementTypeHasNaN = typeof(T) == typeof(float) || typeof(T) == typeof(double)
        || typeof(T) == typeof(Half);

    // The [k, batch] 0/1 matrix whose row j has its one at the j-th longer row: rank(b) = longer rows up to and
    // including b, so row j is "longer and rank == j + 1". Recorded operations on the mask, in the layer's own element
    // type: integer indices computed inside a compiled trace are not something a gather can be recorded through.
    private Tensor<T> Selection(Tensor<T> longer, int count, int batch)
    {
        var shape = new[] { count, batch };
        var rank = Engine.TensorBroadcastTo(Engine.Reshape(Engine.TensorCumSum(longer, 0), new[] { 1, batch }), shape);
        // The ranks 1..count as a host constant (a linspace needs two points).
        var wanted = new Tensor<T>(new[] { count, 1 });
        if (count == 1)
        {
            wanted.Fill(NumOps.One);
        }
        else
        {
            using (AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording())
                wanted = Engine.Reshape(Engine.TensorLinspace(NumOps.One, NumOps.FromDouble(count), count), new[] { count, 1 });
        }
        var isRank = Engine.TensorEquals(rank, Engine.TensorBroadcastTo(wanted, shape));
        return Engine.TensorMultiply(isRank, Engine.TensorBroadcastTo(Engine.Reshape(longer, new[] { 1, batch }), shape));
    }

    // The longer rows of source, in order: selection x source with every other row zeroed first, so that a non-finite
    // value in a row that is not selected cannot reach the result through 0 x inf.
    private Tensor<T> Select(Tensor<T> selection, Tensor<T> longer, Tensor<T> source, int count)
    {
        int batch = source.Shape[0];
        var kept = Engine.TensorWhere(RowMask(longer, source._shape), source, new Tensor<T>(source._shape));
        var rows = Engine.TensorMatMul(selection, Engine.Reshape(kept, new[] { batch, source.Length / batch }));
        var shape = (int[])source._shape.Clone();
        shape[0] = count;
        return Engine.Reshape(rows, shape);
    }

    // The [batch] 0/1 flags broadcast over shape (whose first axis is the batch), as a condition.
    private Tensor<T> RowMask(Tensor<T> longer, int[] shape)
    {
        var lead = new int[shape.Length];
        lead[0] = shape[0];
        for (int i = 1; i < lead.Length; i++) lead[i] = 1;
        return Engine.TensorGreaterThan(Engine.TensorBroadcastTo(Engine.Reshape(longer, lead), shape), NumOps.Zero);
    }

    // One row of the batch, from this layer's seeded stream: a generator per draw, keyed by the draw's number.
    private int NextRandomLongerRow(int batch)
    {
        long counter = System.Threading.Interlocked.Increment(ref _forwardCounter);
        var rng = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(unchecked((int)((uint)RandomSeed.Value * 2654435761u ^ (uint)counter)))
            : RandomHelper.CreateSecureRandom();
        return rng.Next(batch);
    }

    private Tensor<T> Gather(Tensor<T> source, Tensor<int> indices)
    {
        if (indices.Length == source.Shape[0]) return source;
        return Engine.TensorGather(source, indices, axis: 0);
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
        // A new session restarts the seeded longer-row draw sequence, so a reused layer matches a fresh seeded one.
        System.Threading.Interlocked.Exchange(ref _forwardCounter, 0L);
        _projection.ResetState();
        _localProjection.ResetState();
        _fusion.ResetState();
        _norm.ResetState();
    }
}
