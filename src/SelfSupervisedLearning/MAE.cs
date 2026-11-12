using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// MAE: Masked Autoencoder for Self-Supervised Vision Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MAE is a simple yet powerful self-supervised method.
/// It randomly masks a large portion (75%) of image patches, encodes only the visible
/// patches, and trains a decoder to reconstruct the original pixels of the masked patches.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>High masking ratio:</b> 75% of patches are masked (vs ~15% in BERT)</item>
/// <item><b>Asymmetric encoder-decoder:</b> Encoder only sees visible patches</item>
/// <item><b>Efficient training:</b> Encoder processes only 25% of patches</item>
/// <item><b>Reconstruction target:</b> Normalized pixel values of masked patches</item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Input → Patchify → Random Mask → Visible Encoder → Add mask tokens → Decoder → Reconstruct
/// Loss: MSE on masked patches only
/// </code>
///
/// <para><b>Reference:</b> He et al., "Masked Autoencoders Are Scalable Vision Learners"
/// (CVPR 2022)</para>
///
/// <para><b>Best for:</b> Efficient pretraining, generative understanding.</para>
/// <para><b>Pros:</b> Efficient (only encode visible patches), scalable.</para>
/// <para><b>Cons:</b> May require fine-tuning for best downstream performance.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Masked Autoencoders Are Scalable Vision Learners", "https://arxiv.org/abs/2111.06377", Year = 2022, Authors = "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick")]
public partial class MAE<T> : SelfSupervisedLearningMethodBase<T>
{
    private readonly INeuralNetwork<T>? _decoder;
    private readonly int _patchSize;
    private readonly int _numPatches;
    private readonly double _maskRatio;

    /// <inheritdoc />
    public override string Name => "MAE";

    /// <inheritdoc />
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.Generative;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => false;

    /// <summary>
    /// Gets the mask ratio (proportion of patches masked).
    /// </summary>
    public double MaskRatio => _maskRatio;

    /// <summary>
    /// Gets the patch size.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <summary>
    /// Initializes a new instance of the MAE class.
    /// </summary>
    /// <param name="encoder">The encoder (ViT) that processes visible patches.</param>
    /// <param name="decoder">The decoder that reconstructs masked patches.</param>
    /// <param name="patchSize">Size of each patch (default: 16).</param>
    /// <param name="imageSize">Size of input images (default: 224).</param>
    /// <param name="maskRatio">Ratio of patches to mask (default: 0.75).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MAE(
        INeuralNetwork<T> encoder,
        INeuralNetwork<T>? decoder = null,
        int patchSize = 16,
        int imageSize = 224,
        double maskRatio = 0.75,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(encoder, null, config ?? new SelfSupervisedLearningConfig<T>())
    {
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize), "The patch size must be positive.");
        if (maskRatio <= 0 || maskRatio >= 1 || double.IsNaN(maskRatio))
            throw new ArgumentOutOfRangeException(nameof(maskRatio), "The mask ratio must lie strictly between 0 and 1.");
        int patchesPerSide = imageSize / patchSize;
        // MAE masks some patches and encodes the rest, so it needs at least one of each: one patch would leave the
        // encoder nothing to see and the loss a division by zero.
        if (patchesPerSide * patchesPerSide < 2)
            throw new ArgumentException(
                $"An image of {imageSize} split into {patchSize}-pixel patches has {patchesPerSide * patchesPerSide} patch(es); MAE needs at least 2.",
                nameof(imageSize));

        _decoder = decoder;
        _patchSize = patchSize;
        _numPatches = patchesPerSide * patchesPerSide;
        _maskRatio = maskRatio;
    }

    [Scratch]
    private Random? _maskRandom;

    /// <inheritdoc />
    /// <remarks>MAE learns from one view: reconstruction, not a comparison of two views.</remarks>
    protected override (Tensor<T> View1, Tensor<T> View2) CreateViews(
        Tensor<T> batch, SelfSupervisedLearningAugmentationContext<T>? augmentationContext) => (batch, batch);

    /// <inheritdoc />
    /// <remarks>
    /// He et al. 2022, Sec. 3: a random maskRatio of the patches is removed, the encoder sees only the visible ones,
    /// the decoder receives the encoded patches at their positions (mask tokens elsewhere), and the loss is the mean
    /// squared error between the reconstruction and the per-patch-normalized pixels, on the masked patches only.
    /// Without a decoder, the encoder's output must already be a patch (its width equal to the patch size).
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var patches = Patchify(view1);                       // [B, N, P], data
        int batch = patches.Shape[0], numPatches = patches.Shape[1], patchDim = patches.Shape[2];
        if (numPatches < 2)
            throw new ArgumentException(
                $"The images split into {numPatches} patch(es) of {_patchSize} pixels; MAE masks some and encodes the rest, so it needs at least 2.",
                nameof(view1));
        int numMasked = Math.Max(1, Math.Min(numPatches - 1, (int)Math.Round(numPatches * _maskRatio)));
        int numVisible = numPatches - numMasked;
        var rng = _maskRandom ??= _config.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_config.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // One shuffle per sample, in batch order; the first numVisible positions stay visible (kept sorted).
        var visibleIndex = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            var order = Enumerable.Range(0, numPatches).OrderBy(_ => rng.Next()).ToArray();
            visibleIndex[b] = order.Take(numVisible).OrderBy(i => i).ToArray();
        }

        // Visible patches gathered by flat (sample, patch) row: [B, V, P]
        var flatVisible = visibleIndex.SelectMany((indices, b) => indices.Select(i => b * numPatches + i)).ToArray();
        var visible = Engine.TensorIndexSelect(
                patches.Reshape(batch * numPatches, patchDim),
                new Tensor<int>(flatVisible, new[] { flatVisible.Length }), 0)
            .Reshape(batch, numVisible, patchDim);

        // weight[b, n] = 1 for a masked patch: one minus the count of visible one-hots landing on n
        var visibleOneHot = Engine.TensorOneHot<T>(
            new Tensor<int>(visibleIndex.SelectMany(indices => indices).ToArray(), new[] { batch * numVisible }), numPatches);
        var weight = Engine.ScalarMinusTensor(NumOps.One,
            Engine.ReduceSum(visibleOneHot.Reshape(batch, numVisible, numPatches), new[] { 1 }, keepDims: false));

        var encoded = EncodeForTraining(visible);            // [B, V, E]
        int embed = encoded.Shape[encoded.Rank - 1];

        // Scatter the visible encodings back to their positions with a constant selection matrix per sample, so the
        // tape carries the gradient to the encoder. With a decoder, masked positions are the (zero) mask token and the
        // decoder predicts them. Without one, a masked patch is predicted as the mean of the sample's visible
        // encodings, the parameter-free reconstruction this class documents; otherwise nothing would link a masked
        // patch to the encoder and the loss could never fall.
        var rows = new List<Tensor<T>>(batch);
        var share = NumOps.FromDouble(1.0 / numVisible);
        for (int b = 0; b < batch; b++)
        {
            // select[visibleIndex[b][v], v] = 1
            var select = Engine.TensorTranspose(Engine.TensorOneHot<T>(
                new Tensor<int>(visibleIndex[b], new[] { numVisible }), numPatches));
            if (_decoder is null)
            {
                // Masked rows share the visible encodings equally
                var maskedRows = Engine.TensorTile(
                    Engine.TensorSlice(weight, new[] { b, 0 }, new[] { 1, numPatches }).Reshape(numPatches, 1),
                    new[] { 1, numVisible });
                var shareRows = Engine.TensorAddScalar(new Tensor<T>(new[] { numPatches, numVisible }), share);
                select = Engine.TensorWhere(maskedRows, shareRows, select);
            }
            var sample = Engine.Reshape(Engine.TensorSlice(encoded, new[] { b, 0, 0 }, new[] { 1, numVisible, embed }),
                new[] { numVisible, embed });
            rows.Add(Engine.Reshape(Engine.TensorMatMul(select, sample), new[] { 1, numPatches, embed }));
        }
        var decoderInput = Engine.Concat(rows.ToArray(), 0); // [B, N, E]

        Tensor<T> reconstruction;
        if (_decoder is not null)
            reconstruction = EncodeForTraining(_decoder, decoderInput);
        else if (embed == patchDim)
            reconstruction = decoderInput;
        else
            throw new InvalidOperationException(
                $"MAE without a decoder needs the encoder to output a patch ({patchDim} values); it outputs {embed}.");

        // Per-patch normalized target (Sec. 3, "normalized pixels"): (x - mean) / sqrt(var + 1e-6) over each patch.
        var patchDimT = NumOps.FromDouble(patchDim);
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(patches, new[] { 2 }, keepDims: true), patchDimT);
        var centered = Engine.TensorSubtract(patches, mean);
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSquare(centered), new[] { 2 }, keepDims: true), patchDimT);
        var target = Engine.TensorDivide(centered,
            Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-6))));

        var diff = Engine.TensorSubtract(reconstruction, target);
        var perPatch = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorMultiply(diff, diff), new[] { 2 }, keepDims: false), NumOps.FromDouble(patchDim));
        var maskedSum = Engine.ReduceSum(Engine.TensorMultiply(perPatch, weight), null, keepDims: false);
        return Engine.TensorDivideScalar(maskedSum, NumOps.FromDouble(batch * numMasked));
    }

    /// <inheritdoc />
    /// <remarks>The decoder trains with the encoder.</remarks>
    protected override IReadOnlyList<Tensor<T>> GetOnlineTrainableTensors()
    {
        var tensors = new List<Tensor<T>>(base.GetOnlineTrainableTensors());
        if (_decoder is not null) tensors.AddRange(EncoderTrainableTensors(_decoder));
        return tensors;
    }

    /// <inheritdoc />
    /// <remarks>The paper trains with AdamW (Table 8).</remarks>
    protected override SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.AdamW;

    private Tensor<T> Patchify(Tensor<T> images)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var batchSize = images.Shape[0];
        var numChannels = images.Shape.Length > 1 ? images.Shape[1] : 3;
        var height = images.Shape.Length > 2 ? images.Shape[2] : (int)Math.Sqrt(_numPatches) * _patchSize;
        var width = images.Shape.Length > 3 ? images.Shape[3] : height;

        var patchDim = _patchSize * _patchSize * numChannels;
        var patchesPerRow = width / _patchSize;
        var patchesPerCol = height / _patchSize;
        var numPatchesActual = patchesPerRow * patchesPerCol;
        var cropHeight = patchesPerCol * _patchSize;
        var cropWidth = patchesPerRow * _patchSize;

        // Nothing to read: master's result was the zero-filled patch tensor
        if (batchSize == 0 || numPatchesActual == 0 || patchDim == 0)
        {
            return new Tensor<T>(new[] { batchSize, numPatchesActual, patchDim });
        }

        // Master read images[b, c, y, x], which throws for rank > 4 on the first pixel read; keep that.
        if (images.Shape.Length > 4)
        {
            _ = images[0, 0, 0, 0];
        }

        // Pixel grid [B, C, cropHeight, cropWidth] that the patches tile, in [C, H, W] order
        Tensor<T> grid;
        if (images.Shape.Length >= 4)
        {
            grid = Engine.TensorSlice(images, new[] { 0, 0, 0, 0 },
                new[] { batchSize, numChannels, cropHeight, cropWidth });
        }
        else if (images.Shape.Length == 2)
        {
            // Flat input: pixel (c, y, x) is feature c * height * width + y * width + x, zero past the end
            var needed = numChannels * height * width;
            var available = Math.Min(needed, images.Shape[1]);
            var flat = Engine.TensorSlice(images, new[] { 0, 0 }, new[] { batchSize, available });
            if (available < needed)
            {
                flat = Engine.TensorConcatenate(
                    new[] { flat, new Tensor<T>(new[] { batchSize, needed - available }) }, 1);
            }

            grid = Engine.TensorSlice(flat.Contiguous().Reshape(batchSize, numChannels, height, width),
                new[] { 0, 0, 0, 0 }, new[] { batchSize, numChannels, cropHeight, cropWidth });
        }
        else
        {
            return new Tensor<T>(new[] { batchSize, numPatchesActual, patchDim });
        }

        // [B, C, rows, p, cols, p] -> [B, rows, cols, C, p, p] -> [B, patches, C * p * p]
        var blocks = grid.Contiguous().Reshape(
            batchSize, numChannels, patchesPerCol, _patchSize, patchesPerRow, _patchSize);
        return Engine.TensorPermute(blocks, new[] { 0, 2, 4, 1, 3, 5 })
            .Contiguous()
            .Reshape(batchSize, numPatchesActual, patchDim);
    }

    /// <summary>
    /// Creates an MAE instance with default configuration.
    /// </summary>
    public static MAE<T> Create(
        INeuralNetwork<T> encoder,
        INeuralNetwork<T>? decoder = null,
        int patchSize = 16,
        int imageSize = 224,
        double maskRatio = 0.75)
    {
        return new MAE<T>(encoder, decoder, patchSize, imageSize, maskRatio);
    

}
}
