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

        var visible = new Tensor<T>(new[] { batch, numVisible, patchDim });
        var masked = new bool[batch, numPatches];
        var visibleIndex = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            var order = Enumerable.Range(0, numPatches).OrderBy(_ => rng.Next()).ToArray();
            visibleIndex[b] = order.Take(numVisible).OrderBy(i => i).ToArray();
            foreach (int m in order.Skip(numVisible)) masked[b, m] = true;
            for (int v = 0; v < numVisible; v++)
                for (int d = 0; d < patchDim; d++)
                    visible[b, v, d] = patches[b, visibleIndex[b][v], d];
        }

        var encoded = EncodeForTraining(visible);            // [B, V, E]
        int embed = encoded.Shape[encoded.Rank - 1];

        // Scatter the visible encodings back to their positions with a constant selection matrix per sample, so the
        // tape carries the gradient to the encoder. With a decoder, masked positions are the (zero) mask token and the
        // decoder predicts them. Without one, a masked patch is predicted as the mean of the sample's visible
        // encodings, the parameter-free reconstruction this class documents; otherwise nothing would link a masked
        // patch to the encoder and the loss could never fall.
        var rows = new List<Tensor<T>>(batch);
        for (int b = 0; b < batch; b++)
        {
            var select = new Tensor<T>(new[] { numPatches, numVisible });
            for (int v = 0; v < numVisible; v++) select[visibleIndex[b][v], v] = NumOps.One;
            if (_decoder is null)
            {
                var share = NumOps.FromDouble(1.0 / numVisible);
                for (int n = 0; n < numPatches; n++)
                    if (masked[b, n])
                        for (int v = 0; v < numVisible; v++) select[n, v] = share;
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

        // Per-patch normalized target (Sec. 3, "normalized pixels"), and a constant weight selecting masked patches.
        var target = new Tensor<T>(new[] { batch, numPatches, patchDim });
        var weight = new Tensor<T>(new[] { batch, numPatches });
        for (int b = 0; b < batch; b++)
            for (int n = 0; n < numPatches; n++)
            {
                double mean = 0, sq = 0;
                for (int d = 0; d < patchDim; d++) mean += NumOps.ToDouble(patches[b, n, d]);
                mean /= patchDim;
                for (int d = 0; d < patchDim; d++) { double c = NumOps.ToDouble(patches[b, n, d]) - mean; sq += c * c; }
                double std = Math.Sqrt(sq / patchDim + 1e-6);
                for (int d = 0; d < patchDim; d++)
                    target[b, n, d] = NumOps.FromDouble((NumOps.ToDouble(patches[b, n, d]) - mean) / std);
                weight[b, n] = masked[b, n] ? NumOps.One : NumOps.Zero;
            }

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
        var batchSize = images.Shape[0];
        var numChannels = images.Shape.Length > 1 ? images.Shape[1] : 3;
        var height = images.Shape.Length > 2 ? images.Shape[2] : (int)Math.Sqrt(_numPatches) * _patchSize;
        var width = images.Shape.Length > 3 ? images.Shape[3] : height;

        var patchDim = _patchSize * _patchSize * numChannels;
        var patchesPerRow = width / _patchSize;
        var patchesPerCol = height / _patchSize;
        var numPatchesActual = patchesPerRow * patchesPerCol;

        var patches = new T[batchSize * numPatchesActual * patchDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int py = 0; py < patchesPerCol; py++)
            {
                for (int px = 0; px < patchesPerRow; px++)
                {
                    var patchIdx = py * patchesPerRow + px;
                    var startY = py * _patchSize;
                    var startX = px * _patchSize;

                    var patchOffset = (b * numPatchesActual + patchIdx) * patchDim;

                    // Extract patch pixels in [C, H, W] order within patch
                    for (int c = 0; c < numChannels; c++)
                    {
                        for (int dy = 0; dy < _patchSize; dy++)
                        {
                            for (int dx = 0; dx < _patchSize; dx++)
                            {
                                var dimIdx = c * _patchSize * _patchSize + dy * _patchSize + dx;
                                var y = startY + dy;
                                var x = startX + dx;

                                // Handle different input formats [B, C, H, W] or [B, features]
                                if (images.Shape.Length >= 4)
                                {
                                    patches[patchOffset + dimIdx] = images[b, c, y, x];
                                }
                                else if (images.Shape.Length == 2)
                                {
                                    // Flat input: index into the flat dimension
                                    var flatIdx = c * height * width + y * width + x;
                                    if (flatIdx < images.Shape[1])
                                    {
                                        patches[patchOffset + dimIdx] = images[b, flatIdx];
                                    }
                                    else
                                    {
                                        patches[patchOffset + dimIdx] = NumOps.Zero;
                                    }
                                }
                                else
                                {
                                    patches[patchOffset + dimIdx] = NumOps.Zero;
                                }
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(patches, [batchSize, numPatchesActual, patchDim]);
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
