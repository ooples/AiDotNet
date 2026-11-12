using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// MAE (Masked Autoencoder) Reconstruction Loss for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MAE loss measures how well the model reconstructs
/// masked patches of an image. Only the masked patches contribute to the loss,
/// making it efficient and focused on learning useful representations.</para>
///
/// <para><b>Key insight:</b> By masking a large portion (75%) of patches and
/// reconstructing only those patches, the model learns rich visual representations
/// without requiring contrastive learning or negative samples.</para>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = (1/M) * Σ_{i∈masked} ||x_i - x̂_i||²
/// </code>
/// <para>where M is the number of masked patches.</para>
///
/// <para><b>Reference:</b> He et al., "Masked Autoencoders Are Scalable Vision Learners"
/// (CVPR 2022)</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Restoration)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Masked Autoencoders Are Scalable Vision Learners", "https://arxiv.org/abs/2111.06377", Year = 2022, Authors = "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick")]
public class MAEReconstructionLoss<T> : ContrastiveLossBase<T>
{


    // Engine and NumOps come from ContrastiveLossBase; redeclaring them here shadows the base
    // members rather than adding anything.

    private readonly bool _normalize;
    private readonly bool _perPatchNormalization;

    /// <summary>
    /// Initializes a new instance of the MAEReconstructionLoss class.
    /// </summary>
    /// <param name="normalize">Whether to normalize the loss by patch dimension (default: true).</param>
    /// <param name="perPatchNormalization">Whether to normalize each patch individually (default: true).</param>
    public MAEReconstructionLoss(bool normalize = true, bool perPatchNormalization = true)
    {
        _normalize = normalize;
        _perPatchNormalization = perPatchNormalization;
    }

    /// <summary>
    /// Computes the reconstruction loss for masked patches.
    /// </summary>
    /// <param name="reconstructed">Reconstructed patches [batch_size, num_patches, patch_dim].</param>
    /// <param name="original">Original patches [batch_size, num_patches, patch_dim].</param>
    /// <param name="mask">Binary mask indicating which patches are masked [batch_size, num_patches].</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> reconstructed, Tensor<T> original, Tensor<T> mask)
    {
        if (reconstructed is null) throw new ArgumentNullException(nameof(reconstructed));
        if (original is null) throw new ArgumentNullException(nameof(original));
        if (mask is null) throw new ArgumentNullException(nameof(mask));

        // Masked patches (mask > 0) and their MSE (optionally normalized targets)
        var masked = Engine.TensorGreaterThan(mask, NumOps.Zero);
        int maskedCount = Engine.TensorCountNonzero(masked);

        if (maskedCount == 0)
            return NumOps.Zero;

        var patchLosses = MaskedPatchLosses(PatchDifferences(reconstructed, original), masked);
        T totalLoss = Engine.TensorSum(patchLosses);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(maskedCount));
    }

    /// <summary>
    /// reconstructed - target for every patch, where the target is the original patch, per-patch
    /// standardized when <c>perPatchNormalization</c> is set.
    /// </summary>
    private Tensor<T> PatchDifferences(Tensor<T> reconstructed, Tensor<T> original)
    {
        var target = _perPatchNormalization ? NormalizePatches(original) : original;
        return Engine.TensorSubtract(reconstructed, target);
    }

    /// <summary>
    /// Per-patch sum of squared differences ([batch, patches]), divided by patchDim when
    /// <c>normalize</c> is set; patches outside the mask are selected to zero.
    /// </summary>
    private Tensor<T> MaskedPatchLosses(Tensor<T> diff, Tensor<T> masked)
    {
        var patchDim = diff.Shape[2];
        var patchLosses = Engine.ReduceSum(Engine.TensorSquare(diff), new[] { 2 }, keepDims: false);
        if (_normalize)
        {
            patchLosses = Engine.TensorDivideScalar(patchLosses, NumOps.FromDouble(patchDim));
        }

        return Engine.TensorWhere(masked, patchLosses, new Tensor<T>(new[] { diff.Shape[0], diff.Shape[1] }));
    }

    /// <summary>
    /// Computes reconstruction loss with gradients for backpropagation.
    /// </summary>
    public (T loss, Tensor<T> gradReconstructed) ComputeLossWithGradients(
        Tensor<T> reconstructed, Tensor<T> original, Tensor<T> mask)
    {
        var batchSize = reconstructed.Shape[0];
        var numPatches = reconstructed.Shape[1];
        var patchDim = reconstructed.Shape[2];

        var masked = Engine.TensorGreaterThan(mask, NumOps.Zero);
        int maskedCount = Engine.TensorCountNonzero(masked);

        if (maskedCount == 0)
            return (NumOps.Zero, new Tensor<T>(new[] { batchSize, numPatches, patchDim }));

        var diff = PatchDifferences(reconstructed, original);
        T totalLoss = Engine.TensorSum(MaskedPatchLosses(diff, masked));

        // Gradient: 2 * (recon - orig) / patchDim on masked patches, zero elsewhere
        var grad = Engine.TensorMultiplyScalar(diff, NumOps.FromDouble(2.0));
        if (_normalize)
        {
            grad = Engine.TensorDivideScalar(grad, NumOps.FromDouble(patchDim));
        }

        var maskedElements = Engine.TensorTile(masked.Reshape(batchSize, numPatches, 1), new[] { 1, 1, patchDim });
        grad = Engine.TensorWhere(maskedElements, grad, new Tensor<T>(new[] { batchSize, numPatches, patchDim }));

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(maskedCount));
        var scale = NumOps.FromDouble(1.0 / maskedCount);

        return (avgLoss, Engine.TensorMultiplyScalar(grad, scale));
    }

    /// <summary>
    /// Computes per-sample reconstruction loss (useful for analysis).
    /// </summary>
    public T[] ComputePerSampleLoss(Tensor<T> reconstructed, Tensor<T> original, Tensor<T> mask)
    {
        var batchSize = reconstructed.Shape[0];

        var masked = Engine.TensorGreaterThan(mask, NumOps.Zero);
        var patchLosses = MaskedPatchLosses(PatchDifferences(reconstructed, original), masked);

        // Mean over each sample's masked patches; zero for a sample with none
        var sampleLoss = Engine.ReduceSum(patchLosses, new[] { 1 }, keepDims: false);
        var sampleMaskedCount = Engine.ReduceSum(masked, new[] { 1 }, keepDims: false);
        var losses = Engine.TensorWhere(sampleMaskedCount,
            Engine.TensorDivide(sampleLoss, sampleMaskedCount), new Tensor<T>(new[] { batchSize }));

        return losses.ToArray();
    }

    /// <summary>
    /// Creates a random mask for patches.
    /// </summary>
    /// <param name="batchSize">Number of samples in batch.</param>
    /// <param name="numPatches">Number of patches per sample.</param>
    /// <param name="maskRatio">Ratio of patches to mask (default: 0.75).</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Binary mask tensor [batch_size, num_patches].</returns>
    public static Tensor<T> CreateRandomMask(
        int batchSize, int numPatches, double maskRatio = 0.75, int? seed = null)
    {
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;
        var mask = new T[batchSize * numPatches];
        var numMasked = (int)(numPatches * maskRatio);

        for (int b = 0; b < batchSize; b++)
        {
            // Shuffle indices
            var indices = Enumerable.Range(0, numPatches).OrderBy(_ => rng.Next()).ToArray();

            // Mark first numMasked as masked
            for (int p = 0; p < numPatches; p++)
            {
                mask[b * numPatches + indices[p]] = p < numMasked
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return new Tensor<T>(mask, [batchSize, numPatches]);
    }

    private Tensor<T> NormalizePatches(Tensor<T> patches)
    {
        // Per-patch zero mean, unit std (population variance + 1e-6) over the last axis
        var patchDim = NumOps.FromDouble(patches.Shape[2]);
        var mean = Engine.TensorDivideScalar(Engine.ReduceSum(patches, new[] { 2 }, keepDims: true), patchDim);
        var centered = Engine.TensorSubtract(patches, mean);
        var variance = Engine.TensorDivideScalar(
            Engine.ReduceSum(Engine.TensorSquare(centered), new[] { 2 }, keepDims: true), patchDim);
        var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-6)));

        return Engine.TensorDivide(centered, std);
    }

    /// <summary>
    /// The differentiable MAE reconstruction objective, built entirely from <c>IEngine</c>
    /// operations, over all positions.
    /// </summary>
    /// <summary>
    /// Differentiable reconstruction error, as the all-ones-mask case of
    /// <see cref="ComputeLoss(Tensor{T}, Tensor{T}, Tensor{T})"/>.
    /// </summary>
    /// <param name="view1">Reconstructed patches, rank-3 <c>[batch, patches, patch_dim]</c>.</param>
    /// <param name="view2">Original patches, same shape.</param>
    /// <remarks>
    /// <para>
    /// THE CONFIGURED TRANSFORMS APPLY HERE TOO. This previously computed a plain elementwise MSE
    /// while claiming to be the all-ones-mask case, so a user who constructed the loss with
    /// <c>perPatchNormalization</c> or <c>normalize</c> and then trained through the two-view
    /// interface silently got neither. Both are the constructor DEFAULTS and per-patch normalized
    /// targets are the MAE paper's default, so the common configuration was the one being discarded.
    /// </para>
    /// <para>
    /// Order matches the masked overload exactly: normalize each ORIGINAL patch to zero mean and
    /// unit standard deviation (same 1e-6 epsilon inside the square root), take the squared error
    /// against the reconstruction, divide each patch by <c>patch_dim</c> when <c>normalize</c> is
    /// set, then average over patches. Built from IEngine ops so it carries tape history; the masked
    /// overload branches on mask values in host code and returns a scalar, which cannot.
    /// </para>
    /// <para>
    /// Rank is VALIDATED rather than inferred. The masked overload requires rank-3, and this path
    /// used to accept any shape and divide by <c>Length</c> -- so a rank-2 caller received a
    /// differently normalized number instead of an error.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when either view is not rank-3.</exception>
    public override Tensor<T> ComputeLoss(Tensor<T> view1, Tensor<T> view2)
    {
        if (view1 is null) throw new ArgumentNullException(nameof(view1));
        if (view2 is null) throw new ArgumentNullException(nameof(view2));

        if (view1.Shape.Length != 3 || view2.Shape.Length != 3)
        {
            throw new ArgumentException(
                "MAE reconstruction loss operates on patch tensors of rank 3 [batch, patches, "
                + $"patch_dim]; got rank {view1.Shape.Length} and {view2.Shape.Length}. The masked "
                + "overload has the same contract.",
                nameof(view1));
        }

        // THE TWO VIEWS MUST AGREE. Every divisor below is taken from view2 while the difference is
        // taken against view1, so mismatched shapes either threw from deep inside the element-wise
        // subtract or -- where broadcasting applied -- silently normalized by the wrong patch count.
        for (int axis = 0; axis < 3; axis++)
        {
            if (view1.Shape[axis] != view2.Shape[axis])
            {
                throw new ArgumentException(
                    "MAE reconstruction loss requires both views to have the same shape; got "
                    + $"[{string.Join(", ", view1.Shape)}] and [{string.Join(", ", view2.Shape)}].",
                    nameof(view2));
            }
        }

        int patchDim = view2.Shape[2];
        int patchCount = view2.Shape[0] * view2.Shape[1];
        int lastAxis = view2.Shape.Length - 1;

        // A ZERO DIVISOR PRODUCES NaN RATHER THAN THROWING for double and float, and that NaN then
        // propagates through backpropagation and destroys every parameter in the model with nothing
        // to show where it began. The masked overload already returns zero for this same degenerate
        // input (no patch scored, so no loss), and these two paths must not diverge.
        if (patchCount == 0 || (_perPatchNormalization && patchDim == 0))
        {
            // A fresh Tensor<T> is zero-filled, which is the value the masked overload returns for
            // this input. Same idiom as SparseCategoricalCrossEntropyLoss.
            return new Tensor<T>(new[] { 1 });
        }

        var target = view2;
        if (_perPatchNormalization)
        {
            // Per-patch zero mean, unit standard deviation -- the same transform NormalizePatches
            // applies element-wise in the masked path, expressed over the last axis so it stays
            // differentiable.
            var mean = Engine.TensorDivideScalar(
                Engine.ReduceSum(target, new[] { lastAxis }, keepDims: true),
                NumOps.FromDouble(patchDim));
            var centered = Engine.TensorSubtract(target, mean);
            var variance = Engine.TensorDivideScalar(
                Engine.ReduceSum(Engine.TensorMultiply(centered, centered), new[] { lastAxis }, keepDims: true),
                NumOps.FromDouble(patchDim));
            var std = Engine.TensorSqrt(Engine.TensorAddScalar(variance, NumOps.FromDouble(1e-6)));
            target = Engine.TensorDivide(centered, std);
        }

        var diff = Engine.TensorSubtract(view1, target);
        var squared = Engine.TensorMultiply(diff, diff);
        var total = Engine.ReduceSum(squared, null, keepDims: false);

        // The masked overload divides each PATCH by patch_dim only when _normalize is set, then
        // averages over the patches it scored. Folding both divisors into one keeps that exactly.
        double divisor = _normalize ? (double)patchCount * patchDim : patchCount;
        return Engine.TensorDivideScalar(total, NumOps.FromDouble(divisor));
    }
}
