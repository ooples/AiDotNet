using AiDotNet.Helpers;

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
public class MAEReconstructionLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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

        var batchSize = reconstructed.Shape[0];
        var numPatches = reconstructed.Shape[1];
        var patchDim = reconstructed.Shape[2];

        T totalLoss = NumOps.Zero;
        int maskedCount = 0;

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                // Check if this patch is masked
                if (!NumOps.GreaterThan(mask[b, p], NumOps.Zero))
                    continue;

                maskedCount++;

                // Optionally normalize patches
                T[] origPatch = new T[patchDim];
                T[] reconPatch = new T[patchDim];

                for (int d = 0; d < patchDim; d++)
                {
                    origPatch[d] = original[b, p, d];
                    reconPatch[d] = reconstructed[b, p, d];
                }

                if (_perPatchNormalization)
                {
                    NormalizePatch(origPatch);
                }

                // Compute MSE for this patch
                T patchLoss = NumOps.Zero;
                for (int d = 0; d < patchDim; d++)
                {
                    var diff = NumOps.Subtract(reconPatch[d], origPatch[d]);
                    patchLoss = NumOps.Add(patchLoss, NumOps.Multiply(diff, diff));
                }

                if (_normalize)
                {
                    patchLoss = NumOps.Divide(patchLoss, NumOps.FromDouble(patchDim));
                }

                totalLoss = NumOps.Add(totalLoss, patchLoss);
            }
        }

        if (maskedCount == 0)
            return NumOps.Zero;

        return NumOps.Divide(totalLoss, NumOps.FromDouble(maskedCount));
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

        var gradRecon = new T[batchSize * numPatches * patchDim];
        T totalLoss = NumOps.Zero;
        int maskedCount = 0;

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                if (!NumOps.GreaterThan(mask[b, p], NumOps.Zero))
                    continue;

                maskedCount++;

                T[] origPatch = new T[patchDim];
                for (int d = 0; d < patchDim; d++)
                {
                    origPatch[d] = original[b, p, d];
                }

                if (_perPatchNormalization)
                {
                    NormalizePatch(origPatch);
                }

                T patchLoss = NumOps.Zero;
                for (int d = 0; d < patchDim; d++)
                {
                    var diff = NumOps.Subtract(reconstructed[b, p, d], origPatch[d]);
                    patchLoss = NumOps.Add(patchLoss, NumOps.Multiply(diff, diff));

                    // Gradient: 2 * (recon - orig) / patchDim
                    var grad = NumOps.Multiply(NumOps.FromDouble(2.0), diff);
                    if (_normalize)
                    {
                        grad = NumOps.Divide(grad, NumOps.FromDouble(patchDim));
                    }

                    gradRecon[(b * numPatches + p) * patchDim + d] = grad;
                }

                if (_normalize)
                {
                    patchLoss = NumOps.Divide(patchLoss, NumOps.FromDouble(patchDim));
                }

                totalLoss = NumOps.Add(totalLoss, patchLoss);
            }
        }

        if (maskedCount == 0)
            return (NumOps.Zero, new Tensor<T>(gradRecon, [batchSize, numPatches, patchDim]));

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(maskedCount));
        var scale = NumOps.FromDouble(1.0 / maskedCount);

        for (int i = 0; i < gradRecon.Length; i++)
        {
            gradRecon[i] = NumOps.Multiply(gradRecon[i], scale);
        }

        return (avgLoss, new Tensor<T>(gradRecon, [batchSize, numPatches, patchDim]));
    }

    /// <summary>
    /// Computes per-sample reconstruction loss (useful for analysis).
    /// </summary>
    public T[] ComputePerSampleLoss(Tensor<T> reconstructed, Tensor<T> original, Tensor<T> mask)
    {
        var batchSize = reconstructed.Shape[0];
        var numPatches = reconstructed.Shape[1];
        var patchDim = reconstructed.Shape[2];

        var losses = new T[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            T sampleLoss = NumOps.Zero;
            int sampleMaskedCount = 0;

            for (int p = 0; p < numPatches; p++)
            {
                if (!NumOps.GreaterThan(mask[b, p], NumOps.Zero))
                    continue;

                sampleMaskedCount++;

                T[] origPatch = new T[patchDim];
                for (int d = 0; d < patchDim; d++)
                {
                    origPatch[d] = original[b, p, d];
                }

                if (_perPatchNormalization)
                {
                    NormalizePatch(origPatch);
                }

                T patchLoss = NumOps.Zero;
                for (int d = 0; d < patchDim; d++)
                {
                    var diff = NumOps.Subtract(reconstructed[b, p, d], origPatch[d]);
                    patchLoss = NumOps.Add(patchLoss, NumOps.Multiply(diff, diff));
                }

                if (_normalize)
                {
                    patchLoss = NumOps.Divide(patchLoss, NumOps.FromDouble(patchDim));
                }

                sampleLoss = NumOps.Add(sampleLoss, patchLoss);
            }

            losses[b] = sampleMaskedCount > 0
                ? NumOps.Divide(sampleLoss, NumOps.FromDouble(sampleMaskedCount))
                : NumOps.Zero;
        }

        return losses;
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
        var rng = seed.HasValue ? new Random(seed.Value) : RandomHelper.Shared;
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

    private void NormalizePatch(T[] patch)
    {
        // Compute mean
        T mean = NumOps.Zero;
        for (int i = 0; i < patch.Length; i++)
        {
            mean = NumOps.Add(mean, patch[i]);
        }
        mean = NumOps.Divide(mean, NumOps.FromDouble(patch.Length));

        // Compute std
        T variance = NumOps.Zero;
        for (int i = 0; i < patch.Length; i++)
        {
            var diff = NumOps.Subtract(patch[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(patch.Length));
        var std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-6)));

        // Normalize in-place
        for (int i = 0; i < patch.Length; i++)
        {
            patch[i] = NumOps.Divide(NumOps.Subtract(patch[i], mean), std);
        }
    }
}
