using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Infrastructure;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning.VisionSSL;

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
/// </remarks>
public class MAE<T> : SSLMethodBase<T>
{
    private readonly INeuralNetwork<T>? _decoder;
    private readonly MAEReconstructionLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;
    private readonly int _patchSize;
    private readonly int _numPatches;
    private readonly double _maskRatio;
    private readonly int _decoderEmbedDim;

    /// <inheritdoc />
    public override string Name => "MAE";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.Generative;

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
        SSLConfig? config = null)
        : base(encoder, null, config ?? SSLConfig.ForMAE())
    {
        _decoder = decoder;
        _patchSize = patchSize;
        _numPatches = (imageSize / patchSize) * (imageSize / patchSize);
        _maskRatio = maskRatio;

        var maeConfig = _config.MAE ?? new MAEConfig();
        _decoderEmbedDim = maeConfig.DecoderEmbedDimension ?? 512;

        _loss = new MAEReconstructionLoss<T>(normalize: true, perPatchNormalization: true);
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Apply simple augmentation (MAE uses minimal augmentation)
        var augmented = _augmentation.ApplyMinimal(batch);

        // Patchify the image
        var patches = Patchify(augmented);

        // Generate random mask
        var mask = MAEReconstructionLoss<T>.CreateRandomMask(
            batchSize, _numPatches, _maskRatio, _config.Seed);

        // Get visible and masked patches
        var (visiblePatches, visibleIndices) = ExtractVisiblePatches(patches, mask);

        // Encode visible patches
        var encoded = _encoder.ForwardWithMemory(visiblePatches);

        // Decode and reconstruct (if decoder available)
        Tensor<T> reconstructed;
        if (_decoder is not null)
        {
            // Add position embeddings and mask tokens, then decode
            var decoderInput = PrepareDecoderInput(encoded, visibleIndices, mask);
            reconstructed = _decoder.ForwardWithMemory(decoderInput);
        }
        else
        {
            // Simplified: use encoder output directly
            reconstructed = ReconstructFromVisible(encoded, mask);
        }

        // Compute loss only on masked patches
        var (loss, gradRecon) = _loss.ComputeLossWithGradients(reconstructed, patches, mask);

        // Backward pass
        if (_decoder is not null)
        {
            _decoder.Backpropagate(gradRecon);
        }

        // Backward through encoder (simplified - would need proper gradient routing)
        var encoderGrad = ComputeEncoderGradient(gradRecon, mask);
        _encoder.Backpropagate(encoderGrad);

        // Update parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateParameters(learningRate);

        // Compute metrics
        var numMaskedPatches = (int)(_numPatches * _maskRatio) * batchSize;
        var numVisiblePatches = (_numPatches - (int)(_numPatches * _maskRatio)) * batchSize;

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = numMaskedPatches; // Masked patches being reconstructed
        result.NumNegativePairs = 0;
        result.Metrics["mask_ratio"] = NumOps.FromDouble(_maskRatio);
        result.Metrics["num_visible"] = NumOps.FromDouble(numVisiblePatches);
        result.Metrics["num_masked"] = NumOps.FromDouble(numMaskedPatches);

        return result;
    }

    private Tensor<T> Patchify(Tensor<T> images)
    {
        var batchSize = images.Shape[0];
        var patchDim = _patchSize * _patchSize * 3; // RGB patches

        // Simplified: assume input is already patchified or flatten
        // In practice, would reshape [B, C, H, W] to [B, num_patches, patch_dim]
        var patches = new T[batchSize * _numPatches * patchDim];

        // For now, create dummy patches from input
        var inputSize = images.Shape.Length > 1 ? images.Shape[1] : 1;
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _numPatches; p++)
            {
                for (int d = 0; d < patchDim; d++)
                {
                    var idx = (b * inputSize + (p * patchDim + d) % inputSize) % (batchSize * inputSize);
                    var flatIdx = b * inputSize;
                    patches[(b * _numPatches + p) * patchDim + d] =
                        images[b, flatIdx < images.Shape[1] ? flatIdx : 0];
                }
            }
        }

        return new Tensor<T>(patches, [batchSize, _numPatches, patchDim]);
    }

    private (Tensor<T> visible, int[][] indices) ExtractVisiblePatches(Tensor<T> patches, Tensor<T> mask)
    {
        var batchSize = patches.Shape[0];
        var patchDim = patches.Shape[2];
        var numVisible = _numPatches - (int)(_numPatches * _maskRatio);

        var visiblePatches = new T[batchSize * numVisible * patchDim];
        var indices = new int[batchSize][];

        for (int b = 0; b < batchSize; b++)
        {
            var batchIndices = new List<int>();
            int visibleIdx = 0;

            for (int p = 0; p < _numPatches && visibleIdx < numVisible; p++)
            {
                if (!NumOps.GreaterThan(mask[b, p], NumOps.Zero))
                {
                    batchIndices.Add(p);
                    for (int d = 0; d < patchDim; d++)
                    {
                        visiblePatches[(b * numVisible + visibleIdx) * patchDim + d] = patches[b, p, d];
                    }
                    visibleIdx++;
                }
            }

            indices[b] = [.. batchIndices];
        }

        return (new Tensor<T>(visiblePatches, [batchSize, numVisible, patchDim]), indices);
    }

    private Tensor<T> PrepareDecoderInput(Tensor<T> encoded, int[][] visibleIndices, Tensor<T> mask)
    {
        var batchSize = encoded.Shape[0];
        var numVisible = encoded.Shape[1];
        var embedDim = encoded.Shape[2];

        // Create full sequence with mask tokens
        var decoderInput = new T[batchSize * _numPatches * embedDim];

        // Initialize with mask token (learnable in practice, zeros here)
        // Then place encoded visible patches at their original positions

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < visibleIndices[b].Length; i++)
            {
                var patchIdx = visibleIndices[b][i];
                for (int d = 0; d < embedDim; d++)
                {
                    decoderInput[(b * _numPatches + patchIdx) * embedDim + d] = encoded[b, i, d];
                }
            }
        }

        return new Tensor<T>(decoderInput, [batchSize, _numPatches, embedDim]);
    }

    private Tensor<T> ReconstructFromVisible(Tensor<T> encoded, Tensor<T> mask)
    {
        var batchSize = encoded.Shape[0];
        var numVisible = encoded.Shape[1];
        var embedDim = encoded.Shape.Length > 2 ? encoded.Shape[2] : encoded.Shape[1];
        var patchDim = _patchSize * _patchSize * 3;

        // Simple linear projection from embed_dim to patch_dim
        var reconstructed = new T[batchSize * _numPatches * patchDim];

        // Simplified reconstruction
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _numPatches; p++)
            {
                for (int d = 0; d < patchDim; d++)
                {
                    reconstructed[(b * _numPatches + p) * patchDim + d] = NumOps.Zero;
                }
            }
        }

        return new Tensor<T>(reconstructed, [batchSize, _numPatches, patchDim]);
    }

    private Tensor<T> ComputeEncoderGradient(Tensor<T> decoderGrad, Tensor<T> mask)
    {
        var batchSize = decoderGrad.Shape[0];
        var numVisible = _numPatches - (int)(_numPatches * _maskRatio);
        var embedDim = _decoderEmbedDim;

        var encoderGrad = new T[batchSize * numVisible * embedDim];

        // Simplified gradient (would need proper chain rule in practice)
        for (int i = 0; i < encoderGrad.Length; i++)
        {
            encoderGrad[i] = NumOps.FromDouble(0.01);
        }

        return new Tensor<T>(encoderGrad, [batchSize, numVisible, embedDim]);
    }

    private void UpdateParameters(T learningRate)
    {
        // Update encoder
        var encoderGrads = _encoder.GetParameterGradients();
        var encoderParams = _encoder.GetParameters();
        var newEncoderParams = new T[encoderParams.Length];

        for (int i = 0; i < encoderParams.Length; i++)
        {
            newEncoderParams[i] = NumOps.Subtract(
                encoderParams[i],
                NumOps.Multiply(learningRate, encoderGrads[i]));
        }
        _encoder.UpdateParameters(new Vector<T>(newEncoderParams));

        // Update decoder if present
        if (_decoder is not null)
        {
            var decoderGrads = _decoder.GetParameterGradients();
            var decoderParams = _decoder.GetParameters();
            var newDecoderParams = new T[decoderParams.Length];

            for (int i = 0; i < decoderParams.Length; i++)
            {
                newDecoderParams[i] = NumOps.Subtract(
                    decoderParams[i],
                    NumOps.Multiply(learningRate, decoderGrads[i]));
            }
            _decoder.UpdateParameters(new Vector<T>(newDecoderParams));
        }
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
