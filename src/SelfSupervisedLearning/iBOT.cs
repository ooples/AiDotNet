using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// iBOT: Image BERT Pre-Training with Online Tokenizer - combining DINO with masked image modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> iBOT combines the best of DINO (self-distillation) with
/// masked image modeling (like MAE). It masks patches in the student view and predicts
/// both the CLS token (like DINO) and the masked patches (like BERT for images).</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>Dual objective:</b> CLS token distillation + masked patch prediction</item>
/// <item><b>Online tokenizer:</b> Uses teacher to provide targets for masked patches</item>
/// <item><b>Shared architecture:</b> Single network handles both objectives</item>
/// <item><b>Better representations:</b> Combines global (CLS) and local (patch) learning</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = L_cls (DINO loss on CLS token) + λ * L_mim (masked patch prediction)
/// </code>
///
/// <para><b>Reference:</b> Zhou et al., "iBOT: Image BERT Pre-Training with Online Tokenizer"
/// (ICLR 2022)</para>
/// </remarks>
public class iBOT<T> : TeacherStudentSSL<T>
{
    private readonly DINOLoss<T> _clsLoss;
    private readonly DINOLoss<T> _patchLoss;
    private readonly int _outputDim;
    private readonly double _mimWeight;
    private readonly double _maskRatio;

    /// <inheritdoc />
    public override string Name => "iBOT";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.SelfDistillation;

    /// <summary>
    /// Gets the weight for masked image modeling loss.
    /// </summary>
    public double MIMWeight => _mimWeight;

    /// <summary>
    /// Gets the mask ratio for patches.
    /// </summary>
    public double MaskRatio => _maskRatio;

    /// <summary>
    /// Initializes a new instance of the iBOT class.
    /// </summary>
    /// <param name="studentEncoder">The student encoder (ViT required).</param>
    /// <param name="teacherEncoder">The teacher encoder (momentum-updated copy).</param>
    /// <param name="studentProjector">Projection head for student.</param>
    /// <param name="teacherProjector">Projection head for teacher.</param>
    /// <param name="outputDim">Output dimension of the projection heads.</param>
    /// <param name="mimWeight">Weight for masked image modeling loss (default: 1.0).</param>
    /// <param name="maskRatio">Ratio of patches to mask (default: 0.4).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public iBOT(
        INeuralNetwork<T> studentEncoder,
        IMomentumEncoder<T> teacherEncoder,
        IProjectorHead<T> studentProjector,
        IProjectorHead<T> teacherProjector,
        int outputDim = 8192,
        double mimWeight = 1.0,
        double maskRatio = 0.4,
        SSLConfig? config = null)
        : base(studentEncoder, teacherEncoder, studentProjector, teacherProjector,
               outputDim, config ?? new SSLConfig { Method = SSLMethodType.DINO })
    {
        _outputDim = outputDim;
        _mimWeight = mimWeight;
        _maskRatio = maskRatio;

        var dinoConfig = _config.DINO ?? new DINOConfig();
        var studentTemp = dinoConfig.StudentTemperature ?? 0.1;
        var teacherTemp = dinoConfig.TeacherTemperatureStart ?? 0.04;
        var centerMomentum = dinoConfig.CenterMomentum ?? 0.9;

        // Separate losses for CLS and patch prediction
        _clsLoss = new DINOLoss<T>(outputDim, studentTemp, teacherTemp, centerMomentum);
        _patchLoss = new DINOLoss<T>(outputDim, studentTemp, teacherTemp, centerMomentum);

        NumGlobalCrops = 2;
        NumLocalCrops = 0;
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (globalViews, _) = CreateMultiCropViews(batch);
        var view1 = globalViews[0];
        var view2 = globalViews.Count > 1 ? globalViews[1] : globalViews[0];

        // Teacher forward pass (no masking)
        var teacherOut1 = ForwardTeacher(view1);
        var teacherOut2 = ForwardTeacher(view2);

        // Generate random masks for student views
        var numPatches = EstimateNumPatches(view1);
        var mask1 = GenerateMask(batchSize, numPatches);
        var mask2 = GenerateMask(batchSize, numPatches);

        // Student forward pass (full views - masking applied at loss level)
        var studentOut1 = ForwardStudent(view1);
        var studentOut2 = ForwardStudent(view2);

        // Compute CLS token loss (DINO-style)
        var clsLoss1 = _clsLoss.ComputeLoss(studentOut1, teacherOut2);
        var clsLoss2 = _clsLoss.ComputeLoss(studentOut2, teacherOut1);
        var clsLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(clsLoss1, clsLoss2));

        // Compute masked patch prediction loss
        // Apply mask weighting to simulate computing loss only on masked positions
        var patchLoss1 = ComputeMaskedPatchLoss(studentOut1, teacherOut2, mask1);
        var patchLoss2 = ComputeMaskedPatchLoss(studentOut2, teacherOut1, mask2);
        var patchLoss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(patchLoss1, patchLoss2));

        // Total loss: L_cls + λ * L_mim
        var mimWeightT = NumOps.FromDouble(_mimWeight);
        var loss = NumOps.Add(clsLoss, NumOps.Multiply(mimWeightT, patchLoss));

        // Backward pass - include both CLS and MIM loss gradients
        // Compute CLS gradients from both symmetric views
        var (_, gradCls1) = _clsLoss.ComputeLossWithGradients(studentOut1, teacherOut2);
        var (_, gradCls2) = _clsLoss.ComputeLossWithGradients(studentOut2, teacherOut1);

        // Compute MIM (patch) gradients
        var gradMim1 = ComputeMaskedPatchGradients(studentOut1, teacherOut2, mask1);
        var gradMim2 = ComputeMaskedPatchGradients(studentOut2, teacherOut1, mask2);

        // Combine gradients: avg(CLS) + mimWeight * avg(MIM)
        var half = NumOps.FromDouble(0.5);
        var gradCombined = new T[gradCls1.Length];
        for (int i = 0; i < gradCls1.Length; i++)
        {
            // Average CLS gradients
            var avgCls = NumOps.Multiply(half, NumOps.Add(gradCls1.Data[i], gradCls2.Data[i]));
            // Average MIM gradients
            var avgMim = NumOps.Multiply(half, NumOps.Add(gradMim1.Data[i], gradMim2.Data[i]));
            // Total gradient with MIM weighting
            gradCombined[i] = NumOps.Add(avgCls, NumOps.Multiply(mimWeightT, avgMim));
        }

        var gradH = _projector!.Backward(new Tensor<T>(gradCombined, gradCls1.Shape));
        _encoder.Backpropagate(gradH);

        // Update networks
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateStudent(learningRate);
        UpdateTeacher();

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize * 2;
        result.NumNegativePairs = 0;
        result.Metrics["momentum"] = NumOps.FromDouble(TeacherEncoder.Momentum);
        result.Metrics["cls_loss"] = clsLoss;
        result.Metrics["mim_loss"] = patchLoss;
        result.Metrics["mask_ratio"] = NumOps.FromDouble(_maskRatio);

        return result;
    }

    private int EstimateNumPatches(Tensor<T> view)
    {
        // Estimate patch count based on tensor shape
        // For image tensors [batch, channels, height, width], compute from spatial dimensions
        if (view.Shape.Length >= 4)
        {
            var height = view.Shape[2];
            var width = view.Shape[3];
            var patchSize = 16; // Standard ViT patch size
            return (height / patchSize) * (width / patchSize);
        }

        // For pre-processed sequence inputs [batch, seq_len, dim] or [batch, dim]
        // Assume seq_len dimension represents patches (minus CLS token)
        if (view.Shape.Length == 3)
        {
            return Math.Max(1, view.Shape[1] - 1); // Subtract 1 for CLS token
        }

        // Fallback: assume 196 patches (14x14 for 224x224 images with 16x16 patches)
        return 196;
    }

    private T ComputeMaskedPatchLoss(Tensor<T> studentOut, Tensor<T> teacherOut, Tensor<T> mask)
    {
        var batchSize = studentOut.Shape[0];
        var numPatches = mask.Shape[1];

        // For sequence outputs [batch, seq_len, dim], extract and compute loss on masked positions
        if (studentOut.Shape.Length == 3 && studentOut.Shape[1] > 1)
        {
            var seqLen = studentOut.Shape[1];
            var dim = studentOut.Shape[2];

            // Skip CLS token (position 0), patches start at position 1
            var patchStartIdx = 1;
            var numPatchTokens = Math.Min(seqLen - patchStartIdx, numPatches);

            if (numPatchTokens <= 0)
            {
                return NumOps.Zero;
            }

            // Compute loss only on masked positions
            T totalLoss = NumOps.Zero;
            int maskedCount = 0;

            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < numPatchTokens; p++)
                {
                    // Check if this patch is masked
                    if (NumOps.GreaterThan(mask[b, p], NumOps.FromDouble(0.5)))
                    {
                        var patchIdx = patchStartIdx + p;

                        // Compute MSE loss between student and teacher at this position
                        T patchLoss = NumOps.Zero;
                        for (int d = 0; d < dim; d++)
                        {
                            var diff = NumOps.Subtract(studentOut[b, patchIdx, d], teacherOut[b, patchIdx, d]);
                            patchLoss = NumOps.Add(patchLoss, NumOps.Multiply(diff, diff));
                        }

                        totalLoss = NumOps.Add(totalLoss, patchLoss);
                        maskedCount++;
                    }
                }
            }

            // Average over masked patches
            if (maskedCount > 0)
            {
                return NumOps.Divide(totalLoss, NumOps.FromDouble(maskedCount * studentOut.Shape[2]));
            }

            return NumOps.Zero;
        }

        // For 2D outputs [batch, dim], use DINO-style loss weighted by mask ratio
        // This handles the case where encoder outputs only CLS tokens
        var numMasked = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                if (NumOps.GreaterThan(mask[b, p], NumOps.FromDouble(0.5)))
                {
                    numMasked++;
                }
            }
        }

        var baseLoss = _patchLoss.ComputeLoss(studentOut, teacherOut);

        // Weight loss by proportion of masked patches
        var maskWeight = numMasked > 0 ? (double)numMasked / (batchSize * numPatches) : 0.0;
        return NumOps.Multiply(baseLoss, NumOps.FromDouble(maskWeight));
    }

    /// <summary>
    /// Computes gradients for the MIM (Masked Image Modeling) loss.
    /// </summary>
    /// <remarks>
    /// For MSE loss L = (1/n) * sum((s - t)^2), the gradient w.r.t. s is: dL/ds = (2/n) * (s - t)
    /// This is computed only for masked positions.
    /// </remarks>
    private Tensor<T> ComputeMaskedPatchGradients(Tensor<T> studentOut, Tensor<T> teacherOut, Tensor<T> mask)
    {
        var batchSize = studentOut.Shape[0];
        var numPatches = mask.Shape[1];
        var gradients = new T[studentOut.Length];

        // For sequence outputs [batch, seq_len, dim]
        if (studentOut.Shape.Length == 3 && studentOut.Shape[1] > 1)
        {
            var seqLen = studentOut.Shape[1];
            var dim = studentOut.Shape[2];
            var patchStartIdx = 1; // Skip CLS token
            var numPatchTokens = Math.Min(seqLen - patchStartIdx, numPatches);

            if (numPatchTokens <= 0)
            {
                return new Tensor<T>(gradients, studentOut.Shape);
            }

            // Count masked patches for normalization
            int maskedCount = 0;
            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < numPatchTokens; p++)
                {
                    if (NumOps.GreaterThan(mask[b, p], NumOps.FromDouble(0.5)))
                    {
                        maskedCount++;
                    }
                }
            }

            if (maskedCount == 0)
            {
                return new Tensor<T>(gradients, studentOut.Shape);
            }

            // Compute gradients only for masked positions
            var scale = NumOps.FromDouble(2.0 / (maskedCount * dim));
            for (int b = 0; b < batchSize; b++)
            {
                for (int p = 0; p < numPatchTokens; p++)
                {
                    if (NumOps.GreaterThan(mask[b, p], NumOps.FromDouble(0.5)))
                    {
                        var patchIdx = patchStartIdx + p;
                        for (int d = 0; d < dim; d++)
                        {
                            var idx = (b * seqLen + patchIdx) * dim + d;
                            var diff = NumOps.Subtract(studentOut[b, patchIdx, d], teacherOut[b, patchIdx, d]);
                            gradients[idx] = NumOps.Multiply(scale, diff);
                        }
                    }
                }
            }

            return new Tensor<T>(gradients, studentOut.Shape);
        }

        // For 2D outputs [batch, dim], compute full gradient weighted by mask ratio
        var (_, grad) = _patchLoss.ComputeLossWithGradients(studentOut, teacherOut);

        // Weight by mask ratio
        int numMasked = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < numPatches; p++)
            {
                if (NumOps.GreaterThan(mask[b, p], NumOps.FromDouble(0.5)))
                {
                    numMasked++;
                }
            }
        }

        var maskWeight = numMasked > 0 ? (double)numMasked / (batchSize * numPatches) : 0.0;
        var weightT = NumOps.FromDouble(maskWeight);
        for (int i = 0; i < grad.Length; i++)
        {
            gradients[i] = NumOps.Multiply(grad.Data[i], weightT);
        }

        return new Tensor<T>(gradients, studentOut.Shape);
    }

    private Tensor<T> GenerateMask(int batchSize, int numPatches)
    {
        var rng = RandomHelper.Shared;
        var mask = new T[batchSize * numPatches];
        var numMasked = (int)(numPatches * _maskRatio);

        for (int b = 0; b < batchSize; b++)
        {
            // Shuffle indices
            var indices = Enumerable.Range(0, numPatches).OrderBy(_ => rng.Next()).ToArray();

            for (int p = 0; p < numPatches; p++)
            {
                mask[b * numPatches + indices[p]] = p < numMasked
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return new Tensor<T>(mask, [batchSize, numPatches]);
    }

    /// <summary>
    /// Creates an iBOT instance with default configuration.
    /// </summary>
    public static iBOT<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int outputDim = 8192,
        int hiddenDim = 2048,
        double mimWeight = 1.0,
        double maskRatio = 0.4)
    {
        var studentProjector = new MLPProjector<T>(
            encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);
        var teacherProjector = new MLPProjector<T>(
            encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);

        teacherProjector.SetParameters(studentProjector.GetParameters());

        var encoderCopy = createEncoderCopy(encoder);
        var teacherEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new iBOT<T>(encoder, teacherEncoder, studentProjector, teacherProjector,
                          outputDim, mimWeight, maskRatio);
    }
}
