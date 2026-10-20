using AiDotNet.Attributes;
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
///
/// <para><b>Best for:</b> Combining generative and discriminative approaches.</para>
/// <para><b>Pros:</b> Best of both worlds (DINO + MAE-like objectives).</para>
/// <para><b>Cons:</b> More complex than pure DINO or MAE.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("iBOT: Image BERT Pre-Training with Online Tokenizer", "https://arxiv.org/abs/2111.07832", Year = 2022, Authors = "Jinghao Zhou, Chen Wei, Huiber Wang, Wei Shen, Cihang Xie, Alan Yuille, Tao Kong")]
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
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.SelfDistillation;

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
        SelfSupervisedLearningConfig<T>? config = null)
        : base(studentEncoder, teacherEncoder, studentProjector, teacherProjector,
               outputDim, config ?? new SelfSupervisedLearningConfig<T>())
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

    [Scratch]
    private Tensor<T>? _pendingTeacherCls;
    [Scratch]
    private Tensor<T>? _pendingTeacherPatches;
    [Scratch]
    private Random? _maskRandom;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Zhou et al. 2022, Eq. 3: L = L_[CLS] + lambda * L_MIM. L_[CLS] is DINO's self-distillation between the two global
    /// views' class outputs. L_MIM distills, at each masked patch of a view, the teacher's patch distribution (from the
    /// unmasked view) into the student's (from the masked one). Teachers carry no gradient and follow by EMA; each
    /// loss's center advances once per step.
    /// </para>
    /// <para>
    /// The masked term needs per-token inputs and outputs, [batch, tokens, dim] with the class token first. A generic
    /// encoder has no learnable [MASK] embedding to substitute, so a masked input token is zeroed; on [batch, dim]
    /// outputs only the class term applies.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var mask1 = PatchMask(view1);
        var mask2 = PatchMask(view2);
        var s1 = ForwardStudent(ApplyMask(view1, mask1));
        var s2 = ForwardStudent(ApplyMask(view2, mask2));
        var t1 = ForwardTeacher(view1);
        var t2 = ForwardTeacher(view2);

        var s1Cls = ClassToken(s1);
        var s2Cls = ClassToken(s2);
        var t1Cls = ClassToken(t1);
        var t2Cls = ClassToken(t2);
        _pendingTeacherCls = Engine.Concat(new[] { t1Cls, t2Cls }, 0);
        var loss = Engine.TensorMultiplyScalar(Engine.TensorAdd(
            _clsLoss.ComputeDifferentiableLoss(s1Cls, t2Cls, updateCenter: false),
            _clsLoss.ComputeDifferentiableLoss(s2Cls, t1Cls, updateCenter: false)), NumOps.FromDouble(0.5));

        if (mask1 is null || mask2 is null || s1.Rank != 3 || t1.Rank != 3)
            return loss;

        var (studentRows1, teacherRows1) = MaskedRows(s1, t1, mask1);
        var (studentRows2, teacherRows2) = MaskedRows(s2, t2, mask2);
        if (studentRows1 is null || studentRows2 is null || teacherRows1 is null || teacherRows2 is null)
            return loss;
        _pendingTeacherPatches = Engine.Concat(new[] { teacherRows1, teacherRows2 }, 0);
        var mim = Engine.TensorMultiplyScalar(Engine.TensorAdd(
            _patchLoss.ComputeDifferentiableLoss(studentRows1, teacherRows1, updateCenter: false),
            _patchLoss.ComputeDifferentiableLoss(studentRows2, teacherRows2, updateCenter: false)), NumOps.FromDouble(0.5));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(mim, NumOps.FromDouble(_mimWeight)));
    }

    /// <inheritdoc />
    protected override void AfterOptimizerStep()
    {
        base.AfterOptimizerStep();
        if (_pendingTeacherCls is { } cls) _clsLoss.UpdateCenter(cls);
        if (_pendingTeacherPatches is { } patches) _patchLoss.UpdateCenter(patches);
        _pendingTeacherCls = null;
        _pendingTeacherPatches = null;
    }

    // [batch, tokens, dim] -> its class token [batch, dim]; a [batch, dim] output is already one vector per sample.
    private Tensor<T> ClassToken(Tensor<T> output) =>
        output.Rank == 3
            ? Engine.Reshape(Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { output.Shape[0], 1, output.Shape[2] }),
                new[] { output.Shape[0], output.Shape[2] })
            : output;

    // For a [batch, tokens, dim] input, marks maskRatio of each sample's patch tokens (never the class token at 0).
    private bool[,]? PatchMask(Tensor<T> view)
    {
        if (view.Rank != 3 || view.Shape[1] < 2) return null;
        var rng = _maskRandom ??= _config.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_config.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        int batch = view.Shape[0], tokens = view.Shape[1];
        var mask = new bool[batch, tokens];
        int toMask = Math.Max(1, (int)Math.Round((tokens - 1) * _maskRatio));
        for (int b = 0; b < batch; b++)
        {
            var order = Enumerable.Range(1, tokens - 1).OrderBy(_ => rng.Next()).Take(toMask);
            foreach (int t in order) mask[b, t] = true;
        }
        return mask;
    }

    private Tensor<T> ApplyMask(Tensor<T> view, bool[,]? mask)
    {
        if (mask is null) return view;
        var keep = new Tensor<T>(view.Shape.ToArray());
        int batch = view.Shape[0], tokens = view.Shape[1], dim = view.Shape[2];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < tokens; t++)
                for (int d = 0; d < dim; d++)
                    keep[b, t, d] = mask[b, t] ? NumOps.Zero : NumOps.One;
        return Engine.TensorMultiply(view, keep);
    }

    // The masked tokens' student and teacher outputs as [count, dim] rows.
    private (Tensor<T>? Student, Tensor<T>? Teacher) MaskedRows(Tensor<T> student, Tensor<T> teacher, bool[,] mask)
    {
        var studentRows = new List<Tensor<T>>();
        var teacherRows = new List<Tensor<T>>();
        int batch = student.Shape[0], tokens = Math.Min(student.Shape[1], mask.GetLength(1)), dim = student.Shape[2];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < tokens; t++)
            {
                if (!mask[b, t]) continue;
                studentRows.Add(Engine.Reshape(Engine.TensorSlice(student, new[] { b, t, 0 }, new[] { 1, 1, dim }), new[] { 1, dim }));
                teacherRows.Add(Engine.Reshape(Engine.TensorSlice(teacher, new[] { b, t, 0 }, new[] { 1, 1, dim }), new[] { 1, dim }));
            }
        return studentRows.Count == 0
            ? (null, null)
            : (Engine.Concat(studentRows.ToArray(), 0), Engine.Concat(teacherRows.ToArray(), 0));
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
    /// Computes the gradient of the masked patch loss with respect to studentOut.
    /// For MSE loss on masked patches, grad = 2*(student - teacher)/N for masked positions, 0 otherwise.
    /// </summary>
    private Tensor<T> ComputeMaskedPatchGradient(Tensor<T> studentOut, Tensor<T> teacherOut, Tensor<T> mask)
    {
        var grad = new Tensor<T>(studentOut._shape);
        var batchSize = studentOut.Shape[0];
        var numPatches = mask.Shape[1];

        if (studentOut.Shape.Length == 3 && studentOut.Shape[1] > 1)
        {
            var seqLen = studentOut.Shape[1];
            var dim = studentOut.Shape[2];
            var patchStartIdx = 1;
            var numPatchTokens = Math.Min(seqLen - patchStartIdx, numPatches);

            if (numPatchTokens <= 0)
            {
                return grad;
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
                return grad;
            }

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
                            var diff = NumOps.Subtract(studentOut[b, patchIdx, d], teacherOut[b, patchIdx, d]);
                            grad[b, patchIdx, d] = NumOps.Multiply(diff, scale);
                        }
                    }
                }
            }

            return grad;
        }

        // For 2D outputs, use DINOLoss gradients weighted by mask ratio
        var (_, baseGrad) = _patchLoss.ComputeLossWithGradients(studentOut, teacherOut);
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

        for (int i = 0; i < baseGrad.Length; i++)
        {
            grad[i] = NumOps.Multiply(baseGrad[i], weightT);
        }

        return grad;
    }

    /// <summary>
    /// Combines CLS and MIM loss gradients element-wise: result = clsWeight * gradCls + mimWeight * gradMim.
    /// </summary>
    private Tensor<T> CombineLossGradients(Tensor<T> gradCls, Tensor<T> gradMim, double clsWeight, double mimWeight)
    {
        var result = new Tensor<T>(gradCls._shape);
        var cw = NumOps.FromDouble(clsWeight);
        var mw = NumOps.FromDouble(mimWeight);

        for (int i = 0; i < gradCls.Length; i++)
        {
            result[i] = NumOps.Add(
                NumOps.Multiply(gradCls[i], cw),
                NumOps.Multiply(gradMim[i], mw));
        }

        return result;
    }

    /// <summary>
    /// Updates the student network with pre-accumulated projector gradients.
    /// </summary>
    private void UpdateStudent(T learningRate, Vector<T> accumulatedProjGrads)
    {
        // Update encoder
        var encoderGrads = new Vector<T>(_encoder.GetParameterGradients());
        var encoderParams = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(encoderParams, Engine.Multiply(encoderGrads, learningRate)));

        // Update projector with accumulated gradients from both views
        if (_projector is not null)
        {
            var projParams = _projector.GetParameters();
            _projector.SetParameters(Engine.Subtract(projParams, Engine.Multiply(accumulatedProjGrads, learningRate)));
        }
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
