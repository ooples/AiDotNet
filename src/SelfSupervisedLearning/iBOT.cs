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

    private sealed record StepMaskState(
        Tensor<T> View1,
        Tensor<T> View2,
        Tensor<T>? Keep1,
        Tensor<T>? Keep2,
        Tensor<T>? Rows1,
        Tensor<T>? Rows2);

    [Scratch]
    private StepMaskState? _stepState;
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
        StepMaskState state;
        if (_stepState is { } existing
            && ReferenceEquals(existing.View1, view1)
            && ReferenceEquals(existing.View2, view2))
        {
            state = existing;
        }
        else
        {
            var (k1, r1) = BuildPatchMaskLeaves(view1);
            var (k2, r2) = BuildPatchMaskLeaves(view2);
            state = new StepMaskState(view1, view2, k1, k2, r1, r2);
            _stepState = state;
        }

        var s1 = ForwardStudent(state.Keep1 is null ? view1 : Engine.TensorMultiply(view1, Engine.TensorBroadcastTo(state.Keep1, view1._shape)));
        var s2 = ForwardStudent(state.Keep2 is null ? view2 : Engine.TensorMultiply(view2, Engine.TensorBroadcastTo(state.Keep2, view2._shape)));
        var t1 = ForwardTeacher(view1);
        var t2 = ForwardTeacher(view2);

        var s1Cls = ClassToken(s1);
        var s2Cls = ClassToken(s2);
        var t1Cls = ClassToken(t1);
        var t2Cls = ClassToken(t2);
        var loss = Engine.TensorMultiplyScalar(Engine.TensorAdd(
            _clsLoss.ComputeDifferentiableLoss(s1Cls, t2Cls, updateCenter: false),
            _clsLoss.ComputeDifferentiableLoss(s2Cls, t1Cls, updateCenter: false)), NumOps.FromDouble(0.5));

        if (state.Rows1 is null || state.Rows2 is null || s1.Rank != 3 || t1.Rank != 3)
            return loss;

        var studentRows1 = GatherMaskedRows(s1, state.Rows1);
        var studentRows2 = GatherMaskedRows(s2, state.Rows2);
        var teacherRows1 = GatherMaskedRows(t1, state.Rows1);
        var teacherRows2 = GatherMaskedRows(t2, state.Rows2);
        var mim = Engine.TensorMultiplyScalar(Engine.TensorAdd(
            _patchLoss.ComputeDifferentiableLoss(studentRows1, teacherRows1, updateCenter: false),
            _patchLoss.ComputeDifferentiableLoss(studentRows2, teacherRows2, updateCenter: false)), NumOps.FromDouble(0.5));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(mim, NumOps.FromDouble(_mimWeight)));
    }

    /// <inheritdoc />
    protected override void AfterOptimizerStep()
    {
        var state = _stepState;
        _stepState = null;
        if (state is not null)
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var t1 = ForwardTeacher(state.View1);
            var t2 = ForwardTeacher(state.View2);
            _clsLoss.UpdateCenter(Engine.Concat(new[] { ClassToken(t1), ClassToken(t2) }, 0));
            if (state.Rows1 is not null && state.Rows2 is not null && t1.Rank == 3 && t2.Rank == 3)
            {
                _patchLoss.UpdateCenter(Engine.Concat(new[]
                {
                    GatherMaskedRows(t1, state.Rows1),
                    GatherMaskedRows(t2, state.Rows2),
                }, 0));
            }
        }
        base.AfterOptimizerStep();
    }

    // [batch, tokens, dim] -> its class token [batch, dim]; a [batch, dim] output is already one vector per sample.
    private Tensor<T> ClassToken(Tensor<T> output) =>
        output.Rank == 3
            ? Engine.Reshape(Engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { output.Shape[0], 1, output.Shape[2] }),
                new[] { output.Shape[0], output.Shape[2] })
            : output;

    // Builds external-leaf keep masks [batch, tokens, 1] and flattened masked row indices [batch * toMask]
    // so compiled closure plans have a fixed graph topology across random per-step patch masks.
    private (Tensor<T>? Keep, Tensor<T>? Rows) BuildPatchMaskLeaves(Tensor<T> view)
    {
        if (view.Rank != 3 || view.Shape[1] < 2) return (null, null);
        var rng = _maskRandom ??= _config.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_config.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        int batch = view.Shape[0], tokens = view.Shape[1];
        int toMask = Math.Max(1, (int)Math.Round((tokens - 1) * _maskRatio));
        var keepValues = new double[batch * tokens];
        for (int i = 0; i < keepValues.Length; i++) keepValues[i] = 1.0;
        var rowIndices = new double[batch * toMask];
        for (int b = 0; b < batch; b++)
        {
            var order = Enumerable.Range(1, tokens - 1).OrderBy(_ => rng.Next()).Take(toMask).OrderBy(t => t).ToArray();
            for (int m = 0; m < toMask; m++)
            {
                int flatIdx = b * tokens + order[m];
                keepValues[flatIdx] = 0.0;
                rowIndices[b * toMask + m] = flatIdx;
            }
        }

        var keep = new Tensor<double>(new[] { batch, tokens, 1 }, new Vector<double>(keepValues)).Cast<T>();
        var rows = new Tensor<double>(new[] { batch * toMask }, new Vector<double>(rowIndices)).Cast<T>();
        return (keep, rows);
    }

    private Tensor<T> GatherMaskedRows(Tensor<T> sequence, Tensor<T> flatRowIndices)
    {
        int batch = sequence.Shape[0], tokens = sequence.Shape[1], dim = sequence.Shape[2];
        return Engine.TensorEmbeddingLookupFromFloatIndices(
            Engine.Reshape(sequence, new[] { batch * tokens, dim }), flatRowIndices);
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

        ProjectorParameterTransfer<T>.Copy(studentProjector, teacherProjector);

        var encoderCopy = createEncoderCopy(encoder);
        var teacherEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new iBOT<T>(encoder, teacherEncoder, studentProjector, teacherProjector,
                          outputDim, mimWeight, maskRatio);
    

}
}
