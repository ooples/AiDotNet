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
