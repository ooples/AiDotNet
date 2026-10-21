using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// DINO: Self-Distillation with No Labels - a self-supervised method for Vision Transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DINO is a self-supervised method specifically designed for
/// Vision Transformers (ViT). It learns by having a student network predict the output
/// of a teacher network, where the teacher is an EMA of the student.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>Self-distillation:</b> Student learns from teacher's soft labels</item>
/// <item><b>Centering and sharpening:</b> Prevents collapse without negative samples</item>
/// <item><b>Multi-crop training:</b> Uses global and local crops for efficiency</item>
/// <item><b>Emergent properties:</b> Learns features that segment objects without supervision</item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Global views → Teacher → Softmax(z/τ_t - center) → P_t
/// All views → Student → Softmax(z/τ_s) → P_s
/// Loss: Cross-entropy(P_s, P_t)
/// </code>
///
/// <para><b>Reference:</b> Caron et al., "Emerging Properties in Self-Supervised Vision
/// Transformers" (ICCV 2021)</para>
///
/// <para><b>Best for:</b> Vision Transformers, emergent attention properties.</para>
/// <para><b>Pros:</b> Emergent attention maps, strong ViT performance.</para>
/// <para><b>Cons:</b> Primarily designed for Vision Transformers.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Emerging Properties in Self-Supervised Vision Transformers", "https://arxiv.org/abs/2104.14294", Year = 2021, Authors = "Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin")]
public class DINO<T> : TeacherStudentSSL<T>
{
    private readonly DINOLoss<T> _loss;
    private readonly int _outputDim;

    // The step's teacher outputs; the center advances once from them after the optimizer has stepped.
    [Scratch]
    private Tensor<T>? _pendingTeacherOutputs;

    /// <inheritdoc />
    /// <remarks>
    /// Caron et al. 2021, Alg. 1 with multi-crop (Sec. 3.1): the student's distribution for every crop is matched to the
    /// teacher's sharpened, centered distribution for each global crop other than itself, averaged over those pairs
    /// (2 + 2 x local crops of them). The teacher sees only the two global crops and carries no gradient; it follows by
    /// EMA, and the center advances once per step from the teacher's outputs.
    /// </remarks>
    protected override Tensor<T> ComputeObjective(Tensor<T> view1, Tensor<T> view2)
    {
        var s1 = ForwardStudent(view1);
        var s2 = ForwardStudent(view2);
        var t1 = ForwardTeacher(view1);
        var t2 = ForwardTeacher(view2);
        _pendingTeacherOutputs = Engine.Concat(new[] { t1, t2 }, 0);
        var loss = Engine.TensorAdd(
            _loss.ComputeDifferentiableLoss(s1, t2, updateCenter: false),
            _loss.ComputeDifferentiableLoss(s2, t1, updateCenter: false));
        int pairs = 2;
        foreach (var local in LocalViews)
        {
            var s = ForwardStudent(local);
            loss = Engine.TensorAdd(loss, Engine.TensorAdd(
                _loss.ComputeDifferentiableLoss(s, t1, updateCenter: false),
                _loss.ComputeDifferentiableLoss(s, t2, updateCenter: false)));
            pairs += 2;
        }
        return Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(1.0 / pairs));
    }

    /// <inheritdoc />
    protected override void AfterOptimizerStep()
    {
        base.AfterOptimizerStep();
        if (_pendingTeacherOutputs is { } teacherOutputs)
        {
            _loss.UpdateCenter(teacherOutputs);
            _pendingTeacherOutputs = null;
        }
    }

    /// <inheritdoc />
    public override string Name => "DINO";

    /// <inheritdoc />
    public override SelfSupervisedLearningMethodCategory Category => SelfSupervisedLearningMethodCategory.SelfDistillation;

    /// <summary>
    /// Initializes a new instance of the DINO class.
    /// </summary>
    /// <param name="studentEncoder">The student encoder (ViT recommended).</param>
    /// <param name="teacherEncoder">The teacher encoder (momentum-updated copy).</param>
    /// <param name="studentProjector">Projection head for student.</param>
    /// <param name="teacherProjector">Projection head for teacher.</param>
    /// <param name="outputDim">Output dimension of the projection heads.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public DINO(
        INeuralNetwork<T> studentEncoder,
        IMomentumEncoder<T> teacherEncoder,
        IProjectorHead<T> studentProjector,
        IProjectorHead<T> teacherProjector,
        int outputDim = 65536,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(studentEncoder, teacherEncoder, studentProjector, teacherProjector,
               outputDim, config ?? new SelfSupervisedLearningConfig<T>())
    {
        _outputDim = outputDim;

        var dinoConfig = _config.DINO ?? new DINOConfig();
        var studentTemp = dinoConfig.StudentTemperature ?? 0.1;
        var teacherTemp = dinoConfig.TeacherTemperatureStart ?? 0.04;
        var centerMomentum = dinoConfig.CenterMomentum ?? 0.9;

        _loss = new DINOLoss<T>(outputDim, studentTemp, teacherTemp, centerMomentum);

        // Multi-crop: the paper's two global crops (the views a training step compares) and any number of local ones.
        NumGlobalCrops = dinoConfig.NumGlobalCrops ?? 2;
        if (NumGlobalCrops != 2)
            throw new ArgumentException($"DINO trains on two global crops; NumGlobalCrops is {NumGlobalCrops}.", nameof(config));
        NumLocalCrops = dinoConfig.NumLocalCrops ?? 0;
        if (NumLocalCrops < 0)
            throw new ArgumentException($"NumLocalCrops must be non-negative; got {NumLocalCrops}.", nameof(config));
    }

    /// <summary>
    /// Creates a DINO instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder (ViT recommended).</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for teacher.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 256).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 2048).</param>
    /// <param name="outputDim">Output dimension for softmax (default: 65536).</param>
    /// <returns>A configured DINO instance.</returns>
    public static DINO<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 256,
        int hiddenDim = 2048,
        int outputDim = 65536)
    {
        // Create projectors with DINO-specific output dimension
        var studentProjector = new MLPProjector<T>(
            encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);
        var teacherProjector = new MLPProjector<T>(
            encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);

        // Copy projector parameters
        teacherProjector.SetParameters(studentProjector.GetParameters());

        // Create teacher encoder
        var encoderCopy = createEncoderCopy(encoder);
        var teacherEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new DINO<T>(encoder, teacherEncoder, studentProjector, teacherProjector, outputDim);
    

}
}
