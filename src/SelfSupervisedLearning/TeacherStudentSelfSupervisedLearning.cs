using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Base class for teacher-student self-supervised learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Teacher-student SSL methods use two networks:
/// a student that learns from gradients and a teacher that provides targets.
/// The teacher is typically updated as an exponential moving average (EMA)
/// of the student, providing stable learning targets.</para>
///
/// <para><b>Common components:</b></para>
/// <list type="bullet">
/// <item><b>Student network:</b> Trained with backpropagation</item>
/// <item><b>Teacher network:</b> Updated with EMA (momentum encoder)</item>
/// <item><b>Centering:</b> Prevents collapse by centering teacher outputs</item>
/// <item><b>Multi-crop:</b> Uses multiple augmented views of different sizes</item>
/// </list>
///
/// <para><b>Methods using this pattern:</b> DINO, iBOT, EsViT, DINOv2</para>
/// </remarks>
public abstract partial class TeacherStudentSSL<T> : SelfSupervisedLearningMethodBase<T>
{
    /// <summary>
    /// The teacher encoder (momentum-updated copy of student).
    /// </summary>
    [FrozenParameter]
    protected readonly IMomentumEncoder<T> TeacherEncoder;

    /// <summary>
    /// The teacher projection head.
    /// </summary>
    [FrozenParameter]
    protected readonly IProjectorHead<T> TeacherProjector;

    /// <summary>
    /// Centering mechanism to prevent collapse.
    /// </summary>
    [Buffer]
    protected readonly CenteringMechanism<T> Centering;

    /// <summary>
    /// Base momentum value for teacher updates.
    /// </summary>
    protected readonly double BaseMomentum;

    /// <summary>
    /// Augmentation policies for creating views.
    /// </summary>
    protected readonly SelfSupervisedLearningAugmentationPolicies<T> Augmentation;

    /// <summary>
    /// Number of global crops (larger views used by both student and teacher).
    /// </summary>
    protected int NumGlobalCrops { get; set; } = 2;

    /// <summary>
    /// Number of local crops (smaller views used by student only).
    /// </summary>
    protected int NumLocalCrops { get; set; } = 0;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <summary>
    /// The local crops of the current training step, seen by the student only (empty without multi-crop, or when the
    /// step's views were supplied precomputed).
    /// </summary>
    protected IReadOnlyList<Tensor<T>> LocalViews { get; private set; } = Array.Empty<Tensor<T>>();

    /// <inheritdoc />
    protected override (Tensor<T> View1, Tensor<T> View2) CreateViews(
        Tensor<T> batch, SelfSupervisedLearningAugmentationContext<T>? augmentationContext)
    {
        LocalViews = Array.Empty<Tensor<T>>();
        return base.CreateViews(batch, augmentationContext);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Multi-crop (Caron et al. 2021, Sec. 3.1): two global crops, which both networks see, and
    /// <see cref="NumLocalCrops"/> local crops (5-50% of the image, resized to the input size the encoder takes), which
    /// only the student sees; they are kept in <see cref="LocalViews"/>.
    /// </remarks>
    protected override (Tensor<T> View1, Tensor<T> View2) AugmentViews(
        SelfSupervisedLearningAugmentationPolicies<T> policies, Tensor<T> batch)
    {
        var (globalViews, localViews) = Augmentation.ApplyDINO(batch, 2, NumLocalCrops);
        LocalViews = localViews;
        return (globalViews[0], globalViews[1]);
    }

    /// <inheritdoc />
    /// <remarks>The teacher (encoder and projector) follows the student by EMA.</remarks>
    protected override void AfterOptimizerStep() => UpdateTeacher();

    /// <inheritdoc />
    /// <remarks>The paper trains with AdamW (Sec. 4).</remarks>
    protected override SelfSupervisedLearningOptimizerType DefaultOptimizerType => SelfSupervisedLearningOptimizerType.AdamW;

    /// <inheritdoc />
    protected override double DefaultLearningRate => 5e-4;

    /// <summary>
    /// Initializes a new instance of the TeacherStudentSSL class.
    /// </summary>
    /// <param name="studentEncoder">The student encoder network.</param>
    /// <param name="teacherEncoder">The teacher encoder (momentum-updated).</param>
    /// <param name="studentProjector">Projection head for student.</param>
    /// <param name="teacherProjector">Projection head for teacher.</param>
    /// <param name="outputDim">Output dimension for centering.</param>
    /// <param name="config">Optional SSL configuration.</param>
    protected TeacherStudentSSL(
        INeuralNetwork<T> studentEncoder,
        IMomentumEncoder<T> teacherEncoder,
        IProjectorHead<T> studentProjector,
        IProjectorHead<T> teacherProjector,
        int outputDim,
        SelfSupervisedLearningConfig<T>? config = null)
        : base(studentEncoder, studentProjector, config ?? new SelfSupervisedLearningConfig<T>())
    {
        Guard.NotNull(teacherEncoder);
        TeacherEncoder = teacherEncoder;
        Guard.NotNull(teacherProjector);
        TeacherProjector = teacherProjector;

        BaseMomentum = teacherEncoder.Momentum;
        Centering = new CenteringMechanism<T>(outputDim);
        Augmentation = new SelfSupervisedLearningAugmentationPolicies<T>(_config.Seed);
    }

    /// <summary>
    /// Performs forward pass through student network.
    /// </summary>
    protected virtual Tensor<T> ForwardStudent(Tensor<T> view)
    {
        var h = EncodeForTraining(view);
        var projector = _projector ?? throw new InvalidOperationException("Projector has not been initialized.");
        return projector.Project(h);
    }

    /// <summary>
    /// Performs forward pass through teacher network (no gradients).
    /// </summary>
    protected virtual Tensor<T> ForwardTeacher(Tensor<T> view)
    {
        var h = TeacherEncoder.Encode(view);
        var z = TeacherProjector.Project(h);
        return StopGradient<T>.Detach(z);
    }

    /// <summary>
    /// Updates teacher network with EMA from student.
    /// </summary>
    protected virtual void UpdateTeacher()
    {
        // Update teacher encoder
        TeacherEncoder.UpdateFromMainEncoder(_encoder);

        ProjectorParameterTransfer<T>.Update(
            _projector ?? throw new InvalidOperationException("Projector has not been initialized."),
            TeacherProjector, TeacherEncoder.Momentum);
    }

    /// <inheritdoc />
    public override void OnEpochStart(int epochNumber)
    {
        base.OnEpochStart(epochNumber);

        // Update momentum schedule (cosine from base to 1.0)
        var totalEpochs = _config.PretrainingEpochs ?? 300;
        var newMomentum = MomentumEncoder<T>.ScheduleMomentum(
            BaseMomentum, 1.0, epochNumber, totalEpochs);

        TeacherEncoder.SetMomentum(newMomentum);
    }

}
