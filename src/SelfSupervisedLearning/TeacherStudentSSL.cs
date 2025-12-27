using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
public abstract class TeacherStudentSSL<T> : SSLMethodBase<T>
{
    /// <summary>
    /// The teacher encoder (momentum-updated copy of student).
    /// </summary>
    protected readonly IMomentumEncoder<T> TeacherEncoder;

    /// <summary>
    /// The teacher projection head.
    /// </summary>
    protected readonly IProjectorHead<T> TeacherProjector;

    /// <summary>
    /// Centering mechanism to prevent collapse.
    /// </summary>
    protected readonly CenteringMechanism<T> Centering;

    /// <summary>
    /// Base momentum value for teacher updates.
    /// </summary>
    protected readonly double BaseMomentum;

    /// <summary>
    /// Augmentation policies for creating views.
    /// </summary>
    protected readonly SSLAugmentationPolicies<T> Augmentation;

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
        SSLConfig? config = null)
        : base(studentEncoder, studentProjector, config ?? new SSLConfig())
    {
        TeacherEncoder = teacherEncoder ?? throw new ArgumentNullException(nameof(teacherEncoder));
        TeacherProjector = teacherProjector ?? throw new ArgumentNullException(nameof(teacherProjector));

        BaseMomentum = teacherEncoder.Momentum;
        Centering = new CenteringMechanism<T>(outputDim);
        Augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    /// <summary>
    /// Creates augmented views for teacher-student training.
    /// </summary>
    /// <param name="batch">Input batch.</param>
    /// <returns>Global and local crop views.</returns>
    protected virtual (List<Tensor<T>> globalViews, List<Tensor<T>> localViews) CreateMultiCropViews(Tensor<T> batch)
    {
        var globalViews = new List<Tensor<T>>();
        var localViews = new List<Tensor<T>>();

        // Create global crops (full resolution)
        var (dinoGlobal, dinoLocal) = Augmentation.ApplyDINO(batch, NumGlobalCrops, NumLocalCrops);
        foreach (var view in dinoGlobal)
        {
            globalViews.Add(view);
        }

        // Add local crops (lower resolution) if specified
        foreach (var view in dinoLocal)
        {
            localViews.Add(view);
        }

        return (globalViews, localViews);
    }

    /// <summary>
    /// Performs forward pass through student network.
    /// </summary>
    protected virtual Tensor<T> ForwardStudent(Tensor<T> view)
    {
        var h = _encoder.ForwardWithMemory(view);
        return _projector!.Project(h);
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

        // Update teacher projector with EMA
        var momentum = NumOps.FromDouble(TeacherEncoder.Momentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);

        var studentParams = _projector!.GetParameters();
        var teacherParams = TeacherProjector.GetParameters();
        var newParams = new T[teacherParams.Length];

        for (int i = 0; i < teacherParams.Length; i++)
        {
            newParams[i] = NumOps.Add(
                NumOps.Multiply(momentum, teacherParams[i]),
                NumOps.Multiply(oneMinusMomentum, studentParams[i]));
        }

        TeacherProjector.SetParameters(new Vector<T>(newParams));
    }

    /// <summary>
    /// Updates student network parameters with gradients.
    /// </summary>
    protected virtual void UpdateStudent(T learningRate)
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

        // Update projector
        if (_projector is not null)
        {
            var projGrads = _projector.GetParameterGradients();
            var projParams = _projector.GetParameters();
            var newProjParams = new T[projParams.Length];

            for (int i = 0; i < projParams.Length; i++)
            {
                newProjParams[i] = NumOps.Subtract(
                    projParams[i],
                    NumOps.Multiply(learningRate, projGrads[i]));
            }
            _projector.SetParameters(new Vector<T>(newProjParams));
        }
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

    /// <inheritdoc />
    protected override int GetAdditionalParameterCount()
    {
        return TeacherEncoder.GetParameters().Length + TeacherProjector.ParameterCount;
    }

    /// <inheritdoc />
    protected override Vector<T>? GetAdditionalParameters()
    {
        var teacherEncoderParams = TeacherEncoder.GetParameters();
        var teacherProjectorParams = TeacherProjector.GetParameters();

        var combined = new T[teacherEncoderParams.Length + teacherProjectorParams.Length];
        for (int i = 0; i < teacherEncoderParams.Length; i++)
            combined[i] = teacherEncoderParams[i];
        for (int i = 0; i < teacherProjectorParams.Length; i++)
            combined[teacherEncoderParams.Length + i] = teacherProjectorParams[i];

        return new Vector<T>(combined);
    }
}
