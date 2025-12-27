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
/// </remarks>
public class DINO<T> : TeacherStudentSSL<T>
{
    private readonly DINOLoss<T> _loss;
    private readonly int _outputDim;

    /// <inheritdoc />
    public override string Name => "DINO";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.SelfDistillation;

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
        SSLConfig? config = null)
        : base(studentEncoder, teacherEncoder, studentProjector, teacherProjector,
               outputDim, config ?? new SSLConfig { Method = SSLMethodType.DINO })
    {
        _outputDim = outputDim;

        var dinoConfig = _config.DINO ?? new DINOConfig();
        var studentTemp = dinoConfig.StudentTemperature ?? 0.1;
        var teacherTemp = dinoConfig.TeacherTemperatureStart ?? 0.04;
        var centerMomentum = dinoConfig.CenterMomentum ?? 0.9;

        _loss = new DINOLoss<T>(outputDim, studentTemp, teacherTemp, centerMomentum);

        // Configure multi-crop
        NumGlobalCrops = dinoConfig.NumGlobalCrops ?? 2;
        NumLocalCrops = dinoConfig.NumLocalCrops ?? 0;
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create augmented views
        var (globalViews, localViews) = CreateMultiCropViews(batch);

        // Teacher forward pass (global views only)
        var teacherOutputs = new List<Tensor<T>>();
        foreach (var view in globalViews)
        {
            teacherOutputs.Add(ForwardTeacher(view));
        }

        // Student forward pass (all views)
        var studentOutputs = new List<Tensor<T>>();
        foreach (var view in globalViews)
        {
            studentOutputs.Add(ForwardStudent(view));
        }
        foreach (var view in localViews)
        {
            studentOutputs.Add(ForwardStudent(view));
        }

        // Compute DINO loss
        var loss = _loss.ComputeMultiCropLoss(studentOutputs, teacherOutputs);

        // Backward pass through student - accumulate gradients from all student/teacher pairs
        for (int s = 0; s < studentOutputs.Count; s++)
        {
            for (int t = 0; t < teacherOutputs.Count; t++)
            {
                var (_, gradStudent) = _loss.ComputeLossWithGradients(
                    studentOutputs[s], teacherOutputs[t]);

                var gradH = _projector!.Backward(gradStudent);
                _encoder.Backpropagate(gradH);
            }
        }

        // Update networks
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateStudent(learningRate);
        UpdateTeacher();

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize * (NumGlobalCrops + NumLocalCrops);
        result.NumNegativePairs = 0;
        result.Metrics["momentum"] = NumOps.FromDouble(TeacherEncoder.Momentum);
        result.Metrics["center_norm"] = Centering.CenterNorm();

        return result;
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
