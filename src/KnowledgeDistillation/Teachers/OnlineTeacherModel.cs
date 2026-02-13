using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Online teacher model that updates its parameters during student training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike standard distillation where the teacher is frozen,
/// online distillation allows the teacher to continue learning during student training.
/// This is useful for:
/// - Continuous learning scenarios
/// - Evolving data distributions
/// - Co-training teacher and student simultaneously</para>
///
/// <para><b>How It Works:</b>
/// 1. Initialize teacher model (can be pre-trained or random)
/// 2. During student training, also update teacher with new data
/// 3. Teacher provides evolving knowledge to student
/// 4. Both models improve together</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine a mentor and apprentice both continuing to learn as they work together.
/// The mentor (teacher) doesn't just transfer old knowledge - they also learn from new
/// experiences and share those insights with the apprentice (student).</para>
///
/// <para><b>Use Cases:</b>
/// - **Streaming Data**: New data arrives continuously
/// - **Domain Adaptation**: Distribution shifts over time
/// - **Co-training**: Teacher and student help each other
/// - **Incremental Learning**: Models must adapt to new classes/tasks</para>
///
/// <para><b>Update Strategies:</b>
/// - **EMA (Exponential Moving Average)**: Smooth updates, stable teacher
/// - **Periodic Sync**: Update teacher every N steps
/// - **Gradient-based**: Teacher trained with separate loss
/// - **Momentum**: Teacher follows student with momentum</para>
///
/// <para><b>Advantages:</b>
/// - Adapts to changing data
/// - No need for pre-trained teacher
/// - Can improve teacher and student together
/// - Suitable for lifelong learning</para>
///
/// <para><b>Challenges:</b>
/// - Risk of teacher forgetting/degrading
/// - Need careful update rate tuning
/// - More complex training dynamics
/// - Harder to debug</para>
///
/// <para><b>References:</b>
/// - Zhang et al. (2018). Deep Mutual Learning. CVPR.
/// - Anil et al. (2018). Large Scale Distributed Neural Network Training through Online Distillation.</para>
/// </remarks>
public class OnlineTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>>? _teacherForward;
    private readonly IJitCompilable<T>? _jitCompilableModel;
    private readonly Action<Vector<T>, Vector<T>>? _teacherUpdate;
    private readonly OnlineUpdateMode _updateMode;
    private readonly double _updateRate;
    private readonly int _updateFrequency;
    private int _updateCounter;
    private readonly int _inputDim;

    /// <summary>
    /// Gets the output dimension of the teacher model.
    /// </summary>
    public override int OutputDimension { get; }

    /// <summary>
    /// Gets or sets whether the teacher is currently updating.
    /// </summary>
    public bool IsUpdating { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the OnlineTeacherModel class using function delegates.
    /// </summary>
    /// <param name="teacherForward">Function to perform forward pass through teacher.</param>
    /// <param name="inputDimension">Input dimension of the teacher.</param>
    /// <param name="outputDimension">Output dimension of the teacher.</param>
    /// <param name="teacherUpdate">Optional function to update teacher parameters (input, gradient).</param>
    /// <param name="updateMode">How to update the teacher (default: EMA).</param>
    /// <param name="updateRate">Update rate for EMA or learning rate (default: 0.999 for EMA).</param>
    /// <param name="updateFrequency">How often to update (default: every step).</param>
    /// <remarks>
    /// <para><b>Note:</b> This constructor creates a non-JIT-compilable teacher.
    /// For JIT support, use the constructor that accepts an IJitCompilable model.</para>
    /// </remarks>
    public OnlineTeacherModel(
        Func<Vector<T>, Vector<T>> teacherForward,
        int inputDimension,
        int outputDimension,
        Action<Vector<T>, Vector<T>>? teacherUpdate = null,
        OnlineUpdateMode updateMode = OnlineUpdateMode.EMA,
        double updateRate = 0.999,
        int updateFrequency = 1)
    {
        Guard.NotNull(teacherForward);
        _teacherForward = teacherForward;
        _teacherUpdate = teacherUpdate;
        _inputDim = inputDimension;
        OutputDimension = outputDimension;
        _updateMode = updateMode;
        _updateRate = updateRate;
        _updateFrequency = updateFrequency;
        _updateCounter = 0;
        _jitCompilableModel = null;

        if (updateFrequency < 1)
            throw new ArgumentException("Update frequency must be at least 1", nameof(updateFrequency));
        if (updateRate <= 0 || updateRate > 1)
            throw new ArgumentException("Update rate must be in (0, 1]", nameof(updateRate));
    }

    /// <summary>
    /// Initializes a new instance of the OnlineTeacherModel class using a JIT-compilable model.
    /// </summary>
    /// <param name="jitCompilableModel">A JIT-compilable model for forward pass.</param>
    /// <param name="inputDimension">Input dimension of the teacher.</param>
    /// <param name="outputDimension">Output dimension of the teacher.</param>
    /// <param name="teacherUpdate">Optional function to update teacher parameters.</param>
    /// <param name="updateMode">How to update the teacher (default: EMA).</param>
    /// <param name="updateRate">Update rate for EMA or learning rate (default: 0.999 for EMA).</param>
    /// <param name="updateFrequency">How often to update (default: every step).</param>
    /// <remarks>
    /// <para><b>JIT Support:</b> This constructor enables JIT compilation for inference
    /// when the underlying model supports it. Note that updates still use the teacherUpdate
    /// function if provided.</para>
    /// </remarks>
    public OnlineTeacherModel(
        IJitCompilable<T> jitCompilableModel,
        int inputDimension,
        int outputDimension,
        Action<Vector<T>, Vector<T>>? teacherUpdate = null,
        OnlineUpdateMode updateMode = OnlineUpdateMode.EMA,
        double updateRate = 0.999,
        int updateFrequency = 1)
    {
        Guard.NotNull(jitCompilableModel);
        _jitCompilableModel = jitCompilableModel;
        _teacherUpdate = teacherUpdate;
        _inputDim = inputDimension;
        OutputDimension = outputDimension;
        _updateMode = updateMode;
        _updateRate = updateRate;
        _updateFrequency = updateFrequency;
        _updateCounter = 0;
        _teacherForward = null;

        if (updateFrequency < 1)
            throw new ArgumentException("Update frequency must be at least 1", nameof(updateFrequency));
        if (updateRate <= 0 || updateRate > 1)
            throw new ArgumentException("Update rate must be in (0, 1]", nameof(updateRate));
    }

    /// <summary>
    /// Gets logits from the teacher model.
    /// </summary>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        if (_jitCompilableModel != null)
        {
            // IJitCompilable doesn't have execution methods - need to cast to a model interface
            if (_jitCompilableModel is IModel<Vector<T>, Vector<T>, ModelMetadata<T>> model)
            {
                return model.Predict(input);
            }

            throw new InvalidOperationException(
                "Underlying model must implement IModel<Vector<T>, Vector<T>, ModelMetadata<T>> to execute predictions. " +
                "IJitCompilable only provides computation graph export for JIT compilation.");
        }

        if (_teacherForward == null)
            throw new InvalidOperationException("No forward function or JIT-compilable model configured");

        return _teacherForward(input);
    }

    /// <summary>
    /// Updates the teacher model with new data.
    /// </summary>
    /// <param name="input">Input that was used for prediction.</param>
    /// <param name="targetOutput">Target output for the teacher (can be ground truth or student prediction).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this after each batch to update the teacher.
    /// The teacher learns from either:
    /// - Ground truth labels (teacher improves on task)
    /// - Student predictions (mutual learning - teacher learns from student too!)</para>
    ///
    /// <para>Update modes:
    /// - **EMA**: Teacher smoothly tracks student, no explicit gradient
    /// - **GradientBased**: Teacher trained with standard gradient descent
    /// - **MomentumBased**: Teacher follows student with momentum</para>
    /// </remarks>
    public void Update(Vector<T> input, Vector<T> targetOutput)
    {
        if (!IsUpdating)
            return;

        _updateCounter++;

        // Only update at specified frequency
        if (_updateCounter % _updateFrequency != 0)
            return;

        switch (_updateMode)
        {
            case OnlineUpdateMode.EMA:
                UpdateEMA(input, targetOutput);
                break;

            case OnlineUpdateMode.GradientBased:
                UpdateGradient(input, targetOutput);
                break;

            case OnlineUpdateMode.MomentumBased:
                UpdateMomentum(input, targetOutput);
                break;

            default:
                throw new NotImplementedException($"Update mode {_updateMode} not implemented");
        }
    }

    /// <summary>
    /// Updates teacher using exponential moving average.
    /// </summary>
    private void UpdateEMA(Vector<T> input, Vector<T> targetOutput)
    {
        // Get current teacher prediction
        var currentOutput = GetLogits(input);

        // Compute EMA update: new = alpha * current + (1-alpha) * target
        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            // Gradient pushes teacher toward target
            var diff = NumOps.Subtract(targetOutput[i], currentOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(1.0 - _updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate?.Invoke(input, gradient);
    }

    /// <summary>
    /// Updates teacher using gradient-based learning.
    /// </summary>
    private void UpdateGradient(Vector<T> input, Vector<T> targetOutput)
    {
        // Get current prediction
        var currentOutput = GetLogits(input);

        // Compute MSE gradient: 2 * (current - target)
        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            var diff = NumOps.Subtract(currentOutput[i], targetOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(2.0 * _updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate?.Invoke(input, gradient);
    }

    /// <summary>
    /// Updates teacher using momentum.
    /// </summary>
    private void UpdateMomentum(Vector<T> input, Vector<T> targetOutput)
    {
        // Similar to EMA but with momentum factor
        var currentOutput = GetLogits(input);

        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            var diff = NumOps.Subtract(targetOutput[i], currentOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(_updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate?.Invoke(input, gradient);
    }

    /// <summary>
    /// Pauses teacher updates (freezes teacher).
    /// </summary>
    public void PauseUpdates() => IsUpdating = false;

    /// <summary>
    /// Resumes teacher updates.
    /// </summary>
    public void ResumeUpdates() => IsUpdating = true;

    /// <summary>
    /// Resets the update counter.
    /// </summary>
    public void ResetCounter() => _updateCounter = 0;

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if constructed with an IJitCompilable model that supports JIT compilation;
    /// <c>false</c> if constructed with function delegates which cannot be exported as a computation graph.
    /// </value>
    public override bool SupportsJitCompilation => _jitCompilableModel?.SupportsJitCompilation ?? false;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <exception cref="NotSupportedException">Thrown when using function delegates instead of an IJitCompilable model.</exception>
    /// <remarks>
    /// <para>
    /// When constructed with an IJitCompilable model, this method delegates to the underlying model's
    /// computation graph export. When constructed with function delegates, JIT compilation is not supported
    /// because function delegates can contain arbitrary code that cannot be represented as tensor operations.
    /// </para>
    /// <para>
    /// To enable JIT compilation, use the constructor that accepts an IJitCompilable model
    /// instead of using function delegates.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_jitCompilableModel != null && _jitCompilableModel.SupportsJitCompilation)
        {
            return _jitCompilableModel.ExportComputationGraph(inputNodes);
        }

        return ThrowJitNotSupported(
            nameof(OnlineTeacherModel<T>),
            "it uses function delegates which cannot be exported as a computation graph. Use the constructor that accepts an IJitCompilable model instead");
    }
}

/// <summary>
/// Defines how an online teacher model is updated during training.
/// </summary>
public enum OnlineUpdateMode
{
    /// <summary>
    /// Exponential Moving Average - smooth, stable updates.
    /// Teacher slowly tracks toward target without explicit gradients.
    /// </summary>
    EMA,

    /// <summary>
    /// Gradient-based updates - standard gradient descent on teacher.
    /// Teacher optimized with its own loss function.
    /// </summary>
    GradientBased,

    /// <summary>
    /// Momentum-based updates - teacher follows with momentum.
    /// Combines aspects of EMA and gradient-based.
    /// </summary>
    MomentumBased
}
