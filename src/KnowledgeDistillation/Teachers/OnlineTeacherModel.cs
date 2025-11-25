using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
    private readonly Func<Vector<T>, Vector<T>> _teacherForward;
    private readonly Action<Vector<T>, Vector<T>> _teacherUpdate;
    private readonly OnlineUpdateMode _updateMode;
    private readonly double _updateRate;
    private readonly int _updateFrequency;
    private int _updateCounter;

    /// <summary>
    /// Gets the output dimension of the teacher model.
    /// </summary>
    public override int OutputDimension { get; }

    /// <summary>
    /// Gets or sets whether the teacher is currently updating.
    /// </summary>
    public bool IsUpdating { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the OnlineTeacherModel class.
    /// </summary>
    /// <param name="teacherForward">Function to perform forward pass through teacher.</param>
    /// <param name="teacherUpdate">Function to update teacher parameters (input, gradient).</param>
    /// <param name="outputDimension">Output dimension of the teacher.</param>
    /// <param name="updateMode">How to update the teacher (default: EMA).</param>
    /// <param name="updateRate">Update rate for EMA or learning rate (default: 0.999 for EMA).</param>
    /// <param name="updateFrequency">How often to update (default: every step).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create an online teacher by providing:
    /// - Forward function: Gets teacher predictions
    /// - Update function: Updates teacher parameters
    /// - Update mode: How to update (EMA recommended for stability)</para>
    ///
    /// <para>Example:
    /// <code>
    /// // Teacher model with forward and backward functions
    /// Func&lt;Vector&lt;double&gt;, Vector&lt;double&gt;&gt; teacherForward = input => teacherModel.Forward(input);
    /// Action&lt;Vector&lt;double&gt;, Vector&lt;double&gt;&gt; teacherUpdate = (input, grad) => teacherModel.Backward(grad);
    ///
    /// var onlineTeacher = new OnlineTeacherModel&lt;double&gt;(
    ///     teacherForward: teacherForward,
    ///     teacherUpdate: teacherUpdate,
    ///     outputDimension: 10,
    ///     updateMode: OnlineUpdateMode.EMA,
    ///     updateRate: 0.999  // Slow, stable updates
    /// );
    /// </code>
    /// </para>
    ///
    /// <para><b>Choosing Update Parameters:</b>
    /// - **EMA rate 0.99-0.999**: Slow, stable teacher evolution
    /// - **EMA rate 0.9-0.99**: Faster adaptation to new data
    /// - **Gradient-based**: Use small learning rate (0.0001-0.001)
    /// - **Update frequency**: Every step (1) for continuous, or every N steps for stability</para>
    /// </remarks>
    public OnlineTeacherModel(
        Func<Vector<T>, Vector<T>> teacherForward,
        Action<Vector<T>, Vector<T>> teacherUpdate,
        int outputDimension,
        OnlineUpdateMode updateMode = OnlineUpdateMode.EMA,
        double updateRate = 0.999,
        int updateFrequency = 1)
    {
        _teacherForward = teacherForward ?? throw new ArgumentNullException(nameof(teacherForward));
        _teacherUpdate = teacherUpdate ?? throw new ArgumentNullException(nameof(teacherUpdate));
        OutputDimension = outputDimension;
        _updateMode = updateMode;
        _updateRate = updateRate;
        _updateFrequency = updateFrequency;
        _updateCounter = 0;

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
        var currentOutput = _teacherForward(input);

        // Compute EMA update: new = alpha * current + (1-alpha) * target
        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            // Gradient pushes teacher toward target
            var diff = NumOps.Subtract(targetOutput[i], currentOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(1.0 - _updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate(input, gradient);
    }

    /// <summary>
    /// Updates teacher using gradient-based learning.
    /// </summary>
    private void UpdateGradient(Vector<T> input, Vector<T> targetOutput)
    {
        // Get current prediction
        var currentOutput = _teacherForward(input);

        // Compute MSE gradient: 2 * (current - target)
        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            var diff = NumOps.Subtract(currentOutput[i], targetOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(2.0 * _updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate(input, gradient);
    }

    /// <summary>
    /// Updates teacher using momentum.
    /// </summary>
    private void UpdateMomentum(Vector<T> input, Vector<T> targetOutput)
    {
        // Similar to EMA but with momentum factor
        var currentOutput = _teacherForward(input);

        var gradient = new Vector<T>(currentOutput.Length);
        for (int i = 0; i < currentOutput.Length; i++)
        {
            var diff = NumOps.Subtract(targetOutput[i], currentOutput[i]);
            var scaled = NumOps.Multiply(diff, NumOps.FromDouble(_updateRate));
            gradient[i] = scaled;
        }

        _teacherUpdate(input, gradient);
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
    /// Always <c>false</c>. OnlineTeacherModel uses function delegates which cannot be
    /// exported as a computation graph.
    /// </value>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Not supported for OnlineTeacherModel.
    /// </summary>
    /// <param name="inputNodes">Not used.</param>
    /// <returns>Never returns normally.</returns>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    /// <remarks>
    /// <para>
    /// OnlineTeacherModel uses function delegates for forward pass and updates which are
    /// opaque to the JIT compiler. Function delegates can contain arbitrary code that
    /// cannot be represented as tensor operations.
    /// </para>
    /// <para>
    /// To enable JIT compilation, use a teacher model that wraps an IJitCompilable model
    /// directly instead of using function delegates.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        return ThrowJitNotSupported(
            nameof(OnlineTeacherModel<T>),
            "it uses function delegates which cannot be exported as a computation graph");
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
