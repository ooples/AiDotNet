using System.Text.Json;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Learning without Forgetting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LwFOptions<T>
{
    /// <summary>
    /// Gets or sets the distillation temperature.
    /// Higher values produce softer probability distributions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>T = 1: Normal softmax (sharp distribution)</description></item>
    /// <item><description>T = 2-5: Soft distributions (typical for distillation)</description></item>
    /// <item><description>T > 10: Very soft (all classes become similar)</description></item>
    /// </list>
    /// <para>Higher temperature helps transfer more information about uncertainty.</para>
    /// </remarks>
    public double? Temperature { get; set; }

    /// <summary>
    /// Gets or sets the weight for distillation loss relative to task loss.
    /// </summary>
    /// <remarks>
    /// <para>L_total = α * L_task + (1-α) * L_distill where α = 1/(1 + DistillationWeight)</para>
    /// <para>Higher values give more importance to preserving old knowledge.</para>
    /// </remarks>
    public double? DistillationWeight { get; set; }

    /// <summary>
    /// Gets or sets whether to use warmup for distillation weight.
    /// Gradually increases distillation weight during training.
    /// </summary>
    public bool? UseWarmup { get; set; }

    /// <summary>
    /// Gets or sets the number of warmup epochs.
    /// </summary>
    public int? WarmupEpochs { get; set; }

    /// <summary>
    /// Gets or sets the distillation loss type.
    /// </summary>
    public DistillationLossType? LossType { get; set; }

    /// <summary>
    /// Gets or sets whether to distill from intermediate layers.
    /// Helps preserve richer representations but requires compatible architectures.
    /// </summary>
    public bool? UseFeatureDistillation { get; set; }
}

/// <summary>
/// Types of distillation loss functions.
/// </summary>
public enum DistillationLossType
{
    /// <summary>
    /// KL Divergence - standard distillation loss.
    /// </summary>
    KLDivergence,

    /// <summary>
    /// Mean Squared Error between soft targets.
    /// </summary>
    MSE,

    /// <summary>
    /// Cross-entropy with soft targets.
    /// </summary>
    SoftCrossEntropy,

    /// <summary>
    /// Symmetric KL Divergence (KL(T||S) + KL(S||T)).
    /// </summary>
    SymmetricKL
}

/// <summary>
/// Learning without Forgetting (LwF) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> LwF prevents forgetting by using knowledge distillation.
/// Instead of storing old data, it stores the model's predictions (knowledge) and
/// trains the new model to match those predictions while also learning the new task.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Before learning a new task, save the current model as "teacher"</description></item>
/// <item><description>For each training sample, get predictions from both teacher and student</description></item>
/// <item><description>Train to minimize: L_task (new task) + λ * L_distill (match teacher)</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>L_distill = T² * KL(softmax(z_teacher/T) || softmax(z_student/T))</para>
/// <para>Where T is the temperature and z are the logits (pre-softmax outputs).</para>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>No need to store previous task data (memory efficient)</description></item>
/// <item><description>Works well when tasks share output space (e.g., multi-class classification)</description></item>
/// <item><description>Fast - no gradient projection or quadratic programming</description></item>
/// </list>
///
/// <para><b>Disadvantages:</b></para>
/// <list type="bullet">
/// <item><description>Requires outputs to be comparable across tasks</description></item>
/// <item><description>May not work well for very different task distributions</description></item>
/// <item><description>Teacher model doubles memory during training</description></item>
/// </list>
///
/// <para><b>Reference:</b> Li and Hoiem "Learning without Forgetting" (ECCV 2016)</para>
/// </remarks>
public class LearningWithoutForgetting<T, TInput, TOutput>
    : ContinualLearningStrategyBase<T, TInput, TOutput>,
      IDistillationStrategy<T, TInput, TOutput>
{
    private readonly T _temperature;
    private readonly T _distillationWeight;
    private readonly bool _useWarmup;
    private readonly int _warmupEpochs;
    private readonly DistillationLossType _lossType;
    private readonly bool _useFeatureDistillation;

    // Teacher model (frozen copy from before current task)
    private IFullModel<T, TInput, TOutput>? _teacherModel;

    // Training state
    private int _currentEpoch;
    private int _totalDistillationCalls;
    private T _totalDistillationLoss;

    /// <summary>
    /// Initializes a new LwF strategy with default options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="distillationTemperature">Temperature for softening distributions.</param>
    /// <param name="distillationWeight">Weight for distillation loss.</param>
    public LearningWithoutForgetting(
        ILossFunction<T> lossFunction,
        double distillationTemperature = 2.0,
        double distillationWeight = 1.0)
        : this(lossFunction, new LwFOptions<T>
        {
            Temperature = distillationTemperature,
            DistillationWeight = distillationWeight
        })
    {
    }

    /// <summary>
    /// Initializes a new LwF strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public LearningWithoutForgetting(ILossFunction<T> lossFunction, LwFOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new LwFOptions<T>();

        _temperature = NumOps.FromDouble(opts.Temperature ?? 2.0);
        _distillationWeight = NumOps.FromDouble(opts.DistillationWeight ?? 1.0);
        _useWarmup = opts.UseWarmup ?? false;
        _warmupEpochs = opts.WarmupEpochs ?? 5;
        _lossType = opts.LossType ?? DistillationLossType.KLDivergence;
        _useFeatureDistillation = opts.UseFeatureDistillation ?? false;

        _totalDistillationLoss = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override string Name => "LwF";

    /// <inheritdoc/>
    public override bool RequiresMemoryBuffer => false;

    /// <inheritdoc/>
    public override bool ModifiesArchitecture => false;

    /// <inheritdoc/>
    public override long MemoryUsageBytes
    {
        get
        {
            // Teacher model memory (approximately same as main model)
            if (_teacherModel != null)
            {
                return _teacherModel.ParameterCount * GetTypeSize();
            }
            return 0;
        }
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput>? TeacherModel => _teacherModel;

    /// <inheritdoc/>
    public T Temperature => _temperature;

    /// <inheritdoc/>
    public T DistillationWeight => GetEffectiveDistillationWeight();

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // Reset epoch counter
        _currentEpoch = 0;
        _totalDistillationCalls = 0;
        _totalDistillationLoss = NumOps.Zero;

        // Clone current model as teacher (if we have learned tasks)
        if (TaskCount > 0)
        {
            _teacherModel = model.Clone();
            RecordMetric($"Task{TaskCount}_TeacherCreated", DateTime.UtcNow);
        }
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // LwF doesn't use parameter-based regularization
        // Distillation loss is computed per-sample during training via ComputeDistillationLoss
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // LwF doesn't adjust gradients directly
        // The distillation loss contributes gradients through backpropagation
        return gradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Update teacher model with the trained model
        var currentParams = model.GetParameters();

        if (_teacherModel == null)
        {
            _teacherModel = model.Clone();
        }
        else
        {
            _teacherModel.SetParameters(currentParams);
        }

        // Record metrics
        if (_totalDistillationCalls > 0)
        {
            var avgLoss = NumOps.Divide(_totalDistillationLoss, NumOps.FromDouble(_totalDistillationCalls));
            RecordMetric($"Task{TaskCount}_AvgDistillationLoss", Convert.ToDouble(avgLoss));
        }
        RecordMetric($"Task{TaskCount}_TotalDistillationCalls", _totalDistillationCalls);

        TaskCount++;
    }

    /// <inheritdoc/>
    public T ComputeDistillationLoss(TOutput teacherOutput, TOutput studentOutput)
    {
        // For Vector<T> outputs (classification)
        if (teacherOutput is Vector<T> teacherVec && studentOutput is Vector<T> studentVec)
        {
            return ComputeDistillationLoss(teacherVec, studentVec);
        }

        // For other output types, return zero (not supported)
        return NumOps.Zero;
    }

    /// <summary>
    /// Computes the distillation loss between teacher and student predictions.
    /// </summary>
    /// <param name="teacherOutput">Logits from the teacher model.</param>
    /// <param name="studentOutput">Logits from the student model.</param>
    /// <returns>The distillation loss value.</returns>
    public T ComputeDistillationLoss(Vector<T> teacherOutput, Vector<T> studentOutput)
    {
        if (teacherOutput.Length != studentOutput.Length)
            throw new ArgumentException("Teacher and student outputs must have the same length");

        _totalDistillationCalls++;

        T loss;
        switch (_lossType)
        {
            case DistillationLossType.KLDivergence:
                loss = ComputeKLDivergence(teacherOutput, studentOutput);
                break;
            case DistillationLossType.MSE:
                loss = ComputeMSELoss(teacherOutput, studentOutput);
                break;
            case DistillationLossType.SoftCrossEntropy:
                loss = ComputeSoftCrossEntropy(teacherOutput, studentOutput);
                break;
            case DistillationLossType.SymmetricKL:
                loss = ComputeSymmetricKL(teacherOutput, studentOutput);
                break;
            default:
                loss = ComputeKLDivergence(teacherOutput, studentOutput);
                break;
        }

        // Apply distillation weight
        var effectiveWeight = GetEffectiveDistillationWeight();
        loss = NumOps.Multiply(loss, effectiveWeight);

        _totalDistillationLoss = NumOps.Add(_totalDistillationLoss, loss);

        return loss;
    }

    /// <summary>
    /// Computes KL divergence with temperature scaling.
    /// </summary>
    private T ComputeKLDivergence(Vector<T> teacherLogits, Vector<T> studentLogits)
    {
        var softTeacher = SoftmaxWithTemperature(teacherLogits, _temperature);
        var softStudent = SoftmaxWithTemperature(studentLogits, _temperature);

        T kl = NumOps.Zero;
        for (int i = 0; i < softTeacher.Length; i++)
        {
            var teacherProb = Convert.ToDouble(softTeacher[i]);
            if (teacherProb > 1e-10)
            {
                var studentProb = Math.Max(Convert.ToDouble(softStudent[i]), 1e-10);
                var logRatio = Math.Log(teacherProb / studentProb);
                var term = teacherProb * logRatio;
                kl = NumOps.Add(kl, NumOps.FromDouble(term));
            }
        }

        // Scale by T^2 (gradient scaling factor)
        var tempSquared = NumOps.Multiply(_temperature, _temperature);
        return NumOps.Multiply(kl, tempSquared);
    }

    /// <summary>
    /// Computes MSE between soft targets.
    /// </summary>
    private T ComputeMSELoss(Vector<T> teacherLogits, Vector<T> studentLogits)
    {
        var softTeacher = SoftmaxWithTemperature(teacherLogits, _temperature);
        var softStudent = SoftmaxWithTemperature(studentLogits, _temperature);

        T mse = NumOps.Zero;
        for (int i = 0; i < softTeacher.Length; i++)
        {
            var diff = NumOps.Subtract(softTeacher[i], softStudent[i]);
            var squared = NumOps.Multiply(diff, diff);
            mse = NumOps.Add(mse, squared);
        }

        return NumOps.Divide(mse, NumOps.FromDouble(softTeacher.Length));
    }

    /// <summary>
    /// Computes soft cross-entropy loss.
    /// </summary>
    private T ComputeSoftCrossEntropy(Vector<T> teacherLogits, Vector<T> studentLogits)
    {
        var softTeacher = SoftmaxWithTemperature(teacherLogits, _temperature);
        var softStudent = SoftmaxWithTemperature(studentLogits, _temperature);

        T ce = NumOps.Zero;
        for (int i = 0; i < softTeacher.Length; i++)
        {
            var studentProb = Math.Max(Convert.ToDouble(softStudent[i]), 1e-10);
            var teacherProb = Convert.ToDouble(softTeacher[i]);
            var term = -teacherProb * Math.Log(studentProb);
            ce = NumOps.Add(ce, NumOps.FromDouble(term));
        }

        return ce;
    }

    /// <summary>
    /// Computes symmetric KL divergence.
    /// </summary>
    private T ComputeSymmetricKL(Vector<T> teacherLogits, Vector<T> studentLogits)
    {
        var forward = ComputeKLDivergence(teacherLogits, studentLogits);
        var backward = ComputeKLDivergence(studentLogits, teacherLogits);
        return NumOps.Divide(NumOps.Add(forward, backward), NumOps.FromDouble(2.0));
    }

    /// <summary>
    /// Applies softmax with temperature scaling.
    /// </summary>
    private Vector<T> SoftmaxWithTemperature(Vector<T> logits, T temperature)
    {
        var scaled = new T[logits.Length];

        // Find max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (Convert.ToDouble(logits[i]) > Convert.ToDouble(maxLogit))
                maxLogit = logits[i];
        }

        // Compute exp((logit - max) / T) and sum
        T sum = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            var shifted = NumOps.Subtract(logits[i], maxLogit);
            var divided = NumOps.Divide(shifted, temperature);
            var expValue = NumOps.FromDouble(Math.Exp(Convert.ToDouble(divided)));
            scaled[i] = expValue;
            sum = NumOps.Add(sum, expValue);
        }

        // Normalize
        for (int i = 0; i < scaled.Length; i++)
        {
            scaled[i] = NumOps.Divide(scaled[i], sum);
        }

        return new Vector<T>(scaled);
    }

    /// <summary>
    /// Gets the effective distillation weight, accounting for warmup.
    /// </summary>
    private T GetEffectiveDistillationWeight()
    {
        if (!_useWarmup || _currentEpoch >= _warmupEpochs)
        {
            return _distillationWeight;
        }

        // Linear warmup
        var factor = (double)_currentEpoch / _warmupEpochs;
        return NumOps.Multiply(_distillationWeight, NumOps.FromDouble(factor));
    }

    /// <summary>
    /// Advances to the next epoch (for warmup tracking).
    /// </summary>
    public void AdvanceEpoch()
    {
        _currentEpoch++;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _teacherModel = null;
        _currentEpoch = 0;
        _totalDistillationCalls = 0;
        _totalDistillationLoss = NumOps.Zero;
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["Temperature"] = Convert.ToDouble(_temperature);
        state["DistillationWeight"] = Convert.ToDouble(_distillationWeight);
        state["LossType"] = _lossType.ToString();
        state["UseWarmup"] = _useWarmup;
        state["WarmupEpochs"] = _warmupEpochs;
        return state;
    }

    /// <summary>
    /// Gets distillation statistics.
    /// </summary>
    public (int TotalCalls, double AverageLoss) GetDistillationStats()
    {
        if (_totalDistillationCalls == 0)
            return (0, 0);

        var avgLoss = Convert.ToDouble(_totalDistillationLoss) / _totalDistillationCalls;
        return (_totalDistillationCalls, avgLoss);
    }
}
