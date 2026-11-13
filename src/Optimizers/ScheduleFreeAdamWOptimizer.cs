using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Schedule-Free AdamW (Defazio et al. 2024): no learning-rate schedule, by averaging instead.
/// </summary>
/// <remarks>
/// <para>
/// Three sequences rather than one. <c>z</c> is the fast iterate that takes the AdamW step,
/// <c>x</c> is an equal-weighted running average of it, and gradients are evaluated at
/// <c>y = (1 - beta) z + beta x</c>, which sits between them. The averaging does the work that
/// annealing a learning rate normally does, so nothing needs to know the length of the run in
/// advance — which is the point: a scheduled run stopped early never finished decaying, and one
/// extended past its horizon decayed too soon.
/// </para>
/// <para>
/// From the paper, equations 3 to 5: y_t = (1 - beta) z_t + beta x_t, z_{t+1} = z_t - gamma g_t,
/// and x_{t+1} = (1 - c_{t+1}) x_t + c_{t+1} z_{t+1} with c_{t+1} = 1/(t+1). Beta interpolates
/// between Polyak-Ruppert averaging at 0 and primal averaging at 1.
/// </para>
/// <para>
/// The model's parameters always hold <c>y</c>, never <c>z</c> or <c>x</c>. That is not an
/// implementation convenience: the gradient must be taken AT y for the method to be what the paper
/// describes, and the surrounding trainer computes gradients wherever the parameters currently are.
/// Reading the weights mid-run therefore gives y, which is also the point one should evaluate.
/// </para>
/// <para><b>For Beginners:</b> Normal training slows down near the end so it can settle. That
/// requires knowing when the end is. This keeps a running average of where it has been and reports
/// that instead, so it is settled whenever you stop.
/// </para>
/// </remarks>
public class ScheduleFreeAdamWOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    private ScheduleFreeAdamWOptimizerOptions<T, TInput, TOutput> _options
        => (ScheduleFreeAdamWOptimizerOptions<T, TInput, TOutput>)Options;

    /// <summary>The fast iterate, which takes the AdamW step.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _z = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>The running average of the fast iterate, and what the method converges on.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _x = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>Second moment for the AdamW step.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _v = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private int _step;

    /// <summary>
    /// Step count of the flat-vector path (<see cref="UpdateParameters"/>), kept apart from the tape path's
    /// <see cref="_step"/>: the two paths keep separate state, so switching between them must not restart the
    /// other's step size, averaging weight or bias correction.
    /// </summary>
    private int _flatStep;

    /// <summary>Creates a schedule-free AdamW optimizer for a model.</summary>
    public ScheduleFreeAdamWOptimizer(
        IFullModel<T, TInput, TOutput>? model = null,
        ScheduleFreeAdamWOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new ScheduleFreeAdamWOptimizerOptions<T, TInput, TOutput>())
    {
    }

    /// <inheritdoc />
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions() => _options;

    /// <summary>The step size, ramped during warmup and flat afterwards.</summary>
    /// <remarks>
    /// The only schedule this optimizer keeps, and the paper keeps it too: averaging cannot
    /// stabilise the earliest updates because there is almost nothing yet to average.
    /// </remarks>
    private double StepSize(int step)
    {
        double rate = _options.InitialLearningRate;
        if (_options.WarmupSteps <= 0 || step >= _options.WarmupSteps) return rate;

        // 1-indexed so the first step is not a no-op, for the same reason the recipe path ramps
        // from one step's worth rather than from zero.
        return rate * step / _options.WarmupSteps;
    }

    /// <inheritdoc />
    protected override void StepCore(TapeStepContext<T> context)
    {
        PrepareTapeState(context);
        _step++;

        double beta = _options.Interpolation;
        double beta2 = _options.Beta2;
        double epsilon = _options.Epsilon;
        double decay = _options.WeightDecay;
        bool applyWeightDecay = double.IsNaN(decay) || Math.Abs(decay) > 0.0;
        double gamma = StepSize(_step);

        // Equal-weighted averaging: c = 1/t makes x the running mean of every z so far.
        double c = 1.0 / _step;

        foreach (var param in context.Parameters)
        {
            if (SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(
                    context, param, Engine, out var gradient))
            {
                StepParameter(
                    param, gradient, beta, beta2, epsilon, decay, applyWeightDecay, gamma, c);
            }
        }
    }

    private void StepParameter(
        Tensor<T> parameter,
        Tensor<T> gradient,
        double beta,
        double beta2,
        double epsilon,
        double decay,
        bool applyWeightDecay,
        double gamma,
        double averagingWeight)
    {
        int length = parameter.Length;
        if (length == 0) return;
        if (gradient.Length != length)
        {
            throw new ArgumentException(
                $"Gradient length ({gradient.Length}) must match parameter length ({length}).", nameof(gradient));
        }

        var engine = Engine;
        var g = gradient._shape.SequenceEqual(parameter._shape) ? gradient : engine.Reshape(gradient, parameter._shape);

        // z and x both start at the initial point, which is what the parameters hold before the
        // first step. Seeding them from y afterwards would fold the interpolation into itself.
        if (!_z.TryGetValue(parameter, out var z))
        {
            z = parameter.Clone();
            _z[parameter] = z;
        }

        if (!_x.TryGetValue(parameter, out var x))
        {
            x = parameter.Clone();
            _x[parameter] = x;
        }

        var v = _v.GetOrAdd(parameter, _ => new Tensor<T>(parameter._shape));

        var y = ScheduleFreeStep(g, z, x, v, _step, beta, beta2, epsilon, decay, applyWeightDecay, gamma, averagingWeight);
        engine.TensorCopy(y, parameter);
    }

    /// <summary>
    /// One schedule-free AdamW step on whole tensors: updates the second moment v, the iterate z (decoupled decay
    /// acting on z) and its running average x in place, and returns y = (1 - beta) * z + beta * x, the point the
    /// next gradient is taken at.
    /// </summary>
    private Tensor<T> ScheduleFreeStep(
        Tensor<T> g,
        Tensor<T> z,
        Tensor<T> x,
        Tensor<T> v,
        int step,
        double beta,
        double beta2,
        double epsilon,
        double decay,
        bool applyWeightDecay,
        double gamma,
        double averagingWeight)
    {
        var engine = Engine;
        var second = engine.TensorAddScaled(
            v, engine.TensorMultiply(g, g), NumOps.FromDouble(beta2), NumOps.FromDouble(1 - beta2));
        engine.TensorCopy(second, v);
        var corrected = engine.TensorDivideScalar(second, NumOps.FromDouble(1 - Math.Pow(beta2, step)));

        // z takes the step. Weight decay is decoupled, so it acts on z directly rather
        // than entering the second moment.
        var newZ = engine.TensorSubtract(z, engine.TensorDivide(
            engine.TensorMultiplyScalar(g, NumOps.FromDouble(gamma)),
            engine.TensorAddScalar(engine.TensorSqrt(corrected), NumOps.FromDouble(epsilon))));
        if (applyWeightDecay)
            newZ = engine.TensorSubtract(newZ, engine.TensorMultiplyScalar(newZ, NumOps.FromDouble(gamma * decay)));
        engine.TensorCopy(newZ, z);

        // x is the running average of z, and y is where the next gradient will be taken.
        var newX = engine.TensorAddScaled(x, newZ, NumOps.FromDouble(1 - averagingWeight), NumOps.FromDouble(averagingWeight));
        engine.TensorCopy(newX, x);

        return engine.TensorAddScaled(newZ, newX, NumOps.FromDouble(1 - beta), NumOps.FromDouble(beta));
    }

    /// <inheritdoc />
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) must match gradient vector length "
                + $"({gradient.Length}).",
                nameof(gradient));
        }

        if (_flatZ.Length != parameters.Length)
        {
            // A differently sized vector is a different parameter set, so its schedule restarts too: the step
            // count drives the step size, the averaging weight c = 1/t and the second-moment bias correction.
            _flatStep = 0;
            _flatZ = parameters.Clone();
            _flatX = parameters.Clone();
            _flatV = new Vector<T>(parameters.Length);
        }

        _flatStep++;
        double beta = _options.Interpolation;
        double beta2 = _options.Beta2;
        double epsilon = _options.Epsilon;
        double decay = _options.WeightDecay;
        bool applyWeightDecay = double.IsNaN(decay) || Math.Abs(decay) > 0.0;
        double gamma = StepSize(_flatStep);
        double c = 1.0 / _flatStep;

        return ScheduleFreeStep(gradient.AsTensor(), _flatZ.AsTensor(), _flatX.AsTensor(), _flatV.AsTensor(), _flatStep,
            beta, beta2, epsilon, decay, applyWeightDecay, gamma, c).AsVector();
    }

    private Vector<T> _flatZ = new Vector<T>(0);
    private Vector<T> _flatX = new Vector<T>(0);
    private Vector<T> _flatV = new Vector<T>(0);

    /// <summary>The running average, which is the point the method converges on.</summary>
    /// <remarks>
    /// Exposed because it is what a paper means by "the model": the parameters hold y, the point
    /// gradients are taken at, and x is what should be evaluated and saved.
    /// </remarks>
    public Vector<T> AveragedParameters()
    {
        return _flatX.Clone();
    }

    /// <inheritdoc />
    public override OptimizationResult<T, TInput, TOutput> Optimize(
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _z.Clear();
        _x.Clear();
        _v.Clear();
        _flatZ = new Vector<T>(0);
        _flatX = new Vector<T>(0);
        _flatV = new Vector<T>(0);
        _step = 0;
        _flatStep = 0;
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                currentSolution = UpdateSolution(currentSolution, gradient);
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <inheritdoc />
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameterized = InterfaceGuard.Parameterizable(currentSolution);
        var updated = UpdateParameters(parameterized.GetParameters(), gradient);
        return parameterized.WithParameters(updated);
    }
}
