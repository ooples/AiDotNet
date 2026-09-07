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
    private readonly ConcurrentDictionary<Tensor<T>, T[]> _z = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>The running average of the fast iterate, and what the method converges on.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, T[]> _x = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>Second moment for the AdamW step.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, T[]> _v = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private int _step;

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
    public override void Step(TapeStepContext<T> context)
    {
        PrepareTapeState(context);
        _step++;

        double beta = _options.Interpolation;
        double beta2 = _options.Beta2;
        double epsilon = _options.Epsilon;
        double decay = _options.WeightDecay;
        double gamma = StepSize(_step);

        // Equal-weighted averaging: c = 1/t makes x the running mean of every z so far.
        double c = 1.0 / _step;

        foreach (var param in context.Parameters)
        {
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, Engine, out var grad))
                continue;

            var paramSpan = param.AsWritableSpan();
            var gradSpan = grad.AsSpan();
            int length = Math.Min(paramSpan.Length, gradSpan.Length);
            if (length == 0) continue;

            // z and x both start at the initial point, which is what the parameters hold before the
            // first step. Seeding them from y afterwards would fold the interpolation into itself.
            if (!_z.TryGetValue(param, out var z))
            {
                z = Snapshot(paramSpan, length);
                _z[param] = z;
            }

            if (!_x.TryGetValue(param, out var x))
            {
                x = Snapshot(paramSpan, length);
                _x[param] = x;
            }

            var v = _v.GetOrAdd(param, _ => new T[length]);

            for (int i = 0; i < length; i++)
            {
                double g = NumOps.ToDouble(gradSpan[i]);

                double second = beta2 * NumOps.ToDouble(v[i]) + (1 - beta2) * g * g;
                v[i] = NumOps.FromDouble(second);
                double corrected = second / (1 - Math.Pow(beta2, _step));

                // z takes the step. Weight decay is decoupled, so it acts on z directly rather
                // than entering the second moment.
                double zi = NumOps.ToDouble(z[i]);
                zi -= gamma * g / (Math.Sqrt(corrected) + epsilon);
                if (decay != 0.0) zi -= gamma * decay * zi;
                z[i] = NumOps.FromDouble(zi);

                // x is the running average of z, and y is where the next gradient will be taken.
                double xi = (1 - c) * NumOps.ToDouble(x[i]) + c * zi;
                x[i] = NumOps.FromDouble(xi);

                paramSpan[i] = NumOps.FromDouble((1 - beta) * zi + beta * xi);
            }
        }
    }

    private T[] Snapshot(ReadOnlySpan<T> values, int length)
    {
        var copy = new T[length];
        for (int i = 0; i < length; i++) copy[i] = values[i];
        return copy;
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

        _step++;
        if (_flatZ.Length != parameters.Length)
        {
            _flatZ = new double[parameters.Length];
            _flatX = new double[parameters.Length];
            _flatV = new double[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                _flatZ[i] = NumOps.ToDouble(parameters[i]);
                _flatX[i] = _flatZ[i];
            }
        }

        double beta = _options.Interpolation;
        double beta2 = _options.Beta2;
        double epsilon = _options.Epsilon;
        double decay = _options.WeightDecay;
        double gamma = StepSize(_step);
        double c = 1.0 / _step;

        var updated = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);

            _flatV[i] = beta2 * _flatV[i] + (1 - beta2) * g * g;
            double corrected = _flatV[i] / (1 - Math.Pow(beta2, _step));

            _flatZ[i] -= gamma * g / (Math.Sqrt(corrected) + epsilon);
            if (decay != 0.0) _flatZ[i] -= gamma * decay * _flatZ[i];

            _flatX[i] = (1 - c) * _flatX[i] + c * _flatZ[i];
            updated[i] = NumOps.FromDouble((1 - beta) * _flatZ[i] + beta * _flatX[i]);
        }

        return updated;
    }

    private double[] _flatZ = [];
    private double[] _flatX = [];
    private double[] _flatV = [];

    /// <summary>The running average, which is the point the method converges on.</summary>
    /// <remarks>
    /// Exposed because it is what a paper means by "the model": the parameters hold y, the point
    /// gradients are taken at, and x is what should be evaluated and saved.
    /// </remarks>
    public Vector<T> AveragedParameters()
    {
        var result = new Vector<T>(_flatX.Length);
        for (int i = 0; i < _flatX.Length; i++) result[i] = NumOps.FromDouble(_flatX[i]);
        return result;
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
        _flatZ = [];
        _flatX = [];
        _flatV = [];
        _step = 0;
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
