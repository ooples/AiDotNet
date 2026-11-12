using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Enums;
namespace AiDotNet.Optimizers;

/// <summary>
/// Adafactor (Shazeer and Stern 2018): Adam-style adaptivity with O(n+m) state instead of O(nm).
/// </summary>
/// <remarks>
/// <para>
/// Adam keeps a second-moment estimate per weight. Adafactor keeps, for a matrix, only a per-row
/// and a per-column sum of squared gradients, and reconstructs an approximation of the full matrix
/// as their outer product divided by their total. For an n by m weight that is n+m numbers rather
/// than nm, which is what makes it the optimizer of choice for very large embedding and projection
/// layers. Vectors are not factored — there is nothing to gain — and the paper keeps a full moment
/// for them, as this does.
/// </para>
/// <para>
/// Two further departures from Adam, both load-bearing rather than incidental. There is no first
/// moment by default, which is most of the memory saving. And the step is clipped by its own RMS
/// rather than the gradient being clipped by norm, which is what lets the paper's schedules run
/// without warmup.
/// </para>
/// <para><b>For Beginners:</b> Optimizers that adapt per-weight step sizes have to remember
/// something about every weight, which for a large model can cost as much memory as the model
/// itself. Adafactor remembers a summary per row and per column instead and reconstructs the rest,
/// trading a little precision for the ability to train models that otherwise would not fit.
/// </para>
/// <para>
/// This implementation is deliberately plain: unlike the older optimizers here it has no GPU or
/// sparse-embedding fast path. It is correct and readable rather than tuned, and the factored
/// reduction it needs is not one of the engine's existing tensor operations.
/// </para>
/// </remarks>
public class AdafactorOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    private AdafactorOptimizerOptions<T, TInput, TOutput> _options
        => (AdafactorOptimizerOptions<T, TInput, TOutput>)Options;

    /// <summary>Per-tensor row sums of squared gradients, for parameters that are factored.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _rows = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>Per-tensor column sums of squared gradients, for parameters that are factored.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _columns = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>Full second moment, for parameters with too few dimensions to factor.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _full = new(TensorReferenceComparer<Tensor<T>>.Instance);

    /// <summary>First moment, kept only when the recipe asks for one.</summary>
    private readonly ConcurrentDictionary<Tensor<T>, Tensor<T>> _momentum = new(TensorReferenceComparer<Tensor<T>>.Instance);

    private int _step;

    /// <summary>Creates an Adafactor optimizer for a model.</summary>
    public AdafactorOptimizer(
        IFullModel<T, TInput, TOutput>? model = null,
        AdafactorOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new AdafactorOptimizerOptions<T, TInput, TOutput>())
    {
    }

    /// <inheritdoc />
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions() => _options;

    /// <summary>The second-moment decay for this step.</summary>
    /// <remarks>
    /// The paper's schedule is 1 - t^-0.8: it forgets quickly at first, when the estimate is built
    /// from almost no data, and settles as evidence accumulates. A recipe that states a fixed beta2
    /// overrides it.
    /// </remarks>
    private double SecondMomentDecay(int step)
        => double.IsNaN(_options.Beta2)
            ? 1.0 - Math.Pow(step, -0.8)
            : _options.Beta2;

    /// <summary>The step size for this step, before the update is scaled by it.</summary>
    /// <remarks>
    /// With the relative rule the size is min(1e-2, 1/sqrt(t)) scaled by the RMS of the parameters,
    /// so the step stays proportional to the weights it moves — a layer whose weights are tiny takes
    /// tiny steps without anyone tuning a rate for it. The floor keeps a zero-initialised layer from
    /// being frozen at a step size of zero.
    /// </remarks>
    private double StepSize(int step, double parameterRms)
    {
        double rate = _options.UseRelativeStepSize
            ? Math.Min(1e-2, 1.0 / Math.Sqrt(step))
            : _options.InitialLearningRate > 0 ? _options.InitialLearningRate : NumOps.ToDouble(CurrentLearningRate);
        bool scaleByParameters = _options.MultiplyByParameterScale is bool explicitScale ? explicitScale : _options.UseRelativeStepSize;
        return scaleByParameters ? rate * Math.Max(_options.ParameterScaleFloor, parameterRms) : rate;
    }

    /// <summary>Root mean square of every element, as one engine sum-of-squares reduction.</summary>
    private double Rms(Tensor<T> values)
    {
        if (values.Length == 0) return 0.0;
        return Math.Sqrt(NumOps.ToDouble(Engine.TensorSumOfSquares(values)) / values.Length);
    }

    /// <summary>Whether a parameter has enough structure for the factored estimate to save anything.</summary>
    /// <remarks>
    /// Rank one is a vector, where a row and a column sum together cost as much as the full moment
    /// and approximate it worse. Higher ranks collapse to a matrix of (everything else) by (last
    /// dimension), which is how the reference implementations treat convolution kernels.
    /// </remarks>
    private static bool IsFactorable(int[] shape) => shape.Length >= 2 && shape[shape.Length - 1] > 1;

    private static (int Rows, int Columns) AsMatrix(int[] shape)
    {
        int columns = shape[shape.Length - 1];
        int rows = 1;
        for (int i = 0; i < shape.Length - 1; i++) rows *= shape[i];
        return (rows, columns);
    }

    /// <inheritdoc />
    protected override void StepCore(TapeStepContext<T> context)
    {
        PrepareTapeState(context);
        _step++;

        double beta2 = SecondMomentDecay(_step);
        double epsilon = _options.Epsilon;
        double clip = _options.UpdateClippingThreshold;
        double decay = _options.WeightDecay;
        bool applyWeightDecay = double.IsNaN(decay) || Math.Abs(decay) > 0.0;
        bool keepMomentum = !double.IsNaN(_options.Beta1);
        double beta1 = keepMomentum ? _options.Beta1 : 0.0;

        foreach (var param in context.Parameters)
        {
            if (SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(
                    context, param, Engine, out var gradient))
            {
                StepParameter(
                    param, gradient, beta2, epsilon, clip, decay, applyWeightDecay,
                    keepMomentum, beta1);
            }
        }
    }

    private void StepParameter(
        Tensor<T> parameter,
        Tensor<T> gradient,
        double beta2,
        double epsilon,
        double clip,
        double decay,
        bool applyWeightDecay,
        bool keepMomentum,
        double beta1)
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

        // The reconstructed second moment for each element, however it was estimated.
        var estimate = _options.Factored && IsFactorable(parameter._shape)
            ? UpdateFactoredSecondMoment(parameter, g, beta2, epsilon)
            : UpdateFullSecondMoment(parameter, g, beta2, epsilon);

        // U = G / sqrt(V), then scaled down if its RMS exceeds the threshold. Clipping the
        // UPDATE rather than the gradient is what keeps this stable without a warmup.
        var update = engine.TensorDivide(
            g, engine.TensorSqrt(engine.TensorMax(estimate, NumOps.FromDouble(epsilon))));
        double updateRms = Math.Sqrt(NumOps.ToDouble(engine.TensorSumOfSquares(update)) / length);
        double scale = updateRms > clip && clip > 0 ? clip / updateRms : 1.0;
        // The first moment averages the clipped update already multiplied by this step's size, as the reference
        // implementations do (transformers' Adafactor: clip, multiply by lr, then average).
        double alpha = StepSize(_step, Rms(parameter));
        update = engine.TensorMultiplyScalar(update, NumOps.FromDouble(scale * alpha));

        if (keepMomentum)
        {
            var moment = _momentum.GetOrAdd(parameter, _ => new Tensor<T>(parameter._shape));
            var m = engine.TensorAddScaled(moment, update, NumOps.FromDouble(beta1), NumOps.FromDouble(1 - beta1));
            engine.TensorCopy(m, moment);
            update = m;
        }

        var next = engine.TensorSubtract(parameter, update);

        // Decoupled: applied to the weight, not folded into the gradient, so it does not
        // enter the second-moment estimate.
        if (applyWeightDecay)
            next = engine.TensorSubtract(next, engine.TensorMultiplyScalar(parameter, NumOps.FromDouble(alpha * decay)));

        engine.TensorCopy(next, parameter);
    }

    private Tensor<T> UpdateFactoredSecondMoment(
        Tensor<T> parameter,
        Tensor<T> gradient,
        double beta2,
        double epsilon)
    {
        var engine = Engine;
        (int rows, int columns) = AsMatrix(parameter._shape);
        var rowState = _rows.GetOrAdd(parameter, _ => new Tensor<T>(new[] { rows }));
        var columnState = _columns.GetOrAdd(parameter, _ => new Tensor<T>(new[] { columns }));

        // Row and column sums of the squared gradient (plus epsilon) over the (rows, columns) view.
        var squared = engine.TensorAddScalar(
            engine.TensorMultiply(gradient, gradient), NumOps.FromDouble(epsilon));
        var asMatrix = engine.Reshape(squared, new[] { rows, columns });
        var rowSums = engine.ReduceSum(asMatrix, new[] { 1 }, keepDims: false);
        var columnSums = engine.ReduceSum(asMatrix, new[] { 0 }, keepDims: false);

        T decay = NumOps.FromDouble(beta2);
        T oneMinusDecay = NumOps.FromDouble(1 - beta2);
        engine.TensorCopy(engine.TensorAddScaled(rowState, rowSums, decay, oneMinusDecay), rowState);
        engine.TensorCopy(engine.TensorAddScaled(columnState, columnSums, decay, oneMinusDecay), columnState);
        double total = NumOps.ToDouble(engine.TensorSum(rowState));

        // V ~= outer(R, C) / sum(R). The normalisation is what makes the rank-one
        // reconstruction agree with the true second moment in total mass.
        if (total <= 0.0) total = epsilon;
        var reconstructed = engine.TensorDivideScalar(engine.TensorOuter(rowState, columnState), NumOps.FromDouble(total));
        return engine.Reshape(reconstructed, parameter._shape);
    }

    private Tensor<T> UpdateFullSecondMoment(
        Tensor<T> parameter,
        Tensor<T> gradient,
        double beta2,
        double epsilon)
    {
        var engine = Engine;
        var state = _full.GetOrAdd(parameter, _ => new Tensor<T>(parameter._shape));
        var squared = engine.TensorAddScalar(
            engine.TensorMultiply(gradient, gradient), NumOps.FromDouble(epsilon));
        var updated = engine.TensorAddScaled(state, squared, NumOps.FromDouble(beta2), NumOps.FromDouble(1 - beta2));
        engine.TensorCopy(updated, state);
        return updated;
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

        // A flat vector carries no shape, so there is nothing to factor over. That is not a
        // shortcut: the paper keeps a full second moment for parameters it cannot factor, which is
        // exactly this case.
        _step++;
        double beta2 = SecondMomentDecay(_step);
        double epsilon = _options.Epsilon;
        double clip = _options.UpdateClippingThreshold;

        if (_flatState.Length != parameters.Length) _flatState = new Vector<T>(parameters.Length);

        var engine = Engine;
        var g = gradient.AsTensor();
        var state = _flatState.AsTensor();
        var squared = engine.TensorAddScalar(engine.TensorMultiply(g, g), NumOps.FromDouble(epsilon));
        var newState = engine.TensorAddScaled(state, squared, NumOps.FromDouble(beta2), NumOps.FromDouble(1 - beta2));
        engine.TensorCopy(newState, state);
        var update = engine.TensorDivide(
            g, engine.TensorSqrt(engine.TensorMax(newState, NumOps.FromDouble(epsilon))));

        double sumSquares = parameters.Length == 0 ? 0.0 : NumOps.ToDouble(engine.TensorSumOfSquares(update));
        double updateRms = Math.Sqrt(sumSquares / Math.Max(1, parameters.Length));
        double scale = updateRms > clip && clip > 0 ? clip / updateRms : 1.0;
        double decay = _options.WeightDecay;
        bool applyWeightDecay = double.IsNaN(decay) || Math.Abs(decay) > 0.0;

        var p = parameters.AsTensor();
        double alpha = StepSize(_step, Rms(p));
        update = engine.TensorMultiplyScalar(update, NumOps.FromDouble(scale * alpha));
        if (!double.IsNaN(_options.Beta1))
        {
            if (_flatMomentum.Length != parameters.Length) _flatMomentum = new Vector<T>(parameters.Length);
            var mState = _flatMomentum.AsTensor();
            var m = engine.TensorAddScaled(
                mState, update, NumOps.FromDouble(_options.Beta1), NumOps.FromDouble(1 - _options.Beta1));
            engine.TensorCopy(m, mState);
            update = m;
        }

        var next = engine.TensorSubtract(p, update);
        if (applyWeightDecay)
            next = engine.TensorSubtract(next, engine.TensorMultiplyScalar(p, NumOps.FromDouble(alpha * decay)));

        return next.AsVector();
    }

    private Vector<T> _flatState = new Vector<T>(0);
    private Vector<T> _flatMomentum = new Vector<T>(0);

    /// <inheritdoc />
    public override OptimizationResult<T, TInput, TOutput> Optimize(
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        // A reused instance must start a fresh run rather than inherit the previous one's moments.
        _rows.Clear();
        _columns.Clear();
        _full.Clear();
        _momentum.Clear();
        _flatState = new Vector<T>(0);
        _flatMomentum = new Vector<T>(0);
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
