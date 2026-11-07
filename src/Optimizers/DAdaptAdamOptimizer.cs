using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Optimizers;

/// <summary>
/// D-Adaptation Adam (Defazio and Mishchenko 2023, "Learning-Rate-Free Learning by D-Adaptation"): Adam whose step size
/// d·lr comes from a running lower bound on the distance to the solution, estimated from the iterates.
/// </summary>
/// <remarks>
/// <para>
/// Each step, over all parameters together: <c>m ← β₁m + d·lr(1 − β₁)g</c>, <c>v ← β₂v + (1 − β₂)g²</c>,
/// <c>s ← β₂s + d·lr(1 − β₂)g</c>, and the weighted sums <c>Σ g²/(√v + ε)</c>, <c>Σ s²/(√v + ε)</c> and <c>Σ|s|</c>; then
/// <c>gsq ← β₂·gsq + (d·lr)²(1 − β₂)Σ g²/(√v + ε)</c>, <c>d̂ = (Σ s²/(√v + ε)/(1 − β₂) − gsq)/Σ|s|</c>,
/// <c>d ← max(d, min(d̂, d·growth))</c>; then each parameter takes <c>θ ← θ − wd·d·lr·θ − m/(√v + ε)</c>. The update
/// follows the reference implementation exactly (facebookresearch/dadaptation <c>DAdaptAdam</c>, vendored in
/// audiocraft as <c>audiocraft/optim/dadam.py</c>), including returning without an update while <c>Σ|s|</c> is zero.
/// </para>
/// <para><b>For Beginners:</b> Adam with a step size the optimizer works out for itself.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
public class DAdaptAdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    private DAdaptAdamOptimizerOptions<T, TInput, TOutput> _options => (DAdaptAdamOptimizerOptions<T, TInput, TOutput>)Options;

    private readonly ConcurrentDictionary<Tensor<T>, (double[] S, double[] M, double[] V)> _state =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private (double[] S, double[] M, double[] V)? _flatState;
    private double _distance;
    private double _gsqWeighted;
    private bool _distanceInitialized;

    /// <summary>Creates the optimizer.</summary>
    public DAdaptAdamOptimizer(IFullModel<T, TInput, TOutput>? model = null, DAdaptAdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new DAdaptAdamOptimizerOptions<T, TInput, TOutput>())
    {
    }

    /// <summary>The current distance estimate d.</summary>
    public double Distance => _distanceInitialized ? _distance : _options.InitialDistance;

    /// <inheritdoc />
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions() => _options;

    /// <inheritdoc />
    protected override void StepCore(TapeStepContext<T> context)
    {
        PrepareTapeState(context);
        if (GradientOptions.EnableGradientClipping)
            ApplyTapeGlobalNormGradientClipping(context, GradientOptions.MaxGradientNorm);
        if (HasAnomalousTapeGradients(context)) return;
        var parameters = new List<Tensor<T>>();
        var groups = new List<(double[] Values, double[] Gradient, (double[] S, double[] M, double[] V) State)>();
        foreach (var parameter in context.Parameters)
        {
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, parameter, Engine, out var gradient)) continue;
            var gradientSpan = gradient.AsSpan();
            var valueSpan = parameter.AsSpan();
            var g = new double[parameter.Length];
            var values = new double[parameter.Length];
            for (int i = 0; i < g.Length; i++)
            {
                g[i] = NumOps.ToDouble(gradientSpan[i]);
                values[i] = NumOps.ToDouble(valueSpan[i]);
            }
            parameters.Add(parameter);
            groups.Add((values, g, _state.GetOrAdd(parameter, p => (new double[p.Length], new double[p.Length], new double[p.Length]))));
        }
        if (!Step(groups)) return;
        for (int k = 0; k < parameters.Count; k++)
        {
            var span = parameters[k].AsWritableSpan();
            var values = groups[k].Values;
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble(values[i]);
        }
    }

    /// <inheritdoc />
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
            throw new ArgumentException($"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).", nameof(gradient));
        if (_flatState is null || _flatState.Value.S.Length != parameters.Length)
            _flatState = (new double[parameters.Length], new double[parameters.Length], new double[parameters.Length]);
        var g = new double[gradient.Length];
        var values = new double[parameters.Length];
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = NumOps.ToDouble(gradient[i]);
            values[i] = NumOps.ToDouble(parameters[i]);
        }
        Step(new List<(double[], double[], (double[], double[], double[]))> { (values, g, _flatState.Value) });
        var updated = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) updated[i] = NumOps.FromDouble(values[i]);
        return updated;
    }

    // One D-Adaptation step over parameter groups (values, gradient, state), updating the values in place. Returns
    // false when the step is skipped (Σ|s| = 0), as the reference does before touching the parameters.
    private bool Step(List<(double[] Values, double[] Gradient, (double[] S, double[] M, double[] V) State)> groups)
    {
        var o = _options;
        if (!_distanceInitialized)
        {
            _distance = o.InitialDistance;
            _distanceInitialized = true;
        }
        double lr = GetCurrentLearningRate();
        double beta1 = o.Beta1, beta2 = o.Beta2, eps = o.Epsilon, decay = o.WeightDecay;
        double dlr = _distance * lr;
        double gSq = 0, skSqWeighted = 0, skL1 = 0;
        for (int k = 0; k < groups.Count; k++)
        {
            var (values, grad, (s, m, v)) = groups[k];
            for (int i = 0; i < grad.Length; i++)
            {
                double g = grad[i];
                // Coupled weight decay enters the gradient (and so the moments) before anything else.
                if (decay != 0 && !o.DecoupleWeightDecay) g += decay * values[i];
                if (lr <= 0) continue;
                m[i] = beta1 * m[i] + dlr * (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                double denominator = Math.Sqrt(v[i]) + eps;
                gSq += g * g / denominator;
                s[i] = beta2 * s[i] + dlr * (1 - beta2) * g;
                skSqWeighted += s[i] * s[i] / denominator;
                skL1 += Math.Abs(s[i]);
            }
        }
        double gsqWeighted = beta2 * _gsqWeighted + gSq * dlr * dlr * (1 - beta2);
        if (skL1 == 0) return false;
        if (lr > 0)
        {
            double dHat = (skSqWeighted / (1 - beta2) - gsqWeighted) / skL1;
            _distance = Math.Max(_distance, Math.Min(dHat, _distance * o.GrowthRate));
        }
        _gsqWeighted = gsqWeighted;
        foreach (var (values, _, (_, m, v)) in groups)
        {
            for (int i = 0; i < values.Length; i++)
            {
                double denominator = Math.Sqrt(v[i]) + eps;
                if (decay != 0 && o.DecoupleWeightDecay && lr > 0) values[i] += values[i] * (-decay * dlr);
                values[i] -= m[i] / denominator;
            }
        }
        return true;
    }

    /// <inheritdoc />
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);
        var currentSolution = InitializeWorkingSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        Reset();
        InitializeAdaptiveParameters();
        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize, epoch);
            foreach (var (xBatch, yBatch, _) in batcher.GetBatches())
                currentSolution = UpdateSolution(currentSolution, CalculateGradient(currentSolution, xBatch, yBatch));
            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);
            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
                return CreateOptimizationResult(bestStepData, inputData);
            if (IsConvergedAgainstPreviousEpoch(epoch, currentStepData, previousStepData, _options.Tolerance))
                break;
            previousStepData = currentStepData;
        }
        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _state.Clear();
        _flatState = null;
        _distanceInitialized = false;
        _gsqWeighted = 0;
    }
}
