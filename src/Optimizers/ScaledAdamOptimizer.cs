using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Optimizers;

/// <summary>
/// ScaledAdam (k2-fsa/icefall, as GPT-SoVITS vendors it in <c>AR/modules/optim.py</c>): Adam whose update to each
/// tensor is scaled by that tensor's root-mean-square, which also learns each tensor's overall scale, with momentum kept
/// as a parameter change and gradients clipped against the recent median gradient norm.
/// </summary>
/// <remarks>
/// <para>
/// Per tensor with more than one element: <c>δ ← β₁δ</c>; its scale gradient <c>Σ θ·g</c> is recorded each step and every
/// <c>size_update_period</c> steps the recorded rms is refreshed and the scale takes an Adam-like step
/// (<c>δ += (1 − β₁)·θ·s</c>, held at zero below <c>param_min_rms</c> and shrinking above <c>param_max_rms</c>); then
/// <c>v ← β₂v + (1 − β₂)g²</c>, bias-corrected while the correction is below 0.99, and
/// <c>δ += −lr(1 − β₁)·max(rms, param_min_rms)·g/(√v + ε)</c>; <c>θ += δ</c>. A scalar takes plain Adam without a first
/// moment at <c>lr·scalar_lr_scale</c>, clamped to ±<c>scalar_max</c> before its change is added.
/// </para>
/// <para>
/// Clipping, when enabled, records the norm of the rms-scaled gradients each step and every
/// <c>clipping_update_period</c> steps sets the threshold to <c>clipping_scale</c> times the median; from then on the
/// returned factor is <c>min(1, threshold/norm)</c>. As in the reference, the factor scales only the gradient the scale
/// learning records; the Adam and scalar updates read the unscaled gradient.
/// </para>
/// <para>The reference stacks tensors of the same shape into batches; every quantity is per tensor (or global), so the
/// tensors are kept separate here with the same result.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
public class ScaledAdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    private ScaledAdamOptimizerOptions<T, TInput, TOutput> _options => (ScaledAdamOptimizerOptions<T, TInput, TOutput>)Options;

    private sealed class State
    {
        public double[] Delta = Array.Empty<double>();
        public double[] ExpAvgSq = Array.Empty<double>();
        public double ParamRms;
        public double ScaleExpAvgSq;
        public double[] ScaleGrads = Array.Empty<double>();
    }

    private readonly ConcurrentDictionary<Tensor<T>, State> _state = new(TensorReferenceComparer<Tensor<T>>.Instance);
    private State? _flatState;
    private int _step;
    private double[]? _modelNorms;
    private double? _threshold;

    /// <summary>Creates the optimizer.</summary>
    public ScaledAdamOptimizer(IFullModel<T, TInput, TOutput>? model = null, ScaledAdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new ScaledAdamOptimizerOptions<T, TInput, TOutput>())
    {
    }

    /// <summary>The steps taken.</summary>
    public int StepCount => _step;

    /// <inheritdoc />
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions() => _options;

    /// <inheritdoc />
    protected override void StepCore(TapeStepContext<T> context)
    {
        PrepareTapeState(context);
        if (HasAnomalousTapeGradients(context)) return;
        var parameters = new List<Tensor<T>>();
        var groups = new List<(double[] Values, double[] Gradient, State State)>();
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
            groups.Add((values, g, _state.GetOrAdd(parameter, _ => Initial(values))));
        }
        Step(groups);
        for (int k = 0; k < parameters.Count; k++)
        {
            var span = parameters[k].AsWritableSpan();
            var values = groups[k].Values;
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble(values[i]);
        }
    }

    /// <inheritdoc />
    /// <remarks>The flat vector is one tensor (one rms and one learned scale).</remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (parameters.Length != gradient.Length)
            throw new ArgumentException($"Parameter vector length ({parameters.Length}) must match gradient vector length ({gradient.Length}).", nameof(gradient));
        var g = new double[gradient.Length];
        var values = new double[parameters.Length];
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = NumOps.ToDouble(gradient[i]);
            values[i] = NumOps.ToDouble(parameters[i]);
        }
        if (_flatState is null || _flatState.Delta.Length != parameters.Length) _flatState = Initial(values);
        Step(new List<(double[], double[], State)> { (values, g, _flatState) });
        var updated = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) updated[i] = NumOps.FromDouble(values[i]);
        return updated;
    }

    // _init_state: zero momentum and second moment; a tensor's rms and its scale statistics.
    private State Initial(double[] values)
    {
        var state = new State { Delta = new double[values.Length], ExpAvgSq = new double[values.Length] };
        if (values.Length > 1)
        {
            state.ParamRms = Rms(values);
            state.ScaleGrads = new double[_options.SizeUpdatePeriod];
        }
        return state;
    }

    private static double Rms(double[] values) => Math.Sqrt(values.Sum(v => v * v) / values.Length);

    // _get_clipping_scale
    private double ClippingScale(List<(double[] Values, double[] Gradient, State State)> groups)
    {
        var o = _options;
        if (o.ClippingScale is not { } clippingScale || _step == 0) return 1.0;
        int period = o.ClippingUpdatePeriod;
        double sumSq = 0;
        foreach (var (values, gradient, state) in groups)
        {
            if (values.Length == 1) sumSq += gradient[0] * gradient[0];
            else foreach (double g in gradient) sumSq += g * state.ParamRms * (g * state.ParamRms);
        }
        double norm = Math.Sqrt(sumSq);
        _modelNorms ??= new double[period];
        _modelNorms[_step % period] = norm;
        if (_step % period == 0)
        {
            var sorted = (double[])_modelNorms.Clone();
            Array.Sort(sorted);
            _threshold = clippingScale * sorted[Math.Min(period - 1, period / 4 * 2)];
        }
        if (_step < period || _threshold is not { } threshold) return 1.0;
        return Math.Min(1.0, threshold / (norm + 1e-20));
    }

    private void Step(List<(double[] Values, double[] Gradient, State State)> groups)
    {
        var o = _options;
        double lr = GetCurrentLearningRate();
        double beta1 = o.Beta1, beta2 = o.Beta2, eps = o.Epsilon;
        int period = o.SizeUpdatePeriod;
        double clipping = ClippingScale(groups);
        foreach (var (values, gradient, state) in groups)
        {
            for (int i = 0; i < state.Delta.Length; i++) state.Delta[i] *= beta1;
            if (values.Length > 1)
            {
                double scaleGrad = 0;
                for (int i = 0; i < values.Length; i++) scaleGrad += values[i] * (gradient[i] * clipping);
                state.ScaleGrads[_step % period] = scaleGrad;
                if (_step % period == period - 1)
                {
                    state.ParamRms = Rms(values);
                    if (_step > 0) SizeUpdate(values, state, lr);
                }
                // _step
                double biasCorrection2 = 1 - Math.Pow(beta2, _step + 1);
                double alpha = -lr * (1 - beta1) * Math.Max(state.ParamRms, o.ParameterMinRms);
                for (int i = 0; i < values.Length; i++)
                {
                    double g = gradient[i];
                    state.ExpAvgSq[i] = beta2 * state.ExpAvgSq[i] + (1 - beta2) * g * g;
                    double v = biasCorrection2 < 0.99 ? state.ExpAvgSq[i] * (1.0 / biasCorrection2) : state.ExpAvgSq[i];
                    state.Delta[i] += g / (Math.Sqrt(v) + eps) * alpha;
                    values[i] += state.Delta[i];
                }
            }
            else
            {
                // _step_scalar
                double g = gradient[0];
                state.ExpAvgSq[0] = beta2 * state.ExpAvgSq[0] + (1 - beta2) * g * g;
                double biasCorrection2 = 1 - Math.Pow(beta2, _step + 1);
                double denominator = Math.Sqrt(state.ExpAvgSq[0] / biasCorrection2) + eps;
                state.Delta[0] += g / denominator * (-lr * o.ScalarLearningRateScale * (1 - beta1));
                values[0] = Math.Min(Math.Max(values[0], -o.ScalarMax), o.ScalarMax);
                values[0] += state.Delta[0];
            }
        }
        _step++;
    }

    // _size_update
    private void SizeUpdate(double[] values, State state, double lr)
    {
        var o = _options;
        int period = o.SizeUpdatePeriod;
        double sizeLr = lr * o.ScalarLearningRateScale;
        double beta2Corrected = Math.Pow(o.Beta2, period);
        state.ScaleExpAvgSq = state.ScaleExpAvgSq * beta2Corrected + state.ScaleGrads.Average(s => s * s) * (1 - beta2Corrected);
        int sizeStep = (_step + 1) / period;
        double biasCorrection2 = 1 - Math.Pow(beta2Corrected, sizeStep);
        double denominator = Math.Sqrt(state.ScaleExpAvgSq) + o.Epsilon;
        double scaleStep = -sizeLr * Math.Sqrt(biasCorrection2) * state.ScaleGrads.Sum() / denominator;
        if (state.ParamRms < o.ParameterMinRms) scaleStep = 0.0;
        if (state.ParamRms > o.ParameterMaxRms) scaleStep = -sizeLr * period;
        for (int i = 0; i < values.Length; i++) state.Delta[i] += values[i] * scaleStep * (1 - o.Beta1);
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
        _step = 0;
        _modelNorms = null;
        _threshold = null;
    }
}
