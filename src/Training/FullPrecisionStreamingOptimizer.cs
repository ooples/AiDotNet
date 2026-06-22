using AiDotNet.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// #1662 lever #1: shared per-parameter FULL-PRECISION state machinery for the streaming
/// optimizer-in-backward path. Mirrors <see cref="BlockQuantizedStreamingOptimizer{T}"/>'s
/// contract (per-tensor state keyed by reference, one global step counter, per-element update)
/// but stores moments as plain <c>double[]</c> with NO 8-bit block quantization and NO update
/// clamping — so the per-parameter fused update is bit-identical to the classic whole-vector
/// optimizer step. This is what lets single-pass fused optimizer-in-backward be the DEFAULT for
/// models that fit in memory (a pure memory + cache-locality win over collect-then-step) without
/// any accuracy drift, rather than only the OOM-survival path the 8-bit variants serve.
///
/// <para>Adam-family updates are element-wise independent given a shared step <c>t</c>, so
/// applying the update per parameter tensor (in topological backward order) with per-element
/// full-precision moments and a once-per-step <c>t</c> produces exactly the same result as the
/// classic optimizer applying it over the concatenated parameter vector.</para>
/// </summary>
internal abstract class FullPrecisionStreamingOptimizer<T> : IStreamingOptimizer<T>, IStreamingOptimizerLearningRate
{
    private readonly Dictionary<Tensor<T>, double[][]> _moments =
        new(TensorReferenceComparer<Tensor<T>>.Instance);
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();
    private readonly int _momentCount;
    private readonly string _optimizerName;

    /// <summary>Global optimizer step counter, advanced once per <see cref="BeginStep"/> —
    /// identical timing to the classic optimizer's <c>_t++</c>, so bias-correction matches.</summary>
    protected int Step { get; private set; }

    protected double LearningRate { get; private set; }

    protected FullPrecisionStreamingOptimizer(string optimizerName, double learningRate, int momentCount)
    {
        _optimizerName = optimizerName;
        LearningRate = learningRate;
        _momentCount = momentCount;
    }

    public void SetLearningRate(double learningRate)
    {
        if (learningRate > 0.0 && !double.IsNaN(learningRate) && !double.IsInfinity(learningRate))
            LearningRate = learningRate;
    }

    public void BeginStep() => Step++;

    // First-order optimizers update in place during Apply; nothing to flush.
    public virtual void EndStep() { }

    public void Apply(Tensor<T> param, Tensor<T> grad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (grad is null) throw new ArgumentNullException(nameof(grad));
        int length = param.Length;
        if (length == 0)
            throw new ArgumentException(
                $"{_optimizerName}.Apply: param is empty (length == 0); a zero-length parameter should never be registered as a training source.",
                nameof(param));
        if (grad.Length != length)
            throw new ArgumentException(
                $"{_optimizerName}.Apply: gradient length {grad.Length} does not match param length {length}.",
                nameof(grad));

        if (!_moments.TryGetValue(param, out var moments) || moments[0].Length != length)
        {
            moments = new double[_momentCount][];
            for (int m = 0; m < _momentCount; m++) moments[m] = new double[length];
            _moments[param] = moments;
        }

        Span<double> cur = stackalloc double[_momentCount];
        Span<double> next = stackalloc double[_momentCount];
        for (int i = 0; i < length; i++)
        {
            for (int m = 0; m < _momentCount; m++) cur[m] = moments[m][i];
            double p = _ops.ToDouble(param[i]);
            double g = _ops.ToDouble(grad[i]);
            double updated = UpdateElement(p, g, cur, next, i);
            for (int m = 0; m < _momentCount; m++) moments[m][i] = next[m];
            param[i] = _ops.FromDouble(updated);
        }
    }

    /// <summary>Per-element update rule. Reads the current moments, writes the next moments,
    /// and returns the new parameter value. Implementations MUST match the classic optimizer's
    /// formula exactly (no update clamping) to preserve bit-identicality.</summary>
    protected abstract double UpdateElement(
        double parameter, double gradient, ReadOnlySpan<double> moments, Span<double> nextMoments, int index);

    /// <summary>Decoupled (AdamW-style) weight decay; no-op when weightDecay == 0.</summary>
    protected double ApplyDecoupledWeightDecay(double parameter, double weightDecay)
        => weightDecay == 0.0 ? parameter : parameter - LearningRate * weightDecay * parameter;
}
