using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// EnCodec's loss balancer (Défossez et al. 2022, §3.4, Eq. 5; reference <c>encodec/balancer.py</c>): for losses that
/// depend on the model only through its output x̂, each loss's gradient g_i = ∂ℓ_i/∂x̂ is rescaled to
/// <c>g̃_i = R · λ_i / Σ_j λ_j · g_i / ⟨‖g_i‖₂⟩_β</c>, where ⟨·⟩_β is a debiased exponential moving average of the norm
/// over training batches, and Σ_i g̃_i is backpropagated into the network instead of Σ_i λ_i g_i.
/// </summary>
/// <remarks>
/// <para>The gradients with respect to x̂ are taken on a nested tape over a detached copy of x̂, inside its own tensor arena
/// (a tape's disposal resets the current arena). The balanced gradient then enters the outer tape through the surrogate
/// <c>Σ x̂ · stop(Σ_i g̃_i)</c>, whose gradient with respect to x̂ is exactly Σ_i g̃_i.</para>
/// <para>Norms are taken per batch item and averaged (reference <c>per_batch_item=True</c>); with one item per step that
/// is the norm of the whole gradient.</para>
/// </remarks>
internal sealed class LossBalancer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly IReadOnlyDictionary<string, double> _weights;
    private readonly double _totalNorm, _decay, _epsilon;
    private readonly Dictionary<string, double> _total = new(), _fix = new();

    public LossBalancer(IEngine engine, IReadOnlyDictionary<string, double> weights, double totalNorm = 1.0, double emaDecay = 0.999,
        double epsilon = 1e-12)
    {
        _engine = engine;
        _weights = weights;
        _totalNorm = totalNorm;
        _decay = emaDecay;
        _epsilon = epsilon;
    }

    /// <summary>The averaged gradient norm of each loss after the last step (for monitoring).</summary>
    public IReadOnlyDictionary<string, double> AveragedNorms { get; private set; } = new Dictionary<string, double>();

    /// <summary>The weighted sum of the losses' values at <paramref name="output"/> (no gradient), for reporting.</summary>
    public double LastWeightedLoss { get; private set; }

    /// <summary>
    /// A scalar on the outer tape whose gradient with respect to <paramref name="output"/> is the balanced Σ_i g̃_i.
    /// </summary>
    /// <param name="output">The model output x̂ (on the outer tape).</param>
    /// <param name="losses">Each loss as a function of an output tensor.</param>
    public Tensor<T> Surrogate(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses)
    {
        var gradients = new Dictionary<string, Tensor<T>>();
        var norms = new Dictionary<string, double>();
        double weighted = 0;
        foreach (var (name, loss) in losses)
        {
            Tensor<T> gradient;
            using (AiDotNet.Tensors.Helpers.TensorArena.Create())
            {
                var leaf = new Tensor<T>(output._shape, output.ToVector());
                using var tape = new GradientTape<T>();
                var value = loss(leaf);
                weighted += _weights[name] * NumOps.ToDouble(value[0]);
                var grads = tape.ComputeGradients(value, new[] { leaf });
                gradient = grads.TryGetValue(leaf, out var g) ? new Tensor<T>(g._shape, g.ToVector()) : new Tensor<T>(output._shape);
            }
            double sum = 0;
            var span = gradient.Data.Span;
            for (int i = 0; i < span.Length; i++)
            {
                double v = NumOps.ToDouble(span[i]);
                sum += v * v;
            }
            gradients[name] = gradient;
            norms[name] = Math.Sqrt(sum);
        }
        LastWeightedLoss = weighted;

        // averager(beta): total ← β·total + value, fix ← β·fix + 1, average = total / fix.
        var averaged = new Dictionary<string, double>();
        foreach (var (name, norm) in norms)
        {
            _total[name] = (_total.TryGetValue(name, out var t) ? t : 0) * _decay + norm;
            _fix[name] = (_fix.TryGetValue(name, out var f) ? f : 0) * _decay + 1;
            averaged[name] = _total[name] / _fix[name];
        }
        AveragedNorms = averaged;

        double weightTotal = 0;
        foreach (var name in averaged.Keys) weightTotal += _weights[name];
        Tensor<T>? balanced = null;
        foreach (var (name, average) in averaged)
        {
            double scale = _weights[name] / weightTotal * _totalNorm / (_epsilon + average);
            var term = _engine.TensorMultiplyScalar(gradients[name], NumOps.FromDouble(scale));
            balanced = balanced is null ? term : _engine.TensorAdd(balanced, term);
        }
        var stopped = new Tensor<T>(balanced!._shape, balanced.ToVector());
        var product = _engine.TensorMultiply(output, stopped);
        return _engine.ReduceSum(product, System.Linq.Enumerable.Range(0, product.Rank).ToArray(), keepDims: false);
    }
}
