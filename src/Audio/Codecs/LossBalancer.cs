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
        var balanced = new Tensor<T>(new[] { output.Length + 1 });
        Balance(output, losses, balanced.AsWritableSpan());
        var stopped = _engine.Reshape(_engine.TensorSlice(balanced, new[] { 0 }, new[] { output.Length }), output._shape);
        return SurrogateOf(output, stopped);
    }

    /// <summary>The number of balanced steps (each updates the norms' moving averages once).</summary>
    internal int Steps { get; private set; }

    /// <summary>
    /// <see cref="Surrogate(Tensor{T}, IReadOnlyDictionary{string, Func{Tensor{T}, Tensor{T}}})"/> that a compiled training
    /// plan replays: under a trace the balancing (its nested gradients, the host norms and their moving averages) is one
    /// node that reruns it on every replay with that step's output and <paramref name="context"/>.
    /// </summary>
    /// <param name="output">The model output x̂ (on the outer tape or trace).</param>
    /// <param name="context">The tensors the losses read (the real recording, say); the node's other inputs.</param>
    /// <param name="lossesFor">The losses for given context values. The latest call's factory serves every replay, so a
    /// step's choices (its discriminator) reach the plan when the step's objective runs.</param>
    /// <returns>The surrogate (gradient Σ_i g̃_i on x̂) and the detached weighted sum of the losses' values.</returns>
    public (Tensor<T> Surrogate, Tensor<T> WeightedLoss) TracedSurrogate(Tensor<T> output, Tensor<T>[] context,
        Func<IReadOnlyList<Tensor<T>>, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>>> lossesFor)
    {
        _lossesFor = lossesFor;
        int n = output.Length;
        var inputs = new Tensor<T>[context.Length + 1];
        inputs[0] = output;
        Array.Copy(context, 0, inputs, 1, context.Length);
        Tensor<T> packed;
        var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.Current;
        if (scope is null)
        {
            packed = new Tensor<T>(new[] { n + 1 });
            Balance(output, lossesFor(context), packed.AsWritableSpan());
        }
        else
        {
            // The node may hold only values a plan's signature can compare (tensors, numbers), so it names this
            // balancer by id; the balancer's moving averages and the step's losses stay outside the plan.
            long id = _id;
            Live[id] = new WeakReference<LossBalancer<T>>(this);
            packed = scope.RecordVariadic(
                AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom, "EnCodecLossBalancer",
                inputs, new[] { n + 1 },
                (replayEngine, result) =>
                {
                    if (!Live.TryGetValue(id, out var reference) || !reference.TryGetTarget(out var balancer))
                        throw new InvalidOperationException("The loss balancer of this training plan no longer exists.");
                    balancer.Replay(replayEngine, inputs, result);
                },
                static (_, _, _, _, _, _) => { }, Array.Empty<object>());
        }
        var stopped = _engine.Reshape(_engine.TensorSlice(packed, new[] { 0 }, new[] { n }), output._shape);
        return (SurrogateOf(output, stopped), _engine.Reshape(_engine.TensorSlice(packed, new[] { n }, new[] { 1 }), Array.Empty<int>()));
    }

    private Func<IReadOnlyList<Tensor<T>>, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>>>? _lossesFor;

    private static long s_nextId;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<long, WeakReference<LossBalancer<T>>> Live = new();
    private readonly long _id = System.Threading.Interlocked.Increment(ref s_nextId);

    ~LossBalancer() => Live.TryRemove(_id, out _);

    private void Replay(IEngine engine, Tensor<T>[] inputs, Tensor<T> result)
    {
        var factory = _lossesFor ?? throw new InvalidOperationException("The balancer has no losses.");
        var value = new Tensor<T>(result._shape);
        Balance(inputs[0], factory(new ArraySegment<Tensor<T>>(inputs, 1, inputs.Length - 1)), value.AsWritableSpan());
        AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(engine, value, result);
    }

    private Tensor<T> SurrogateOf(Tensor<T> output, Tensor<T> stopped)
    {
        var product = _engine.TensorMultiply(output, stopped);
        return _engine.ReduceSum(product, System.Linq.Enumerable.Range(0, product.Rank).ToArray(), keepDims: false);
    }

    // Σ_i λ_i ℓ_i(x̂) on the engine.
    private Tensor<T> WeightedLoss(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses)
    {
        var names = System.Linq.Enumerable.ToArray(losses.Keys);
        if (names.Length == 0) return new Tensor<T>(new[] { 1 });
        var weights = AiDotNet.Helpers.TensorCopyHelper.CastLogical<double, T>(new Tensor<double>(new[] { names.Length },
            new Vector<double>(System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(names, name => _weights[name])))));
        var values = _engine.TensorConcatenate(System.Linq.Enumerable.ToArray(
            System.Linq.Enumerable.Select(names, name => _engine.Reshape(losses[name](output), new[] { 1 }))), 0);
        return _engine.ReduceSum(_engine.TensorMultiply(values, weights), new[] { 0 }, keepDims: true);
    }

    // Writes Σ_i g̃_i (x̂'s elements) and then the weighted loss into result; updates the moving averages once.
    // Without gradients (a no-grad evaluation, such as a training step's shape warm-up) there is nothing to balance:
    // only the weighted loss is written, and the averages stay where the training steps left them.
    private void Balance(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses, Span<T> result)
    {
        if (NoGradScope<T>.IsSuppressed)
        {
            result.Clear();
            result[result.Length - 1] = WeightedLoss(output, losses).GetFlat(0);
            return;
        }
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
        Steps++;

        double weightTotal = 0;
        foreach (var name in averaged.Keys) weightTotal += _weights[name];
        Tensor<T>? balanced = null;
        using (new NoGradScope<T>())
        {
            foreach (var (name, average) in averaged)
            {
                double scale = _weights[name] / weightTotal * _totalNorm / (_epsilon + average);
                var term = _engine.TensorMultiplyScalar(gradients[name], NumOps.FromDouble(scale));
                balanced = balanced is null ? term : _engine.TensorAdd(balanced, term);
            }
            balanced!.Data.Span.CopyTo(result);
        }
        result[result.Length - 1] = NumOps.FromDouble(weighted);
    }
}
