using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
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
    private static readonly ConditionalWeakTable<AiDotNet.Tensors.Engines.Compilation.LazyTensorScope, StrongBox<int>> ScopeOrdinals = new();
    private static readonly ConditionalWeakTable<object, BoxedObjectId> ObjectIds = new();
    private static long s_nextId;
    private static long s_nextObjectId;
    private static readonly ConcurrentDictionary<long, WeakReference<LossBalancer<T>>> Live = new();

    private readonly IEngine _engine;
    private readonly IReadOnlyDictionary<string, double> _weights;
    private readonly double _totalNorm, _decay, _epsilon;
    private readonly Dictionary<string, double> _total = new(), _fix = new();
    private readonly ConcurrentDictionary<SlotKey, long> _slotIdsByKey = new();
    private readonly ConcurrentDictionary<long, Func<IReadOnlyList<Tensor<T>>, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>>>> _factoriesBySlotId = new();
    private readonly long _id = Interlocked.Increment(ref s_nextId);
    private long _nextSlotId;

    public LossBalancer(IEngine engine, IReadOnlyDictionary<string, double> weights, double totalNorm = 1.0, double emaDecay = 0.999,
        double epsilon = 1e-12)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _weights = weights ?? throw new ArgumentNullException(nameof(weights));
        _totalNorm = totalNorm;
        _decay = emaDecay;
        _epsilon = epsilon;
    }

    ~LossBalancer() => Live.TryRemove(_id, out _);

    /// <summary>The averaged gradient norm of each loss after the last step (for monitoring).</summary>
    public IReadOnlyDictionary<string, double> AveragedNorms { get; private set; } = new Dictionary<string, double>();

    /// <summary>The weighted sum of the losses' values at <c>output</c> (no gradient), for reporting.</summary>
    public double LastWeightedLoss { get; private set; }

    /// <summary>The number of balanced steps (each updates the norms' moving averages once).</summary>
    internal int Steps { get; private set; }

    /// <summary>
    /// A scalar on the outer tape whose gradient with respect to <paramref name="output"/> is the balanced Σ_i g̃_i.
    /// </summary>
    /// <param name="output">The model output x̂ (on the outer tape).</param>
    /// <param name="losses">Each loss as a function of an output tensor.</param>
    public Tensor<T> Surrogate(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (losses is null) throw new ArgumentNullException(nameof(losses));
        var balanced = new Tensor<T>(new[] { output.Length + 1 });
        Balance(output, losses, balanced.AsWritableSpan());
        var stopped = _engine.Reshape(_engine.TensorSlice(balanced, new[] { 0 }, new[] { output.Length }), output._shape);
        return SurrogateOf(output, stopped);
    }

    /// <summary>
    /// <see cref="Surrogate(Tensor{T}, IReadOnlyDictionary{string, Func{Tensor{T}, Tensor{T}}})"/> that a compiled training
    /// plan replays: under a trace the balancing (its nested gradients, the host norms and their moving averages) is one
    /// node that reruns it on every replay with that step's output and <paramref name="context"/>.
    /// </summary>
    /// <param name="output">The model output x̂ (on the outer tape or trace).</param>
    /// <param name="context">The tensors the losses read (the real recording, say); the node's other inputs.</param>
    /// <param name="lossesFor">The losses for given context values.</param>
    /// <param name="stateKey">Optional explicit state token included in the compiled plan signature alongside the factory identity.</param>
    /// <returns>The surrogate (gradient Σ_i g̃_i on x̂) and the detached weighted sum of the losses' values.</returns>
    public (Tensor<T> Surrogate, Tensor<T> WeightedLoss) TracedSurrogate(
        Tensor<T> output,
        Tensor<T>[] context,
        Func<IReadOnlyList<Tensor<T>>, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>>> lossesFor,
        object? stateKey = null)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (context is null) throw new ArgumentNullException(nameof(context));
        if (lossesFor is null) throw new ArgumentNullException(nameof(lossesFor));

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
            int callOrdinal = ScopeOrdinals.GetValue(scope, static _ => new StrongBox<int>(0)).Value++;
            object[] signatureTokens = BuildSignatureTokens(_id, callOrdinal, _weights, lossesFor, stateKey);
            var slotKey = new SlotKey(signatureTokens);
            long slotId = ResolveOrBindSlot(slotKey, lossesFor, scope.IsProbe);
            long id = _id;
            if (!scope.IsProbe)
            {
                Live[id] = new WeakReference<LossBalancer<T>>(this);
            }

            packed = scope.RecordVariadic(
                AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom, "EnCodecLossBalancer",
                inputs, new[] { n + 1 },
                (replayEngine, result) =>
                {
                    if (!Live.TryGetValue(id, out var reference) || !reference.TryGetTarget(out var balancer))
                        throw new InvalidOperationException("The loss balancer of this training plan no longer exists.");
                    balancer.Replay(replayEngine, slotId, inputs, result);
                },
                static (_, _, _, _, _, _) => { }, signatureTokens);

            if (packed.LazySource is AiDotNet.Tensors.Engines.Compilation.LazyNode<T> lazyNode)
            {
                lazyNode.ReplaysHostState = true;
            }
        }

        var stopped = _engine.Reshape(_engine.TensorSlice(packed, new[] { 0 }, new[] { n }), output._shape);
        return (SurrogateOf(output, stopped), _engine.Reshape(_engine.TensorSlice(packed, new[] { n }, new[] { 1 }), Array.Empty<int>()));
    }

    private long ResolveOrBindSlot(
        SlotKey key,
        Func<IReadOnlyList<Tensor<T>>, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>>> lossesFor,
        bool isProbe)
    {
        if (_slotIdsByKey.TryGetValue(key, out long existingSlotId))
        {
            if (!isProbe)
            {
                _factoriesBySlotId[existingSlotId] = lossesFor;
            }

            return existingSlotId;
        }

        if (isProbe)
        {
            // Probes must never allocate or overwrite replay slots or mutate balancer state.
            return -1L;
        }

        long allocated = Interlocked.Increment(ref _nextSlotId);
        long slotId = _slotIdsByKey.GetOrAdd(key, allocated);
        _factoriesBySlotId[slotId] = lossesFor;
        return slotId;
    }

    private void Replay(IEngine engine, long slotId, Tensor<T>[] inputs, Tensor<T> result)
    {
        if (!_factoriesBySlotId.TryGetValue(slotId, out var factory))
            throw new InvalidOperationException("The balancer has no losses bound for this compiled plan slot.");
        var value = new Tensor<T>(result._shape);
        Balance(inputs[0], factory(new ArraySegment<Tensor<T>>(inputs, 1, inputs.Length - 1)), value.AsWritableSpan());
        AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(engine, value, result);
    }

    private Tensor<T> SurrogateOf(Tensor<T> output, Tensor<T> stopped)
    {
        var product = _engine.TensorMultiply(output, stopped);
        return _engine.ReduceSum(product, Enumerable.Range(0, product.Rank).ToArray(), keepDims: false);
    }

    // Σ_i λ_i ℓ_i(x̂) on the engine.
    private Tensor<T> WeightedLoss(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses)
    {
        var names = losses.Keys.ToArray();
        if (names.Length == 0) return new Tensor<T>(new[] { 1 });
        var weights = TensorCopyHelper.CastLogical<double, T>(new Tensor<double>(new[] { names.Length },
            new Vector<double>(names.Select(name => _weights[name]).ToArray())));
        var values = _engine.TensorConcatenate(
            names.Select(name => _engine.Reshape(losses[name](output), new[] { 1 })).ToArray(), 0);
        return _engine.ReduceSum(_engine.TensorMultiply(values, weights), new[] { 0 }, keepDims: true);
    }

    // Writes Σ_i g̃_i (x̂'s elements) and then the weighted loss into result; updates the moving averages once.
    // Without gradients (a no-grad evaluation, such as a training step's shape warm-up or non-step forward) there is
    // nothing to balance: only the weighted loss is written, and the averages stay where the training steps left them.
    private void Balance(Tensor<T> output, IReadOnlyDictionary<string, Func<Tensor<T>, Tensor<T>>> losses, Span<T> result)
    {
        if (NoGradScope<T>.IsSuppressed || AiDotNet.Training.CompiledStepRandom<T>.InNonStepForward)
        {
            result.Clear();
            result[result.Length - 1] = WeightedLoss(output, losses).GetFlat(0);
            return;
        }

        if (losses.Count == 0)
        {
            result.Clear();
            return;
        }

        var gradients = new Dictionary<string, Tensor<T>>(losses.Count, StringComparer.Ordinal);
        var norms = new Dictionary<string, double>(losses.Count, StringComparer.Ordinal);
        double weighted = 0;
        foreach (var (name, loss) in losses)
        {
            if (!_weights.TryGetValue(name, out double lossWeight))
                throw new KeyNotFoundException($"Loss balancer has no configured weight for loss '{name}'.");

            Tensor<T> gradient;
            using (AiDotNet.Tensors.Helpers.TensorArena.Create())
            {
                var leaf = new Tensor<T>(output._shape, output.ToVector());
                using var tape = new GradientTape<T>();
                var value = loss(leaf);
                weighted += lossWeight * NumOps.ToDouble(value[0]);
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

        // Guard EMA and step counters against non-finite losses or gradients so a skipped/divergent step never poisons
        // _total, _fix, AveragedNorms, LastWeightedLoss, or Steps.
        if (!double.IsFinite(weighted) || norms.Values.Any(static norm => !double.IsFinite(norm)))
        {
            T nan = NumOps.FromDouble(double.NaN);
            result.Fill(nan);
            return;
        }

        double weightTotal = 0;
        foreach (var name in norms.Keys) weightTotal += _weights[name];
        if (!double.IsFinite(weightTotal) || weightTotal <= 0.0)
        {
            result.Clear();
            result[result.Length - 1] = NumOps.FromDouble(weighted);
            return;
        }

        LastWeightedLoss = weighted;

        // averager(beta): total ← β·total + value, fix ← β·fix + 1, average = total / fix.
        var averaged = new Dictionary<string, double>(norms.Count, StringComparer.Ordinal);
        foreach (var (name, norm) in norms)
        {
            _total[name] = (_total.TryGetValue(name, out var t) ? t : 0) * _decay + norm;
            _fix[name] = (_fix.TryGetValue(name, out var f) ? f : 0) * _decay + 1;
            averaged[name] = _total[name] / _fix[name];
        }

        AveragedNorms = averaged;
        Steps++;

        Tensor<T>? balanced = null;
        using (new NoGradScope<T>())
        {
            foreach (var (name, average) in averaged)
            {
                double scale = _weights[name] / weightTotal * _totalNorm / (_epsilon + average);
                var term = _engine.TensorMultiplyScalar(gradients[name], NumOps.FromDouble(scale));
                balanced = balanced is null ? term : _engine.TensorAdd(balanced, term);
            }

            if (balanced is not null)
            {
                balanced.Data.Span.CopyTo(result);
            }
            else
            {
                result.Slice(0, result.Length - 1).Clear();
            }
        }

        result[result.Length - 1] = NumOps.FromDouble(weighted);
    }

    private static object[] BuildSignatureTokens(
        long balancerId,
        int callOrdinal,
        IReadOnlyDictionary<string, double> weights,
        Delegate factory,
        object? stateKey)
    {
        var tokens = new List<object>(32)
        {
            "balancer",
            balancerId,
            "ordinal",
            callOrdinal,
            "weights",
            weights.Count,
        };

        foreach (var kv in weights.OrderBy(static kv => kv.Key, StringComparer.Ordinal))
        {
            tokens.Add(kv.Key);
            tokens.Add(kv.Value);
        }

        var visited = new HashSet<object>(ReferenceEqualityComparer.Instance);
        tokens.Add("factory");
        AppendSignatureTokens(tokens, factory, visited, depth: 0);
        tokens.Add("stateKey");
        AppendSignatureTokens(tokens, stateKey, visited, depth: 0);
        return tokens.ToArray();
    }

    private static void AppendSignatureTokens(List<object> tokens, object? value, HashSet<object> visited, int depth)
    {
        const int maxDepth = 8;
        if (value is null)
        {
            tokens.Add("<null>");
            return;
        }

        if (depth > maxDepth)
        {
            tokens.Add("<depth>");
            tokens.Add(value.GetType());
            if (!value.GetType().IsValueType)
            {
                tokens.Add(GetStableObjectId(value));
            }
            return;
        }

        switch (value)
        {
            case string s:
                tokens.Add("str");
                tokens.Add(s);
                return;
            case bool or byte or sbyte or short or ushort or int or uint or long or ulong
                or float or double or char or IntPtr or UIntPtr or decimal or DateTime or Guid:
                tokens.Add(value.GetType());
                tokens.Add(value);
                return;
            case Enum:
                tokens.Add(value.GetType());
                tokens.Add(Convert.ToInt64(value, CultureInfo.InvariantCulture));
                return;
            case Type typeValue:
                tokens.Add("type");
                tokens.Add(typeValue);
                return;
            case IEngine engine:
                tokens.Add("engine");
                tokens.Add(engine.GetType());
                return;
            case Tensor<T> typedTensor:
                tokens.Add("tensor");
                tokens.Add(GetStableObjectId(typedTensor));
                tokens.Add(GetStableObjectId(typedTensor._storage));
                tokens.Add(typedTensor.Version);
                tokens.Add(typedTensor.Rank);
                for (int d = 0; d < typedTensor.Rank; d++)
                {
                    tokens.Add(typedTensor._shape[d]);
                }
                return;
            case Delegate del:
                var list = del.GetInvocationList();
                tokens.Add("delegate");
                tokens.Add(list.Length);
                for (int i = 0; i < list.Length; i++)
                {
                    var item = list[i];
                    var method = item.Method;
                    tokens.Add(method.DeclaringType ?? typeof(void));
                    tokens.Add(method.MetadataToken);
                    tokens.Add(method.Module.ModuleVersionId);
                    if (method.IsGenericMethod)
                    {
                        var genericArgs = method.GetGenericArguments();
                        tokens.Add(genericArgs.Length);
                        for (int g = 0; g < genericArgs.Length; g++)
                        {
                            tokens.Add(genericArgs[g]);
                        }
                    }
                    else
                    {
                        tokens.Add(0);
                    }

                    AppendSignatureTokens(tokens, item.Target, visited, depth + 1);
                }
                return;
            case IDictionary dict:
                if (!visited.Add(dict))
                {
                    tokens.Add("<cycle>");
                    return;
                }
                try
                {
                    tokens.Add("dict");
                    tokens.Add(dict.Count);
                    var entries = new List<DictionaryEntry>(dict.Count);
                    foreach (DictionaryEntry entry in dict) entries.Add(entry);
                    entries.Sort(static (a, b) => string.CompareOrdinal(
                        Convert.ToString(a.Key, CultureInfo.InvariantCulture),
                        Convert.ToString(b.Key, CultureInfo.InvariantCulture)));
                    for (int i = 0; i < entries.Count; i++)
                    {
                        AppendSignatureTokens(tokens, entries[i].Key, visited, depth + 1);
                        AppendSignatureTokens(tokens, entries[i].Value, visited, depth + 1);
                    }
                }
                finally
                {
                    visited.Remove(dict);
                }
                return;
            case Array array:
                if (!visited.Add(array))
                {
                    tokens.Add("<cycle>");
                    return;
                }
                try
                {
                    tokens.Add("array");
                    tokens.Add(array.GetType());
                    tokens.Add(array.Length);
                    foreach (var elem in array)
                    {
                        AppendSignatureTokens(tokens, elem, visited, depth + 1);
                    }
                }
                finally
                {
                    visited.Remove(array);
                }
                return;
        }

        var objType = value.GetType();
        if (objType.IsValueType || IsCompilerGeneratedClosure(objType))
        {
            if (!objType.IsValueType && !visited.Add(value))
            {
                tokens.Add("<cycle>");
                return;
            }
            try
            {
                tokens.Add("fields");
                tokens.Add(objType);
                var fields = objType.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                Array.Sort(fields, static (a, b) => string.CompareOrdinal(a.Name, b.Name));
                for (int i = 0; i < fields.Length; i++)
                {
                    tokens.Add(fields[i].Name);
                    AppendSignatureTokens(tokens, fields[i].GetValue(value), visited, depth + 1);
                }
            }
            finally
            {
                if (!objType.IsValueType) visited.Remove(value);
            }
            return;
        }

        if (value is IList listValue)
        {
            if (!visited.Add(listValue))
            {
                tokens.Add("<cycle>");
                return;
            }
            try
            {
                tokens.Add("list");
                tokens.Add(objType);
                tokens.Add(listValue.Count);
                for (int i = 0; i < listValue.Count; i++)
                {
                    AppendSignatureTokens(tokens, listValue[i], visited, depth + 1);
                }
            }
            finally
            {
                visited.Remove(listValue);
            }
            return;
        }

        tokens.Add("object");
        tokens.Add(objType);
        tokens.Add(GetStableObjectId(value));
    }

    private static bool IsCompilerGeneratedClosure(Type type)
        => type.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false)
           || type.Name.IndexOf('<') >= 0;

    private static long GetStableObjectId(object target)
        => ObjectIds.GetValue(target, static _ => new BoxedObjectId(Interlocked.Increment(ref s_nextObjectId))).Id;

    private sealed class BoxedObjectId
    {
        public BoxedObjectId(long id) => Id = id;
        public long Id { get; }
    }

    private sealed class ReferenceEqualityComparer : IEqualityComparer<object>
    {
        public static readonly ReferenceEqualityComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
    }

    private readonly struct SlotKey : IEquatable<SlotKey>
    {
        private readonly object[] _tokens;
        private readonly int _hash;

        public SlotKey(object[] tokens)
        {
            _tokens = tokens;
            int h = 17;
            for (int i = 0; i < tokens.Length; i++)
            {
                h = unchecked(h * 31 + tokens[i].GetHashCode());
            }
            _hash = h;
        }

        public bool Equals(SlotKey other)
        {
            if (_hash != other._hash || _tokens.Length != other._tokens.Length) return false;
            for (int i = 0; i < _tokens.Length; i++)
            {
                if (!Equals(_tokens[i], other._tokens[i])) return false;
            }
            return true;
        }

        public override bool Equals(object? obj) => obj is SlotKey other && Equals(other);
        public override int GetHashCode() => _hash;
    }
}
