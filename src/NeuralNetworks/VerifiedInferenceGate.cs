using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Reusable correctness gate that makes compiled inference safe to enable by default (#1622 L3b).
/// A compiled plan can compile "successfully" yet compute the WRONG answer for some model families,
/// so this gate runs the eager forward (the source of truth) once per input shape, compares it to the
/// compiled candidate, and adopts compiled for that shape ONLY on numerical parity — otherwise it stays
/// eager for that shape forever (no recompile thrash). On top sits a collision-safe value-hash memo
/// that returns the cached output for a confirmed-identical repeated input in O(1).
/// </summary>
/// <remarks>
/// <para>
/// Composed by <see cref="NeuralNetworkBase{T}"/> and <c>NoisePredictorBase&lt;T&gt;</c> so both the
/// neural-network and diffusion noise-predictor forward funnels get the same verify-then-trust + memo
/// behavior from a single implementation. Output is numerically identical to the eager forward for
/// every input.
/// </para>
/// <para>
/// All state is scoped to a caller-supplied monotonic structure version; when the layer graph changes
/// the caller bumps the version and the gate re-verifies from scratch. Thread-safe for concurrent
/// inference on a shared model (per-request serving pools).
/// </para>
/// </remarks>
/// <typeparam name="T">The tensor element type.</typeparam>
internal sealed class VerifiedInferenceGate<T>
{
    private const byte VerdictTrusted = 1;
    private const byte VerdictRejected = 2;
    private const int MemoMaxShapes = 8;

    private readonly System.Collections.Concurrent.ConcurrentDictionary<long, (byte Verdict, int Version)> _verdicts = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<long, MemoEntry> _memo = new();
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    private long _memoHits;
    private long _verifyCount;

    /// <summary>Count of value-memo hits (identical-input short-circuits). Diagnostic/test hook.</summary>
    public long MemoHits => System.Threading.Interlocked.Read(ref _memoHits);

    /// <summary>Count of verify-then-trust verifications performed (one per new shape+version). Diagnostic/test hook.</summary>
    public long VerifyCount => System.Threading.Interlocked.Read(ref _verifyCount);

    private sealed class MemoEntry
    {
        public long ValueHash;
        public int Version;
        public T[] Input = System.Array.Empty<T>();
        public int[] InputShape = System.Array.Empty<int>();
        public T[] Output = System.Array.Empty<T>();
        public int[] OutputShape = System.Array.Empty<int>();
    }

    /// <summary>
    /// Runs the gated forward: memo lookup → per-shape verdict (trusted replay / rejected eager /
    /// unknown verify). Always returns output numerically identical to <paramref name="eager"/>.
    /// </summary>
    /// <param name="input">The inference input.</param>
    /// <param name="structureVersion">Caller's monotonic layer-graph version.</param>
    /// <param name="eager">The eager forward (source of truth).</param>
    /// <param name="compiled">The compiled forward (candidate; may itself fall back to eager internally).</param>
    /// <param name="onDecision">Optional callback invoked when a shape's verdict is first decided (enabled, reason).</param>
    public Tensor<T> Run(
        Tensor<T> input,
        int structureVersion,
        System.Func<Tensor<T>> eager,
        System.Func<Tensor<T>> compiled,
        System.Action<bool, string>? onDecision = null)
    {
        long shapeKey = ComputeShapeKey(input._shape);

        long valueHash = ComputeValueHash(input);
        if (_memo.TryGetValue(shapeKey, out var memo)
            && memo.Version == structureVersion
            && memo.ValueHash == valueHash
            && InputMatches(memo, input))
        {
            System.Threading.Interlocked.Increment(ref _memoHits);
            return new Tensor<T>((T[])memo.Output.Clone(), (int[])memo.OutputShape.Clone());
        }

        _verdicts.TryGetValue(shapeKey, out var v);
        bool decided = v.Version == structureVersion && (v.Verdict == VerdictTrusted || v.Verdict == VerdictRejected);

        Tensor<T> output;
        if (decided && v.Verdict == VerdictTrusted)
        {
            output = compiled();
        }
        else if (decided && v.Verdict == VerdictRejected)
        {
            output = eager();
        }
        else
        {
            System.Threading.Interlocked.Increment(ref _verifyCount);
            var reference = eager();
            Tensor<T> candidate;
            try
            {
                candidate = compiled();
            }
            catch (System.Exception ex) when (
                ex is not System.OutOfMemoryException &&
                ex is not System.StackOverflowException &&
                ex is not System.AccessViolationException)
            {
                _verdicts[shapeKey] = (VerdictRejected, structureVersion);
                onDecision?.Invoke(false, $"verify-throw:{ex.GetType().Name}");
                output = reference;
                StoreMemo(shapeKey, valueHash, structureVersion, input, output);
                return output;
            }

            bool parity = OutputsClose(reference, candidate);
            _verdicts[shapeKey] = (parity ? VerdictTrusted : VerdictRejected, structureVersion);
            onDecision?.Invoke(parity, parity ? "compiled-matches-eager" : "compiled-diverged-stay-eager");
            // The eager reference is the source of truth on the verify call, so a divergent (rejected)
            // compiled plan never leaks a wrong value to the caller.
            output = reference;
        }

        StoreMemo(shapeKey, valueHash, structureVersion, input, output);
        return output;
    }

    /// <summary>Drops all verdicts and memo entries (e.g. on entering training, where weights change).</summary>
    public void Clear()
    {
        _verdicts.Clear();
        _memo.Clear();
    }

    /// <summary>
    /// Returns the current verdict for a shape at <paramref name="structureVersion"/>:
    /// 0 = unknown, 1 = trusted, 2 = rejected. Test/diagnostic hook.
    /// </summary>
    public int VerdictFor(int[] shape, int structureVersion)
    {
        long key = ComputeShapeKey(shape);
        return _verdicts.TryGetValue(key, out var v) && v.Version == structureVersion ? v.Verdict : 0;
    }

    private void StoreMemo(long shapeKey, long valueHash, int version, Tensor<T> input, Tensor<T> output)
    {
        if (!_memo.ContainsKey(shapeKey) && _memo.Count >= MemoMaxShapes)
        {
            foreach (var k in _memo.Keys)
            {
                if (_memo.TryRemove(k, out _)) break;
            }
        }

        _memo[shapeKey] = new MemoEntry
        {
            ValueHash = valueHash,
            Version = version,
            Input = (T[])input.ToArray().Clone(),
            InputShape = (int[])input._shape.Clone(),
            Output = (T[])output.ToArray().Clone(),
            OutputShape = (int[])output._shape.Clone(),
        };
    }

    private bool InputMatches(MemoEntry memo, Tensor<T> input)
    {
        if (memo.InputShape.Length != input._shape.Length) return false;
        for (int i = 0; i < memo.InputShape.Length; i++)
            if (memo.InputShape[i] != input._shape[i]) return false;

        var span = input.AsSpan();
        if (memo.Input.Length != span.Length) return false;
        for (int i = 0; i < span.Length; i++)
            if (!_numOps.Equals(memo.Input[i], span[i])) return false;
        return true;
    }

    private long ComputeValueHash(Tensor<T> input)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        var span = input.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            long bits = System.BitConverter.DoubleToInt64Bits(_numOps.ToDouble(span[i]));
            hash ^= bits;
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }

    private bool OutputsClose(Tensor<T> a, Tensor<T> b)
    {
        if (a._shape.Length != b._shape.Length) return false;
        for (int i = 0; i < a._shape.Length; i++)
            if (a._shape[i] != b._shape[i]) return false;

        var sa = a.AsSpan();
        var sb = b.AsSpan();
        if (sa.Length != sb.Length) return false;

        double rel = typeof(T) == typeof(float) ? 2e-3 : 1e-6;
        const double abs = 1e-5;
        for (int i = 0; i < sa.Length; i++)
        {
            double x = _numOps.ToDouble(sa[i]);
            double y = _numOps.ToDouble(sb[i]);
            if (double.IsNaN(x) != double.IsNaN(y)) return false;
            if (double.IsNaN(x)) continue;
            double diff = System.Math.Abs(x - y);
            double tol = abs + rel * System.Math.Max(System.Math.Abs(x), System.Math.Abs(y));
            if (diff > tol) return false;
        }
        return true;
    }

    private static long ComputeShapeKey(int[] shape)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }
}
