using System.Collections.Concurrent;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Shared building blocks of the reference time-series transformers (Informer, Autoformer): per-shape constant
/// tensors built once and reused by every forward, dropout masks drawn by the engine's own generator, and a top-k
/// choice made of comparisons so it runs on the device and replays in a compiled training plan.
/// </summary>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal sealed class SequenceNetworkOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly ConcurrentDictionary<string, Tensor<T>> _constants = new();
    private readonly IEngine _engine;

    public SequenceNetworkOps(IEngine engine) => _engine = engine;

    /// <summary>
    /// The constant <paramref name="key"/> names, built by <paramref name="build"/> on first use. A constant is built
    /// on the host once; a GPU engine uploads it once and keeps it resident, so no forward rebuilds or re-uploads it.
    /// </summary>
    public Tensor<T> Constant(string key, Func<Tensor<T>> build) => _constants.GetOrAdd(key, _ => build());

    /// <summary>A [count, rows, columns] constant: <paramref name="count"/> copies of one [rows, columns] pattern.</summary>
    public Tensor<T> Tiled(string name, int count, int rows, int columns, Func<int, int, double> value)
        => Constant($"{name}:{count}x{rows}x{columns}", () =>
        {
            var pattern = new double[rows * columns];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < columns; j++) pattern[i * columns + j] = value(i, j);
            var tiled = new double[count * pattern.Length];
            for (int r = 0; r < count; r++) Array.Copy(pattern, 0, tiled, r * pattern.Length, pattern.Length);
            return FromDoubles(tiled, new[] { count, rows, columns });
        });

    /// <summary>A tensor of <paramref name="shape"/> filled with ones.</summary>
    public static Tensor<T> Ones(int[] shape)
    {
        var ones = new Tensor<T>(shape);
        ones.Fill(NumOps.One);
        return ones;
    }

    /// <summary>A tensor of <paramref name="shape"/> holding <paramref name="values"/> (row-major).</summary>
    public static Tensor<T> FromDoubles(double[] values, int[] shape)
        => new(shape, new Vector<T>(values.Select(NumOps.FromDouble).ToArray()));

    /// <summary>
    /// The one-hot rows <c>[rows, top, length]</c> of the <paramref name="top"/> largest entries of each row of
    /// <paramref name="values"/> <c>[rows, length]</c>, in descending order, ties keeping the earlier entry (a stable
    /// descending sort's order). Entry i's rank is the number of entries that beat it; rank r &lt; top fills row r.
    /// Comparisons and sums only: no host read, no index tensor, and nothing for a gradient to flow through.
    /// </summary>
    public Tensor<T> TopRows(Tensor<T> values, int rows, int length, int top)
    {
        var mine = _engine.TensorTile(_engine.Reshape(values, new[] { rows, length, 1 }), new[] { 1, 1, length });   // [i, j] = v_i
        var theirs = _engine.TensorTile(_engine.Reshape(values, new[] { rows, 1, length }), new[] { 1, length, 1 }); // [i, j] = v_j
        var earlier = Tiled("earlier", rows, length, length, (i, j) => j < i ? 1.0 : 0.0);
        var beats = _engine.TensorAdd(_engine.TensorGreaterThan(theirs, mine),
            _engine.TensorMultiply(_engine.TensorEquals(theirs, mine), earlier));
        var rank = _engine.ReduceSum(beats, new[] { 2 }, keepDims: false);                                         // [rows, L]
        var rankTiled = _engine.TensorTile(_engine.Reshape(rank, new[] { rows, 1, length }), new[] { 1, top, 1 });
        var slot = Tiled("slot", rows, top, length, (u, _) => u);
        return _engine.TensorEquals(rankTiled, slot);
    }

    /// <summary>
    /// An inverted-dropout mask for <paramref name="x"/>: zero with probability <paramref name="rate"/>, otherwise
    /// <c>1 / (1 − rate)</c>. The engine draws it (on a GPU engine, on the device) from a seed the step's
    /// <paramref name="random"/> supplies, and a compiled training plan draws a fresh mask on every replay without a
    /// host round trip.
    /// </summary>
    public static Tensor<T> DropoutMask(Tensor<T> x, double rate, Random random)
    {
        int[] shape = (int[])x._shape.Clone();
        // A forward run only to resolve lazy shapes draws nothing, so the generator advances as it would eagerly.
        if (AiDotNet.Training.CompiledStepRandom<T>.InNonStepForward)
            return new Tensor<T>(shape);
        var rateValue = NumOps.FromDouble(rate);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        return AiDotNet.Training.CompiledStepDeviceValue<T>.Compute("SequenceDropoutMask", x, shape,
            engine => engine.TensorDropoutMask(shape, rateValue, keep, random.Next()));
    }
}
