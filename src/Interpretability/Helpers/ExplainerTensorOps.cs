using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Helpers;

/// <summary>
/// Whole-array building blocks shared by the explainers: path interpolation, path integration,
/// finite-difference gradients and row/tensor conversions, all expressed as engine operations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class ExplainerTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Indices that order a tensor's values descending, equal values kept in index order, as LINQ
    /// <c>OrderByDescending</c> does.
    /// </summary>
    public static int[] StableDescendingOrder(Tensor<T> values)
    {
        int count = values.Length;
        if (count == 0)
            return Array.Empty<int>();

        var engine = AiDotNetEngine.Current;
        var flat = values.Rank == 1 ? values : engine.Reshape(values, new[] { count });
        var sorted = engine.TensorSort(flat, 0, false).Item1;

        // descending rank #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}; the engine argsort is not
        // documented as stable, so it sorts the distinct key (rank, index), whose order is unique
        var atMost = engine.TensorSearchSorted(sorted, flat, true).AsSpan().ToArray();
        var keys = new long[count];
        for (int j = 0; j < count; j++)
            keys[j] = (long)(count - atMost[j]) * count + j;

        return engine.TensorArgsort(new Tensor<long>(keys, new[] { count }), 0, false).AsSpan().ToArray();
    }

    /// <summary>
    /// A [1, n] tensor holding a copy of <paramref name="row"/>.
    /// </summary>
    public static Tensor<T> RowTensor(Vector<T> row)
    {
        return new Tensor<T>(new[] { 1, row.Length }, row.Clone());
    }

    /// <summary>
    /// A 1-row matrix holding a copy of <paramref name="row"/>.
    /// </summary>
    public static Matrix<T> RowMatrix(Vector<T> row)
    {
        var matrix = new Matrix<T>(1, row.Length);
        AiDotNetEngine.Current.SetRow(matrix, 0, row);
        return matrix;
    }

    /// <summary>
    /// <paramref name="epsilon"/> as a T, raised to the smallest positive normal Half when T is Half (Half rounds
    /// 1e-10 to 0, which disables "above epsilon" guards); other types keep <paramref name="epsilon"/>.
    /// </summary>
    public static T Tolerance(double epsilon) =>
        AiDotNet.Preprocessing.DimensionalityReduction.DimensionalityReductionOps.Tolerance<T>(epsilon);

    /// <summary>
    /// The sums of squares of <paramref name="numerator"/> and <paramref name="denominator"/>, both divided by the
    /// square of the largest magnitude in either, so their ratio is exact while neither sum overflows (a Half square
    /// overflows above 256). Returns (0, 0) when both tensors are all zero.
    /// </summary>
    public static (T Numerator, T Denominator) ScaledSumsOfSquares(Tensor<T> numerator, Tensor<T> denominator)
    {
        var engine = AiDotNetEngine.Current;
        T scale = MathHelper.Max(
            engine.TensorMaxValue(engine.TensorAbs(numerator)), engine.TensorMaxValue(engine.TensorAbs(denominator)));
        if (!NumOps.GreaterThan(scale, NumOps.Zero))
        {
            return (NumOps.Zero, NumOps.Zero);
        }

        return (engine.TensorSumOfSquares(engine.TensorDivideScalar(numerator, scale)),
            engine.TensorSumOfSquares(engine.TensorDivideScalar(denominator, scale)));
    }

    /// <summary>
    /// Index of the largest entry (first one on ties).
    /// </summary>
    public static int ArgMax(Vector<T> values)
    {
        if (values.Length == 0)
            throw new ArgumentException("Cannot take the arg max of an empty vector.", nameof(values));

        // The engine arg max does not promise the first index on ties
        return AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMax(values.AsTensor());
    }

    /// <summary>
    /// Index of the first largest entry strictly above <paramref name="seed"/> (NaN never is), or -1 when no entry is:
    /// the scan "best = seed; move only on a strictly larger entry" without its default index.
    /// </summary>
    public static int FirstArgMaxAbove(Vector<T> values, T seed)
    {
        var engine = AiDotNetEngine.Current;
        var tensor = values.AsTensor();
        var above = engine.TensorGreaterThan(tensor, seed);
        if (values.Length == 0 || engine.TensorCountNonzero(above) == 0)
        {
            return -1;
        }

        var candidates = engine.TensorWhere(above, tensor, engine.TensorFullLike(tensor, seed));
        var isMax = engine.TensorMultiply(engine.TensorEquals(candidates, engine.TensorMaxValue(candidates)), above);
        return AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMax(isMax);
    }

    /// <summary>
    /// Index of the first largest entry, skipping NaN, or 0 when every entry is NaN or the type's minimum: what the
    /// scan "best = MinValue; move only on a strictly larger entry" returns.
    /// </summary>
    public static int ArgMaxSkippingNaN(Vector<T> values)
    {
        return Math.Max(0, FirstArgMaxAbove(values, NumOps.MinValue));
    }

    /// <summary>
    /// Rows baseline + alpha_s * (input - baseline) for each alpha_s, as an [alphas, n] tensor.
    /// </summary>
    public static Tensor<T> InterpolationPath(Vector<T> baseline, Vector<T> input, Tensor<T> alphas)
    {
        var engine = AiDotNetEngine.Current;
        var offsets = engine.TensorOuter(alphas, engine.Subtract(input, baseline).AsTensor());
        var baselineRows = engine.TensorBroadcastTo(baseline.AsTensor().Reshape(new[] { 1, baseline.Length }), offsets.Shape.ToArray());
        return engine.TensorAdd(offsets, baselineRows);
    }

    /// <summary>
    /// alpha_s = s / steps for s = 0..steps (both end points), as a [steps + 1] tensor.
    /// </summary>
    public static Tensor<T> UniformAlphasWithEndpoints(int steps)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivideScalar(engine.TensorArange<T>(0, steps + 1, 1), NumOps.FromDouble(steps));
    }

    /// <summary>
    /// Trapezoid-rule average of gradients sampled at s / steps for s = 0..steps: the end points weigh 0.5,
    /// interior points 1, and the sum is divided by steps.
    /// </summary>
    public static Vector<T> TrapezoidAverage(IReadOnlyList<Vector<T>> gradients)
    {
        var engine = AiDotNetEngine.Current;
        int count = gradients.Count;
        int steps = count - 1;
        var weights = engine.Fill(count, NumOps.Divide(NumOps.One, NumOps.FromDouble(steps)));
        T halfWeight = NumOps.Divide(NumOps.FromDouble(0.5), NumOps.FromDouble(steps));
        weights[0] = halfWeight;
        weights[steps] = halfWeight;
        return WeightedSum(gradients, weights);
    }

    /// <summary>
    /// Sum over i of weights[i] * vectors[i].
    /// </summary>
    public static Vector<T> WeightedSum(IReadOnlyList<Vector<T>> vectors, Vector<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        var stacked = Stack(vectors);
        return engine.TensorMatMul(weights.AsTensor().Reshape(new[] { 1, vectors.Count }), stacked).ToVector();
    }

    /// <summary>
    /// Element-wise mean of equally long vectors.
    /// </summary>
    public static Vector<T> Mean(IReadOnlyList<Vector<T>> vectors)
    {
        return AiDotNetEngine.Current.ReduceMean(Stack(vectors), new[] { 0 }, false).ToVector();
    }

    /// <summary>
    /// Equally long vectors as the rows of a [count, length] tensor.
    /// </summary>
    public static Tensor<T> Stack(IReadOnlyList<Vector<T>> vectors)
    {
        var rows = new Tensor<T>[vectors.Count];
        for (int i = 0; i < rows.Length; i++)
        {
            rows[i] = vectors[i].AsTensor();
        }

        return AiDotNetEngine.Current.TensorStack(rows, 0);
    }

    /// <summary>
    /// Central-difference gradient (f(x + eps e_j) - f(x - eps e_j)) / (2 eps) of output
    /// <paramref name="outputIndex"/> for each of the first <paramref name="count"/> features, evaluating the
    /// plus then the minus input for each feature in turn.
    /// </summary>
    /// <param name="predict">The model's prediction for one input.</param>
    /// <param name="input">The point to differentiate at.</param>
    /// <param name="outputIndex">The output to differentiate.</param>
    /// <param name="epsilon">The finite-difference step.</param>
    /// <param name="count">How many leading features to differentiate.</param>
    /// <param name="outOfRangeIsZero">When true, an output index past the prediction yields zero gradients
    /// instead of throwing.</param>
    public static Vector<T> CentralDifferenceGradient(
        Func<Vector<T>, Vector<T>> predict, Vector<T> input, int outputIndex, double epsilon, int count,
        bool outOfRangeIsZero = false)
    {
        var engine = AiDotNetEngine.Current;
        T eps = NumOps.FromDouble(epsilon);
        var inputTensor = input.AsTensor();
        var plusAll = engine.TensorAddScalar(inputTensor, eps);
        var minusAll = engine.TensorSubtractScalar(inputTensor, eps);
        var positions = engine.TensorArange<int>(0, input.Length, 1);

        var plusOutputs = new Tensor<T>[count];
        var minusOutputs = new Tensor<T>[count];
        for (int j = 0; j < count; j++)
        {
            var isFeature = engine.TensorEqScalar(positions, j);
            plusOutputs[j] = predict(engine.TensorWhere(isFeature, plusAll, inputTensor).ToVector()).AsTensor();
            minusOutputs[j] = predict(engine.TensorWhere(isFeature, minusAll, inputTensor).ToVector()).AsTensor();
        }

        if (count == 0)
            return new Vector<T>(0);

        int outputs = plusOutputs[0].Length;
        if (outOfRangeIsZero && outputIndex >= outputs)
            return new Vector<T>(count);
        if (outputIndex < 0 || outputIndex >= outputs)
            throw new ArgumentOutOfRangeException(nameof(outputIndex),
                $"Output index {outputIndex} is out of bounds for prediction with {outputs} outputs.");

        var plus = engine.TensorSliceAxis(engine.TensorStack(plusOutputs, 0), 1, outputIndex);
        var minus = engine.TensorSliceAxis(engine.TensorStack(minusOutputs, 0), 1, outputIndex);
        return engine.TensorDivideScalar(engine.TensorSubtract(plus, minus), NumOps.FromDouble(2 * epsilon)).ToVector();
    }

    /// <summary>
    /// "+" for entries at or above zero and "" for negative ones (a negative value prints its own sign).
    /// </summary>
    public static string[] SignPrefixes(IEnumerable<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var vector = new Vector<T>(values).AsTensor();
        var atOrAboveZero = engine.TensorGreaterEqual(vector, engine.TensorFullLike(vector, NumOps.Zero)).Cast<int>().ToVector();
        return Enumerable.Select(atOrAboveZero, flag => flag != 0 ? "+" : "").ToArray();
    }

    /// <summary>
    /// Dense ranks of the values (0 for the smallest, equal values share a rank), so that comparing or
    /// ordering by rank (a stable LINQ order on an int key) matches comparing or ordering by value.
    /// </summary>
    public static int[] DenseRanks(Tensor<T> values)
    {
        int n = values.Length;
        var ranks = new int[n];
        if (n <= 1)
        {
            return ranks;
        }

        var engine = AiDotNetEngine.Current;
        var (sorted, order) = engine.TensorSort(values);
        var increases = engine.TensorGreaterThan(
            engine.TensorSlice(sorted, new[] { 1 }, new[] { n - 1 }),
            engine.TensorSlice(sorted, new[] { 0 }, new[] { n - 1 }));
        // Count in int: a 0/1 running sum in Half stops growing at 2048
        var denseRanks = engine.TensorCumSum(increases.Cast<int>(), 0).ToArray();
        var positions = order.ToArray();
        for (int k = 1; k < n; k++)
        {
            ranks[positions[k]] = denseRanks[k - 1];
        }

        return ranks;
    }

    /// <summary>
    /// The first <paramref name="length"/> entries of <paramref name="values"/>, zero-padded when it is shorter.
    /// </summary>
    public static Vector<T> PadOrTruncate(Vector<T> values, int length)
    {
        int count = Math.Min(values.Length, length);
        if (count == length)
        {
            return values.GetSubVector(0, length);
        }

        if (count == 0)
        {
            return new Vector<T>(length);
        }

        var padded = AiDotNetEngine.Current.TensorConcatenate(
            new[] { values.GetSubVector(0, count).AsTensor(), new Tensor<T>(new[] { length - count }) }, 0);
        return padded.ToVector();
    }

    /// <summary>
    /// A 0/1 vector of length <paramref name="length"/> that is 1 at indices in [start, end).
    /// </summary>
    public static Tensor<T> RangeIndicator(int length, int start, int end)
    {
        var engine = AiDotNetEngine.Current;
        // Compare indices in int: Half cannot represent every index above 2048
        var indices = engine.TensorArange<int>(0, length, 1);
        var atOrAfterStart = engine.TensorGreaterThan(indices, start - 1);
        var beforeEnd = engine.TensorLessThan(indices, end);
        return engine.TensorMultiply(atOrAfterStart, beforeEnd).Cast<T>();
    }

    /// <summary>
    /// (x - min) / (max - min), with a range below <paramref name="minimumRange"/> replaced by 1.
    /// </summary>
    public static Tensor<T> MinMaxNormalize(Tensor<T> values, double minimumRange)
    {
        if (values.Length == 0)
        {
            return values;
        }

        var engine = AiDotNetEngine.Current;
        T minValue = engine.TensorMinValue(values);
        T range = NumOps.Subtract(engine.TensorMaxValue(values), minValue);
        if (NumOps.LessThan(range, NumOps.FromDouble(minimumRange)))
        {
            range = NumOps.One;
        }

        return engine.TensorDivideScalar(engine.TensorSubtractScalar(values, minValue), range);
    }

    /// <summary>
    /// Bilinear resize of a [height, width] map, sampling source coordinate i / target * source along each axis
    /// (clamped at the last row and column), as R_h * map * R_w^T with per-axis interpolation matrices.
    /// </summary>
    public static Tensor<T> BilinearResize(Tensor<T> map, int targetHeight, int targetWidth)
    {
        var engine = AiDotNetEngine.Current;
        var rows = InterpolationMatrix(map.Shape[0], targetHeight);
        var columns = InterpolationMatrix(map.Shape[1], targetWidth);
        return engine.TensorMatMul(engine.TensorMatMul(rows, map), engine.TensorTranspose(columns));
    }

    /// <summary>
    /// The [target, source] linear-interpolation matrix: row i weighs (1 - f) at floor(s) and f at
    /// min(floor(s) + 1, source - 1), where s = i / target * source and f = s - floor(s).
    /// </summary>
    private static Tensor<T> InterpolationMatrix(int source, int target)
    {
        var engine = AiDotNetEngine.Current;
        var lower = Enumerable.Range(0, target).Select(i => Math.Min((int)((long)i * source / target), source - 1)).ToArray();
        var upper = lower.Select(index => Math.Min(index + 1, source - 1)).ToArray();
        var lowerIndex = new Tensor<int>(lower, new[] { target });

        // f = s - floor(s) = ((i * source) mod target) / target, from int index arithmetic (an index tensor in Half
        // cannot represent positions above 2048, so s - floor(s) taken in T loses the fraction)
        var remainders = Enumerable.Range(0, target).Select(i => (int)((long)i * source % target)).ToArray();
        var fraction = engine.TensorDivideScalar(
            new Tensor<int>(remainders, new[] { target }).Cast<T>(), NumOps.FromDouble(target)).Reshape(new[] { target, 1 });
        var complement = engine.TensorSubtract(engine.TensorFullLike(fraction, NumOps.One), fraction);

        var identity = engine.TensorEye<T>(source);
        var lowerRows = engine.TensorIndexSelect(identity, lowerIndex, 0);
        var upperRows = engine.TensorIndexSelect(identity, new Tensor<int>(upper, new[] { target }), 0);
        var shape = new[] { target, source };
        return engine.TensorAdd(
            engine.TensorMultiply(lowerRows, engine.TensorBroadcastTo(complement, shape)),
            engine.TensorMultiply(upperRows, engine.TensorBroadcastTo(fraction, shape)));
    }
}
