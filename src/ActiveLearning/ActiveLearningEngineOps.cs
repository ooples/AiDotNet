namespace AiDotNet.ActiveLearning;

/// <summary>
/// Whole-array engine operations shared by the active learning strategies (softmax, entropy,
/// distances, score statistics and greedy diversity selection).
/// </summary>
internal static class ActiveLearningEngineOps
{
    /// <summary>
    /// <paramref name="epsilon"/> as a T, raised to Half's smallest positive normal value (2^-14) for Half,
    /// where 1e-10 and similar guards would otherwise round to zero (turning "+ epsilon" denominators and
    /// log(p + epsilon) back into divisions by zero and log(0)); float, double and decimal keep it unchanged.
    /// </summary>
    internal static T Tolerance<T>(double epsilon)
    {
        const double HalfSmallestNormal = 6.103515625e-05;
        return MathHelper.GetNumericOperations<T>().FromDouble(
            typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);
    }

    /// <summary>
    /// counts / total as T, with the division done in double so that counts above 2048 (not exact in
    /// Half) keep their ratio.
    /// </summary>
    internal static Tensor<T> CountFractions<T>(Tensor<int> counts, int total)
    {
        return AiDotNetEngine.Current.TensorDivideScalar(counts.Cast<double>(), (double)Math.Max(1, total)).Cast<T>();
    }

    /// <summary>
    /// Returns the first <paramref name="rows"/> x <paramref name="cols"/> elements of
    /// <paramref name="data"/> (row-major) as a [rows, cols] tensor.
    /// </summary>
    internal static Tensor<T> AsRows<T>(Tensor<T> data, int rows, int cols)
    {
        int count = rows * cols;
        var flat = data.Reshape(new[] { data.Length });
        if (count != data.Length)
        {
            flat = AiDotNetEngine.Current.TensorSlice(flat, new[] { 0 }, new[] { count });
        }

        return flat.Reshape(new[] { rows, cols });
    }

    /// <summary>
    /// Returns the samples of a pool as a [numSamples, featureSize] tensor, where
    /// featureSize = pool.Length / pool.Shape[0].
    /// </summary>
    internal static Tensor<T> PoolRows<T>(Tensor<T> pool, int numSamples)
    {
        int featureSize = pool.Shape[0] == 0 ? 0 : pool.Length / pool.Shape[0];
        return AsRows(pool, numSamples, featureSize);
    }

    /// <summary>
    /// Applies a numerically stable softmax to each sample's logits.
    /// </summary>
    /// <returns>A [numSamples, numClasses] tensor of probabilities.</returns>
    internal static Tensor<T> SoftmaxRows<T>(Tensor<T> predictions, int numSamples, int numClasses)
    {
        var logits = AsRows(predictions, numSamples, numClasses);
        if (numSamples == 0 || numClasses == 0)
        {
            return logits;
        }

        return AiDotNetEngine.Current.Softmax(logits, -1);
    }

    /// <summary>
    /// Computes H = -sum((p + eps) * log(p + eps)) along the last axis of a [n, c] tensor.
    /// </summary>
    /// <returns>A [n] tensor of entropies.</returns>
    internal static Tensor<T> EntropyRows<T>(Tensor<T> probabilities, T epsilon)
    {
        var engine = AiDotNetEngine.Current;
        if (probabilities.Shape[0] == 0 || probabilities.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { probabilities.Shape[0] });
        }

        var shifted = engine.TensorAddScalar(probabilities, epsilon);
        var terms = engine.TensorMultiply(shifted, engine.TensorLog(shifted));
        return engine.TensorNegate(engine.ReduceSum(terms, new[] { 1 }, keepDims: false));
    }

    /// <summary>
    /// Largest value of each row of a [n, c] tensor of non-negative values, floored at zero
    /// (an empty row gives zero).
    /// </summary>
    /// <returns>A [n] tensor.</returns>
    internal static Tensor<T> MaxRows<T>(Tensor<T> probabilities)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        if (probabilities.Shape[0] == 0 || probabilities.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { probabilities.Shape[0] });
        }

        return engine.TensorMax(engine.ReduceMax(probabilities, new[] { 1 }, keepDims: false), numOps.Zero);
    }

    /// <summary>
    /// Largest and second-largest value (counting repeats) of each row of a [n, c] tensor of
    /// non-negative values, both floored at zero; missing entries give zero.
    /// </summary>
    /// <returns>Two [n] tensors.</returns>
    internal static (Tensor<T> First, Tensor<T> Second) TopTwoRows<T>(Tensor<T> probabilities)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int n = probabilities.Shape[0];
        int c = probabilities.Shape[1];
        if (n == 0 || c < 2)
        {
            return (MaxRows(probabilities), new Tensor<T>(new[] { n }));
        }

        var (values, _) = engine.TopK(probabilities, 2, 1, true);
        var first = engine.TensorSlice(values, new[] { 0, 0 }, new[] { n, 1 }).Reshape(new[] { n });
        var second = engine.TensorSlice(values, new[] { 0, 1 }, new[] { n, 1 }).Reshape(new[] { n });
        return (engine.TensorMax(first, numOps.Zero), engine.TensorMax(second, numOps.Zero));
    }

    /// <summary>
    /// Euclidean distances from row <paramref name="index"/> of a [n, d] tensor to every row.
    /// </summary>
    /// <returns>A [n] tensor of distances.</returns>
    internal static Tensor<T> DistancesToRow<T>(Tensor<T> rows, int index)
    {
        if (rows.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { rows.Shape[0] });
        }

        var point = AiDotNetEngine.Current.TensorSlice(rows, new[] { index, 0 }, new[] { 1, rows.Shape[1] });
        return DistancesToPoint(rows, point);
    }

    /// <summary>
    /// Euclidean distances from a [1, d] point to every row of a [n, d] tensor, computed as
    /// sqrt(sum((row - point)^2)) without the Gram-matrix shortcut.
    /// </summary>
    /// <returns>A [n] tensor of distances.</returns>
    internal static Tensor<T> DistancesToPoint<T>(Tensor<T> rows, Tensor<T> point)
    {
        var engine = AiDotNetEngine.Current;
        if (rows.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { rows.Shape[0] });
        }

        var diff = engine.TensorSubtract(rows, engine.TensorBroadcastTo(point, new[] { rows.Shape[0], rows.Shape[1] }));
        return ScaledNorms(diff, 1);
    }

    /// <summary>
    /// Euclidean norms along <paramref name="axis"/>, computed as s * sqrt(sum((x / s)^2)) with
    /// s = max |x| along that axis, so that the squares cannot overflow (Half overflows once a
    /// component exceeds 255) or underflow; all-zero slices get norm 0.
    /// </summary>
    internal static Tensor<T> ScaledNorms<T>(Tensor<T> values, int axis)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = values.Shape.ToArray();
        var scale = engine.ReduceMax(engine.TensorAbs(values), new[] { axis }, keepDims: true);
        var ones = new Tensor<T>(scale.Shape.ToArray());
        ones.Fill(numOps.One);
        var safeScale = engine.TensorWhere(engine.TensorGreaterThan(scale, numOps.Zero), scale, ones);
        var unit = engine.TensorDivide(values, engine.TensorBroadcastTo(safeScale, shape));
        var norms = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(unit, unit), new[] { axis }, keepDims: false));
        var scaled = engine.TensorMultiply(norms, safeScale.Reshape(norms.Shape.ToArray()));

        // The scaled form rounds differently for every slice (s differs), so two slices with the same exact norm
        // can come out one ulp apart and a distance tie is then decided by rounding instead of by index. Where the
        // direct squares neither overflow nor underflow, the direct norm is used, as the scalar loops computed it.
        var plainSums = engine.ReduceSum(engine.TensorMultiply(values, values), new[] { axis }, keepDims: false);
        var usable = engine.TensorMultiply(
            engine.TensorGreaterThan(plainSums, numOps.Zero), engine.TensorLessThan(plainSums, numOps.MaxValue));
        return engine.TensorWhere(usable, engine.TensorSqrt(plainSums), scaled);
    }

    /// <summary>
    /// The rows of a [n, d] tensor divided by their Euclidean norms (rows with norm 0 stay 0), so
    /// that cosine similarities are dot products of unit rows instead of dot / (|a| |b|), whose
    /// numerator and denominator overflow in Half.
    /// </summary>
    internal static Tensor<T> UnitRows<T>(Tensor<T> rows, Tensor<T> norms)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = rows.Shape[0];
        var column = norms.Reshape(new[] { n, 1 });
        var ones = new Tensor<T>(new[] { n, 1 });
        ones.Fill(numOps.One);
        var safe = engine.TensorWhere(engine.TensorGreaterThan(column, numOps.Zero), column, ones);
        return engine.TensorDivide(rows, engine.TensorBroadcastTo(safe, rows.Shape.ToArray()));
    }

    /// <summary>
    /// Cosine similarities between rows [start, start + count) of a [n, d] tensor and all of its
    /// rows, as dot products of unit rows (see <see cref="UnitRows{T}"/>).
    /// </summary>
    /// <returns>A [count, n] tensor.</returns>
    internal static Tensor<T> CosineBlock<T>(Tensor<T> rows, Tensor<T> norms, int start, int count)
    {
        return DotBlock(UnitRows(rows, norms), start, count);
    }

    /// <summary>
    /// Mean of the rows of a [n, d] tensor as a [1, d] tensor, dividing before summing so that the
    /// sum cannot overflow (a Half sum overflows past 65504).
    /// </summary>
    internal static Tensor<T> MeanRow<T>(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        var scaled = engine.TensorDivideScalar(rows, MathHelper.GetNumericOperations<T>().FromDouble(Math.Max(1, n)));
        return engine.ReduceSum(scaled, new[] { 0 }, keepDims: true);
    }

    /// <summary>
    /// A [n, d] tensor divided by its largest absolute entry (unchanged when all entries are 0).
    /// k-means++ seeding and nearest-centroid assignment are invariant to this global rescale, and
    /// it keeps their squared distances finite in Half.
    /// </summary>
    internal static Tensor<T> UnitScale<T>(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        if (rows.Length == 0)
        {
            return rows;
        }

        T max = engine.TensorMaxValue(engine.TensorAbs(rows));
        return numOps.GreaterThan(max, numOps.Zero) ? engine.TensorDivideScalar(rows, max) : rows;
    }

    /// <summary>
    /// Number of rows per block for blockwise pairwise computations on a [n, d] tensor, sized so
    /// that a [rows, n, d] intermediate stays around four million elements.
    /// </summary>
    internal static int PairwiseBlockRows(int n, int d)
    {
        return BlockRows(n, (long)n * Math.Max(1, d));
    }

    /// <summary>
    /// Number of rows (at least one, at most <paramref name="totalRows"/>) per block when each row
    /// produces <paramref name="elementsPerRow"/> intermediate elements, targeting about four
    /// million elements per block.
    /// </summary>
    internal static int BlockRows(int totalRows, long elementsPerRow)
    {
        long rows = (1L << 22) / Math.Max(1L, elementsPerRow);
        return (int)Math.Max(1L, Math.Min(totalRows, rows));
    }

    /// <summary>
    /// Exact Euclidean distances between rows [start, start + count) of a [n, d] tensor and all of
    /// its rows, computed as sqrt(sum((a - b)^2)) without the Gram-matrix shortcut.
    /// </summary>
    /// <returns>A [count, n] tensor.</returns>
    internal static Tensor<T> DistanceBlock<T>(Tensor<T> rows, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int n = rows.Shape[0];
        int d = rows.Shape[1];
        if (d == 0)
        {
            return new Tensor<T>(new[] { count, n });
        }

        var shape = new[] { count, n, d };
        var block = engine.TensorSlice(rows, new[] { start, 0 }, new[] { count, d }).Reshape(new[] { count, 1, d });
        var all = rows.Reshape(new[] { 1, n, d });
        var diff = engine.TensorSubtract(engine.TensorBroadcastTo(block, shape), engine.TensorBroadcastTo(all, shape));
        return ScaledNorms(diff, 2);
    }

    /// <summary>
    /// Euclidean distances between every row of a [m, d] tensor and every row of a [k, d] tensor,
    /// computed exactly (no Gram-matrix shortcut, overflow-safe via <see cref="ScaledNorms{T}"/>)
    /// blockwise over the rows of <paramref name="a"/>.
    /// </summary>
    /// <returns>A [m, k] tensor.</returns>
    internal static Tensor<T> Distances<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        int m = a.Shape[0];
        int k = b.Shape[0];
        int d = a.Shape[1];
        if (d == 0 || m == 0 || k == 0)
        {
            return new Tensor<T>(new[] { m, k });
        }

        var blockRows = BlockRows(m, (long)k * d);
        var all = b.Reshape(new[] { 1, k, d });
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < m; start += blockRows)
        {
            int count = Math.Min(blockRows, m - start);
            var shape = new[] { count, k, d };
            var block = engine.TensorSlice(a, new[] { start, 0 }, new[] { count, d }).Reshape(new[] { count, 1, d });
            var diff = engine.TensorSubtract(engine.TensorBroadcastTo(block, shape), engine.TensorBroadcastTo(all, shape));
            blocks.Add(ScaledNorms(diff, 2));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <summary>
    /// Squared Euclidean distances: the square of <see cref="Distances{T}"/>. Squares of distances
    /// above about 255 are infinite in Half; callers that need finite values rescale first
    /// (see <see cref="UnitScale{T}"/>).
    /// </summary>
    /// <returns>A [m, k] tensor.</returns>
    internal static Tensor<T> SquaredDistances<T>(Tensor<T> a, Tensor<T> b)
    {
        var distances = Distances(a, b);
        return AiDotNetEngine.Current.TensorMultiply(distances, distances);
    }

    /// <summary>
    /// Stacks equal-length vectors into a [count, length] tensor.
    /// </summary>
    internal static Tensor<T> StackVectors<T>(IReadOnlyList<Vector<T>> vectors)
    {
        var rows = new Tensor<T>[vectors.Count];
        for (int i = 0; i < vectors.Count; i++)
        {
            rows[i] = Tensor<T>.FromVector(vectors[i]);
        }

        return AiDotNetEngine.Current.TensorStack(rows, 0);
    }

    /// <summary>
    /// Dot products between rows [start, start + count) of a [n, d] tensor and all of its rows.
    /// </summary>
    /// <returns>A [count, n] tensor.</returns>
    internal static Tensor<T> DotBlock<T>(Tensor<T> rows, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int d = rows.Shape[1];
        if (d == 0)
        {
            return new Tensor<T>(new[] { count, rows.Shape[0] });
        }

        var block = engine.TensorSlice(rows, new[] { start, 0 }, new[] { count, d });
        return engine.TensorMatMul(block, engine.TensorTranspose(rows));
    }

    /// <summary>
    /// Euclidean norm of every row of a [n, d] tensor.
    /// </summary>
    /// <returns>A [n] tensor.</returns>
    internal static Tensor<T> RowNorms<T>(Tensor<T> rows)
    {
        if (rows.Shape[1] == 0)
        {
            return new Tensor<T>(new[] { rows.Shape[0] });
        }

        return ScaledNorms(rows, 1);
    }

    /// <summary>
    /// Mask over a [count, n] block that is true where the block row is the same sample as the
    /// column (row r of the block is sample start + r).
    /// </summary>
    internal static bool[] SelfMask(int start, int count, int n)
    {
        var mask = new bool[count * n];
        for (int r = 0; r < count && start + r < n; r++)
        {
            mask[r * n + start + r] = true;
        }

        return mask;
    }

    /// <summary>
    /// Computes min, max and mean of a non-empty score vector.
    /// </summary>
    internal static (T Min, T Max, T Mean) Statistics<T>(Vector<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var tensor = Tensor<T>.FromVector(scores);
        return (engine.TensorMinValue(tensor), engine.TensorMaxValue(tensor), engine.TensorMean(tensor));
    }

    /// <summary>
    /// Index of the first maximum of a [1, n] candidate tensor among the entries that are not
    /// excluded and are strictly greater than <paramref name="floor"/>, or -1 when none qualifies
    /// (NaN never qualifies).
    /// </summary>
    internal static int ArgMaxAbove<T>(Tensor<T> candidates, bool[] excluded, T floor)
    {
        var engine = AiDotNetEngine.Current;
        var masked = engine.TensorMaskedFill(candidates, excluded, floor);
        var valid = engine.TensorGreaterThan(masked, floor);
        if (engine.TensorCountNonzero(valid) == 0)
        {
            return -1;
        }

        var floorTensor = new Tensor<T>(masked.Shape.ToArray());
        floorTensor.Fill(floor);
        var safe = engine.TensorWhere(valid, masked, floorTensor);
        return engine.TensorArgMax(safe, 1).GetFlat(0);
    }

    /// <summary>
    /// Greedily selects samples maximizing score x (minimum distance to the samples selected so far);
    /// the first pick uses the score alone. Ties go to the lowest index.
    /// </summary>
    internal static int[] SelectWithDiversity<T>(Vector<T> scores, Tensor<T> pool, int batchSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int n = scores.Length;
        var selected = new List<int>();
        if (n == 0 || batchSize <= 0)
        {
            return [.. selected];
        }

        var rows = PoolRows(pool, n);
        var scoreRow = Tensor<T>.FromVector(scores).Reshape(new[] { 1, n });
        var excluded = new bool[n];
        Tensor<T>? minDistance = null;

        while (selected.Count < batchSize && selected.Count < n)
        {
            var combined = minDistance is null ? scoreRow : engine.TensorMultiply(scoreRow, minDistance);
            int best = ArgMaxAbove(combined, excluded, numOps.MinValue);
            if (best < 0)
            {
                // No remaining candidate has a comparable score (e.g. NaN): stop instead of spinning.
                break;
            }

            selected.Add(best);
            excluded[best] = true;

            var distances = DistancesToRow(rows, best).Reshape(new[] { 1, n });
            minDistance = minDistance is null ? distances : engine.TensorMin(minDistance, distances);
        }

        return [.. selected];
    }

    /// <summary>
    /// Softmax probabilities of each prediction tensor, each [numSamples, numClasses].
    /// </summary>
    internal static List<Tensor<T>> SoftmaxAll<T>(List<Tensor<T>> predictions, int numSamples, int numClasses)
    {
        var probs = new List<Tensor<T>>(predictions.Count);
        foreach (var prediction in predictions)
        {
            probs.Add(SoftmaxRows(prediction, numSamples, numClasses));
        }

        return probs;
    }

    /// <summary>
    /// BALD scores I(y; θ|x) = H(mean_m p_m) - mean_m H(p_m) for every sample, from the per-pass
    /// [numSamples, numClasses] probabilities of the Monte Carlo forward passes.
    /// </summary>
    /// <returns>A [numSamples] tensor.</returns>
    internal static Tensor<T> BaldScores<T>(List<Tensor<T>> mcProbabilities, int numSamples)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var epsilon = ActiveLearningEngineOps.Tolerance<T>(1e-10);

        // Sum the per-pass probabilities and entropies across MC samples
        Tensor<T>? probSum = null;
        Tensor<T>? entropySum = null;
        foreach (var probs in mcProbabilities)
        {
            var entropy = EntropyRows(probs, epsilon);
            probSum = probSum is null ? probs : engine.TensorAdd(probSum, probs);
            entropySum = entropySum is null ? entropy : engine.TensorAdd(entropySum, entropy);
        }

        if (probSum is null || entropySum is null)
        {
            return new Tensor<T>(new[] { numSamples });
        }

        // H(y|x): entropy of the average prediction
        var count = numOps.FromDouble(mcProbabilities.Count);
        var entropyOfAvg = EntropyRows(engine.TensorDivideScalar(probSum, count), epsilon);

        // E[H(y|x, θ)]: average of the individual entropies
        var avgEntropy = engine.TensorDivideScalar(entropySum, count);
        return engine.TensorSubtract(entropyOfAvg, avgEntropy);
    }

    /// <summary>
    /// Selects the <paramref name="batchSize"/> highest-scoring indices, ties in index order
    /// (the order of the stable LINQ OrderByDescending this replaces).
    /// </summary>
    internal static int[] SelectTopScoring<T>(Vector<T> scores, int batchSize)
    {
        var order = StableOrder(Tensor<T>.FromVector(scores), descending: true);
        int take = Math.Max(0, Math.Min(batchSize, order.Length));
        var result = new int[take];
        Array.Copy(order, result, take);
        return result;
    }

    /// <summary>
    /// Positions of <paramref name="values"/> in stable sorted order (ties keep their original order),
    /// ascending or descending.
    /// </summary>
    /// <remarks>
    /// The engine sort is not guaranteed stable, so position j gets the unique key
    /// (number of values sorting strictly before v[j]) * count + j, found with one sort and a binary
    /// search; sorting those keys gives the stable order in O(count log count).
    /// </remarks>
    internal static int[] StableOrder<TValue>(Tensor<TValue> values, bool descending)
    {
        int count = values.Length;
        if (count == 0)
        {
            return [];
        }

        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(values.Contiguous(), [count]);
        var sorted = engine.TensorSort(flat, 0, false).Values;

        // ascending: #{m : v[m] < v[j]}; descending: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var before = descending
            ? engine.ScalarMinusTensor(count, engine.TensorSearchSorted(sorted, flat, true))
            : engine.TensorSearchSorted(sorted, flat, false);
        var position = engine.TensorArange<long>(0, count, 1);
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(before.Cast<long>(), (long)count), position);
        return engine.TensorArgsort(key, 0, false).ToArray();
    }
}
