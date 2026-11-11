namespace AiDotNet.ActiveLearning;

/// <summary>
/// Whole-array engine operations shared by the active learning strategies (softmax, entropy,
/// distances, score statistics and greedy diversity selection).
/// </summary>
internal static class ActiveLearningEngineOps
{
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
        return engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, keepDims: false));
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
    /// Selects the <paramref name="batchSize"/> highest-scoring indices, ties in index order.
    /// </summary>
    /// <remarks>
    /// Kept as a stable LINQ sort until the engine provides a stable argsort
    /// (TensorArgSortStable in missing-ops): IEngine.ArgSort/TopK use an unstable sort and would
    /// reorder tied scores.
    /// </remarks>
    internal static int[] SelectTopScoring<T>(Vector<T> scores, int batchSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var indexedScores = new List<(int Index, T Score)>();
        for (int i = 0; i < scores.Length; i++)
        {
            indexedScores.Add((i, scores[i]));
        }

        return indexedScores
            .OrderByDescending(x => numOps.ToDouble(x.Score))
            .Take(batchSize)
            .Select(x => x.Index)
            .ToArray();
    }
}
