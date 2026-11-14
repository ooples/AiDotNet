using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// Scores one embedding against many with whole-tensor engine operations.
/// </summary>
/// <typeparam name="T">The numeric type of the embeddings.</typeparam>
internal static class EmbeddingScores<T>
{
    /// <summary>
    /// Cosine similarity between <paramref name="query"/> and each of <paramref name="candidates"/>, with the
    /// semantics of <c>VectorHelper.CosineSimilarity</c>: clamped to [-1, 1], and 0 when the product of
    /// the two norms is not above 1e-10 (which includes NaN norms).
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="query">The embedding scored against every candidate (read flattened).</param>
    /// <param name="candidates">The embeddings to score (each read flattened, each as long as the query).</param>
    /// <returns>A <c>[candidates.Length]</c> tensor of similarities.</returns>
    internal static Tensor<T> CosineToEach(IEngine engine, Tensor<T> query, Tensor<T>[] candidates)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (candidates is null) throw new ArgumentNullException(nameof(candidates));

        int count = candidates.Length;
        if (count == 0)
            return new Tensor<T>([0]);

        int dim = query.Length;
        var rows = new Tensor<T>[count];
        for (int c = 0; c < count; c++)
        {
            var candidate = candidates[c] ?? throw new ArgumentNullException(nameof(candidates));
            if (candidate.Length != dim)
                throw new ArgumentException(
                    $"Vectors must have equal length for cosine similarity; got {dim} and {candidate.Length}.");
            rows[c] = candidate.Reshape([1, dim]);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var unitRows = UnitRows(engine, engine.TensorConcatenate(rows, 0), out var rowNorms);
        var unitQuery = UnitRows(engine, query.Reshape([1, dim]), out var queryNorm);
        var cosine = engine.TensorClamp(
            engine.TensorMatMul(unitRows, unitQuery.Reshape([dim, 1])).Reshape([count]),
            numOps.FromDouble(-1.0), numOps.One);

        // A norm product that overflows a narrow type becomes +infinity, which still passes the guard.
        var normProduct = engine.TensorMultiply(rowNorms, queryNorm).Reshape([count]);
        var valid = engine.TensorGreaterThan(normProduct, numOps.FromDouble(1e-10));
        return engine.TensorWhere(valid, cosine, new Tensor<T>([count]));
    }

    /// <summary>
    /// SigLIP's per-pair match probability <c>sigmoid(cos(query, candidate) / temperature + bias)</c>
    /// (Zhai et al., 2023), for each candidate on its own: the scores are not normalized across candidates.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="query">The embedding scored against every candidate.</param>
    /// <param name="candidates">The embeddings to score.</param>
    /// <param name="temperature">The temperature (1 / the paper's logit scale t).</param>
    /// <param name="bias">The logit bias b.</param>
    /// <returns>A <c>[candidates.Length]</c> tensor of probabilities.</returns>
    internal static Tensor<T> SigmoidPairScores(IEngine engine, Tensor<T> query, Tensor<T>[] candidates, T temperature,
        T bias) => engine.TensorSigmoid(engine.TensorAddScalar(
            engine.TensorDivideScalar(CosineToEach(engine, query, candidates), temperature), bias));

    /// <summary>
    /// Divides each row of a <c>[n, d]</c> tensor by its L2 norm. Each row is first scaled by its largest
    /// magnitude, so the squares summed stay within [0, d] and cannot overflow a narrow type such as Half.
    /// A zero row stays zero.
    /// </summary>
    /// <param name="engine">Engine to run on.</param>
    /// <param name="rows">The <c>[n, d]</c> rows.</param>
    /// <param name="norms">The <c>[n, 1]</c> L2 norm of each row.</param>
    /// <returns>The unit rows.</returns>
    private static Tensor<T> UnitRows(IEngine engine, Tensor<T> rows, out Tensor<T> norms)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = rows.Shape[0];
        int d = rows.Shape[1];
        var ones = Tensor<T>.CreateDefault([n, 1], numOps.One);

        var maxAbs = engine.ReduceMax(engine.TensorAbs(rows), [1], true);
        var positive = engine.TensorGreaterThan(maxAbs, numOps.Zero);
        var scale = engine.TensorWhere(positive, maxAbs, ones);
        var scaled = engine.TensorDivide(rows, engine.TensorBroadcastTo(scale, [n, d]));

        var scaledNorm = engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(scaled, scaled), [1], true));
        norms = engine.TensorMultiply(scaledNorm, scale);
        var safeNorm = engine.TensorWhere(positive, scaledNorm, ones);
        return engine.TensorDivide(scaled, engine.TensorBroadcastTo(safeNorm, [n, d]));
    }
}
