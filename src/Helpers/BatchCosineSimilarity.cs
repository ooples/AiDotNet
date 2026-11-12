using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Scores one query vector against many candidates with the semantics of
/// <see cref="VectorHelper.CosineSimilarity{T}(Vector{T}, Vector{T}, double)"/>, as whole-tensor engine operations.
/// </summary>
/// <remarks>
/// Each score is <c>dot / sqrt(|q|^2 |c|^2)</c> clamped to [-1, 1], or 0 when that denominator is not above
/// <c>epsilon</c> (zero vectors and NaN norms score 0, as in the scalar helper). Reductions and
/// normalization use double precision so low-precision norm products do not overflow.
/// </remarks>
internal static class BatchCosineSimilarity
{
    /// <summary>Cosine similarity of <paramref name="query"/> with each candidate, in candidate order.</summary>
    /// <exception cref="ArgumentNullException">The query or a candidate is null.</exception>
    /// <exception cref="ArgumentException">A candidate's length differs from the query's.</exception>
    public static Tensor<T> Scores<T>(Vector<T> query, IEnumerable<Vector<T>> candidates, double epsilon = 1e-10)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (candidates is null) throw new ArgumentNullException(nameof(candidates));

        var rows = new List<Tensor<T>>();
        foreach (var candidate in candidates)
        {
            if (candidate is null) throw new ArgumentNullException(nameof(candidates));
            if (candidate.Length != query.Length)
                throw new ArgumentException(
                    $"Vectors must have equal length for cosine similarity; got {query.Length} and {candidate.Length}.");
            rows.Add(Tensor<T>.FromVector(candidate));
        }

        if (rows.Count == 0) return new Tensor<T>(new[] { 0 });
        return RowScores(query, Tensor<T>.Stack(rows.ToArray()), epsilon);
    }

    /// <summary>Cosine similarity of <paramref name="query"/> with each row of a <c>[n, query.Length]</c> tensor.</summary>
    public static Tensor<T> RowScores<T>(Vector<T> query, Tensor<T> rows, double epsilon = 1e-10)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (rows is null) throw new ArgumentNullException(nameof(rows));
        if (rows.Rank != 2 || rows.Shape[1] != query.Length)
            throw new ArgumentException(
                $"Vectors must have equal length for cosine similarity; got {query.Length} and {(rows.Rank == 2 ? rows.Shape[1] : rows.Length)}.");

        int count = rows.Shape[0];
        if (count == 0 || query.Length == 0)
            return new Tensor<T>(new[] { count });

        var engine = AiDotNetEngine.Current;
        var queryColumn = Tensor<T>.FromVector(query, new[] { query.Length, 1 }).Cast<double>();
        var wideRows = rows.Cast<double>();
        var dots = engine.Reshape(engine.TensorMatMul(wideRows, queryColumn), new[] { count });
        double querySquaredNorm = engine.DotProduct(queryColumn.ToVector(), queryColumn.ToVector());
        var rowSquaredNorms = engine.ReduceSum(engine.TensorSquare(wideRows), new[] { 1 }, false);
        var denominators = engine.TensorSqrt(engine.TensorMultiplyScalar(rowSquaredNorms, querySquaredNorm));

        var valid = engine.TensorGreaterThan(denominators, epsilon);
        var ones = new Tensor<double>(new[] { count });
        ones.Fill(1.0);
        var safeDenominators = engine.TensorWhere(valid, denominators, ones);
        var ratios = engine.TensorClamp(engine.TensorDivide(dots, safeDenominators), -1.0, 1.0);
        return engine.TensorWhere(valid, ratios, new Tensor<double>(new[] { count })).Cast<T>();
    }
}
