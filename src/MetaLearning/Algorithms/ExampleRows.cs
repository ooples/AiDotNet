namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Whole-tensor helpers for meta-learners that read a flattened feature vector as per-example rows and solve
/// small symmetric positive-definite systems over them (GP regression, ridge reconstruction).
/// </summary>
internal static class ExampleRows
{
    /// <summary>
    /// Reads up to <paramref name="maxRows"/> consecutive <paramref name="width"/>-wide rows from
    /// <paramref name="flat"/>; the last row may be shorter and is zero-padded, and rows that would start past
    /// the end are dropped.
    /// </summary>
    /// <returns>The [count, width] rows, a matching 0/1 mask of the entries that came from
    /// <paramref name="flat"/>, and the row count.</returns>
    internal static (Tensor<T> Rows, Tensor<T> Mask, int Count) Split<T>(
        IEngine engine, Vector<T> flat, int maxRows, int width)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int count = Math.Max(0, Math.Min(maxRows, (flat.Length + width - 1) / width));
        int size = count * width;
        int used = Math.Min(flat.Length, size);

        var values = used == size ? flat.GetSubVector(0, size)
            : Vector<T>.Concatenate(flat.GetSubVector(0, used), new Vector<T>(size - used));
        var present = used == size ? engine.Fill(size, numOps.One)
            : Vector<T>.Concatenate(engine.Fill(used, numOps.One), new Vector<T>(size - used));

        var shape = new[] { count, width };
        return (engine.Reshape(Tensor<T>.FromVector(values), shape),
            engine.Reshape(Tensor<T>.FromVector(present), shape), count);
    }

    /// <summary>
    /// Solves a x = b for a symmetric positive-definite <paramref name="a"/> [n, n] and right-hand sides
    /// <paramref name="b"/> [n, m] through a Cholesky factor (as GPyTorch and scikit-learn do for GP and ridge
    /// systems). A matrix that is not numerically positive definite falls back to a pivoted LU solve, and a
    /// singular one gives x = 0.
    /// </summary>
    internal static Tensor<T> SolveSymmetricPositiveDefinite<T>(Tensor<T> a, Tensor<T> b)
    {
        var (factor, info) = LinalgDispatch<T>.CholeskyEx(a);
        if (info.ToArray().All(code => code == 0))
        {
            return LinalgDispatch<T>.CholeskySolve(b, factor);
        }

        var (solution, solveInfo) = LinalgDispatch<T>.SolveEx(a, b);
        return solveInfo.ToArray().All(code => code == 0) ? solution : new Tensor<T>(b.Shape.ToArray());
    }
}
