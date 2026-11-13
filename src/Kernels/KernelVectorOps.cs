namespace AiDotNet.Kernels;

/// <summary>
/// Whole-vector engine helpers shared by kernels whose per-dimension terms need a guard.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class KernelVectorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes sum_i (x1_i - x2_i)^2 / (x1_i + x2_i), skipping dimensions where x1_i + x2_i is zero.
    /// </summary>
    public static T SafeChiSquaredSum(Vector<T> x1, Vector<T> x2)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.Subtract(x1, x2).AsTensor();
        var numerator = engine.TensorMultiply(diff, diff);
        var denominator = engine.Add(x1, x2).AsTensor();
        var isZero = engine.TensorEquals(denominator, NumOps.Zero);
        var ones = Tensor<T>.CreateDefault(new[] { x1.Length }, NumOps.One);
        var zeros = Tensor<T>.CreateDefault(new[] { x1.Length }, NumOps.Zero);
        var safeDenominator = engine.TensorWhere(isZero, ones, denominator);
        var terms = engine.TensorWhere(isZero, zeros, engine.TensorDivide(numerator, safeDenominator));
        return engine.TensorSum(terms);
    }

    /// <summary>
    /// The symmetric Gram matrix K[i, j] = k(points[i], points[j]); each pair is evaluated once.
    /// </summary>
    public static Matrix<T> SymmetricGram(IKernelFunction<T> kernel, Vector<T>[] points)
    {
        int n = points.Length;
        if (n == 0)
            return new Matrix<T>(0, 0);

        T zero = NumOps.Zero;
        var engine = AiDotNetEngine.Current;

        // Strict upper triangle U and diagonal d, then K = U + U^T + diag(d).
        var upper = new Matrix<T>(Enumerable.Range(0, n)
            .Select(i => Enumerable.Range(0, n).Select(j => j > i ? kernel.Calculate(points[i], points[j]) : zero)));
        var diagonal = new Vector<T>(points.Select(p => kernel.Calculate(p, p)));

        return engine.MatrixAdd(engine.MatrixAdd(upper, engine.MatrixTranspose(upper)), Matrix<T>.CreateDiagonal(diagonal));
    }

    /// <summary>
    /// (K_1 ⊗ K_2 ⊗ ... ⊗ K_G) v without forming the Kronecker product: v is viewed as a
    /// [n_1, ..., n_G] tensor and each square factor is applied along its own mode.
    /// </summary>
    public static Vector<T> KroneckerMultiply(Matrix<T>[] factors, Vector<T> v)
    {
        var engine = AiDotNetEngine.Current;
        int total = v.Length;
        var current = v.AsTensor();

        for (int g = factors.Length - 1; g >= 0; g--)
        {
            int n = factors[g].Rows;
            int rightSize = 1;
            for (int gg = g + 1; gg < factors.Length; gg++)
                rightSize *= factors[gg].Rows;
            int leftSize = total / (n * rightSize);

            // [left, n, right] -> [left, right, n] @ K^T -> [left, right, n] -> [left, n, right]
            var modeLast = engine.TensorPermute(engine.Reshape(current, new[] { leftSize, n, rightSize }), new[] { 0, 2, 1 });
            var kTransposed = Tensor<T>.FromMatrix(engine.MatrixTranspose(factors[g]));
            var applied = engine.TensorMatMul(modeLast, kTransposed);
            current = engine.Reshape(engine.TensorPermute(applied, new[] { 0, 2, 1 }), new[] { total });
        }

        return current.ToVector();
    }

    /// <summary>
    /// scale * log|K|, with log|K| = 2 * sum(log(diag(L))) from the Cholesky factor L of a symmetric matrix.
    /// Returns false when K is not positive definite.
    /// </summary>
    public static bool TryCholeskyLogDeterminant(Matrix<T> matrix, double scale, out T logDeterminant)
    {
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
        {
            bool success = KernelVectorOps<double>.TryCholeskyLogDeterminant(
                Tensor<T>.FromMatrix(matrix).Cast<double>().ToMatrix(), scale, out double wide);
            logDeterminant = NumOps.FromDouble(wide);
            return success;
        }
        var engine = AiDotNetEngine.Current;
        var (factor, info) = LinalgDispatch<T>.CholeskyEx(Tensor<T>.FromMatrix(matrix));
        if (info.GetFlatIndexValue(0) != 0)
        {
            logDeterminant = NumOps.Zero;
            return false;
        }

        var diagonal = LinalgDispatch<T>.Diagonal(factor);
        logDeterminant = NumOps.Multiply(NumOps.FromDouble(2.0 * scale), engine.TensorSum(engine.TensorLog(diagonal)));
        return true;
    }
}
