namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Whole-array engine helpers shared by the Gaussian process models (diagonal updates, finiteness checks).
/// </summary>
internal static class GaussianProcessEngineOps
{
    /// <summary>
    /// Returns <paramref name="matrix"/> + <paramref name="value"/> * I as a new matrix.
    /// </summary>
    internal static Matrix<T> AddToDiagonal<T>(Matrix<T> matrix, T value)
    {
        var engine = AiDotNetEngine.Current;
        var identity = engine.TensorEye<T>(matrix.Rows);
        if (matrix.Columns != matrix.Rows)
            identity = engine.TensorSlice(
                engine.TensorEye<T>(Math.Max(matrix.Rows, matrix.Columns)),
                new[] { 0, 0 },
                new[] { matrix.Rows, matrix.Columns });

        return engine.TensorAdd(Tensor<T>.FromMatrix(matrix), engine.TensorMultiplyScalar(identity, value)).ToMatrix();
    }

    /// <summary>
    /// Returns <paramref name="matrix"/> + diag(<paramref name="diagonal"/>) as a new square matrix.
    /// </summary>
    internal static Matrix<T> AddToDiagonal<T>(Matrix<T> matrix, Vector<T> diagonal)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(Tensor<T>.FromMatrix(matrix), engine.TensorDiag(Tensor<T>.FromVector(diagonal))).ToMatrix();
    }

    /// <summary>
    /// Adds <paramref name="value"/> to every diagonal entry of <paramref name="matrix"/> in place.
    /// </summary>
    internal static void AddToDiagonalInPlace<T>(Matrix<T> matrix, T value)
    {
        var updated = AddToDiagonal(matrix, value);
        updated.AsSpan().CopyTo(matrix.AsWritableSpan());
    }

    /// <summary>
    /// Adds diag(<paramref name="diagonal"/>) to <paramref name="matrix"/> in place.
    /// </summary>
    internal static void AddToDiagonalInPlace<T>(Matrix<T> matrix, Vector<T> diagonal)
    {
        var updated = AddToDiagonal(matrix, diagonal);
        updated.AsSpan().CopyTo(matrix.AsWritableSpan());
    }

    /// <summary>
    /// Main diagonal of a matrix as a vector.
    /// </summary>
    internal static Vector<T> Diagonal<T>(Matrix<T> matrix)
    {
        return AiDotNetEngine.Current.TensorDiagonal(Tensor<T>.FromMatrix(matrix)).ToVector();
    }

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty tensor is finite.
    /// </summary>
    internal static bool AllFinite<T>(Tensor<T> values)
    {
        if (values.Length == 0)
            return true;

        var engine = AiDotNetEngine.Current;
        return engine.TensorMinValue(engine.TensorIsFinite(values)) != AiDotNet.Tensors.Bit.False;
    }

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty vector is finite.
    /// </summary>
    internal static bool AllFinite<T>(Vector<T> values) => AllFinite(Tensor<T>.FromVector(values));

    /// <summary>
    /// True when every element is finite (no NaN and no infinity). An empty matrix is finite.
    /// </summary>
    internal static bool AllFinite<T>(Matrix<T> values) => AllFinite(Tensor<T>.FromMatrix(values));

    /// <summary>
    /// Adds the smallest jitter from 1e-6, 1e-5, ..., 1e-1 to the diagonal of <paramref name="K"/> (in place) for
    /// which the Cholesky factor exists and is well conditioned (max/min diagonal of L below 1e8). Kernels such as
    /// Exponential or Laplacian give matrices where Cholesky succeeds but substitution amplifies errors to NaN; the
    /// diagonal ratio catches them. When no jitter qualifies, 1e-1 is added.
    /// </summary>
    internal static void AddAdaptiveJitter<T>(Matrix<T> K)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double jitterValue = 1e-6;
        const double maxJitter = 1e-1;

        while (jitterValue <= maxJitter)
        {
            if (TryAddJitter(K, jitterValue))
                return;

            jitterValue *= 10;
        }

        AddToDiagonalInPlace(K, numOps.FromDouble(maxJitter));
    }

    /// <summary>
    /// Adds <paramref name="jitterValue"/> to the diagonal of <paramref name="K"/> and keeps it when the
    /// Cholesky factor exists and is well conditioned (max/min diagonal ratio below 1e8); otherwise
    /// removes it again and returns false.
    /// </summary>
    private static bool TryAddJitter<T>(Matrix<T> K, double jitterValue)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var jitter = numOps.FromDouble(jitterValue);
        AddToDiagonalInPlace(K, jitter);

        try
        {
            var chol = new CholeskyDecomposition<T>(K);
            var diagonal = Tensor<T>.FromVector(Diagonal(chol.L));
            bool wellConditioned = diagonal.Length == 0;
            if (!wellConditioned && AllFinite(diagonal))
            {
                T minDiag = engine.TensorMinValue(diagonal);
                T maxDiag = engine.TensorMaxValue(diagonal);
                wellConditioned = numOps.GreaterThan(minDiag, numOps.Zero)
                    && numOps.LessThan(numOps.Divide(maxDiag, minDiag), numOps.FromDouble(1e8));
            }

            if (wellConditioned)
                return true;
        }
        catch (ArgumentException)
        {
            // Cholesky failed - remove this jitter so a larger value can be tried.
        }

        AddToDiagonalInPlace(K, numOps.Negate(jitter));
        return false;
    }

    /// <summary>
    /// Rows of <paramref name="matrix"/> at <paramref name="rows"/>, in the given order.
    /// </summary>
    internal static Matrix<T> SelectRows<T>(Matrix<T> matrix, int[] rows)
    {
        var index = new Tensor<int>(rows, new[] { rows.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(matrix), index, 0).ToMatrix();
    }

    /// <summary>
    /// Elements of <paramref name="vector"/> at <paramref name="indices"/>, in the given order.
    /// </summary>
    internal static Vector<T> Select<T>(Vector<T> vector, int[] indices)
    {
        var index = new Tensor<int>(indices, new[] { indices.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromVector(vector), index, 0).ToVector();
    }

    /// <summary>
    /// Engine Cholesky factor of <paramref name="A"/> when it equals the pivot-floored factor
    /// L_jj = sqrt(max(A_jj - Σ_k L_jk², <paramref name="floor"/>)); null when a pivot needs the floor.
    /// </summary>
    /// <remarks>
    /// When the standard factorization succeeds and no pivot falls below the floor the clamped and standard
    /// factors are identical. On null the caller runs the sequential clamped factorization, because clamping
    /// one pivot changes every later column.
    /// </remarks>
    internal static Tensor<T>? UnclampedCholesky<T>(Matrix<T> A, T floor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        if (A.Rows == 0)
            return new Tensor<T>(new[] { 0, 0 });

        var (factor, info) = LinalgDispatch<T>.CholeskyEx(Tensor<T>.FromMatrix(A), upper: false);
        if (engine.TensorCountNonzero(info) == 0 && AllFinite(factor)
            && !numOps.LessThan(engine.TensorMinValue(engine.TensorSquare(engine.TensorDiagonal(factor))), floor))
        {
            return factor;
        }

        return null;
    }
}
