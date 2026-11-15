namespace AiDotNet.Metrics;

/// <summary>
/// Shared Gaussian-statistics math for the Frechet distances (FID, FVD), written with engine ops.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class FrechetDistanceHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Sample mean and unbiased covariance (divisor n - 1) of the rows of <paramref name="features"/>.
    /// </summary>
    public static (Vector<T> Mean, Matrix<T> Covariance) ComputeStatistics(Matrix<T> features)
    {
        int numSamples = features.Rows;
        int dim = features.Columns;

        if (numSamples < 2)
        {
            throw new ArgumentException(
                $"Need at least 2 samples to compute covariance, got {numSamples}.",
                nameof(features));
        }

        var engine = AiDotNetEngine.Current;
        var data = Tensor<T>.FromMatrix(features);
        var mean = engine.ReduceMean(data, new[] { 0 }, keepDims: true);
        var centered = engine.TensorSubtract(data, engine.TensorBroadcastTo(mean, new[] { numSamples, dim }));

        // Covariance: (1/(n-1)) * centered^T * centered
        var covariance = engine.TensorMultiplyScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered),
            NumOps.FromDouble(1.0 / (numSamples - 1)));

        return (engine.Reshape(mean, new[] { dim }).ToVector(), covariance.ToMatrix());
    }

    /// <summary>
    /// Frechet distance between N(mean1, cov1) and N(mean2, cov2) (Heusel et al. 2017):
    /// ||mean1 - mean2||^2 + Tr(cov1) + Tr(cov2) - 2 Tr(sqrt(cov1 cov2)).
    /// </summary>
    /// <remarks>
    /// Tr(sqrt(cov1 cov2)) is computed as Tr(sqrt(S cov2 S)) with S = sqrt(cov1): the two matrices
    /// are similar, and S cov2 S is symmetric positive semi-definite, so its square root comes from a
    /// symmetric eigendecomposition (the value scipy.linalg.sqrtm gives in pytorch-fid).
    /// </remarks>
    public static T FrechetDistance(Vector<T> mean1, Matrix<T> cov1, Vector<T> mean2, Matrix<T> cov2)
    {
        var engine = AiDotNetEngine.Current;
        var meanDiff = engine.Subtract(mean1, mean2);
        T meanDiffSq = engine.DotProduct(meanDiff, meanDiff);

        var sigma1 = Tensor<T>.FromMatrix(cov1);
        var sigma2 = Tensor<T>.FromMatrix(cov2);
        // The scale below skips NaN (max(NaN, x) is x), and an all-NaN covariance would then look like a zero
        // one and return the mean term alone. A NaN covariance has no distance: report NaN.
        if (engine.TensorCountNonzero(engine.TensorIsNan(sigma1)) > 0
            || engine.TensorCountNonzero(engine.TensorIsNan(sigma2)) > 0)
            return NumOps.FromDouble(double.NaN);
        // A common covariance scale prevents the sandwich product from underflowing or overflowing.
        T scale = MathHelper.Max(engine.TensorMaxValue(engine.TensorAbs(sigma1)),
            engine.TensorMaxValue(engine.TensorAbs(sigma2)));
        if (!NumOps.GreaterThan(scale, NumOps.Zero)) return meanDiffSq;
        var divisor = Tensor<T>.CreateDefault(sigma1.Shape.ToArray(), scale);
        sigma1 = engine.TensorDivide(sigma1, divisor);
        sigma2 = engine.TensorDivide(sigma2, divisor);
        T traceCov = NumOps.Add(engine.TensorTrace(sigma1), engine.TensorTrace(sigma2));
        var sqrtSigma1 = SymmetricSquareRoot(sigma1);
        var inner = engine.TensorMatMul(engine.TensorMatMul(sqrtSigma1, sigma2), sqrtSigma1);
        var (innerEigenvalues, _) = ScaledEigh(inner);
        T traceSqrtProduct = engine.TensorSum(engine.TensorSqrt(engine.TensorClampMin(innerEigenvalues, NumOps.Zero)));
        T covarianceDistance = NumOps.Subtract(traceCov, NumOps.Multiply(NumOps.FromDouble(2.0), traceSqrtProduct));
        return NumOps.Add(meanDiffSq, NumOps.Multiply(scale, MathHelper.Max(NumOps.Zero, covarianceDistance)));
    }

    /// <summary>
    /// V diag(sqrt(max(lambda, 0))) V^T for a symmetric matrix with eigendecomposition V diag(lambda) V^T.
    /// </summary>
    private static Tensor<T> SymmetricSquareRoot(Tensor<T> matrix)
    {
        var engine = AiDotNetEngine.Current;
        var (eigenvalues, eigenvectors) = ScaledEigh(matrix);
        int n = eigenvalues.Length;
        var sqrtEigenvalues = engine.TensorSqrt(engine.TensorClampMin(eigenvalues, NumOps.Zero));
        var scaledVectors = engine.TensorMultiply(eigenvectors,
            engine.TensorBroadcastTo(engine.Reshape(sqrtEigenvalues, new[] { 1, n }), new[] { n, n }));
        return engine.TensorMatMul(scaledVectors, engine.TensorTranspose(eigenvectors));
    }

    /// <summary>
    /// Symmetric eigendecomposition of the matrix divided by its largest absolute entry, with the
    /// eigenvalues scaled back.
    /// </summary>
    /// <remarks>
    /// The eigensolver's convergence threshold is absolute (1e-12), so a matrix whose entries are all far
    /// below it (covariances of tiny features) would skip every rotation; normalizing first makes the
    /// threshold relative. Numeric types the solver does not support (for example decimal) are
    /// decomposed in double, the solver's widest type, and converted back.
    /// </remarks>
    private static (Tensor<T> Values, Tensor<T> Vectors) ScaledEigh(Tensor<T> matrix)
    {
        var engine = AiDotNetEngine.Current;
        int n = matrix.Shape[0];
        T scale = engine.TensorMaxValue(engine.TensorAbs(matrix));
        if (!NumOps.GreaterThan(scale, NumOps.Zero))
        {
            // The zero matrix: every eigenvalue is 0 and any orthonormal basis is an eigenbasis.
            return (new Tensor<T>(new[] { n }), Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(n)));
        }

        var normalized = engine.TensorDivide(matrix, Tensor<T>.CreateDefault(matrix.Shape.ToArray(), scale));
        Tensor<T> values;
        Tensor<T> vectors;
        if (typeof(T) == typeof(double) || typeof(T) == typeof(float))
        {
            (values, vectors) = LinalgDispatch<T>.Eigh(normalized);
        }
        else
        {
            var (doubleValues, doubleVectors) = LinalgDispatch<double>.Eigh(normalized.Cast<double>());
            values = doubleValues.Cast<T>();
            vectors = doubleVectors.Cast<T>();
        }

        return (engine.TensorMultiplyScalar(values, scale), vectors);
    }

    /// <summary>
    /// Row of FeatureDimension features: the first min(output length, featureDimension) network outputs,
    /// zero-padded when the network returns fewer.
    /// </summary>
    public static Vector<T> FeatureRow(Tensor<T> output, int featureDimension)
    {
        int featureCount = Math.Min(output.Length, featureDimension);
        var row = featureCount > 0
            ? AiDotNetEngine.Current.Reshape(output, new[] { output.Length }).ToVector().GetSubVector(0, featureCount)
            : new Vector<T>(0);
        return featureCount < featureDimension
            ? Vector<T>.Concatenate(row, new Vector<T>(featureDimension - featureCount))
            : row;
    }
}
