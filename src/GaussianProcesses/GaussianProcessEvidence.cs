namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Linear algebra for type-II maximum likelihood (ML-II): choosing a Gaussian process's
/// hyperparameters by maximizing the log marginal likelihood of the training targets.
/// </summary>
/// <remarks>
/// Everything runs in the model's numeric type through engine and Linalg ops: Cholesky factors,
/// Cholesky solves and log determinants, so the evidence and its gradient carry the same precision
/// as the fitted model.
/// </remarks>
internal static class GaussianProcessEvidence
{
    /// <summary>
    /// Lower Cholesky factor of a symmetric matrix, or null when the matrix is not positive definite
    /// (a pivot that is not positive, or a factor that is not finite).
    /// </summary>
    internal static Tensor<T>? Cholesky<T>(Tensor<T> matrix)
    {
        var (lower, info) = LinalgDispatch<T>.CholeskyEx(matrix, upper: false);
        if (AiDotNetEngine.Current.TensorCountNonzero(info) > 0 || !GaussianProcessEngineOps.AllFinite(lower))
            return null;

        return lower;
    }

    /// <summary>
    /// Solves A X = B for X given the lower Cholesky factor of A; <paramref name="rhs"/> is [n, k].
    /// </summary>
    internal static Tensor<T> Solve<T>(Tensor<T> lower, Tensor<T> rhs)
    {
        return LinalgDispatch<T>.CholeskySolve(rhs, lower, upper: false);
    }

    /// <summary>
    /// A^-1 from the lower Cholesky factor of A.
    /// </summary>
    internal static Tensor<T> Inverse<T>(Tensor<T> lower)
    {
        return LinalgDispatch<T>.CholeskyInverse(lower, upper: false);
    }

    /// <summary>
    /// log |A| = 2 Σ log L_ii from the lower Cholesky factor of A.
    /// </summary>
    internal static T LogDeterminant<T>(Tensor<T> lower)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        return numOps.Multiply(numOps.FromDouble(2.0), engine.TensorSum(engine.TensorLog(engine.TensorDiagonal(lower))));
    }

    /// <summary>
    /// Σ_c log N(y_c | 0, A) over the columns y_c of <paramref name="columns"/> ([n, c]),
    /// given the lower Cholesky factor of A.
    /// </summary>
    internal static T LogMarginalLikelihood<T>(Tensor<T> lower, Tensor<T> columns)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int n = lower.Shape[0];
        int count = columns.Shape[1];
        var alpha = Solve(lower, columns);
        T quadratic = engine.TensorSum(engine.TensorMultiply(columns, alpha));
        T half = numOps.FromDouble(0.5);

        return numOps.Subtract(
            numOps.Subtract(
                numOps.Negate(numOps.Multiply(half, quadratic)),
                numOps.Multiply(numOps.FromDouble(0.5 * count), LogDeterminant(lower))),
            numOps.FromDouble(0.5 * n * count * Math.Log(2.0 * Math.PI)));
    }

    /// <summary>
    /// Fits K = s^2 K_x + (sigma^2 + jitter) I to the columns of <paramref name="columns"/> ([n, c]) by
    /// Adam on log s^2 and log sigma^2, ascending the analytic gradient of the evidence
    /// 1/2 tr((Σ_c a_c a_c^T - c K^-1) dK/dtheta), a_c = K^-1 y_c. The best point seen is returned;
    /// a fixed noise is not moved.
    /// </summary>
    internal static (T SignalVariance, T NoiseVariance) FitSignalAndNoise<T>(
        Tensor<T> inputKernel,
        Tensor<T> columns,
        double? fixedNoise,
        int steps,
        double jitter)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        int n = inputKernel.Shape[0];
        int count = columns.Shape[1];

        // Average sample variance of the columns (denominator n - 1, at least 1)
        var means = engine.ReduceMean(columns, new[] { 0 }, keepDims: true);
        var deviations = engine.TensorSubtract(columns, engine.TensorBroadcastTo(means, new[] { n, count }));
        T averageVariance = numOps.Divide(
            engine.TensorSumOfSquares(deviations), numOps.FromDouble(Math.Max(n - 1, 1) * (double)count));

        T floor = numOps.FromDouble(1e-8);
        T logSignal = numOps.Log(MathHelper.Max(averageVariance, floor));
        bool learnNoise = !fixedNoise.HasValue;
        T fixedNoiseValue = numOps.FromDouble(fixedNoise ?? 0.0);
        T logNoise = learnNoise
            ? numOps.Log(MathHelper.Max(numOps.Multiply(numOps.FromDouble(0.1), averageVariance), floor))
            : numOps.Zero;
        T NoiseOf(T log) => learnNoise ? numOps.Exp(log) : fixedNoiseValue;

        var identity = engine.TensorEye<T>(n);
        T jitterValue = numOps.FromDouble(jitter);
        Tensor<T>? FactorOf(T signal, T noise)
        {
            var matrix = engine.TensorAdd(
                engine.TensorMultiplyScalar(inputKernel, signal),
                engine.TensorMultiplyScalar(identity, numOps.Add(noise, jitterValue)));
            return Cholesky(matrix);
        }

        var kernelTransposed = engine.TensorTranspose(inputKernel);
        T bestSignal = numOps.Exp(logSignal);
        T bestNoise = NoiseOf(logNoise);
        T bestEvidence = numOps.FromDouble(double.NegativeInfinity);
        T m1Signal = numOps.Zero, v1Signal = numOps.Zero, m1Noise = numOps.Zero, v1Noise = numOps.Zero;
        const double LearningRate = 0.05, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8;
        T beta1 = numOps.FromDouble(Beta1), oneMinusBeta1 = numOps.FromDouble(1 - Beta1);
        T beta2 = numOps.FromDouble(Beta2), oneMinusBeta2 = numOps.FromDouble(1 - Beta2);
        T half = numOps.FromDouble(0.5);
        T countValue = numOps.FromDouble(count);

        T AdamStep(T gradient, ref T first, ref T second, int step)
        {
            first = numOps.Add(numOps.Multiply(beta1, first), numOps.Multiply(oneMinusBeta1, gradient));
            second = numOps.Add(numOps.Multiply(beta2, second),
                numOps.Multiply(oneMinusBeta2, numOps.Multiply(gradient, gradient)));
            T firstHat = numOps.Divide(first, numOps.FromDouble(1 - Math.Pow(Beta1, step + 1)));
            T secondHat = numOps.Divide(second, numOps.FromDouble(1 - Math.Pow(Beta2, step + 1)));
            return numOps.Divide(numOps.Multiply(numOps.FromDouble(LearningRate), firstHat),
                numOps.Add(numOps.Sqrt(secondHat), numOps.FromDouble(Epsilon)));
        }

        for (int step = 0; step <= steps; step++)
        {
            T signal = numOps.Exp(logSignal);
            T noise = NoiseOf(logNoise);
            var lower = FactorOf(signal, noise);
            if (lower is null)
                break;

            T evidence = LogMarginalLikelihood(lower, columns);
            if (numOps.GreaterThan(evidence, bestEvidence))
            {
                bestEvidence = evidence;
                bestSignal = signal;
                bestNoise = noise;
            }

            if (step == steps)
                break;

            // W = Σ_c a_c a_c^T - c K^-1
            var alpha = Solve(lower, columns);
            var w = engine.TensorSubtract(
                engine.TensorMatMul(alpha, engine.TensorTranspose(alpha)),
                engine.TensorMultiplyScalar(Inverse(lower), countValue));

            T kernelTrace = engine.TensorSum(engine.TensorMultiply(w, kernelTransposed));
            T trace = engine.TensorTrace(w);

            T gradientSignal = numOps.Multiply(half, numOps.Multiply(signal, kernelTrace));
            logSignal = numOps.Add(logSignal, AdamStep(gradientSignal, ref m1Signal, ref v1Signal, step));

            if (learnNoise)
            {
                T gradientNoise = numOps.Multiply(half, numOps.Multiply(noise, trace));
                logNoise = numOps.Add(logNoise, AdamStep(gradientNoise, ref m1Noise, ref v1Noise, step));
            }
        }

        return (bestSignal, bestNoise);
    }
}
