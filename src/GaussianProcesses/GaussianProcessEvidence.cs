namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Double-precision linear algebra for type-II maximum likelihood (ML-II): choosing a Gaussian process's
/// hyperparameters by maximizing the log marginal likelihood of the training targets.
/// </summary>
/// <remarks>
/// The hyperparameter search runs in double regardless of the model's numeric type: the log determinant
/// and the solves are exactly where float loses the digits a line search needs. The fitted model keeps
/// its own type; only the evidence is computed here.
/// </remarks>
internal static class GaussianProcessEvidence
{
    /// <summary>
    /// Lower Cholesky factor of a symmetric matrix, or null when the matrix is not positive definite.
    /// </summary>
    internal static double[,]? Cholesky(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var lower = new double[n, n];
        for (int j = 0; j < n; j++)
        {
            double pivot = matrix[j, j];
            for (int k = 0; k < j; k++)
                pivot -= lower[j, k] * lower[j, k];
            if (!(pivot > 0) || double.IsInfinity(pivot))
                return null;

            double diagonal = Math.Sqrt(pivot);
            lower[j, j] = diagonal;
            for (int i = j + 1; i < n; i++)
            {
                double sum = matrix[i, j];
                for (int k = 0; k < j; k++)
                    sum -= lower[i, k] * lower[j, k];
                lower[i, j] = sum / diagonal;
            }
        }

        return lower;
    }

    /// <summary>
    /// Solves (L L^T) x = b for a lower Cholesky factor L.
    /// </summary>
    internal static double[] Solve(double[,] lower, double[] rhs)
    {
        int n = rhs.Length;
        var z = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = rhs[i];
            for (int k = 0; k < i; k++)
                sum -= lower[i, k] * z[k];
            z[i] = sum / lower[i, i];
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = z[i];
            for (int k = i + 1; k < n; k++)
                sum -= lower[k, i] * x[k];
            x[i] = sum / lower[i, i];
        }

        return x;
    }

    /// <summary>
    /// The inverse of L L^T for a lower Cholesky factor L.
    /// </summary>
    internal static double[,] Inverse(double[,] lower)
    {
        int n = lower.GetLength(0);
        var inverse = new double[n, n];
        var unit = new double[n];
        for (int column = 0; column < n; column++)
        {
            Array.Clear(unit, 0, n);
            unit[column] = 1.0;
            var solved = Solve(lower, unit);
            for (int row = 0; row < n; row++)
                inverse[row, column] = solved[row];
        }

        return inverse;
    }

    /// <summary>
    /// log |L L^T| for a lower Cholesky factor L.
    /// </summary>
    internal static double LogDeterminant(double[,] lower)
    {
        double sum = 0;
        for (int i = 0; i < lower.GetLength(0); i++)
            sum += Math.Log(lower[i, i]);
        return 2.0 * sum;
    }

    /// <summary>
    /// log N(y | 0, K) summed over independent target columns sharing K = L L^T.
    /// </summary>
    internal static double LogMarginalLikelihood(double[,] lower, IReadOnlyList<double[]> columns)
    {
        int n = lower.GetLength(0);
        double quadratic = 0;
        foreach (var column in columns)
        {
            var alpha = Solve(lower, column);
            for (int i = 0; i < n; i++)
                quadratic += column[i] * alpha[i];
        }

        return -0.5 * quadratic
            - 0.5 * columns.Count * LogDeterminant(lower)
            - 0.5 * n * columns.Count * Math.Log(2.0 * Math.PI);
    }

    /// <summary>
    /// Fits the signal variance s^2 and the noise variance sigma^2 of K = s^2 K_x + (sigma^2 + jitter) I by
    /// maximizing the log marginal likelihood summed over the target columns.
    /// </summary>
    /// <remarks>
    /// Adam on (log s^2, log sigma^2) with the analytic gradient d log p / d theta = 1/2 tr(W dK/dtheta),
    /// W = sum_c a_c a_c^T - C K^-1, a_c = K^-1 y_c: 1/2 s^2 sum_ij W_ij K_x,ji for the signal and
    /// 1/2 sigma^2 tr(W) for the noise. It starts from s^2 = the mean column variance and sigma^2 = a tenth of
    /// that, so scaled targets give proportionally scaled hyperparameters. The best point seen is returned;
    /// a fixed noise is never moved.
    /// </remarks>
    internal static (double SignalVariance, double NoiseVariance) FitSignalAndNoise(
        double[,] inputKernel,
        IReadOnlyList<double[]> columns,
        double? fixedNoise,
        int steps,
        double jitter)
    {
        int n = inputKernel.GetLength(0);
        double averageVariance = 0;
        foreach (var column in columns)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += column[i] / n;
            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += (column[i] - mean) * (column[i] - mean) / Math.Max(n - 1, 1);
            averageVariance += variance / columns.Count;
        }

        double logSignal = Math.Log(Math.Max(averageVariance, 1e-8));
        bool learnNoise = !fixedNoise.HasValue;
        double logNoise = learnNoise ? Math.Log(Math.Max(0.1 * averageVariance, 1e-8)) : 0.0;
        double NoiseOf(double log) => learnNoise ? Math.Exp(log) : fixedNoise ?? 0.0;

        double[,]? FactorOf(double signal, double noise)
        {
            var matrix = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    matrix[i, j] = signal * inputKernel[i, j];
                matrix[i, i] += noise + jitter;
            }

            return Cholesky(matrix);
        }

        double bestSignal = Math.Exp(logSignal);
        double bestNoise = NoiseOf(logNoise);
        double bestEvidence = double.NegativeInfinity;
        double m1Signal = 0, v1Signal = 0, m1Noise = 0, v1Noise = 0;
        const double LearningRate = 0.05, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8;

        for (int step = 0; step <= steps; step++)
        {
            double signal = Math.Exp(logSignal);
            double noise = NoiseOf(logNoise);
            var lower = FactorOf(signal, noise);
            if (lower is null)
                break;

            double evidence = LogMarginalLikelihood(lower, columns);
            if (evidence > bestEvidence)
            {
                bestEvidence = evidence;
                bestSignal = signal;
                bestNoise = noise;
            }

            if (step == steps)
                break;

            var inverse = Inverse(lower);
            var w = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    w[i, j] = -columns.Count * inverse[i, j];
            foreach (var column in columns)
            {
                var alpha = Solve(lower, column);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        w[i, j] += alpha[i] * alpha[j];
            }

            double kernelTrace = 0, trace = 0;
            for (int i = 0; i < n; i++)
            {
                trace += w[i, i];
                for (int j = 0; j < n; j++)
                    kernelTrace += w[i, j] * inputKernel[j, i];
            }

            double gradientSignal = 0.5 * signal * kernelTrace;
            m1Signal = Beta1 * m1Signal + (1 - Beta1) * gradientSignal;
            v1Signal = Beta2 * v1Signal + (1 - Beta2) * gradientSignal * gradientSignal;
            logSignal += LearningRate * (m1Signal / (1 - Math.Pow(Beta1, step + 1)))
                / (Math.Sqrt(v1Signal / (1 - Math.Pow(Beta2, step + 1))) + Epsilon);

            if (learnNoise)
            {
                double gradientNoise = 0.5 * noise * trace;
                m1Noise = Beta1 * m1Noise + (1 - Beta1) * gradientNoise;
                v1Noise = Beta2 * v1Noise + (1 - Beta2) * gradientNoise * gradientNoise;
                logNoise += LearningRate * (m1Noise / (1 - Math.Pow(Beta1, step + 1)))
                    / (Math.Sqrt(v1Noise / (1 - Math.Pow(Beta2, step + 1))) + Epsilon);
            }
        }

        return (bestSignal, bestNoise);
    }

}
