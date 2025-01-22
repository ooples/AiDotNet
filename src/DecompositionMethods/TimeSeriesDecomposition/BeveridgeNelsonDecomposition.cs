namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class BeveridgeNelsonDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _maxLag;
    private readonly T _differenceThreshold;
    private readonly IMatrixDecomposition<T>? _decomposition;

    public BeveridgeNelsonDecomposition(Vector<T> timeSeries, int maxLag = 10, double differenceThreshold = 1e-6, IMatrixDecomposition<T>? decomposition = null)
        : base(timeSeries)
    {
        _maxLag = maxLag;
        _decomposition = decomposition;
        _differenceThreshold = NumOps.FromDouble(differenceThreshold);
    }

    public void Decompose()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> cycle = new Vector<T>(n, NumOps);

        // Compute first differences
        Vector<T> differences = ComputeDifferences(TimeSeries);

        // Estimate ARIMA model (simplified approach using AR model)
        Vector<T> arCoefficients = FitARModel(differences, _maxLag);

        // Compute long-run multiplier
        T longRunMultiplier = ComputeLongRunMultiplier(arCoefficients);

        // Compute trend and cycle components
        trend[0] = TimeSeries[0];
        for (int t = 1; t < n; t++)
        {
            trend[t] = NumOps.Add(trend[t - 1], NumOps.Multiply(longRunMultiplier, differences[t - 1]));
            cycle[t] = NumOps.Subtract(TimeSeries[t], trend[t]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Cycle, cycle);
    }

    private Vector<T> ComputeDifferences(Vector<T> series)
    {
        int n = series.Length;
        Vector<T> differences = new Vector<T>(n - 1, NumOps);
        for (int i = 0; i < n - 1; i++)
        {
            differences[i] = NumOps.Subtract(series[i + 1], series[i]);
        }

        return differences;
    }

    private Vector<T> FitARModel(Vector<T> series, int maxLag)
    {
        if (series == null || series.Length == 0)
        {
            throw new ArgumentException("Series cannot be null or empty", nameof(series));
        }

        if (maxLag <= 0 || maxLag >= series.Length)
        {
            throw new ArgumentException("Invalid maxLag value", nameof(maxLag));
        }

        // Compute ACF up to maxLag
        Vector<T> acf = ComputeAutocorrelation(series, maxLag);

        // Use AIC for model selection
        int bestLag = SelectBestLagUsingAIC(series, acf, maxLag);

        // Construct Toeplitz matrix and right-hand side vector
        Matrix<T> toeplitz = ConstructToeplitzMatrix(acf.GetSubVector(0, bestLag));
        Vector<T> rhs = acf.GetSubVector(1, bestLag);

        // Solve Yule-Walker equations
        Vector<T> arCoefficients;
        try
        {
            var decomposition = _decomposition ?? new LuDecomposition<T>(toeplitz);
            arCoefficients = MatrixSolutionHelper.SolveLinearSystem(rhs, decomposition);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to solve Yule-Walker equations", ex);
        }

        // Ensure stability of the AR model
        if (!IsStableARModel(arCoefficients))
        {
            throw new InvalidOperationException("Estimated AR model is not stable");
        }

        return arCoefficients;
    }

    private int SelectBestLagUsingAIC(Vector<T> series, Vector<T> acf, int maxLag)
    {
        T minAIC = NumOps.MaxValue;
        int bestLag = 1;

        for (int p = 1; p <= maxLag; p++)
        {
            Matrix<T> toeplitz = ConstructToeplitzMatrix(acf.GetSubVector(0, p));
            Vector<T> rhs = acf.GetSubVector(1, p);

            try
            {
                var decomposition = _decomposition ?? new LuDecomposition<T>(toeplitz);
                Vector<T> arCoefficients = MatrixSolutionHelper.SolveLinearSystem(rhs, decomposition);
                T aic = CalculateAIC(series, arCoefficients, p);

                if (NumOps.LessThan(aic, minAIC))
                {
                    minAIC = aic;
                    bestLag = p;
                }
            }
            catch
            {
                // If solving fails for this lag, skip it
                continue;
            }
        }

        return bestLag;
    }

    private bool IsStableARModel(Vector<T> arCoefficients)
    {
        // Construct the characteristic polynomial
        int p = arCoefficients.Length;
        Vector<Complex<T>> polynomial = new Vector<Complex<T>>(p + 1);
        polynomial[0] = new Complex<T>(NumOps.One, NumOps.Zero);
        for (int i = 0; i < p; i++)
        {
            polynomial[i + 1] = new Complex<T>(NumOps.Negate(arCoefficients[i]), NumOps.Zero);
        }

        // Find the roots of the characteristic polynomial
        Vector<Complex<T>> roots = FindPolynomialRoots(polynomial);

        // Check if all roots lie outside the unit circle
        for (int i = 0; i < roots.Length; i++)
        {
            if (NumOps.LessThanOrEquals(roots[i].Magnitude, NumOps.One))
            {
                return false; // Model is not stable
            }
        }

        return true; // Model is stable
    }

    private Vector<Complex<T>> FindPolynomialRoots(Vector<Complex<T>> coefficients)
    {
        int n = coefficients.Length - 1;
        Matrix<Complex<T>> companion = new Matrix<Complex<T>>(n, n);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Construct the companion matrix
        for (int i = 0; i < n - 1; i++)
        {
            companion[i + 1, i] = new Complex<T>(NumOps.One, NumOps.Zero);
        }
        for (int i = 0; i < n; i++)
        {
            companion[i, n - 1] = complexOps.Divide(complexOps.Negate(coefficients[i]), coefficients[n]);
        }

        // Compute eigenvalues of the companion matrix
        var eigenDecomp = new EigenDecomposition<Complex<T>>(companion);
        return eigenDecomp.EigenValues;
    }

    private Vector<T> ComputeAutocorrelation(Vector<T> series, int maxLag)
    {
        int n = series.Length;
        Vector<T> acf = new Vector<T>(maxLag + 1, NumOps);
        T mean = series.Mean();
        T variance = series.Variance();

        for (int lag = 0; lag <= maxLag; lag++)
        {
            T sum = NumOps.Zero;
            for (int t = lag; t < n; t++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(
                    NumOps.Subtract(series[t], mean),
                    NumOps.Subtract(series[t - lag], mean)));
            }
            acf[lag] = NumOps.Divide(sum, NumOps.Multiply(NumOps.FromDouble(n - lag), variance));
        }

        return acf;
    }

    private T CalculateAIC(Vector<T> series, Vector<T> arCoefficients, int p)
    {
        int n = series.Length;
        T logLikelihood = CalculateLogLikelihood(series, arCoefficients);
        return NumOps.Add(
            NumOps.Multiply(NumOps.FromDouble(-2), logLikelihood),
            NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(p))
        );
    }

    private T CalculateLogLikelihood(Vector<T> series, Vector<T> arCoefficients)
    {
        int n = series.Length;
        int p = arCoefficients.Length;
        T sumSquaredResiduals = NumOps.Zero;

        for (int t = p; t < n; t++)
        {
            T predicted = NumOps.Zero;
            for (int i = 0; i < p; i++)
            {
                predicted = NumOps.Add(predicted, NumOps.Multiply(arCoefficients[i], series[t - i - 1]));
            }
            T residual = NumOps.Subtract(series[t], predicted);
            sumSquaredResiduals = NumOps.Add(sumSquaredResiduals, NumOps.Square(residual));
        }

        T variance = NumOps.Divide(sumSquaredResiduals, NumOps.FromDouble(n - p));
        return NumOps.Multiply(
            NumOps.FromDouble(-0.5 * (n - p)),
            NumOps.Add(NumOps.Log(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), variance)), NumOps.One)
        );
    }

    private Matrix<T> ConstructToeplitzMatrix(Vector<T> acf)
    {
        int size = acf.Length - 1;
        Matrix<T> toeplitz = new Matrix<T>(size, size, NumOps);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                toeplitz[i, j] = acf[Math.Abs(i - j)];
            }
        }

        return toeplitz;
    }

    private T ComputeLongRunMultiplier(Vector<T> arCoefficients)
    {
        T sum = NumOps.One;
        for (int i = 0; i < arCoefficients.Length; i++)
        {
            sum = NumOps.Add(sum, arCoefficients[i]);
        }

        return NumOps.Divide(NumOps.One, sum);
    }
}