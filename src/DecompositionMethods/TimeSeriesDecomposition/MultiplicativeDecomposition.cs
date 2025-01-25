namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class MultiplicativeDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly MultiplicativeDecompositionAlgorithm _algorithm;
    private readonly int _seasonalPeriod;

    public MultiplicativeDecomposition(Vector<T> timeSeries, MultiplicativeDecompositionAlgorithm algorithm = MultiplicativeDecompositionAlgorithm.GeometricMovingAverage, int seasonalPeriod = 12)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        _seasonalPeriod = seasonalPeriod;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case MultiplicativeDecompositionAlgorithm.GeometricMovingAverage:
                DecomposeGeometricMovingAverage();
                break;
            case MultiplicativeDecompositionAlgorithm.MultiplicativeExponentialSmoothing:
                DecomposeMultiplicativeExponentialSmoothing();
                break;
            case MultiplicativeDecompositionAlgorithm.LogTransformedSTL:
                DecomposeLogTransformedSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Multiplicative decomposition algorithm.");
        }
    }

    private void DecomposeGeometricMovingAverage()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> seasonal = new Vector<T>(n, NumOps);
        Vector<T> residual = new Vector<T>(n, NumOps);

        // Calculate trend using geometric moving average
        int halfWindow = _seasonalPeriod / 2;
        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n - 1, i + halfWindow);
            T product = NumOps.One;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                product = NumOps.Multiply(product, TimeSeries[j]);
                count++;
            }

            trend[i] = NumOps.Power(product, NumOps.Divide(NumOps.One, NumOps.FromDouble(count)));
        }

        // Calculate seasonal component
        for (int i = 0; i < n; i++)
        {
            seasonal[i] = NumOps.Divide(TimeSeries[i], trend[i]);
        }

        // Normalize seasonal component
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            T seasonalSum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += _seasonalPeriod)
            {
                seasonalSum = NumOps.Add(seasonalSum, seasonal[j]);
                count++;
            }
            T averageSeasonal = NumOps.Divide(seasonalSum, NumOps.FromDouble(count));
            for (int j = i; j < n; j += _seasonalPeriod)
            {
                seasonal[j] = NumOps.Divide(seasonal[j], averageSeasonal);
            }
        }

        // Calculate residual
        for (int i = 0; i < n; i++)
        {
            residual[i] = NumOps.Divide(TimeSeries[i], NumOps.Multiply(trend[i], seasonal[i]));
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    private void DecomposeMultiplicativeExponentialSmoothing()
    {
        int n = TimeSeries.Length;
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> seasonal = new Vector<T>(n, NumOps);
        Vector<T> residual = new Vector<T>(n, NumOps);

        T alpha = NumOps.FromDouble(0.2); // Smoothing factor for level
        T beta = NumOps.FromDouble(0.1);  // Smoothing factor for trend
        T gamma = NumOps.FromDouble(0.3); // Smoothing factor for seasonal

        T level = TimeSeries[0];
        T trendComponent = NumOps.One;

        // Initialize seasonal factors
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            seasonal[i] = NumOps.Divide(TimeSeries[i], level);
        }

        for (int t = 0; t < n; t++)
        {
            T value = TimeSeries[t];
            T lastLevel = level;
            T lastTrend = trendComponent;
            T lastSeasonal = seasonal[t % _seasonalPeriod];

            // Update level
            level = NumOps.Add(
                NumOps.Multiply(alpha, NumOps.Divide(value, lastSeasonal)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Multiply(lastLevel, lastTrend))
            );

            // Update trend
            trendComponent = NumOps.Add(
                NumOps.Multiply(beta, NumOps.Divide(level, lastLevel)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), lastTrend)
            );

            // Update seasonal
            seasonal[t % _seasonalPeriod] = NumOps.Add(
                NumOps.Multiply(gamma, NumOps.Divide(value, level)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), lastSeasonal)
            );

            // Calculate components
            trend[t] = NumOps.Multiply(level, trendComponent);
            residual[t] = NumOps.Divide(value, NumOps.Multiply(trend[t], seasonal[t % _seasonalPeriod]));
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    private void DecomposeLogTransformedSTL()
    {
        int n = TimeSeries.Length;
        Vector<T> logTimeSeries = new Vector<T>(n, NumOps);

        // Log transform the time series
        for (int i = 0; i < n; i++)
        {
            logTimeSeries[i] = NumOps.Log(TimeSeries[i]);
        }

        // Perform STL decomposition on log-transformed data
        var stlOptions = new STLDecompositionOptions<T>
        {
            SeasonalPeriod = _seasonalPeriod,
            RobustIterations = 1,
            InnerLoopPasses = 2,
            SeasonalDegree = 1,
            TrendDegree = 1,
            SeasonalJump = 1,
            TrendJump = 1,
            SeasonalBandwidth = 0.75,
            TrendBandwidth = 0.75,
            LowPassBandwidth = 0.75
        };

        var stlDecomposition = new STLDecomposition<T>(stlOptions);
        stlDecomposition.Train(new Matrix<T>(logTimeSeries.Length, 1, NumOps), logTimeSeries);

        Vector<T> logTrend = stlDecomposition.GetTrend();
        Vector<T> logSeasonal = stlDecomposition.GetSeasonal();
        Vector<T> logResidual = stlDecomposition.GetResidual();

        // Transform components back to original scale
        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> seasonal = new Vector<T>(n, NumOps);
        Vector<T> residual = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            trend[i] = NumOps.Exp(logTrend[i]);
            seasonal[i] = NumOps.Exp(logSeasonal[i]);
            residual[i] = NumOps.Exp(logResidual[i]);
        }

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }
}