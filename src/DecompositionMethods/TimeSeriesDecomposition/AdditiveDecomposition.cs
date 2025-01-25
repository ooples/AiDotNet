namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class AdditiveDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly AdditiveDecompositionAlgorithm _algorithm;

    public AdditiveDecomposition(Vector<T> timeSeries, AdditiveDecompositionAlgorithm algorithm = AdditiveDecompositionAlgorithm.MovingAverage)
        : base(timeSeries)
    {
        _algorithm = algorithm;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case AdditiveDecompositionAlgorithm.MovingAverage:
                DecomposeMovingAverage();
                break;
            case AdditiveDecompositionAlgorithm.ExponentialSmoothing:
                DecomposeExponentialSmoothing();
                break;
            case AdditiveDecompositionAlgorithm.STL:
                DecomposeSTL();
                break;
            default:
                throw new ArgumentException("Unsupported Additive decomposition algorithm.");
        }
    }

    private void DecomposeMovingAverage()
    {
        // Implementation of Moving Average decomposition
        Vector<T> trend = CalculateTrendMovingAverage();
        Vector<T> seasonal = CalculateSeasonalMovingAverage(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    private void DecomposeExponentialSmoothing()
    {
        // Implementation of Exponential Smoothing decomposition
        Vector<T> trend = CalculateTrendExponentialSmoothing();
        Vector<T> seasonal = CalculateSeasonalExponentialSmoothing(trend);
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    private void DecomposeSTL()
    {
        // Implementation of Seasonal and Trend decomposition using Loess (STL)
        (Vector<T> trend, Vector<T> seasonal) = PerformSTLDecomposition();
        Vector<T> residual = CalculateResidual(trend, seasonal);

        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Residual, residual);
    }

    private (Vector<T>, Vector<T>) PerformSTLDecomposition()
    {
        int n = TimeSeries.Length;
        int seasonalPeriod = 12;
        int nInner = 2;
        int nOuter = 1;
        int trendWindow = (int)Math.Ceiling(1.5 * seasonalPeriod / (1 - 1.5 / seasonalPeriod));
        trendWindow = trendWindow % 2 == 0 ? trendWindow + 1 : trendWindow;

        Vector<T> trend = new Vector<T>(n, NumOps);
        Vector<T> seasonal = new Vector<T>(n, NumOps);
        Vector<T> detrended = new Vector<T>(n, NumOps);

        for (int i = 0; i < nOuter; i++)
        {
            for (int j = 0; j < nInner; j++)
            {
                // Step 1: Detrending
                detrended = SubtractVectors(TimeSeries, trend);

                // Step 2: Cycle-subseries Smoothing
                seasonal = CycleSubseriesSmoothing(detrended, seasonalPeriod);

                // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
                Vector<T> lowPassSeasonal = LowPassFilter(seasonal, seasonalPeriod);

                // Step 4: Detrending of Smoothed Cycle-subseries
                seasonal = SubtractVectors(seasonal, lowPassSeasonal);

                // Step 5: Deseasonalizing
                Vector<T> deseasonalized = SubtractVectors(TimeSeries, seasonal);

                // Step 6: Trend Smoothing
                trend = LoessSmoothing(deseasonalized, trendWindow);
            }
        }

        return (trend, seasonal);
    }

    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        int n = a.Length;
        Vector<T> result = new Vector<T>(n, NumOps);
        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.Subtract(a[i], b[i]);
        }

        return result;
    }

    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period)
    {
        int n = data.Length;
        Vector<T> smoothed = new Vector<T>(n, NumOps);

        for (int i = 0; i < period; i++)
        {
            List<(T x, T y)> subseries = new List<(T x, T y)>();
            for (int j = i; j < n; j += period)
            {
                subseries.Add((NumOps.FromDouble(j), data[j]));
            }

            Vector<T> smoothedSubseries = LoessSmoothing(subseries, 0.75);

            int index = 0;
            for (int j = i; j < n; j += period)
            {
                smoothed[j] = smoothedSubseries[index++];
            }
        }

        return smoothed;
    }

    private Vector<T> LowPassFilter(Vector<T> data, int period)
    {
        int n = data.Length;
        Vector<T> filtered = new Vector<T>(n, NumOps);
        int windowSize = period + 1;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize / 2);
            int end = Math.Min(n - 1, i + windowSize / 2);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, data[j]);
                count++;
            }

            filtered[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return filtered;
    }

    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> smoothed = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize / 2);
            int end = Math.Min(n - 1, i + windowSize / 2);
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = start; j <= end; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(NumOps.FromDouble(j), NumOps.FromDouble(i)));
                T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize / 2)));
                weightedPoints.Add((distance, weight, data[j]));
            }

            smoothed[i] = WeightedLeastSquares(weightedPoints);
        }

        return smoothed;
    }

    private Vector<T> LoessSmoothing(List<(T x, T y)> data, double span)
    {
        int n = data.Count;
        Vector<T> smoothed = new Vector<T>(n, NumOps);
        int windowSize = (int)(span * n);

        for (int i = 0; i < n; i++)
        {
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(data[j].x, data[i].x));
                T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize / 2)));
                weightedPoints.Add((distance, weight, data[j].y));
            }

            smoothed[i] = WeightedLeastSquares(weightedPoints);
        }

        return smoothed;
    }

    private T TriCube(T x)
    {
        if (NumOps.GreaterThan(x, NumOps.One))
        {
            return NumOps.Zero;
        }
        T oneMinusX = NumOps.Subtract(NumOps.One, x);

        return NumOps.Multiply(NumOps.Multiply(oneMinusX, oneMinusX), oneMinusX);
    }

    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;

        foreach (var (_, weight, y) in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(weight, y));
        }

        return NumOps.Divide(sumWeightedY, sumWeights);
    }

    private Vector<T> CalculateTrendMovingAverage()
    {
        int windowSize = 7;
        Vector<T> trend = new Vector<T>(TimeSeries.Length, NumOps);

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            int start = Math.Max(0, i - windowSize / 2);
            int end = Math.Min(TimeSeries.Length - 1, i + windowSize / 2);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, TimeSeries[j]);
                count++;
            }

            trend[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return trend;
    }

    private Vector<T> CalculateSeasonalMovingAverage(Vector<T> trend)
    {
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length, NumOps);
        int seasonalPeriod = 12;

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            seasonal[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        // Calculate average seasonal component for each period
        Vector<T> averageSeasonal = new Vector<T>(seasonalPeriod, NumOps);
        for (int i = 0; i < seasonalPeriod; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < TimeSeries.Length; j += seasonalPeriod)
            {
                sum = NumOps.Add(sum, seasonal[j]);
                count++;
            }
            averageSeasonal[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal component
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            seasonal[i] = averageSeasonal[i % seasonalPeriod];
        }

        return seasonal;
    }

    private Vector<T> CalculateTrendExponentialSmoothing()
    {
        T alpha = NumOps.FromDouble(0.2); // Smoothing factor
        Vector<T> trend = new Vector<T>(TimeSeries.Length, NumOps);
        trend[0] = TimeSeries[0];

        for (int i = 1; i < TimeSeries.Length; i++)
        {
            T prevSmoothed = trend[i - 1];
            T observation = TimeSeries[i];
            trend[i] = NumOps.Add(
                NumOps.Multiply(alpha, observation),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), prevSmoothed)
            );
        }

        return trend;
    }

    private Vector<T> CalculateSeasonalExponentialSmoothing(Vector<T> trend)
    {
        T gamma = NumOps.FromDouble(0.3); // Seasonal smoothing factor
        int seasonalPeriod = 12;
        Vector<T> seasonal = new Vector<T>(TimeSeries.Length, NumOps);

        // Initialize seasonal components
        for (int i = 0; i < seasonalPeriod; i++)
        {
            seasonal[i] = NumOps.Subtract(TimeSeries[i], trend[i]);
        }

        for (int i = seasonalPeriod; i < TimeSeries.Length; i++)
        {
            int seasonIndex = i % seasonalPeriod;
            T observation = TimeSeries[i];
            T levelTrend = trend[i];
            T prevSeasonal = seasonal[i - seasonalPeriod];

            seasonal[i] = NumOps.Add(
                NumOps.Multiply(gamma, NumOps.Subtract(observation, levelTrend)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), prevSeasonal)
            );
        }

        return seasonal;
    }
}