using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

public class X11Decomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _seasonalPeriod;
    private readonly int _trendCycleMovingAverageWindow;
    private readonly X11AlgorithmType _algorithmType;

    public X11Decomposition(Vector<T> timeSeries, int seasonalPeriod = 12, int trendCycleMovingAverageWindow = 13, X11AlgorithmType algorithmType = X11AlgorithmType.Standard) 
        : base(timeSeries)
    {
        if (seasonalPeriod <= 0)
        {
            throw new ArgumentException("Seasonal period must be a positive integer.", nameof(seasonalPeriod));
        }
        if (trendCycleMovingAverageWindow <= 0 || trendCycleMovingAverageWindow % 2 == 0)
        {
            throw new ArgumentException("Trend-cycle moving average window must be a positive odd integer.", nameof(trendCycleMovingAverageWindow));
        }

        _seasonalPeriod = seasonalPeriod;
        _trendCycleMovingAverageWindow = trendCycleMovingAverageWindow;
        _algorithmType = algorithmType;
        Decompose();
    }

    protected override void Decompose()
    {
        switch (_algorithmType)
        {
            case X11AlgorithmType.Standard:
                DecomposeStandard();
                break;
            case X11AlgorithmType.MultiplicativeAdjustment:
                DecomposeMultiplicative();
                break;
            case X11AlgorithmType.LogAdditiveAdjustment:
                DecomposeLogAdditive();
                break;
            default:
                throw new ArgumentException("Unsupported X11 algorithm type.");
        }
    }

    private void DecomposeStandard()
    {
        // Step 1: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(TimeSeries, _trendCycleMovingAverageWindow);

        // Step 2: Initial seasonal-irregular estimate
        Vector<T> seasonalIrregular = TimeSeries.Subtract(trendCycle);

        // Step 3: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 4: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = TimeSeries.Subtract(seasonalFactors);

        // Step 5: Final trend-cycle estimate
        trendCycle = CenteredMovingAverage(seasonallyAdjusted, _trendCycleMovingAverageWindow);

        // Step 6: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.Subtract(trendCycle);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trendCycle);
        AddComponent(DecompositionComponentType.Seasonal, seasonalFactors);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    private void DecomposeMultiplicative()
    {
        // Step 1: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(TimeSeries, _trendCycleMovingAverageWindow);

        // Step 2: Initial seasonal-irregular ratios
        Vector<T> seasonalIrregular = TimeSeries.ElementwiseDivide(trendCycle);

        // Step 3: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactorsMultiplicative(seasonalIrregular);

        // Step 4: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = TimeSeries.ElementwiseDivide(seasonalFactors);

        // Step 5: Refined trend-cycle estimate
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13); // Using Henderson 13-term moving average

        // Step 6: Refined seasonal-irregular ratios
        seasonalIrregular = TimeSeries.ElementwiseDivide(trendCycle);

        // Step 7: Final seasonal factors
        seasonalFactors = EstimateSeasonalFactorsMultiplicative(seasonalIrregular);

        // Step 8: Final seasonally adjusted series
        seasonallyAdjusted = TimeSeries.ElementwiseDivide(seasonalFactors);

        // Step 9: Final trend-cycle estimate
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 10: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.ElementwiseDivide(trendCycle);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trendCycle);
        AddComponent(DecompositionComponentType.Seasonal, seasonalFactors);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    private Vector<T> EstimateSeasonalFactorsMultiplicative(Vector<T> seasonalIrregular)
    {
        int n = seasonalIrregular.Length;
        Vector<T> seasonalFactors = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            T product = NumOps.One;
            int count = 0;

            for (int j = i % _seasonalPeriod; j < n; j += _seasonalPeriod)
            {
                product = NumOps.Multiply(product, seasonalIrregular[j]);
                count++;
            }

            seasonalFactors[i] = NumOps.Power(product, NumOps.Divide(NumOps.One, NumOps.FromDouble(count)));
        }

        // Normalize seasonal factors
        T totalFactor = NumOps.Zero;
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            totalFactor = NumOps.Add(totalFactor, seasonalFactors[i]);
        }
        T averageFactor = NumOps.Divide(totalFactor, NumOps.FromDouble(_seasonalPeriod));

        for (int i = 0; i < n; i++)
        {
            seasonalFactors[i] = NumOps.Divide(seasonalFactors[i], averageFactor);
        }

        return seasonalFactors;
    }

    private Vector<T> HendersonMovingAverage(Vector<T> data, int terms)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n, NumOps);
        T[] weights = CalculateHendersonWeights(terms);

        int halfTerms = terms / 2;

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            T weightSum = NumOps.Zero;

            for (int j = -halfTerms; j <= halfTerms; j++)
            {
                int index = i + j;
                if (index >= 0 && index < n)
                {
                    T weight = weights[j + halfTerms];
                    sum = NumOps.Add(sum, NumOps.Multiply(data[index], weight));
                    weightSum = NumOps.Add(weightSum, weight);
                }
            }

            result[i] = NumOps.Divide(sum, weightSum);
        }

        return result;
    }

    private T[] CalculateHendersonWeights(int terms)
    {
        int m = (terms - 1) / 2;
        T[] weights = new T[terms];

        for (int i = -m; i <= m; i++)
        {
            T t = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(m + 1.0));
            T tSquared = NumOps.Multiply(t, t);
        
            T factor1 = NumOps.Subtract(NumOps.One, tSquared);
            T factor2 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(5), tSquared));
            T factor3 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(7), tSquared));
            T factor4 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(9), tSquared));
            T factor5 = NumOps.Subtract(NumOps.One, NumOps.Multiply(NumOps.FromDouble(11), tSquared));

            T numerator = NumOps.Multiply(NumOps.FromDouble(315), 
                NumOps.Multiply(factor1, NumOps.Multiply(factor2, NumOps.Multiply(factor3, NumOps.Multiply(factor4, factor5)))));

            weights[i + m] = NumOps.Divide(numerator, NumOps.FromDouble(320.0));
        }

        return weights;
    }

    private void DecomposeLogAdditive()
    {
        // Step 1: Apply logarithmic transformation to the time series
        Vector<T> logTimeSeries = TimeSeries.Transform(x => NumOps.Log(x));

        // Step 2: Initial trend-cycle estimate
        Vector<T> trendCycle = CenteredMovingAverage(logTimeSeries, _trendCycleMovingAverageWindow);

        // Step 3: Initial seasonal-irregular estimate
        Vector<T> seasonalIrregular = logTimeSeries.Subtract(trendCycle);

        // Step 4: Initial seasonal factors
        Vector<T> seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 5: Seasonally adjusted series
        Vector<T> seasonallyAdjusted = logTimeSeries.Subtract(seasonalFactors);

        // Step 6: Refined trend-cycle estimate using Henderson moving average
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 7: Refined seasonal-irregular estimate
        seasonalIrregular = logTimeSeries.Subtract(trendCycle);

        // Step 8: Final seasonal factors
        seasonalFactors = EstimateSeasonalFactors(seasonalIrregular);

        // Step 9: Final seasonally adjusted series
        seasonallyAdjusted = logTimeSeries.Subtract(seasonalFactors);

        // Step 10: Final trend-cycle estimate
        trendCycle = HendersonMovingAverage(seasonallyAdjusted, 13);

        // Step 11: Final irregular component
        Vector<T> irregular = seasonallyAdjusted.Subtract(trendCycle);

        // Step 12: Apply exponential transformation to convert back to original scale
        AddComponent(DecompositionComponentType.Trend, ApplyExp(trendCycle));
        AddComponent(DecompositionComponentType.Seasonal, ApplyExp(seasonalFactors));
        AddComponent(DecompositionComponentType.Irregular, ApplyExp(irregular));

        // Step 13: Ensure multiplicative consistency
        EnsureMultiplicativeConsistency();
    }

    private Vector<T> ApplyExp(Vector<T> vector)
    {
        return vector.Transform(x => NumOps.Exp(x));
    }

    private void EnsureMultiplicativeConsistency()
    {
        Vector<T> trend = (Vector<T>?)GetComponent(DecompositionComponentType.Trend) ?? new Vector<T>(TimeSeries.Length, NumOps);
        Vector<T> seasonal = (Vector<T>?)GetComponent(DecompositionComponentType.Seasonal) ?? new Vector<T>(TimeSeries.Length, NumOps);
        Vector<T> irregular = (Vector<T>?)GetComponent(DecompositionComponentType.Irregular) ?? new Vector<T>(TimeSeries.Length, NumOps);

        Vector<T> reconstructed = new Vector<T>(TimeSeries.Length, NumOps);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            reconstructed[i] = NumOps.Multiply(NumOps.Multiply(trend[i], seasonal[i]), irregular[i]);
        }

        Vector<T> adjustmentFactor = new Vector<T>(TimeSeries.Length, NumOps);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            adjustmentFactor[i] = NumOps.Divide(TimeSeries[i], reconstructed[i]);
        }

        // Distribute the adjustment factor equally among the components
        T cubicRoot = NumOps.FromDouble(1.0 / 3.0);
        Vector<T> componentAdjustment = new Vector<T>(TimeSeries.Length, NumOps);
        for (int i = 0; i < TimeSeries.Length; i++)
        {
            componentAdjustment[i] = NumOps.Power(adjustmentFactor[i], cubicRoot);
        }

        Vector<T> adjustedTrend = new Vector<T>(TimeSeries.Length, NumOps);
        Vector<T> adjustedSeasonal = new Vector<T>(TimeSeries.Length, NumOps);
        Vector<T> adjustedIrregular = new Vector<T>(TimeSeries.Length, NumOps);

        for (int i = 0; i < TimeSeries.Length; i++)
        {
            adjustedTrend[i] = NumOps.Multiply(trend[i], componentAdjustment[i]);
            adjustedSeasonal[i] = NumOps.Multiply(seasonal[i], componentAdjustment[i]);
            adjustedIrregular[i] = NumOps.Multiply(irregular[i], componentAdjustment[i]);
        }

        AddComponent(DecompositionComponentType.Trend, adjustedTrend);
        AddComponent(DecompositionComponentType.Seasonal, adjustedSeasonal);
        AddComponent(DecompositionComponentType.Irregular, adjustedIrregular);
    }

    private Vector<T> CenteredMovingAverage(Vector<T> data, int window)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n, NumOps);

        int halfWindow = window / 2;

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - halfWindow);
            int end = Math.Min(n - 1, i + halfWindow);
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = start; j <= end; j++)
            {
                sum = NumOps.Add(sum, data[j]);
                count++;
            }

            result[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        return result;
    }

    private Vector<T> EstimateSeasonalFactors(Vector<T> seasonalIrregular)
    {
        int n = seasonalIrregular.Length;
        Vector<T> seasonalFactors = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;

            for (int j = i % _seasonalPeriod; j < n; j += _seasonalPeriod)
            {
                sum = NumOps.Add(sum, seasonalIrregular[j]);
                count++;
            }

            seasonalFactors[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal factors
        T totalFactor = NumOps.Zero;
        for (int i = 0; i < _seasonalPeriod; i++)
        {
            totalFactor = NumOps.Add(totalFactor, seasonalFactors[i]);
        }
        T averageFactor = NumOps.Divide(totalFactor, NumOps.FromDouble(_seasonalPeriod));

        for (int i = 0; i < n; i++)
        {
            seasonalFactors[i] = NumOps.Subtract(seasonalFactors[i], averageFactor);
        }

        return seasonalFactors;
    }
}