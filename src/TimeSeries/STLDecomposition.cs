using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.TimeSeries;

public class STLDecomposition<T> : TimeSeriesModelBase<T>
{
    private STLDecompositionOptions<T> _stlOptions;
    private Vector<T> _trend;
    private Vector<T> _seasonal;
    private Vector<T> _residual;

    public STLDecomposition(STLDecompositionOptions<T>? options = null) 
        : base(options ?? new STLDecompositionOptions<T>())
    {
        _stlOptions = options ?? new STLDecompositionOptions<T>();
        _trend = Vector<T>.Empty();
        _seasonal = Vector<T>.Empty();
        _residual = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (y.Length < _stlOptions.SeasonalPeriod * 2)
        {
            throw new ArgumentException("Time series is too short for the specified seasonal period.");
        }

        int n = y.Length;
        _trend = new Vector<T>(n);
        _seasonal = new Vector<T>(n);
        _residual = new Vector<T>(n);

        switch (_stlOptions.AlgorithmType)
        {
            case STLAlgorithmType.Standard:
                PerformStandardSTL(y);
                break;
            case STLAlgorithmType.Robust:
                PerformRobustSTL(y);
                break;
            case STLAlgorithmType.Fast:
                PerformFastSTL(y);
                break;
            default:
                throw new ArgumentException("Invalid STL algorithm type.");
        }
    }

    private void PerformStandardSTL(Vector<T> y)
    {
        Vector<T> detrended = y;

        // Step 1: Detrending
        detrended = SubtractVectors(y, _trend);

        // Step 2: Cycle-subseries Smoothing
        _seasonal = CycleSubseriesSmoothing(detrended, _stlOptions.SeasonalPeriod, _stlOptions.SeasonalLoessWindow);

        // Step 3: Low-pass Filtering of Smoothed Cycle-subseries
        Vector<T> lowPassSeasonal = LowPassFilter(_seasonal, _stlOptions.LowPassFilterWindowSize);

        // Step 4: Detrending of Smoothed Cycle-subseries
        _seasonal = SubtractVectors(_seasonal, lowPassSeasonal);

        // Step 5: Deseasonalizing
        Vector<T> deseasonalized = SubtractVectors(y, _seasonal);

        // Step 6: Trend Smoothing
        _trend = LoessSmoothing(deseasonalized, _stlOptions.TrendLoessWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);
    }

    private void PerformRobustSTL(Vector<T> y)
    {
        Vector<T> detrended = y;
        Vector<T> robustnessWeights = Vector<T>.CreateDefault(y.Length, NumOps.One);

        for (int iteration = 0; iteration < _stlOptions.RobustIterations; iteration++)
        {
            PerformStandardSTL(y);

            // Apply robustness weights if not the last iteration
            if (iteration < _stlOptions.RobustIterations - 1)
            {
                robustnessWeights = CalculateRobustWeights(_residual);
                y = ApplyRobustnessWeights(y, robustnessWeights);
            }
        }
    }

    private void PerformFastSTL(Vector<T> y)
    {
        int n = y.Length;
        int period = _stlOptions.SeasonalPeriod;
        int trendWindow = _stlOptions.TrendWindowSize;
        int seasonalWindow = _stlOptions.SeasonalLoessWindow;

        // Step 1: Initial Trend Estimation (using moving average)
        _trend = MovingAverage(y, trendWindow);

        // Step 2: Detrending
        Vector<T> detrended = SubtractVectors(y, _trend);

        // Step 3: Initial Seasonal Estimation
        _seasonal = new Vector<T>(n);
        for (int i = 0; i < period; i++)
        {
            T seasonalValue = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += period)
            {
                seasonalValue = NumOps.Add(seasonalValue, detrended[j]);
                count++;
            }
            seasonalValue = NumOps.Divide(seasonalValue, NumOps.FromDouble(count));
            for (int j = i; j < n; j += period)
            {
                _seasonal[j] = seasonalValue;
            }
        }

        // Step 4: Seasonal Smoothing
        _seasonal = SmoothSeasonal(_seasonal, period, seasonalWindow);

        // Step 5: Seasonal Adjustment
        Vector<T> seasonallyAdjusted = SubtractVectors(y, _seasonal);

        // Step 6: Final Trend Estimation
        _trend = MovingAverage(seasonallyAdjusted, trendWindow);

        // Step 7: Calculation of Residuals
        _residual = CalculateResiduals(y, _trend, _seasonal);

        // Step 8: Normalize Seasonal Component
        NormalizeSeasonal();
    }

    private Vector<T> SmoothSeasonal(Vector<T> seasonal, int period, int window)
    {
        Vector<T> smoothed = new Vector<T>(seasonal.Length);
        for (int i = 0; i < period; i++)
        {
            List<T> subseries = new List<T>();
            for (int j = i; j < seasonal.Length; j += period)
            {
                subseries.Add(seasonal[j]);
            }
            Vector<T> smoothedSubseries = MovingAverage(new Vector<T>(subseries), window);
            for (int j = 0; j < smoothedSubseries.Length; j++)
            {
                smoothed[i + j * period] = smoothedSubseries[j];
            }
        }

        return smoothed;
    }

    private void NormalizeSeasonal()
    {
        T seasonalMean = _seasonal.Sum();
        seasonalMean = NumOps.Divide(seasonalMean, NumOps.FromDouble(_seasonal.Length));
        _seasonal = _seasonal.Transform(s => NumOps.Subtract(s, seasonalMean));
        _trend = _trend.Transform(t => NumOps.Add(t, seasonalMean));
    }

    private Vector<T> MovingAverage(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);
        T windowSum = NumOps.Zero;
        int effectiveWindow = Math.Min(windowSize, n);

        // Initialize the first window
        for (int i = 0; i < effectiveWindow; i++)
        {
            windowSum = NumOps.Add(windowSum, data[i]);
        }

        // Calculate moving average
        for (int i = 0; i < n; i++)
        {
            if (i < effectiveWindow / 2 || i >= n - effectiveWindow / 2)
            {
                // Edge case: use available data points
                int start = Math.Max(0, i - effectiveWindow / 2);
                int end = Math.Min(n, i + effectiveWindow / 2 + 1);
                T sum = NumOps.Zero;
                for (int j = start; j < end; j++)
                {
                    sum = NumOps.Add(sum, data[j]);
                }
                result[i] = NumOps.Divide(sum, NumOps.FromDouble(end - start));
            }
            else
            {
                // Regular case: use full window
                result[i] = NumOps.Divide(windowSum, NumOps.FromDouble(effectiveWindow));
                if (i + effectiveWindow / 2 + 1 < n)
                {
                    windowSum = NumOps.Add(windowSum, data[i + effectiveWindow / 2 + 1]);
                }
                if (i - effectiveWindow / 2 >= 0)
                {
                    windowSum = NumOps.Subtract(windowSum, data[i - effectiveWindow / 2]);
                }
            }
        }

        return result;
    }

    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        return new Vector<T>(a.Zip(b, (x, y) => NumOps.Subtract(x, y)));
    }

    private Vector<T> CycleSubseriesSmoothing(Vector<T> data, int period, int loessWindow)
    {
        Vector<T> smoothed = new Vector<T>(data.Length);
        for (int i = 0; i < period; i++)
        {
            List<(T x, T y)> subseries = new List<(T x, T y)>();
            for (int j = i; j < data.Length; j += period)
            {
                subseries.Add((NumOps.FromDouble(j), data[j]));
            }
            Vector<T> smoothedSubseries = LoessSmoothing(subseries, loessWindow);
            for (int j = 0; j < smoothedSubseries.Length; j++)
            {
                smoothed[i + j * period] = smoothedSubseries[j];
            }
        }
        return smoothed;
    }

    private Vector<T> LowPassFilter(Vector<T> data, int windowSize)
    {
        return MovingAverage(MovingAverage(MovingAverage(data, windowSize), windowSize), 3);
    }

    private Vector<T> CalculateResiduals(Vector<T> y, Vector<T> trend, Vector<T> seasonal)
    {
        return new Vector<T>(y.Zip(trend, (a, b) => NumOps.Subtract(a, b))
                 .Zip(seasonal, (a, b) => NumOps.Subtract(a, b)));
    }

    private Vector<T> CalculateRobustWeights(Vector<T> residuals)
    {
        Vector<T> absResiduals = residuals.Transform(r => NumOps.Abs(r));
        T median = absResiduals.Median();
        T threshold = NumOps.Multiply(NumOps.FromDouble(6), median);

        return absResiduals.Transform(r => 
        {
            if (NumOps.LessThan(r, threshold))
            {
                T weight = NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Divide(r, threshold), NumOps.FromDouble(2)));
                return MathHelper.Max(weight, NumOps.FromDouble(_stlOptions.RobustWeightThreshold));
            }
            else
            {
                return NumOps.FromDouble(_stlOptions.RobustWeightThreshold);
            }
        });
    }

    private Vector<T> ApplyRobustnessWeights(Vector<T> y, Vector<T> weights)
    {
        return new Vector<T>(y.Zip(weights, (a, b) => NumOps.Multiply(a, b)));
    }

    private T TriCube(T x)
    {
        T absX = NumOps.Abs(x);
        if (NumOps.GreaterThan(absX, NumOps.One))
        {
            return NumOps.Zero;
        }

        T oneMinusAbsX = NumOps.Subtract(NumOps.One, absX);
        T cube = NumOps.Multiply(NumOps.Multiply(oneMinusAbsX, oneMinusAbsX), oneMinusAbsX);

        return NumOps.Multiply(cube, cube);
    }

    private Vector<T> LoessSmoothing(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T x = NumOps.FromDouble(i);
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(NumOps.FromDouble(j), x));
                if (NumOps.LessThanOrEquals(distance, NumOps.FromDouble(windowSize)))
                {
                    T weight = TriCube(NumOps.Divide(distance, NumOps.FromDouble(windowSize)));
                    weightedPoints.Add((distance, weight, data[j]));
                }
            }

            if (weightedPoints.Count > 0)
            {
                result[i] = WeightedLeastSquares(weightedPoints);
            }
            else
            {
                result[i] = data[i];
            }
        }

        return result;
    }

    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;

        foreach (var point in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, point.weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(point.weight, point.y));
        }

        return NumOps.Divide(sumWeightedY, sumWeights);
    }

    private Vector<T> LoessSmoothing(List<(T x, T y)> data, double span)
    {
        int n = data.Count;
        Vector<T> result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            T x = data[i].x;
            List<(T distance, T weight, T y)> weightedPoints = new List<(T distance, T weight, T y)>();

            for (int j = 0; j < n; j++)
            {
                T distance = NumOps.Abs(NumOps.Subtract(data[j].x, x));
                weightedPoints.Add((distance, NumOps.Zero, data[j].y));
            }

            weightedPoints.Sort((a, b) => 
            {
                if (NumOps.LessThan(a.distance, b.distance))
                    return -1;
                else if (NumOps.GreaterThan(a.distance, b.distance))
                    return 1;
                else
                    return 0;
            });
            int q = (int)(n * span);
            T maxDistance = weightedPoints[q - 1].distance;

            for (int j = 0; j < q; j++)
            {
                T weight = TriCube(NumOps.Divide(weightedPoints[j].distance, maxDistance));
                weightedPoints[j] = (weightedPoints[j].distance, weight, weightedPoints[j].y);
            }

            result[i] = WeightedLeastSquares(weightedPoints.Take(q).ToList(), x);
        }

        return result;
    }

    private T WeightedLeastSquares(List<(T distance, T weight, T y)> weightedPoints, T x)
    {
        T sumWeights = NumOps.Zero;
        T sumWeightedY = NumOps.Zero;
        T sumWeightedX = NumOps.Zero;
        T sumWeightedXY = NumOps.Zero;
        T sumWeightedX2 = NumOps.Zero;

        foreach (var (_, weight, y) in weightedPoints)
        {
            sumWeights = NumOps.Add(sumWeights, weight);
            sumWeightedY = NumOps.Add(sumWeightedY, NumOps.Multiply(weight, y));
            sumWeightedX = NumOps.Add(sumWeightedX, NumOps.Multiply(weight, x));
            sumWeightedXY = NumOps.Add(sumWeightedXY, NumOps.Multiply(NumOps.Multiply(weight, x), y));
            sumWeightedX2 = NumOps.Add(sumWeightedX2, NumOps.Multiply(NumOps.Multiply(weight, x), x));
        }

        T meanX = NumOps.Divide(sumWeightedX, sumWeights);
        T meanY = NumOps.Divide(sumWeightedY, sumWeights);

        T numerator = NumOps.Subtract(sumWeightedXY, NumOps.Multiply(sumWeightedX, meanY));
        T denominator = NumOps.Subtract(sumWeightedX2, NumOps.Multiply(sumWeightedX, meanX));

        T slope = NumOps.Divide(numerator, denominator);
        T intercept = NumOps.Subtract(meanY, NumOps.Multiply(slope, meanX));

        return NumOps.Add(intercept, NumOps.Multiply(slope, x));
    }

    public Vector<T> GetTrend() => _trend ?? throw new InvalidOperationException("Model has not been trained.");
    public Vector<T> GetSeasonal() => _seasonal ?? throw new InvalidOperationException("Model has not been trained.");
    public Vector<T> GetResidual() => _residual ?? throw new InvalidOperationException("Model has not been trained.");

    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_trend == null || _seasonal == null)
            throw new InvalidOperationException("Model has not been trained.");

        int forecastHorizon = input.Rows;
        Vector<T> forecast = new Vector<T>(forecastHorizon);

        // Extend trend using last trend value
        T lastTrendValue = _trend[_trend.Length - 1];

        // Use last full season for seasonal component
        int seasonLength = _stlOptions.SeasonalPeriod;
        Vector<T> lastSeason = new Vector<T>(seasonLength);
        for (int i = 0; i < seasonLength; i++)
        {
            lastSeason[i] = _seasonal[_seasonal.Length - seasonLength + i];
        }

        for (int i = 0; i < forecastHorizon; i++)
        {
            forecast[i] = NumOps.Add(lastTrendValue, lastSeason[i % seasonLength]);
        }

        return forecast;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        if (_trend == null || _seasonal == null || _residual == null)
            throw new InvalidOperationException("Model has not been trained.");

        Vector<T> yPred = Predict(xTest);
    
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Calculate Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, yPred);

        // Calculate Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, yPred);

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write STL options
        writer.Write(_stlOptions.SeasonalPeriod);
        writer.Write(_stlOptions.TrendWindowSize);
        writer.Write(_stlOptions.SeasonalLoessWindow);
        writer.Write(_stlOptions.TrendLoessWindow);
        writer.Write(_stlOptions.LowPassFilterWindowSize);
        writer.Write(_stlOptions.RobustIterations);
        writer.Write(_stlOptions.RobustWeightThreshold);

        // Write decomposition components
        SerializationHelper<T>.SerializeVector(writer, _trend);
        SerializationHelper<T>.SerializeVector(writer, _seasonal);
        SerializationHelper<T>.SerializeVector(writer, _residual);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read STL options
        int seasonalPeriod = reader.ReadInt32();
        int trendWindowSize = reader.ReadInt32();
        int seasonalLoessWindow = reader.ReadInt32();
        int trendLoessWindow = reader.ReadInt32();
        int lowPassFilterWindowSize = reader.ReadInt32();
        int robustIterations = reader.ReadInt32();
        double robustWeightThreshold = reader.ReadDouble();

        // Create new STLDecompositionOptions<T> with the read values
        _stlOptions = new STLDecompositionOptions<T>
        {
            SeasonalPeriod = seasonalPeriod,
            TrendWindowSize = trendWindowSize,
            SeasonalLoessWindow = seasonalLoessWindow,
            TrendLoessWindow = trendLoessWindow,
            LowPassFilterWindowSize = lowPassFilterWindowSize,
            RobustIterations = robustIterations,
            RobustWeightThreshold = robustWeightThreshold
        };

        // Read decomposition components
        _trend = SerializationHelper<T>.DeserializeVector(reader);
        _seasonal = SerializationHelper<T>.DeserializeVector(reader);
        _residual = SerializationHelper<T>.DeserializeVector(reader);
    }
}