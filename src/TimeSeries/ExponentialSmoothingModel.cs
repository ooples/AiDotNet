namespace AiDotNet.TimeSeries;

public class ExponentialSmoothingModel<T> : TimeSeriesModelBase<T>
{
    private T _alpha; // Smoothing factor for level
    private T _beta;  // Smoothing factor for trend (if applicable)
    private T _gamma; // Smoothing factor for seasonality (if applicable)
    private Vector<T> _initialValues;

    public ExponentialSmoothingModel(ExponentialSmoothingOptions<T> options) : base(options)
    {
        _alpha = NumOps.FromDouble(options.InitialAlpha);
        _beta = options.UseTrend ? NumOps.FromDouble(options.InitialBeta) : NumOps.Zero;
        _gamma = options.UseSeasonal ? NumOps.FromDouble(options.InitialGamma) : NumOps.Zero;
        _initialValues = Vector<T>.Empty();
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Implement Exponential Smoothing training logic
        (_alpha, _beta, _gamma) = EstimateParametersGridSearch(y);
        _initialValues = EstimateInitialValues(y);
    }

    private (T alpha, T beta, T gamma) EstimateParametersGridSearch(Vector<T> y)
    {
        T bestAlpha = NumOps.Zero, bestBeta = NumOps.Zero, bestGamma = NumOps.Zero;
        T bestMSE = NumOps.MaxValue;

        for (double a = 0; a <= 1; a += 0.1)
        {
            for (double b = 0; b <= 1; b += 0.1)
            {
                for (double g = 0; g <= 1; g += 0.1)
                {
                    T alpha = NumOps.FromDouble(a);
                    T beta = NumOps.FromDouble(b);
                    T gamma = NumOps.FromDouble(g);

                    T mse = CalculateMSE(y, alpha, beta, gamma);

                    if (NumOps.LessThan(mse, bestMSE))
                    {
                        bestMSE = mse;
                        bestAlpha = alpha;
                        bestBeta = beta;
                        bestGamma = gamma;
                    }
                }
            }
        }

        return (bestAlpha, bestBeta, bestGamma);
    }

    private T CalculateMSE(Vector<T> y, T alpha, T beta, T gamma)
    {
        Vector<T> predictions = ForecastWithParameters(y, alpha, beta, gamma);
        return StatisticsHelper<T>.CalculateMeanSquaredError(predictions, y);
    }

    private Vector<T> ForecastWithParameters(Vector<T> y, T alpha, T beta, T gamma)
    {
        Vector<T> forecasts = new(y.Length);
        T level = y[0];
        T trend = _options.IncludeTrend ? NumOps.Subtract(y[1], y[0]) : NumOps.Zero;
        Vector<T> seasonalFactors = _options.SeasonalPeriod > 0 ? EstimateInitialSeasonalFactors(y) : Vector<T>.Empty();

        for (int i = 0; i < y.Length; i++)
        {
            T forecast;
            if (_options.SeasonalPeriod > 0)
            {
                forecast = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % _options.SeasonalPeriod]);
            }
            else
            {
                forecast = NumOps.Add(level, trend);
            }

            forecasts[i] = forecast;

            if (i < y.Length - 1)
            {
                T observation = y[i + 1];
                T oldLevel = level;

                // Update level
                level = NumOps.Add(
                    NumOps.Multiply(alpha, NumOps.Divide(observation, _options.SeasonalPeriod > 0 ? seasonalFactors[i % _options.SeasonalPeriod] : NumOps.One)),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, alpha), NumOps.Add(oldLevel, trend))
                );

                // Update trend
                if (_options.IncludeTrend)
                {
                    trend = NumOps.Add(
                        NumOps.Multiply(beta, NumOps.Subtract(level, oldLevel)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, beta), trend)
                    );
                }

                // Update seasonal factors
                if (_options.SeasonalPeriod > 0)
                {
                    int seasonIndex = i % _options.SeasonalPeriod;
                    seasonalFactors[seasonIndex] = NumOps.Add(
                        NumOps.Multiply(gamma, NumOps.Divide(observation, level)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, gamma), seasonalFactors[seasonIndex])
                    );
                }
            }
        }

        return forecasts;
    }

    private Vector<T> EstimateInitialValues(Vector<T> y)
    {
        Vector<T> initialValues = new Vector<T>(_options.SeasonalPeriod > 0 ? _options.SeasonalPeriod + 2 : 2);
        
        // Initial level
        initialValues[0] = y[0];

        // Initial trend
        if (_options.IncludeTrend)
        {
            initialValues[1] = NumOps.Subtract(y[1], y[0]);
        }
        else
        {
            initialValues[1] = NumOps.Zero;
        }

        // Initial seasonal factors
        if (_options.SeasonalPeriod > 0)
        {
            Vector<T> seasonalFactors = EstimateInitialSeasonalFactors(y);
            for (int i = 0; i < _options.SeasonalPeriod; i++)
            {
                initialValues[i + 2] = seasonalFactors[i];
            }
        }

        return initialValues;
    }

    private Vector<T> EstimateInitialSeasonalFactors(Vector<T> y)
    {
        Vector<T> seasonalFactors = new Vector<T>(_options.SeasonalPeriod);
        int seasons = y.Length / _options.SeasonalPeriod;

        for (int i = 0; i < _options.SeasonalPeriod; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < seasons; j++)
            {
                sum = NumOps.Add(sum, y[i + j * _options.SeasonalPeriod]);
            }
            seasonalFactors[i] = NumOps.Divide(sum, NumOps.FromDouble(seasons));
        }

        // Normalize seasonal factors
        T seasonalSum = seasonalFactors.Sum();
        for (int i = 0; i < _options.SeasonalPeriod; i++)
        {
            seasonalFactors[i] = NumOps.Divide(NumOps.Multiply(seasonalFactors[i], NumOps.FromDouble(_options.SeasonalPeriod)), seasonalSum);
        }

        return seasonalFactors;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows);
        T level = _initialValues[0];
        T trend = _options.IncludeTrend ? _initialValues[1] : NumOps.Zero;
        Vector<T> seasonalFactors = _options.SeasonalPeriod > 0 ? new Vector<T>([.. _initialValues.Skip(2)]) : Vector<T>.Empty();

        for (int i = 0; i < predictions.Length; i++)
        {
            T prediction;
            if (_options.SeasonalPeriod > 0)
            {
                prediction = NumOps.Multiply(NumOps.Add(level, trend), seasonalFactors[i % _options.SeasonalPeriod]);
            }
            else
            {
                prediction = NumOps.Add(level, trend);
            }

            predictions[i] = prediction;

            // Update level, trend, and seasonal factors
            T oldLevel = level;
            level = NumOps.Add(NumOps.Multiply(_alpha, prediction), NumOps.Multiply(NumOps.Subtract(NumOps.One, _alpha), NumOps.Add(oldLevel, trend)));

            if (_options.IncludeTrend)
            {
                trend = NumOps.Add(NumOps.Multiply(_beta, NumOps.Subtract(level, oldLevel)), NumOps.Multiply(NumOps.Subtract(NumOps.One, _beta), trend));
            }

            if (_options.SeasonalPeriod > 0)
            {
                int seasonIndex = i % _options.SeasonalPeriod;
                seasonalFactors[seasonIndex] = NumOps.Add(
                    NumOps.Multiply(_gamma, NumOps.Divide(prediction, NumOps.Add(oldLevel, trend))),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, _gamma), seasonalFactors[seasonIndex])
                );
            }
        }

        return predictions;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = [];

        // Calculate MSE
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(predictions, yTest);
        metrics["MSE"] = mse;

        // Calculate RMSE
        T rmse = NumOps.Sqrt(mse);
        metrics["RMSE"] = rmse;

        // Calculate MAPE
        T mape = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(predictions, yTest);
        metrics["MAPE"] = mape;

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(Convert.ToDouble(_alpha));
        writer.Write(Convert.ToDouble(_beta));
        writer.Write(Convert.ToDouble(_gamma));
        writer.Write(_initialValues.Length);

        foreach (var value in _initialValues)
        {
            writer.Write(Convert.ToDouble(value));
        }
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        _alpha = NumOps.FromDouble(reader.ReadDouble());
        _beta = NumOps.FromDouble(reader.ReadDouble());
        _gamma = NumOps.FromDouble(reader.ReadDouble());
        int initialValuesLength = reader.ReadInt32();
        _initialValues = new Vector<T>(initialValuesLength);

        for (int i = 0; i < initialValuesLength; i++)
        {
            _initialValues[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}