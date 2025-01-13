namespace AiDotNet.TimeSeries;

public class TBATSModel<T> : TimeSeriesModelBase<T>
{
    private TBATSModelOptions<T> _tbatsOptions;
    private Vector<T> _level;
    private Vector<T> _trend;
    private List<Vector<T>> _seasonalComponents;
    private Vector<T> _arCoefficients;
    private Vector<T> _maCoefficients;
    private T _boxCoxLambda;

    public TBATSModel(TBATSModelOptions<T>? options = null) : base(options ?? new TBATSModelOptions<T>())
    {
        _tbatsOptions = (TBATSModelOptions<T>)_options;

        _level = new Vector<T>(1, NumOps);
        _trend = new Vector<T>(1, NumOps);
        _seasonalComponents = new List<Vector<T>>();
        foreach (int period in _tbatsOptions.SeasonalPeriods)
        {
            _seasonalComponents.Add(new Vector<T>(period, NumOps));
        }
        _arCoefficients = new Vector<T>(_tbatsOptions.ARMAOrder, NumOps);
        _maCoefficients = new Vector<T>(_tbatsOptions.ARMAOrder, NumOps);
        _boxCoxLambda = NumOps.FromDouble(_tbatsOptions.BoxCoxLambda);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Initialize components
        InitializeComponents(y);

        // Main training loop
        for (int iteration = 0; iteration < _tbatsOptions.MaxIterations; iteration++)
        {
            T oldLogLikelihood = CalculateLogLikelihood(y);

            UpdateComponents(y);
            UpdateARMACoefficients(y);

            T newLogLikelihood = CalculateLogLikelihood(y);

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(newLogLikelihood, oldLogLikelihood)), NumOps.FromDouble(_tbatsOptions.Tolerance)))
            {
                break;
            }
        }
    }

    private T CalculateLogLikelihood(Vector<T> y)
    {
        T logLikelihood = NumOps.Zero;
        Vector<T> predictions = Predict(new Matrix<T>(y.Length, 1, NumOps)); // Create a dummy input matrix

        for (int t = 0; t < y.Length; t++)
        {
            T error = NumOps.Subtract(y[t], predictions[t]);
            T squaredError = NumOps.Multiply(error, error);
        
            // Assuming Gaussian errors, the log-likelihood is proportional to the negative sum of squared errors
            logLikelihood = NumOps.Subtract(logLikelihood, squaredError);
        }

        // Add a penalty term for model complexity
        int totalParameters = 2 + _seasonalComponents.Count + 2 * _tbatsOptions.ARMAOrder;
        T complexityPenalty = NumOps.Multiply(NumOps.FromDouble(totalParameters), NumOps.Log(NumOps.FromDouble(y.Length)));
        logLikelihood = NumOps.Subtract(logLikelihood, complexityPenalty);

        return logLikelihood;
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        Vector<T> predictions = new Vector<T>(input.Rows, NumOps);

        for (int t = 0; t < input.Rows; t++)
        {
            T prediction = _level[_level.Length - 1];
            prediction = NumOps.Add(prediction, _trend[_trend.Length - 1]);

            for (int i = 0; i < _seasonalComponents.Count; i++)
            {
                int period = _tbatsOptions.SeasonalPeriods[i];
                prediction = NumOps.Multiply(prediction, _seasonalComponents[i][t % period]);
            }

            // Apply ARMA effects
            for (int p = 0; p < _tbatsOptions.ARMAOrder; p++)
            {
                if (t - p - 1 >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[p], NumOps.Subtract(predictions[t - p - 1], _level[_level.Length - p - 2])));
                }
            }

            predictions[t] = prediction;
        }

        return predictions;
    }

    private void InitializeComponents(Vector<T> y)
    {
        int n = y.Length;

        // Initialize level using a robust moving median
        int windowSize = Math.Min(14, n); // Use two weeks' worth of data or less if not available
        _level = new Vector<T>(n, NumOps);
        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize + 1);
            int end = i + 1;
            _level[i] = StatisticsHelper<T>.CalculateMedian(y.Slice(start, end));
        }

        // Initialize trend using robust slope estimation
        _trend = new Vector<T>(n, NumOps);
        for (int i = windowSize; i < n; i++)
        {
            Vector<T> x = Vector<T>.Range(0, windowSize);
            Vector<T> yWindow = y.Slice(i - windowSize, i);
            Vector<T> coefficients = RobustLinearRegression(x, yWindow);
            _trend[i] = coefficients[1]; // Slope
        }

        // Extrapolate trend for the first windowSize points
        for (int i = 0; i < windowSize; i++)
        {
            _trend[i] = _trend[windowSize];
        }

        // Initialize seasonal components using STL decomposition with robust fitting
        for (int i = 0; i < _seasonalComponents.Count; i++)
        {
            int period = _tbatsOptions.SeasonalPeriods[i];
            Vector<T> seasonalComponent = InitializeSeasonalComponentRobust(y, period);
            _seasonalComponents[i] = seasonalComponent;
        }

        // Initialize ARMA coefficients using a robust method
        InitializeARMACoefficientsRobust(y);
    }

    private Vector<T> RobustLinearRegression(Vector<T> x, Vector<T> y)
    {
        // Implement Theil-Sen estimator for robust linear regression
        List<T> slopes = new List<T>();
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = i + 1; j < x.Length; j++)
            {
                T slope = NumOps.Divide(NumOps.Subtract(y[j], y[i]), NumOps.Subtract(x[j], x[i]));
                slopes.Add(slope);
            }
        }

        T medianSlope = StatisticsHelper<T>.CalculateMedian(new Vector<T>(slopes.ToArray(), NumOps));
        T intercept = NumOps.Subtract(StatisticsHelper<T>.CalculateMedian(y), 
                                      NumOps.Multiply(medianSlope, StatisticsHelper<T>.CalculateMedian(x)));

        return new Vector<T>(new T[] { intercept, medianSlope }, NumOps);
    }

    private Vector<T> InitializeSeasonalComponentRobust(Vector<T> y, int period)
    {
        int n = y.Length;
        Vector<T> seasonal = new Vector<T>(period, NumOps);
        Vector<T> detrended = new Vector<T>(n, NumOps);

        // Detrend the series
        for (int i = 0; i < n; i++)
        {
            detrended[i] = NumOps.Subtract(y[i], NumOps.Add(_level[i], NumOps.Multiply(_trend[i], NumOps.FromDouble(i))));
        }

        // Calculate seasonal indices using median
        for (int i = 0; i < period; i++)
        {
            List<T> values = new List<T>();
            for (int j = i; j < n; j += period)
            {
                values.Add(detrended[j]);
            }
            seasonal[i] = StatisticsHelper<T>.CalculateMedian(new Vector<T>(values.ToArray(), NumOps));
        }

        // Normalize seasonal component
        T seasonalMedian = StatisticsHelper<T>.CalculateMedian(seasonal);
        for (int i = 0; i < period; i++)
        {
            seasonal[i] = NumOps.Divide(seasonal[i], seasonalMedian);
        }

        return seasonal;
    }

    private void InitializeARMACoefficientsRobust(Vector<T> y)
    {
        int p = _tbatsOptions.ARMAOrder;
        int q = _tbatsOptions.ARMAOrder;

        // Calculate robust autocorrelations
        T[] autocorrelations = CalculateRobustAutocorrelations(y, Math.Max(p, q));

        // Initialize AR coefficients using Yule-Walker method with robust autocorrelations
        Matrix<T> R = new Matrix<T>(p, p, NumOps);
        Vector<T> r = new Vector<T>(p, NumOps);

        for (int i = 0; i < p; i++)
        {
            r[i] = autocorrelations[i + 1];
            for (int j = 0; j < p; j++)
            {
                R[i, j] = autocorrelations[Math.Abs(i - j)];
            }
        }

        _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _tbatsOptions.DecompositionType);

        // Initialize MA coefficients using innovations algorithm with robust autocorrelations
        Vector<T> residuals = CalculateRobustResiduals(y);
        _maCoefficients = new Vector<T>(q, NumOps);
        Vector<T> v = new Vector<T>(q + 1, NumOps);
        v[0] = StatisticsHelper<T>.CalculateMedianAbsoluteDeviation(residuals);

        for (int k = 1; k <= q; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_maCoefficients[j - 1], v[k - j]));
            }
            _maCoefficients[k - 1] = NumOps.Divide(NumOps.Subtract(autocorrelations[k], sum), v[0]);

            v[k] = NumOps.Multiply(NumOps.Subtract(NumOps.One, NumOps.Multiply(_maCoefficients[k - 1], _maCoefficients[k - 1])), v[k - 1]);
        }
    }

    private T[] CalculateRobustAutocorrelations(Vector<T> y, int maxLag)
    {
        T[] autocorrelations = new T[maxLag + 1];
        T median = StatisticsHelper<T>.CalculateMedian(y);
        T mad = StatisticsHelper<T>.CalculateMedianAbsoluteDeviation(y);

        for (int lag = 0; lag <= maxLag; lag++)
        {
            List<T> products = new List<T>();
            int n = y.Length - lag;

            for (int t = 0; t < n; t++)
            {
                T diff1 = NumOps.Divide(NumOps.Subtract(y[t], median), mad);
                T diff2 = NumOps.Divide(NumOps.Subtract(y[t + lag], median), mad);
                products.Add(NumOps.Multiply(diff1, diff2));
            }

            autocorrelations[lag] = StatisticsHelper<T>.CalculateMedian(new Vector<T>(products.ToArray(), NumOps));
        }

        return autocorrelations;
    }

    private Vector<T> CalculateRobustResiduals(Vector<T> y)
    {
        Vector<T> residuals = new Vector<T>(y.Length, NumOps);
        Vector<T> predictions = Predict(new Matrix<T>(y.Length, 1, NumOps)); // Create a dummy input matrix

        for (int t = 0; t < y.Length; t++)
        {
            residuals[t] = NumOps.Subtract(y[t], predictions[t]);
        }

        // Apply Huber's M-estimator to make residuals more robust
        T median = StatisticsHelper<T>.CalculateMedian(residuals);
        T mad = StatisticsHelper<T>.CalculateMedianAbsoluteDeviation(residuals);
        T k = NumOps.Multiply(NumOps.FromDouble(1.345), mad); // Tuning constant for 95% efficiency

        for (int t = 0; t < residuals.Length; t++)
        {
            T scaledResidual = NumOps.Divide(NumOps.Subtract(residuals[t], median), mad);
            if (NumOps.GreaterThan(NumOps.Abs(scaledResidual), k))
            {
                T sign = NumOps.GreaterThan(scaledResidual, NumOps.Zero) ? NumOps.One : NumOps.FromDouble(-1);
                residuals[t] = NumOps.Add(median, NumOps.Multiply(NumOps.Multiply(sign, k), mad));
            }
        }

        return residuals;
    }

    private Vector<T> InitializeSeasonalComponent(Vector<T> y, int period)
    {
        int n = y.Length;
        Vector<T> seasonal = new Vector<T>(period, NumOps);

        // Calculate seasonal indices
        for (int i = 0; i < period; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += period)
            {
                sum = NumOps.Add(sum, y[j]);
                count++;
            }
            seasonal[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal component
        T seasonalMean = StatisticsHelper<T>.CalculateMean(seasonal);
        for (int i = 0; i < period; i++)
        {
            seasonal[i] = NumOps.Divide(seasonal[i], seasonalMean);
        }

        return seasonal;
    }

    private void InitializeARMACoefficients(Vector<T> y)
    {
        int p = _tbatsOptions.ARMAOrder;
        int q = _tbatsOptions.ARMAOrder;

        // Initialize AR coefficients using Yule-Walker method
        T[] autocorrelations = CalculateAutocorrelations(y, Math.Max(p, q));
        Matrix<T> R = new Matrix<T>(p, p, NumOps);
        Vector<T> r = new Vector<T>(p, NumOps);

        for (int i = 0; i < p; i++)
        {
            r[i] = autocorrelations[i + 1];
            for (int j = 0; j < p; j++)
            {
                R[i, j] = autocorrelations[Math.Abs(i - j)];
            }
        }

        _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _tbatsOptions.DecompositionType);

        // Initialize MA coefficients using innovations algorithm
        Vector<T> residuals = CalculateRobustResiduals(y);
        _maCoefficients = new Vector<T>(q, NumOps);
        Vector<T> v = new Vector<T>(q + 1, NumOps);
        v[0] = StatisticsHelper<T>.CalculateMedianAbsoluteDeviation(residuals);

        for (int k = 1; k <= q; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_maCoefficients[j - 1], v[k - j]));
            }
            _maCoefficients[k - 1] = NumOps.Divide(NumOps.Subtract(autocorrelations[k], sum), v[0]);

            v[k] = NumOps.Multiply(NumOps.Subtract(NumOps.One, NumOps.Multiply(_maCoefficients[k - 1], _maCoefficients[k - 1])), v[k - 1]);
        }
    }

    private Vector<T> CalculateResiduals(Vector<T> y)
    {
        Vector<T> residuals = new Vector<T>(y.Length, NumOps);
        Vector<T> predictions = Predict(new Matrix<T>(y.Length, 1, NumOps)); // Create a dummy input matrix

        for (int t = 0; t < y.Length; t++)
        {
            residuals[t] = NumOps.Subtract(y[t], predictions[t]);
        }

        return residuals;
    }

    private void UpdateComponents(Vector<T> y)
    {
        for (int t = 1; t < y.Length; t++)
        {
            T observation = y[t];
            T seasonalFactor = NumOps.One;

            for (int i = 0; i < _seasonalComponents.Count; i++)
            {
                int period = _tbatsOptions.SeasonalPeriods[i];
                seasonalFactor = NumOps.Multiply(seasonalFactor, _seasonalComponents[i][t % period]);
            }

            T newLevel = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.1), NumOps.Divide(observation, seasonalFactor)),
                NumOps.Multiply(NumOps.FromDouble(0.9), NumOps.Add(_level[t - 1], _trend[t - 1]))
            );

            T newTrend = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.01), NumOps.Subtract(newLevel, _level[t - 1])),
                NumOps.Multiply(NumOps.FromDouble(0.99), _trend[t - 1])
            );

            Vector<T> newLevelVector = new Vector<T>(_level.Length + 1);
            Vector<T> newTrendVector = new Vector<T>(_trend.Length + 1);

            for (int i = 0; i < _level.Length; i++)
            {
                newLevelVector[i] = _level[i];
                newTrendVector[i] = _trend[i];
            }

            newLevelVector[_level.Length] = newLevel;
            newTrendVector[_trend.Length] = newTrend;

            _level = newLevelVector;
            _trend = newTrendVector;

            for (int i = 0; i < _seasonalComponents.Count; i++)
            {
                int period = _tbatsOptions.SeasonalPeriods[i];
                T newSeasonal = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(0.1), NumOps.Divide(observation, NumOps.Multiply(newLevel, seasonalFactor))),
                    NumOps.Multiply(NumOps.FromDouble(0.9), _seasonalComponents[i][t % period])
                );
                _seasonalComponents[i][t % period] = newSeasonal;
            }
        }
    }

    private void UpdateARMACoefficients(Vector<T> y)
    {
        int p = _tbatsOptions.ARMAOrder; // AR order
        int q = _tbatsOptions.ARMAOrder; // MA order

        // Calculate autocorrelations
        T[] autocorrelations = CalculateAutocorrelations(y, Math.Max(p, q));

        // Update AR coefficients using Durbin-Levinson algorithm
        Vector<T> arCoefficients = DurbinLevinsonAlgorithm(autocorrelations, p);

        // Update MA coefficients using innovations algorithm
        Vector<T> maCoefficients = InnovationsAlgorithm(autocorrelations, q);

        // Update the model's coefficients
        _arCoefficients = arCoefficients;
        _maCoefficients = maCoefficients;
    }

    private Vector<T> DurbinLevinsonAlgorithm(T[] autocorrelations, int p)
    {
        Vector<T> phi = new Vector<T>(p, NumOps);
        Vector<T> prevPhi = new Vector<T>(p, NumOps);
        T v = autocorrelations[0];

        for (int k = 1; k <= p; k++)
        {
            T alpha = autocorrelations[k];
            for (int j = 1; j < k; j++)
            {
                alpha = NumOps.Subtract(alpha, NumOps.Multiply(prevPhi[j - 1], autocorrelations[k - j]));
            }
            alpha = NumOps.Divide(alpha, v);

            phi[k - 1] = alpha;
            for (int j = 1; j < k; j++)
            {
                phi[j - 1] = NumOps.Subtract(prevPhi[j - 1], NumOps.Multiply(alpha, prevPhi[k - j - 1]));
            }

            v = NumOps.Multiply(v, NumOps.Subtract(NumOps.One, NumOps.Multiply(alpha, alpha)));

            // Copy phi to prevPhi for next iteration
            for (int j = 0; j < k; j++)
            {
                prevPhi[j] = phi[j];
            }
        }

        return phi;
    }

    private Vector<T> InnovationsAlgorithm(T[] autocorrelations, int q)
    {
        Vector<T> theta = new Vector<T>(q, NumOps);
        Vector<T> v = new Vector<T>(q + 1, NumOps);
        v[0] = autocorrelations[0];

        for (int k = 1; k <= q; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(theta[j - 1], v[k - j]));
            }
            theta[k - 1] = NumOps.Divide(NumOps.Subtract(autocorrelations[k], sum), v[0]);

            v[k] = NumOps.Multiply(
                NumOps.Subtract(NumOps.One, NumOps.Multiply(theta[k - 1], theta[k - 1])),
                v[k - 1]
            );
        }

        return theta;
    }

    private T[] CalculateAutocorrelations(Vector<T> y, int maxLag)
    {
        T[] autocorrelations = new T[maxLag + 1];
        T mean = StatisticsHelper<T>.CalculateMean(y);
        T variance = StatisticsHelper<T>.CalculateVariance(y);

        for (int lag = 0; lag <= maxLag; lag++)
        {
            T sum = NumOps.Zero;
            int n = y.Length - lag;

            for (int t = 0; t < n; t++)
            {
                T diff1 = NumOps.Subtract(y[t], mean);
                T diff2 = NumOps.Subtract(y[t + lag], mean);
                sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
            }

            autocorrelations[lag] = NumOps.Divide(sum, NumOps.Multiply(NumOps.FromDouble(n), variance));
        }

        return autocorrelations;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),
            ["MAPE"] = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions)
        };

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize TBATSModel specific data
        writer.Write(_level.Length);
        for (int i = 0; i < _level.Length; i++)
            writer.Write(Convert.ToDouble(_level[i]));

        writer.Write(_trend.Length);
        for (int i = 0; i < _trend.Length; i++)
            writer.Write(Convert.ToDouble(_trend[i]));

        writer.Write(_seasonalComponents.Count);
        foreach (var component in _seasonalComponents)
        {
            writer.Write(component.Length);
            for (int i = 0; i < component.Length; i++)
                writer.Write(Convert.ToDouble(component[i]));
        }

        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_arCoefficients[i]));

        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_maCoefficients[i]));

        writer.Write(Convert.ToDouble(_boxCoxLambda));

        // Serialize TBATSModelOptions
        writer.Write(JsonConvert.SerializeObject(_tbatsOptions));
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize TBATSModel specific data
        int levelLength = reader.ReadInt32();
        _level = new Vector<T>(levelLength, NumOps);
        for (int i = 0; i < levelLength; i++)
            _level[i] = NumOps.FromDouble(reader.ReadDouble());

        int trendLength = reader.ReadInt32();
        _trend = new Vector<T>(trendLength, NumOps);
        for (int i = 0; i < trendLength; i++)
            _trend[i] = NumOps.FromDouble(reader.ReadDouble());

        int seasonalComponentsCount = reader.ReadInt32();
        _seasonalComponents = new List<Vector<T>>();
        for (int j = 0; j < seasonalComponentsCount; j++)
        {
            int componentLength = reader.ReadInt32();
            Vector<T> component = new Vector<T>(componentLength, NumOps);
            for (int i = 0; i < componentLength; i++)
                component[i] = NumOps.FromDouble(reader.ReadDouble());
            _seasonalComponents.Add(component);
        }

        int arCoefficientsLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arCoefficientsLength, NumOps);
        for (int i = 0; i < arCoefficientsLength; i++)
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int maCoefficientsLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maCoefficientsLength, NumOps);
        for (int i = 0; i < maCoefficientsLength; i++)
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        _boxCoxLambda = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize TBATSModelOptions
        string optionsJson = reader.ReadString();
        _tbatsOptions = JsonConvert.DeserializeObject<TBATSModelOptions<T>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize TBATS model options.");
    }
}