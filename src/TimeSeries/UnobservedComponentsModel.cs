namespace AiDotNet.TimeSeries;

public class UnobservedComponentsModel<T> : TimeSeriesModelBase<T>
{
    private readonly UnobservedComponentsOptions<T> _ucOptions;
    private Vector<T> _trend;
    private Vector<T> _previousTrend;
    private Vector<T> _seasonal;
    private Vector<T> _cycle;
    private Vector<T> _irregular;
    private Vector<T> _y;
    private readonly FastFourierTransform<T> _fft;
    private Matrix<T> _stateTransition;
    private Matrix<T> _observationModel;
    private Matrix<T> _processNoise;
    private T _observationNoise;
    private Vector<T> _state;
    private Matrix<T> _stateCovariance;
    private T _convergenceThreshold;
    private List<Vector<T>> _filteredState;
    private List<Matrix<T>> _filteredCovariance;

    public UnobservedComponentsModel(UnobservedComponentsOptions<T>? options = null) 
        : base(options ?? new UnobservedComponentsOptions<T>())
    {
        _ucOptions = options ?? new UnobservedComponentsOptions<T>();
        
        // Initialize model components
        _trend = new Vector<T>(_ucOptions.MaxIterations);
        _previousTrend = new Vector<T>(_ucOptions.MaxIterations);
        _y = new Vector<T>(_ucOptions.MaxIterations);
        _seasonal = new Vector<T>(_ucOptions.MaxIterations);
        _cycle = new Vector<T>(_ucOptions.MaxIterations);
        _irregular = new Vector<T>(_ucOptions.MaxIterations);
        _fft = new FastFourierTransform<T>();
        _filteredState = [];
        _filteredCovariance = [];
        _stateCovariance = Matrix<T>.Empty();
        _state = new Vector<T>(_ucOptions.MaxIterations);
        _stateTransition = Matrix<T>.Empty();
        _observationModel = Matrix<T>.Empty();
        _processNoise = Matrix<T>.Empty();
        _observationNoise = NumOps.Zero;
        _convergenceThreshold = NumOps.Zero;
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        _y = y.Copy();
        int n = _y.Length;

        // Initialize components
        InitializeComponents(_y);
        InitializeKalmanParameters();

        // Kalman filter and smoothing
        for (int iteration = 0; iteration < _ucOptions.MaxIterations; iteration++)
        {
            KalmanFilter(_y);
            KalmanSmoother(_y);

            if (HasConverged())
            {
                break;
            }
        }

        // Optimize parameters if needed
        if (_ucOptions.OptimizeParameters)
        {
            OptimizeParameters(x, _y);
        }
    }

    private void InitializeComponents(Vector<T> y)
    {
        int n = y.Length;

        // Initialize trend using simple moving average
        int windowSize = Math.Min(n, 7); // Use a 7-day window or less if data is shorter
        _trend = MovingAverage(y, windowSize);

        // Initialize seasonal component
        if (_ucOptions.SeasonalPeriod > 1)
        {
            _seasonal = InitializeSeasonal(y, _trend);
        }
        else
        {
            _seasonal = new Vector<T>(n);
        }

        // Initialize cycle component (if applicable)
        if (_ucOptions.IncludeCycle)
        {
            _cycle = InitializeCycle(y, _trend, _seasonal);
        }
        else
        {
            _cycle = new Vector<T>(n);
        }

        // Initialize irregular component
        _irregular = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            _irregular[i] = NumOps.Subtract(y[i], NumOps.Add(_trend[i], NumOps.Add(_seasonal[i], _cycle[i])));
        }
    }

    private Vector<T> MovingAverage(Vector<T> data, int windowSize)
    {
        int n = data.Length;
        Vector<T> result = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            int start = Math.Max(0, i - windowSize + 1);
            int end = i + 1;
            T sum = NumOps.Zero;
            for (int j = start; j < end; j++)
            {
                sum = NumOps.Add(sum, data[j]);
            }

            result[i] = NumOps.Divide(sum, NumOps.FromDouble(end - start));
        }

        return result;
    }

    private Vector<T> InitializeSeasonal(Vector<T> y, Vector<T> trend)
    {
        int n = y.Length;
        int period = _ucOptions.SeasonalPeriod;
        Vector<T> seasonal = new Vector<T>(n);

        // Calculate initial seasonal indices
        Vector<T> seasonalIndices = new Vector<T>(period);
        for (int i = 0; i < period; i++)
        {
            T sum = NumOps.Zero;
            int count = 0;
            for (int j = i; j < n; j += period)
            {
                sum = NumOps.Add(sum, NumOps.Subtract(y[j], trend[j]));
                count++;
            }

            seasonalIndices[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }

        // Normalize seasonal indices
        T seasonalSum = seasonalIndices.Sum();
        T seasonalAdjustment = NumOps.Divide(seasonalSum, NumOps.FromDouble(period));
        for (int i = 0; i < period; i++)
        {
            seasonalIndices[i] = NumOps.Subtract(seasonalIndices[i], seasonalAdjustment);
        }

        // Apply seasonal indices to the full series
        for (int i = 0; i < n; i++)
        {
            seasonal[i] = seasonalIndices[i % period];
        }

        return seasonal;
    }

    private Vector<T> InitializeCycle(Vector<T> y, Vector<T> trend, Vector<T> seasonal)
    {
        int n = y.Length;
        Vector<T> cycle = new Vector<T>(n);

        // Step 1: Calculate residuals after removing trend and seasonal components
        for (int i = 0; i < n; i++)
        {
            cycle[i] = NumOps.Subtract(y[i], NumOps.Add(trend[i], seasonal[i]));
        }

        // Step 2: Apply Hodrick-Prescott filter to separate cycle from noise
        cycle = HodrickPrescottFilter(cycle, _ucOptions.CycleLambda);

        // Step 3: Apply band-pass filter to isolate cycle frequencies
        cycle = BandPassFilter(cycle, _ucOptions.CycleMinPeriod, _ucOptions.CycleMaxPeriod);

        // Step 4: Normalize the cycle component
        T cycleMean = cycle.Average();
        T cycleStd = cycle.StandardDeviation();
        for (int i = 0; i < n; i++)
        {
            cycle[i] = NumOps.Divide(NumOps.Subtract(cycle[i], cycleMean), cycleStd);
        }

        return cycle;
    }

    private Vector<T> HodrickPrescottFilter(Vector<T> data, double lambda)
    {
        int n = data.Length;
        Matrix<T> A = new Matrix<T>(n, n);
        Vector<T> B = new Vector<T>(n);

        // Set up the matrix A
        for (int i = 0; i < n; i++)
        {
            A[i, i] = NumOps.FromDouble(1 + 2 * lambda);
            if (i > 0) A[i, i - 1] = NumOps.FromDouble(-lambda);
            if (i < n - 1) A[i, i + 1] = NumOps.FromDouble(-lambda);
            if (i > 1) A[i, i - 2] = NumOps.FromDouble(lambda);
            if (i < n - 2) A[i, i + 2] = NumOps.FromDouble(lambda);
        }

        // Set up the vector B
        for (int i = 0; i < n; i++)
        {
            B[i] = data[i];
        }

        // Solve the system A * trend = B
        var decomposition = _ucOptions.Decomposition ?? new LuDecomposition<T>(A);
        Vector<T> trend = MatrixSolutionHelper.SolveLinearSystem(B, decomposition);

        return trend;
    }

    private Vector<T> BandPassFilter(Vector<T> data, int minPeriod, int maxPeriod)
    {
        int n = data.Length;
        Vector<T> filtered = new Vector<T>(n);

        // Apply FFT
        Vector<Complex<T>> spectrum = _fft.Forward(data);

        // Apply band-pass filter in frequency domain
        T minFreq = NumOps.FromDouble(1.0 / maxPeriod);
        T maxFreq = NumOps.FromDouble(1.0 / minPeriod);
        for (int i = 0; i < n; i++)
        {
            T freq = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(n));
            if (NumOps.GreaterThanOrEquals(freq, minFreq) && NumOps.LessThanOrEquals(freq, maxFreq))
            {
                // Pass this frequency
                continue;
            }
            else
            {
                // Filter out this frequency
                spectrum[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
        }

        // Apply inverse FFT
        filtered = _fft.Inverse(spectrum);

        return filtered;
    }

    private void InitializeKalmanParameters()
    {
        int stateSize = 3; // trend, seasonal, cycle
        if (!_ucOptions.IncludeCycle) stateSize--;
        if (_ucOptions.SeasonalPeriod <= 1) stateSize--;

        _stateTransition = Matrix<T>.CreateIdentity(stateSize);
        _observationModel = new Matrix<T>(1, stateSize);
        for (int i = 0; i < stateSize; i++)
            _observationModel[0, i] = NumOps.One;

        _processNoise = Matrix<T>.CreateIdentity(stateSize);
        _observationNoise = NumOps.FromDouble(0.1);
        _state = new Vector<T>(stateSize);
        _stateCovariance = Matrix<T>.CreateIdentity(stateSize);
        _convergenceThreshold = NumOps.FromDouble(1e-6);
    }

    private void KalmanFilter(Vector<T> y)
    {
        int n = y.Length;
        var filteredState = new List<Vector<T>>();
        var filteredCovariance = new List<Matrix<T>>();

        for (int t = 0; t < n; t++)
        {
            // Predict
            Vector<T> predictedState = _stateTransition * _state;
            Matrix<T> predictedCovariance = _stateTransition * _stateCovariance * _stateTransition.Transpose() + _processNoise;

            // Update
            T innovation = NumOps.Subtract(y[t], (_observationModel * predictedState)[0]);
            T innovationCovariance = (_observationModel * predictedCovariance * _observationModel.Transpose())[0, 0];
            innovationCovariance = NumOps.Add(innovationCovariance, _observationNoise);

            // Calculate Kalman gain
            Matrix<T> kalmanGainMatrix = predictedCovariance * _observationModel.Transpose();
            Vector<T> kalmanGain = kalmanGainMatrix.GetColumn(0) * NumOps.Divide(NumOps.One, innovationCovariance);

            // Update state
            _state = predictedState + kalmanGain * innovation;

            // Update state covariance
            Matrix<T> temp = Matrix<T>.OuterProduct(kalmanGain, _observationModel.GetRow(0));
            _stateCovariance = predictedCovariance - temp * predictedCovariance;

            filteredState.Add(_state);
            filteredCovariance.Add(_stateCovariance);
        }

        // Store filtered results
        _filteredState = filteredState;
        _filteredCovariance = filteredCovariance;
    }

    private void KalmanSmoother(Vector<T> y)
    {
        int n = y.Length;
        var smoothedState = new List<Vector<T>>();
        var smoothedCovariance = new List<Matrix<T>>();

        Vector<T> nextSmoothedState = _filteredState[n - 1];
        Matrix<T> nextSmoothedCovariance = _filteredCovariance[n - 1];

        smoothedState.Add(nextSmoothedState);
        smoothedCovariance.Add(nextSmoothedCovariance);

        for (int t = n - 2; t >= 0; t--)
        {
            Vector<T> filteredState = _filteredState[t];
            Matrix<T> filteredCovariance = _filteredCovariance[t];

            Vector<T> predictedState = _stateTransition * filteredState;
            Matrix<T> predictedCovariance = _stateTransition * filteredCovariance * _stateTransition.Transpose() + _processNoise;

            Matrix<T> smoother = filteredCovariance * _stateTransition.Transpose() * predictedCovariance.Inverse();
            Vector<T> smoothedStateT = filteredState + smoother * (nextSmoothedState - predictedState);
            Matrix<T> smoothedCovarianceT = filteredCovariance + smoother * (nextSmoothedCovariance - predictedCovariance) * smoother.Transpose();

            smoothedState.Insert(0, smoothedStateT);
            smoothedCovariance.Insert(0, smoothedCovarianceT);

            nextSmoothedState = smoothedStateT;
            nextSmoothedCovariance = smoothedCovarianceT;
        }

        // Update model components with smoothed estimates
        UpdateComponentsFromSmoothedState(smoothedState);
    }

    private void UpdateComponentsFromSmoothedState(List<Vector<T>> smoothedState)
    {
        int n = smoothedState.Count;
        _trend = new Vector<T>(n);
        _seasonal = new Vector<T>(n);
        _cycle = new Vector<T>(n);

        int stateIndex = 0;
        for (int t = 0; t < n; t++)
        {
            _trend[t] = smoothedState[t][stateIndex];
            stateIndex++;

            if (_ucOptions.SeasonalPeriod > 1)
            {
                _seasonal[t] = smoothedState[t][stateIndex];
                stateIndex++;
            }

            if (_ucOptions.IncludeCycle)
            {
                _cycle[t] = smoothedState[t][stateIndex];
                stateIndex++;
            }

            stateIndex = 0; // Reset for next time step
        }

        // Update irregular component
        for (int t = 0; t < n; t++)
        {
            _irregular[t] = NumOps.Subtract(_y[t], NumOps.Add(_trend[t], NumOps.Add(_seasonal[t], _cycle[t])));
        }
    }

    private bool HasConverged()
    {
        T maxDifference = NumOps.Zero;
        int n = _trend.Length;

        for (int i = 0; i < n; i++)
        {
            T difference = NumOps.Abs(NumOps.Subtract(_trend[i], _previousTrend[i]));

            if (NumOps.GreaterThan(difference, maxDifference))
            {
                maxDifference = difference;
            }
        }

        _previousTrend = new Vector<T>(_trend);
        return NumOps.LessThan(maxDifference, _convergenceThreshold);
    }

    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        // Use the user-defined optimizer if provided, otherwise use LBFGSOptimizer as default
        IOptimizer<T> optimizer = _ucOptions.Optimizer ?? new LBFGSOptimizer<T>();

        // Prepare the optimization input data
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        // Run optimization
        OptimizationResult<T> result = optimizer.Optimize(inputData);

        // Update model parameters with optimized values
        UpdateModelParameters(result.BestSolution.Coefficients);
    }

    private void UpdateModelParameters(Vector<T> optimizedParameters)
    {
        int paramIndex = 0;

        // Update trend parameters
        T trendLevel = optimizedParameters[paramIndex++];
        T trendSlope = optimizedParameters[paramIndex++];

        // Update seasonal parameters
        T seasonalVariance = optimizedParameters[paramIndex++];

        // Update cycle parameters (if included)
        T cyclePeriod = NumOps.Zero;
        T cycleVariance = NumOps.Zero;
        if (_ucOptions.IncludeCycle)
        {
            cyclePeriod = optimizedParameters[paramIndex++];
            cycleVariance = optimizedParameters[paramIndex++];
        }

        // Update irregular component variance
        T irregularVariance = optimizedParameters[paramIndex++];

        // Update state transition matrix
        _stateTransition[0, 0] = NumOps.One;
        _stateTransition[0, 1] = NumOps.One;
        _stateTransition[1, 1] = NumOps.One;

        // Update process noise covariance matrix
        _processNoise[0, 0] = trendLevel;
        _processNoise[1, 1] = trendSlope;

        int stateSize = 2; // For trend (level and slope)

        if (_ucOptions.SeasonalPeriod > 1)
        {
            // Update seasonal component
            for (int i = 0; i < _ucOptions.SeasonalPeriod - 1; i++)
            {
                _stateTransition[stateSize + i, stateSize + i] = NumOps.FromDouble(-1);
                for (int j = 0; j < _ucOptions.SeasonalPeriod - 1; j++)
                {
                    _stateTransition[stateSize + i, stateSize + j] = NumOps.FromDouble(-1);
                }
            }
            _processNoise[stateSize, stateSize] = seasonalVariance;
            stateSize += _ucOptions.SeasonalPeriod - 1;
        }

        if (_ucOptions.IncludeCycle)
        {
            // Update cycle component
            T cosCyclePeriod = MathHelper.Cos(NumOps.Divide(NumOps.FromDouble(2 * Math.PI), cyclePeriod));
            T sinCyclePeriod = MathHelper.Sin(NumOps.Divide(NumOps.FromDouble(2 * Math.PI), cyclePeriod));

            _stateTransition[stateSize, stateSize] = cosCyclePeriod;
            _stateTransition[stateSize, stateSize + 1] = sinCyclePeriod;
            _stateTransition[stateSize + 1, stateSize] = NumOps.Negate(sinCyclePeriod);
            _stateTransition[stateSize + 1, stateSize + 1] = cosCyclePeriod;

            _processNoise[stateSize, stateSize] = cycleVariance;
            _processNoise[stateSize + 1, stateSize + 1] = cycleVariance;
        }

        // Update observation noise
        _observationNoise = irregularVariance;

        // Reinitialize state and covariance
        _state = new Vector<T>(_stateTransition.Rows);
        _stateCovariance = Matrix<T>.CreateIdentity(_stateTransition.Rows);

        // Re-run the Kalman filter and smoother here
        // to update the component estimates based on the new parameters
        KalmanFilter(_y);
        KalmanSmoother(_y);
    }

    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> x)
    {
        int timeIndex = Convert.ToInt32(x[0]); // Assume the first column is the time index
        T prediction = NumOps.Add(_trend[timeIndex], _seasonal[timeIndex]);
        prediction = NumOps.Add(prediction, _cycle[timeIndex]);
        prediction = NumOps.Add(prediction, _irregular[timeIndex]);

        return prediction;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            // Mean Absolute Error (MAE)
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),

            // Mean Squared Error (MSE)
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),

            // Root Mean Squared Error (RMSE)
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),

            // R-squared (R2)
            ["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions)
        };

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        writer.Write(_trend.Length);
        for (int i = 0; i < _trend.Length; i++)
        {
            writer.Write(Convert.ToDouble(_trend[i]));
        }

        writer.Write(_seasonal.Length);
        for (int i = 0; i < _seasonal.Length; i++)
        {
            writer.Write(Convert.ToDouble(_seasonal[i]));
        }

        writer.Write(_cycle.Length);
        for (int i = 0; i < _cycle.Length; i++)
        {
            writer.Write(Convert.ToDouble(_cycle[i]));
        }

        writer.Write(_irregular.Length);
        for (int i = 0; i < _irregular.Length; i++)
        {
            writer.Write(Convert.ToDouble(_irregular[i]));
        }

        // Write options
        writer.Write(_ucOptions.MaxIterations);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        int trendLength = reader.ReadInt32();
        _trend = new Vector<T>(trendLength);
        for (int i = 0; i < trendLength; i++)
        {
            _trend[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int seasonalLength = reader.ReadInt32();
        _seasonal = new Vector<T>(seasonalLength);
        for (int i = 0; i < seasonalLength; i++)
        {
            _seasonal[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int cycleLength = reader.ReadInt32();
        _cycle = new Vector<T>(cycleLength);
        for (int i = 0; i < cycleLength; i++)
        {
            _cycle[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int irregularLength = reader.ReadInt32();
        _irregular = new Vector<T>(irregularLength);
        for (int i = 0; i < irregularLength; i++)
        {
            _irregular[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read options
        _ucOptions.MaxIterations = reader.ReadInt32();
    }
}