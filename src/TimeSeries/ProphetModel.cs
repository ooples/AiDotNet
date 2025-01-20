namespace AiDotNet.TimeSeries;

public class ProphetModel<T> : TimeSeriesModelBase<T>
{
    private ProphetOptions<T> _prophetOptions;
    private T _trend;
    private Vector<T> _seasonalComponents;
    private Vector<T> _holidayComponents;
    private T _changepoint;
    private Vector<T> _regressors;

    public ProphetModel(ProphetOptions<T>? options = null) 
        : base(options ?? new ProphetOptions<T>())
    {
        _prophetOptions = options ?? new ProphetOptions<T>();

        // Initialize model components
        _trend = NumOps.FromDouble(_prophetOptions.InitialTrendValue);
        _seasonalComponents = new Vector<T>(_prophetOptions.SeasonalPeriods.Sum());
        _holidayComponents = new Vector<T>(_prophetOptions.Holidays.Count);
        _changepoint = NumOps.FromDouble(_prophetOptions.InitialChangepointValue);
        _regressors = new Vector<T>(_prophetOptions.RegressorCount);
    }

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        Matrix<T> states = new Matrix<T>(n, GetStateSize());

        // Initialize components
        InitializeComponents(x, y);

        // Perform optimization (e.g., using L-BFGS or Stan)
        OptimizeParameters(x, y);

        // Store final state
        states.SetRow(n - 1, GetCurrentState());
    }

    private void InitializeComponents(Matrix<T> x, Vector<T> y)
    {
        // Initialize trend
        _trend = EstimateInitialTrend(y);

        // Initialize seasonal components
        InitializeSeasonalComponents(x, y);

        // Initialize holiday components
        InitializeHolidayComponents(x, y);

        // Initialize changepoint
        _changepoint = EstimateInitialChangepoint(y);

        // Initialize regressors
        InitializeRegressors(x, y);
    }

    private T EstimateInitialTrend(Vector<T> y)
    {
        // Simple linear regression on the first few points
        int n = Math.Min(y.Length, 10);
        Vector<T> x = Vector<T>.CreateDefault(n, NumOps.One);
        for (int i = 0; i < n; i++)
        {
            x[i] = NumOps.FromDouble(i);
        }

        return SimpleLinearRegression(x, y.Subvector(0, n));
    }

    private void InitializeSeasonalComponents(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int index = 0;

        foreach (int period in _prophetOptions.SeasonalPeriods)
        {
            int numHarmonics = Math.Min(period / 2, 10); // Use up to 10 harmonics or period/2, whichever is smaller

            for (int h = 1; h <= numHarmonics; h++)
            {
                Vector<T> sinComponent = new Vector<T>(n);
                Vector<T> cosComponent = new Vector<T>(n);

                for (int i = 0; i < n; i++)
                {
                    T t = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(period));
                    T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * h), t);
                    sinComponent[i] = MathHelper.Sin(angle);
                    cosComponent[i] = MathHelper.Cos(angle);
                }

                // Perform simple linear regression to get initial estimates
                T sinCoefficient = SimpleLinearRegression(sinComponent, y);
                T cosCoefficient = SimpleLinearRegression(cosComponent, y);

                _seasonalComponents[index++] = sinCoefficient;
                _seasonalComponents[index++] = cosCoefficient;
            }
        }
    }

    private T SimpleLinearRegression(Vector<T> x, Vector<T> y)
    {
        T sumX = x.Sum();
        T sumY = y.Sum();
        T sumXY = x.DotProduct(y);
        T sumXSquared = x.DotProduct(x);
        int n = x.Length;

        T numerator = NumOps.Subtract(
            NumOps.Multiply(NumOps.FromDouble(n), sumXY),
            NumOps.Multiply(sumX, sumY)
        );
        T denominator = NumOps.Subtract(
            NumOps.Multiply(NumOps.FromDouble(n), sumXSquared),
            NumOps.Multiply(sumX, sumX)
        );

        return NumOps.Divide(numerator, denominator);
    }

    private void InitializeHolidayComponents(Matrix<T> x, Vector<T> y)
    {
        // Initialize holiday components to zero
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = NumOps.Zero;
        }
    }

    private T EstimateInitialChangepoint(Vector<T> y)
    {
        if (y == null || y.Length < 2)
        {
            throw new ArgumentException("Input vector must have at least two elements.", nameof(y));
        }

        // Calculate first differences
        Vector<T> diffs = new Vector<T>(y.Length - 1);
        for (int i = 1; i < y.Length; i++)
        {
            diffs[i - 1] = NumOps.Subtract(y[i], y[i - 1]);
        }

        // Remove zero differences to avoid issues with median calculation
        List<T> nonZeroDiffs = new List<T>();
        for (int i = 0; i < diffs.Length; i++)
        {
            if (!NumOps.Equals(diffs[i], NumOps.Zero))
            {
                nonZeroDiffs.Add(diffs[i]);
            }
        }

        if (nonZeroDiffs.Count == 0)
        {
            // If all differences are zero, return zero as the changepoint
            return NumOps.Zero;
        }

        // Calculate median of non-zero differences
        nonZeroDiffs.Sort();
        int middleIndex = nonZeroDiffs.Count / 2;

        if (nonZeroDiffs.Count % 2 == 0)
        {
            // Even number of elements, average the two middle values
            T middle1 = nonZeroDiffs[middleIndex - 1];
            T middle2 = nonZeroDiffs[middleIndex];
            return NumOps.Divide(NumOps.Add(middle1, middle2), NumOps.FromDouble(2.0));
        }
        else
        {
            // Odd number of elements, return the middle value
            return nonZeroDiffs[middleIndex];
        }
    }

    private void InitializeRegressors(Matrix<T> x, Vector<T> y)
    {
        // Initialize regressors using OLS
        if (_prophetOptions.RegressorCount > 0)
        {
            Matrix<T> regressorMatrix = x.Submatrix(0, x.Columns - _prophetOptions.RegressorCount, x.Rows, _prophetOptions.RegressorCount);
            _regressors = SimpleMultipleRegression(regressorMatrix, y);
        }
    }

    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Initialize parameters
        Vector<T> initialParameters = new Vector<T>(p + 2); // +2 for trend and changepoint
        for (int i = 0; i < p; i++)
        {
            initialParameters[i] = _regressors[i];
        }
        initialParameters[p] = NumOps.FromDouble(_prophetOptions.InitialTrendValue);
        initialParameters[p + 1] = NumOps.FromDouble(_prophetOptions.InitialChangepointValue);

        // Use the user-defined optimizer if provided, otherwise use LFGSOptimizer as default
        IOptimizer<T> optimizer = _prophetOptions.Optimizer ?? new LBFGSOptimizer<T>();

        // Prepare the optimization input data
        var inputData = new OptimizationInputData<T>
        {
            XTrain = x,
            YTrain = y
        };

        // Run optimization
        OptimizationResult<T> result = optimizer.Optimize(inputData);

        // Update model parameters with optimized values
        Vector<T> optimizedParameters = result.BestSolution.Coefficients;
        for (int i = 0; i < p; i++)
        {
            _regressors[i] = optimizedParameters[i];
        }
        _trend = optimizedParameters[p];
        _changepoint = optimizedParameters[p + 1];
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
        T prediction = _trend;
        prediction = NumOps.Add(prediction, GetSeasonalComponent(x));
        prediction = NumOps.Add(prediction, GetHolidayComponent(x));
        prediction = NumOps.Add(prediction, GetChangepointEffect(x));
        prediction = NumOps.Add(prediction, GetRegressorEffect(x));

        return prediction;
    }

    private T GetSeasonalComponent(Vector<T> x)
    {
        T seasonalComponent = NumOps.Zero;
        int timeIndex = 0; // Assume the time index is the first element of x

        foreach (var period in _prophetOptions.SeasonalPeriods)
        {
            T t = NumOps.Divide(NumOps.FromDouble(timeIndex), NumOps.FromDouble(period));
            for (int j = 0; j < _prophetOptions.FourierOrder; j++)
            {
                int idx = j * 2;
                T cos_t = MathHelper.Cos(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * (j + 1)), t));
                T sin_t = MathHelper.Sin(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * (j + 1)), t));
                seasonalComponent = NumOps.Add(seasonalComponent, NumOps.Multiply(_seasonalComponents[idx], cos_t));
                seasonalComponent = NumOps.Add(seasonalComponent, NumOps.Multiply(_seasonalComponents[idx + 1], sin_t));
            }
        }

        return seasonalComponent;
    }

    private T GetHolidayComponent(Vector<T> x)
    {
        T holidayComponent = NumOps.Zero;
        DateTime currentDate = DateTime.FromOADate(Convert.ToDouble(x[0])); // Assume the date is the first element of x

        for (int i = 0; i < _prophetOptions.Holidays.Count; i++)
        {
            if (currentDate.Date == _prophetOptions.Holidays[i].Date)
            {
                holidayComponent = NumOps.Add(holidayComponent, _holidayComponents[i]);
                break; // Assume only one holiday per day
            }
        }

        return holidayComponent;
    }

    private T GetChangepointEffect(Vector<T> x)
    {
        T changepointEffect = NumOps.Zero;
        T t = x[0]; // Assume the time is the first element of x

        for (int i = 0; i < _prophetOptions.Changepoints.Count; i++)
        {
            if (NumOps.GreaterThan(t, _prophetOptions.Changepoints[i]))
            {
                T delta = NumOps.Subtract(t, _prophetOptions.Changepoints[i]);
                changepointEffect = NumOps.Add(changepointEffect, NumOps.Multiply(_changepoint, delta));
            }
        }

        return changepointEffect;
    }

    private T GetRegressorEffect(Vector<T> x)
    {
        T regressorEffect = NumOps.Zero;

        for (int i = 0; i < _regressors.Length; i++)
        {
            // Assume regressors start from the second element of x
            regressorEffect = NumOps.Add(regressorEffect, NumOps.Multiply(_regressors[i], x[i + 1]));
        }

        return regressorEffect;
    }

    private int GetStateSize()
    {
        return 1 + _seasonalComponents.Length + _holidayComponents.Length + 1 + _regressors.Length;
    }

    private Vector<T> GetCurrentState()
    {
        int stateSize = GetStateSize();
        Vector<T> currentState = new Vector<T>(stateSize);
        int index = 0;

        currentState[index++] = _trend;
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            currentState[index++] = _seasonalComponents[i];
        }
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            currentState[index++] = _holidayComponents[i];
        }
        currentState[index++] = _changepoint;
        for (int i = 0; i < _regressors.Length; i++)
        {
            currentState[index++] = _regressors[i];
        }

        return currentState;
    }

    private Vector<T> SimpleMultipleRegression(Matrix<T> x, Vector<T> y)
    {
        // Add a column of ones to X for the intercept term
        Matrix<T> xWithIntercept = new Matrix<T>(x.Rows, x.Columns + 1);
        for (int i = 0; i < x.Rows; i++)
        {
            xWithIntercept[i, 0] = NumOps.One;
            for (int j = 0; j < x.Columns; j++)
            {
                xWithIntercept[i, j + 1] = x[i, j];
            }
        }

        // Calculate (X^T * X)
        Matrix<T> xTx = xWithIntercept.Transpose().Multiply(xWithIntercept);

        // Calculate (X^T * y)
        Vector<T> xTy = xWithIntercept.Transpose().Multiply(y);

        // Solve the normal equations: (X^T * X) * beta = (X^T * y)
        Vector<T> beta;
        try
        {
            // Try Cholesky decomposition first (faster and more stable for well-conditioned matrices)
            var cholesky = new CholeskyDecomposition<T>(xTx);
            beta = cholesky.Solve(xTy);
        }
        catch (Exception)
        {
            // If Cholesky fails, fall back to SVD (more robust but slower)
            var svd = new SvdDecomposition<T>(xTx);
            beta = svd.Solve(xTy);
        }

        return beta;
    }

    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);;

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);;

        return metrics;
    }

    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model parameters
        writer.Write(Convert.ToDouble(_trend));
        writer.Write(_seasonalComponents.Length);
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            writer.Write(Convert.ToDouble(_seasonalComponents[i]));
        }
        writer.Write(_holidayComponents.Length);
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            writer.Write(Convert.ToDouble(_holidayComponents[i]));
        }
        writer.Write(Convert.ToDouble(_changepoint));
        writer.Write(_regressors.Length);
        for (int i = 0; i < _regressors.Length; i++)
        {
            writer.Write(Convert.ToDouble(_regressors[i]));
        }

        // Write options
        writer.Write(_prophetOptions.SeasonalPeriods.Count);
        foreach (var period in _prophetOptions.SeasonalPeriods)
        {
            writer.Write(period);
        }
        writer.Write(_prophetOptions.Holidays.Count);
        foreach (var holiday in _prophetOptions.Holidays)
        {
            writer.Write(holiday.Ticks);
        }
        writer.Write(_prophetOptions.RegressorCount);
    }

    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model parameters
        _trend = NumOps.FromDouble(reader.ReadDouble());
        int seasonalLength = reader.ReadInt32();
        _seasonalComponents = new Vector<T>(seasonalLength);
        for (int i = 0; i < seasonalLength; i++)
        {
            _seasonalComponents[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        int holidayLength = reader.ReadInt32();
        _holidayComponents = new Vector<T>(holidayLength);
        for (int i = 0; i < holidayLength; i++)
        {
            _holidayComponents[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _changepoint = NumOps.FromDouble(reader.ReadDouble());
        int regressorLength = reader.ReadInt32();
        _regressors = new Vector<T>(regressorLength);
        for (int i = 0; i < regressorLength; i++)
        {
            _regressors[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read options
        _prophetOptions = new ProphetOptions<T>();
        int seasonalPeriodsCount = reader.ReadInt32();
        for (int i = 0; i < seasonalPeriodsCount; i++)
        {
            _prophetOptions.SeasonalPeriods.Add(reader.ReadInt32());
        }
        int holidaysCount = reader.ReadInt32();
        for (int i = 0; i < holidaysCount; i++)
        {
            _prophetOptions.Holidays.Add(new DateTime(reader.ReadInt64()));
        }
        _prophetOptions.RegressorCount = reader.ReadInt32();
    }
}