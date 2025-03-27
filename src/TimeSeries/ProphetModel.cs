namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a Prophet model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Prophet model is a procedure for forecasting time series data based on an additive model 
/// where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
/// </para>
/// <para><b>For Beginners:</b> Think of the Prophet model as a smart crystal ball for predicting future values 
/// in a series of data points over time. It's like predicting weather, but for any kind of data that changes 
/// over time, such as sales, website traffic, or stock prices. The model looks at past patterns, including 
/// seasonal changes (like how sales might increase during holidays) and overall trends, to make educated 
/// guesses about future values.
/// </para>
/// </remarks>
public class ProphetModel<T> : TimeSeriesModelBase<T>
{
    private ProphetOptions<T> _prophetOptions;
    private T _trend;
    private Vector<T> _seasonalComponents;
    private Vector<T> _holidayComponents;
    private T _changepoint;
    private Vector<T> _regressors;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProphetModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the Prophet model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Prophet model with the specified options or default options if none are provided.
    /// It initializes all the components of the model, including trend, seasonal components, holiday effects, 
    /// changepoints, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your prediction tool. You can customize how it works 
    /// by providing options, or let it use default settings. The model prepares itself to handle different aspects 
    /// of your data, such as overall direction (trend), repeating patterns (seasonal components), special events 
    /// (holidays), sudden changes (changepoints), and other factors that might affect your predictions (regressors).
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Trains the Prophet model using the provided input data and target values.
    /// </summary>
    /// <param name="x">The input matrix containing features.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Prophet model by initializing its components based on the input data,
    /// then optimizing the model parameters to best fit the provided target values.
    /// </para>
    /// <para><b>For Beginners:</b> Training is like teaching the model about your data. You give it examples 
    /// of past data (x) and what happened (y), and it learns patterns from this information. The model adjusts 
    /// its internal settings (parameters) to make its predictions as close as possible to the actual values you provided.
    /// After training, the model will be ready to make predictions on new data.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the components of the Prophet model based on the input data.
    /// </summary>
    /// <param name="x">The input matrix containing features.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This method sets initial values for the trend, seasonal components, holiday effects, 
    /// changepoint, and regressors based on the provided data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving the model a starting point. It looks at your data 
    /// and makes initial guesses about things like the overall direction of your data (trend), 
    /// repeating patterns (seasonal components), effects of holidays, points where the data behavior 
    /// changes suddenly (changepoint), and how other factors might be influencing your data (regressors). 
    /// These initial guesses help the model start its learning process from a reasonable position.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Estimates the initial trend of the time series data.
    /// </summary>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <returns>The estimated initial trend value.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a simple linear regression on the first few points of the time series 
    /// to estimate the initial trend.
    /// </para>
    /// <para><b>For Beginners:</b> This is like looking at the start of your data and figuring out 
    /// if it's generally going up or down. The method takes a quick look at the first few data points 
    /// and draws a straight line through them. The slope of this line gives us an idea of the initial 
    /// direction and speed of change in your data.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the seasonal components of the Prophet model.
    /// </summary>
    /// <param name="x">The input matrix containing features.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This method sets up the initial values for seasonal patterns in the data, using Fourier series 
    /// to represent different seasonal periods.
    /// </para>
    /// <para><b>For Beginners:</b> Seasons in data are like repeating patterns. This method looks for 
    /// patterns that repeat over different time periods (like daily, weekly, or yearly patterns). 
    /// It uses a mathematical technique called Fourier series, which is a way to represent these 
    /// repeating patterns using simple wave-like functions. By doing this, the model can understand 
    /// and predict seasonal effects in your data.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Performs a simple linear regression to find the relationship between two variables.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>The slope of the regression line.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the best-fitting straight line through a set of points.
    /// </para>
    /// <para><b>For Beginners:</b> This is like finding the best straight line that fits through 
    /// a bunch of points on a graph. It helps us understand how one thing (x) relates to another (y). 
    /// The result tells us how much y tends to change when x changes by a certain amount.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the holiday components of the model.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output vector.</param>
    /// <remarks>
    /// <para>
    /// This method initializes all holiday components to zero. It's typically called during the model setup phase.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial effect of holidays on our predictions.
    /// We start by assuming holidays have no effect (zero), and the model will learn the actual effects during training.
    /// </para>
    /// </remarks>
    private void InitializeHolidayComponents(Matrix<T> x, Vector<T> y)
    {
        // Initialize holiday components to zero
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Estimates the initial changepoint value based on the input data.
    /// </summary>
    /// <param name="y">The input vector of time series values.</param>
    /// <returns>The estimated initial changepoint value.</returns>
    /// <exception cref="ArgumentException">Thrown when the input vector has fewer than two elements.</exception>
    /// <remarks>
    /// <para>
    /// This method calculates the median of the first differences of the input vector, excluding zero differences.
    /// It's used to initialize the changepoint parameter of the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method tries to find a good starting point for detecting changes in our data.
    /// It looks at how much our data changes from one point to the next and picks a typical value for these changes.
    /// This helps our model start with a reasonable guess about when important shifts in the data might occur.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Initializes the regressor components of the model.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output vector.</param>
    /// <remarks>
    /// <para>
    /// This method initializes the regressor components using Ordinary Least Squares (OLS) regression.
    /// It's only performed if there are regressors specified in the model options.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial relationships between our extra input factors
    /// (regressors) and our output. It uses a simple form of regression to guess how each factor might influence
    /// our predictions before we start the main training process.
    /// </para>
    /// </remarks>
    private void InitializeRegressors(Matrix<T> x, Vector<T> y)
    {
        // Initialize regressors using OLS
        if (_prophetOptions.RegressorCount > 0)
        {
            Matrix<T> regressorMatrix = x.Submatrix(0, x.Columns - _prophetOptions.RegressorCount, x.Rows, _prophetOptions.RegressorCount);
            _regressors = SimpleMultipleRegression(regressorMatrix, y);
        }
    }

    /// <summary>
    /// Optimizes the model parameters using the specified or default optimizer.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output vector.</param>
    /// <remarks>
    /// <para>
    /// This method performs the main optimization of the model parameters. It initializes the parameters,
    /// prepares the optimization input data, runs the optimization process, and updates the model parameters
    /// with the optimized values.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model really learns from the data. It starts with initial
    /// guesses for all its parameters, then uses an optimization algorithm to adjust these parameters to better
    /// fit the data. It's like fine-tuning all the knobs of the model to make its predictions as accurate as possible.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Predicts output values for the given input matrix.
    /// </summary>
    /// <param name="input">The input matrix to make predictions for.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates predictions for each row in the input matrix by calling PredictSingle for each row.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes a bunch of input data and produces predictions for each piece
    /// of that data. It's like running many individual predictions all at once.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Predicts a single output value for the given input vector.
    /// </summary>
    /// <param name="x">The input vector to make a prediction for.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method combines all components of the Prophet model (trend, seasonal, holiday, changepoint, and regressor effects)
    /// to produce a single prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a single point in time. It adds up all the
    /// different effects our model has learned (like overall trends, seasonal patterns, holiday effects, and so on)
    /// to come up with a final prediction.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> x)
    {
        T prediction = _trend;
        prediction = NumOps.Add(prediction, GetSeasonalComponent(x));
        prediction = NumOps.Add(prediction, GetHolidayComponent(x));
        prediction = NumOps.Add(prediction, GetChangepointEffect(x));
        prediction = NumOps.Add(prediction, GetRegressorEffect(x));

        return prediction;
    }

    /// <summary>
    /// Calculates the seasonal component of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the time index.</param>
    /// <returns>The calculated seasonal component.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the seasonal effect using Fourier series for each seasonal period defined in the model options.
    /// It combines sine and cosine terms to represent cyclical patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how much the prediction should be adjusted based on seasonal patterns.
    /// For example, it might account for weekly or yearly cycles in your data. It uses mathematical functions (sine and cosine)
    /// to create smooth, repeating patterns that match the seasonality in your data.
    /// </para>
    /// </remarks>
    private T GetSeasonalComponent(Vector<T> x)
    {
        T _seasonalComponent = NumOps.Zero;
        int _timeIndex = 0; // Assume the time index is the first element of x

        foreach (var _period in _prophetOptions.SeasonalPeriods)
        {
            T _t = NumOps.Divide(NumOps.FromDouble(_timeIndex), NumOps.FromDouble(_period));
            for (int _j = 0; _j < _prophetOptions.FourierOrder; _j++)
            {
                int _idx = _j * 2;
                T _cos_t = MathHelper.Cos(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * (_j + 1)), _t));
                T _sin_t = MathHelper.Sin(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * (_j + 1)), _t));
                _seasonalComponent = NumOps.Add(_seasonalComponent, NumOps.Multiply(_seasonalComponents[_idx], _cos_t));
                _seasonalComponent = NumOps.Add(_seasonalComponent, NumOps.Multiply(_seasonalComponents[_idx + 1], _sin_t));
            }
        }

        return _seasonalComponent;
    }

    /// <summary>
    /// Calculates the holiday component of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the date information.</param>
    /// <returns>The calculated holiday component.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the current date is a holiday and returns the corresponding holiday effect.
    /// It assumes that only one holiday can occur on a given day.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the prediction based on whether the current date is a holiday.
    /// Holidays often have unique patterns that differ from regular days, so this helps the model account for those special days.
    /// </para>
    /// </remarks>
    private T GetHolidayComponent(Vector<T> x)
    {
        T _holidayComponent = NumOps.Zero;
        DateTime _currentDate = DateTime.FromOADate(Convert.ToDouble(x[0])); // Assume the date is the first element of x

        for (int _i = 0; _i < _prophetOptions.Holidays.Count; _i++)
        {
            if (_currentDate.Date == _prophetOptions.Holidays[_i].Date)
            {
                _holidayComponent = NumOps.Add(_holidayComponent, _holidayComponents[_i]);
                break; // Assume only one holiday per day
            }
        }

        return _holidayComponent;
    }

    /// <summary>
    /// Calculates the changepoint effect of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the time information.</param>
    /// <returns>The calculated changepoint effect.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the cumulative effect of all changepoints up to the current time point.
    /// Changepoints represent sudden changes in the trend of the time series.
    /// </para>
    /// <para><b>For Beginners:</b> Changepoints are moments when the overall trend of your data suddenly shifts.
    /// This method calculates how much these shifts affect the prediction at the current time point.
    /// For example, a changepoint might occur when a company releases a new product, causing a sudden increase in sales.
    /// </para>
    /// </remarks>
    private T GetChangepointEffect(Vector<T> x)
    {
        T _changepointEffect = NumOps.Zero;
        T _t = x[0]; // Assume the time is the first element of x

        for (int _i = 0; _i < _prophetOptions.Changepoints.Count; _i++)
        {
            if (NumOps.GreaterThan(_t, _prophetOptions.Changepoints[_i]))
            {
                T _delta = NumOps.Subtract(_t, _prophetOptions.Changepoints[_i]);
                _changepointEffect = NumOps.Add(_changepointEffect, NumOps.Multiply(_changepoint, _delta));
            }
        }

        return _changepointEffect;
    }

    /// <summary>
    /// Calculates the regressor effect of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the regressor values.</param>
    /// <returns>The calculated regressor effect.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the combined effect of all regressors on the prediction.
    /// Regressors are external factors that can influence the time series.
    /// </para>
    /// <para><b>For Beginners:</b> Regressors are additional pieces of information that can help explain changes in your data.
    /// For example, if you're predicting ice cream sales, temperature could be a regressor because it affects how much ice cream people buy.
    /// This method calculates how all these extra factors combine to influence the prediction.
    /// </para>
    /// </remarks>
    private T GetRegressorEffect(Vector<T> x)
    {
        T _regressorEffect = NumOps.Zero;

        for (int _i = 0; _i < _regressors.Length; _i++)
        {
            // Assume regressors start from the second element of x
            _regressorEffect = NumOps.Add(_regressorEffect, NumOps.Multiply(_regressors[_i], x[_i + 1]));
        }

        return _regressorEffect;
    }

    /// <summary>
    /// Calculates the total size of the model's state vector.
    /// </summary>
    /// <returns>The size of the state vector.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the total number of parameters in the model, including trend, seasonal components,
    /// holiday components, changepoint, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method counts how many different pieces of information the model is using to make predictions.
    /// It's like counting all the ingredients in a recipe. This count helps the model know how much information it needs to keep track of.
    /// </para>
    /// </remarks>
    private int GetStateSize()
    {
        return 1 + _seasonalComponents.Length + _holidayComponents.Length + 1 + _regressors.Length;
    }

    /// <summary>
    /// Retrieves the current state of the model as a vector.
    /// </summary>
    /// <returns>A vector representing the current state of the model.</returns>
    /// <remarks>
    /// <para>
    /// This method collects all current parameter values into a single vector, including trend, seasonal components,
    /// holiday components, changepoint, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the current settings of the model into one list.
    /// It's like taking a snapshot of the model at a specific moment. This is useful for saving the model's state
    /// or for certain types of calculations that need all the model's information at once.
    /// </para>
    /// </remarks>
    private Vector<T> GetCurrentState()
    {
        int _stateSize = GetStateSize();
        Vector<T> _currentState = new Vector<T>(_stateSize);
        int _index = 0;

        _currentState[_index++] = _trend;
        for (int _i = 0; _i < _seasonalComponents.Length; _i++)
        {
            _currentState[_index++] = _seasonalComponents[_i];
        }
        for (int _i = 0; _i < _holidayComponents.Length; _i++)
        {
            _currentState[_index++] = _holidayComponents[_i];
        }
        _currentState[_index++] = _changepoint;
        for (int _i = 0; _i < _regressors.Length; _i++)
        {
            _currentState[_index++] = _regressors[_i];
        }

        return _currentState;
    }

    /// <summary>
    /// Performs simple multiple linear regression.
    /// </summary>
    /// <param name="x">The input matrix of regressor values.</param>
    /// <param name="y">The output vector of target values.</param>
    /// <returns>A vector of regression coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This method performs multiple linear regression using either Cholesky decomposition or Singular Value Decomposition (SVD).
    /// It first attempts to use Cholesky decomposition for efficiency, and falls back to SVD if Cholesky fails.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best way to combine different pieces of information (regressors)
    /// to predict an outcome. It's like finding the perfect recipe that tells you how much of each ingredient to use.
    /// The method tries a fast approach first (Cholesky decomposition), but if that doesn't work, it uses a more reliable
    /// but slower method (SVD). The result tells you how important each piece of information is for making predictions.
    /// </para>
    /// </remarks>
    private Vector<T> SimpleMultipleRegression(Matrix<T> x, Vector<T> y)
    {
        // Add a column of ones to X for the intercept term
        Matrix<T> _xWithIntercept = new Matrix<T>(x.Rows, x.Columns + 1);
        for (int _i = 0; _i < x.Rows; _i++)
        {
            _xWithIntercept[_i, 0] = NumOps.One;
            for (int _j = 0; _j < x.Columns; _j++)
            {
                _xWithIntercept[_i, _j + 1] = x[_i, _j];
            }
        }

        // Calculate (X^T * X)
        Matrix<T> _xTx = _xWithIntercept.Transpose().Multiply(_xWithIntercept);

        // Calculate (X^T * y)
        Vector<T> _xTy = _xWithIntercept.Transpose().Multiply(y);

        // Solve the normal equations: (X^T * X) * beta = (X^T * y)
        Vector<T> _beta;
        try
        {
            // Try Cholesky decomposition first (faster and more stable for well-conditioned matrices)
            var _cholesky = new CholeskyDecomposition<T>(_xTx);
            _beta = _cholesky.Solve(_xTy);
        }
        catch (Exception)
        {
            // If Cholesky fails, fall back to SVD (more robust but slower)
            var _svd = new SvdDecomposition<T>(_xTx);
            _beta = _svd.Solve(_xTy);
        }

        return _beta;
    }

    /// <summary>
    /// Evaluates the performance of the Prophet model on a test dataset.
    /// </summary>
    /// <param name="xTest">The input matrix containing the test features.</param>
    /// <param name="yTest">The vector containing the true target values for the test set.</param>
    /// <returns>A dictionary containing various performance metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates several common regression metrics to assess the model's performance:
    /// - Mean Absolute Error (MAE): Measures the average magnitude of errors in the predictions.
    /// - Mean Squared Error (MSE): Measures the average squared difference between predictions and actual values.
    /// - Root Mean Squared Error (RMSE): The square root of MSE, providing a metric in the same units as the target variable.
    /// - R-squared (R2): Represents the proportion of variance in the dependent variable that is predictable from the independent variable(s).
    /// </para>
    /// <para><b>For Beginners:</b> This method helps us understand how well our model is performing. 
    /// It compares the model's predictions to the actual values we know are correct, and calculates 
    /// several numbers that tell us how close our predictions are:
    /// - MAE: On average, how far off are our predictions? (in the same units as our data)
    /// - MSE: Similar to MAE, but punishes big mistakes more (in squared units)
    /// - RMSE: Like MSE, but back in the original units of our data
    /// - R2: How much of the changes in our data does our model explain? (from 0 to 1, where 1 is perfect)
    /// 
    /// These numbers help us decide if our model is good enough or if we need to improve it.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

        return metrics;
    }

    /// <summary>
    /// Serializes the core components of the Prophet model.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the serialized data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the model's parameters and options to a binary stream. It includes the trend,
    /// seasonal components, holiday components, changepoint, regressors, and various model options.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important parts of our model to a file.
    /// It's like taking a snapshot of the model's current state, including all the patterns it has learned.
    /// This allows us to save our trained model and use it later without having to retrain it.
    /// </para>
    /// </remarks>
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

    /// <summary>
    /// Deserializes the core components of the Prophet model.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the serialized data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's parameters and options from a binary stream. It reconstructs
    /// the trend, seasonal components, holiday components, changepoint, regressors, and various model options.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a file.
    /// It's like restoring a snapshot of the model's state, including all the patterns it had learned.
    /// This allows us to use a trained model without having to retrain it every time we want to use it.
    /// </para>
    /// </remarks>
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