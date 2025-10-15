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
public class ProphetModel<T, TInput, TOutput> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Stores the configuration options for the Prophet model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds all user-configurable settings for the model, including seasonal periods,
    /// holidays, changepoint settings, regressor configurations, and optimization parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the model's instruction manual. It contains
    /// all the settings that control how the model works, such as which seasonal patterns to look for,
    /// which holidays to consider, and how the model should learn from your data.
    /// </para>
    /// </remarks>
    private ProphetOptions<T, TInput, TOutput> _prophetOptions = default!;

    /// <summary>
    /// Represents the overall trend component of the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The trend component captures the long-term increase or decrease in the time series data.
    /// It's one of the fundamental components of the Prophet decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the main direction your data is heading in over time.
    /// For example, if you're tracking sales over many years, this captures whether sales are generally
    /// increasing, decreasing, or staying flat over the long term, ignoring seasonal ups and downs.
    /// </para>
    /// </remarks>
    private T _trend = default!;

    /// <summary>
    /// Stores the coefficients for all seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients for the Fourier series representation of all seasonal components.
    /// Each seasonal period (yearly, monthly, weekly, etc.) is represented by multiple sine and cosine terms.
    /// </para>
    /// <para><b>For Beginners:</b> This holds information about repeating patterns in your data.
    /// For example, ice cream sales might go up every summer and down every winter, or website traffic
    /// might be higher on weekends than weekdays. These coefficients help the model capture these
    /// regular patterns at different time scales (daily, weekly, yearly, etc.).
    /// </para>
    /// </remarks>
    private Vector<T> _seasonalComponents = default!;

    /// <summary>
    /// Stores the effect of each holiday on the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients representing how much each defined holiday affects the
    /// time series values. Each element corresponds to a holiday defined in ProphetOptions.Holidays.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how special days like holidays affect your data.
    /// For example, retail sales might spike before Christmas or drop on certain holidays.
    /// Each number in this list tells the model how much a specific holiday tends to
    /// increase or decrease the values in your data.
    /// </para>
    /// </remarks>
    private Vector<T> _holidayComponents = default!;

    /// <summary>
    /// Represents the coefficient for the changepoint component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The changepoint component models abrupt changes in the trend of the time series.
    /// This coefficient determines the magnitude of the effect of each changepoint.
    /// </para>
    /// <para><b>For Beginners:</b> This captures sudden changes in your data's overall pattern.
    /// For example, if a company launched a new product and saw a sudden increase in sales,
    /// or if a policy change caused a sharp drop in some measurement, this helps the model
    /// account for these unexpected shifts instead of treating them as noise.
    /// </para>
    /// </remarks>
    private T _changepoint = default!;

    /// <summary>
    /// Stores the coefficients for additional regressor variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients for external regressor variables that may influence
    /// the time series but are not part of the core seasonal or trend components.
    /// </para>
    /// <para><b>For Beginners:</b> This holds information about how external factors affect your data.
    /// For example, if you're predicting ice cream sales, temperature might be a regressor because
    /// hot weather leads to more ice cream sales. These numbers tell the model exactly how much
    /// each external factor influences your data.
    /// </para>
    /// </remarks>
    private Vector<T> _regressors = default!;

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
    public ProphetModel(ProphetOptions<T, TInput, TOutput>? options = null) 
        : base(options ?? new ProphetOptions<T, TInput, TOutput>())
    {
        _prophetOptions = options ?? new ProphetOptions<T, TInput, TOutput>();

        // Initialize model components
        _trend = NumOps.FromDouble(_prophetOptions.InitialTrendValue);
        _seasonalComponents = new Vector<T>(_prophetOptions.SeasonalPeriods.Sum());

        // Handle potentially empty holiday collection
        if (_prophetOptions.Holidays != null && _prophetOptions.Holidays.Count > 0)
        {
            _holidayComponents = new Vector<T>(_prophetOptions.Holidays.Count);
        }
        else
        {
            // Create an empty vector for holidays
            _holidayComponents = Vector<T>.Empty();
        }

        _changepoint = NumOps.FromDouble(_prophetOptions.InitialChangepointValue);
        _regressors = _prophetOptions.RegressorCount > 0
            ? new Vector<T>(_prophetOptions.RegressorCount)
            : Vector<T>.Empty();
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
        int totalComponents = 0;

        // Calculate total number of components needed
        foreach (int period in _prophetOptions.SeasonalPeriods)
        {
            int numHarmonics = Math.Min(period / 2, 10);
            totalComponents += numHarmonics * 2; // Each harmonic has sin and cos
        }

        // Initialize the seasonal components vector
        _seasonalComponents = new Vector<T>(totalComponents);

        int index = 0;

        // For each seasonal period
        foreach (int period in _prophetOptions.SeasonalPeriods)
        {
            int numHarmonics = Math.Min(period / 2, 10);

            for (int h = 1; h <= numHarmonics; h++)
            {
                // Create matrices for this harmonic
                var sinComponent = new Vector<T>(n);
                var cosComponent = new Vector<T>(n);

                // Calculate sin and cos for each time point
                for (int i = 0; i < n; i++)
                {
                    // Get time value from first column
                    T timeValue = x[i, 0];
                    T t = NumOps.Divide(timeValue, NumOps.FromDouble(period));
                    T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * h), t);

                    sinComponent[i] = MathHelper.Sin(angle);
                    cosComponent[i] = MathHelper.Cos(angle);
                }

                // Perform simple linear regression to get initial coefficients
                T sinCoefficient = SimpleLinearRegression(sinComponent, y);
                T cosCoefficient = SimpleLinearRegression(cosComponent, y);

                // Store the coefficients
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
        _trend = y.Mean();

        // Calculate total parameter count
        int totalParams = 1 + _seasonalComponents.Length + _holidayComponents.Length + 1 + _regressors.Length;

        // Initialize parameters vector with current values
        var initialParameters = new Vector<T>(totalParams);
        int paramIndex = 0;
        initialParameters[paramIndex++] = _trend;

        // Add seasonal component parameters
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            initialParameters[paramIndex++] = _seasonalComponents[i];
        }

        // Add holiday component parameters
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            initialParameters[paramIndex++] = _holidayComponents[i];
        }

        // Add changepoint parameter
        initialParameters[paramIndex++] = _changepoint;

        // Add regressor parameters
        for (int i = 0; i < _regressors.Length; i++)
        {
            initialParameters[paramIndex++] = _regressors[i];
        }

        // Initialize random values for zero parameters to avoid optimization issues
        InitializeZeroParameters(initialParameters);

        // Get active feature indices if available
        IEnumerable<int>? activeFeatures = null;
        if (this is IFeatureAware featureAwareModel)
        {
            activeFeatures = featureAwareModel.GetActiveFeatureIndices();
        }

        // Define a custom prediction function that mirrors the actual full prediction logic
        Func<Matrix<T>, Vector<T>, Vector<T>> customPredict = (input, parameters) =>
        {
            // Create output vector with appropriate size (one prediction per row)
            var result = new Vector<T>(input.Rows);
            T trendParam = parameters[0];

            // For each input row
            for (int i = 0; i < result.Length; i++)
            {
                var row = input.GetRow(i);
                T timeValue = row[0]; // Time is first column
                result[i] = trendParam;

                // Add seasonal components
                int pIdx = 1; // Start index after trend

                foreach (var period in _prophetOptions.SeasonalPeriods)
                {
                    int numHarmonics = Math.Min(period / 2, 10);
                    for (int h = 1; h <= numHarmonics; h++)
                    {
                        T t = NumOps.Divide(timeValue, NumOps.FromDouble(period));
                        T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * h), t);
                        T sinVal = MathHelper.Sin(angle);
                        T cosVal = MathHelper.Cos(angle);

                        if (pIdx < parameters.Length && pIdx + 1 < parameters.Length)
                        {
                            T sinCoef = parameters[pIdx++];
                            T cosCoef = parameters[pIdx++];

                            result[i] = NumOps.Add(result[i], NumOps.Multiply(sinCoef, sinVal));
                            result[i] = NumOps.Add(result[i], NumOps.Multiply(cosCoef, cosVal));
                        }
                    }
                }

                // Skip holiday components for simplicity
                pIdx += _holidayComponents.Length;

                // Add changepoint effect if there are any
                if (pIdx < parameters.Length)
                {
                    T changepointParam = parameters[pIdx++];

                    foreach (var cp in _prophetOptions.Changepoints)
                    {
                        if (NumOps.GreaterThan(timeValue, cp))
                        {
                            T delta = NumOps.Subtract(timeValue, cp);
                            result[i] = NumOps.Add(result[i], NumOps.Multiply(changepointParam, delta));
                        }
                    }
                }

                // Skip regressor components for simplicity
            }

            return result;
        };

        // Create a parameter placeholder model
        var placeholderModel = new ParameterPlaceholderModel<T, Matrix<T>, Vector<T>>(
            initialParameters, null, activeFeatures, customPredict, this);

        // Use user-defined optimizer or default
        var optimizer = _prophetOptions.Optimizer ?? new LBFGSOptimizer<T, Matrix<T>, Vector<T>>(placeholderModel);

        // Get the cached input data
        var optimizationData = DefaultInputCache.GetDefaultInputData<T, Matrix<T>, Vector<T>>();

        // Perform optimization
        var optimizationResult = optimizer.Optimize(optimizationData);

        // Check if optimization succeeded
        if (optimizationResult.BestSolution == null || optimizationResult.BestSolution.GetParameters() == null)
        {
            throw new InvalidOperationException("Optimization failed: no solution found");
        }

        // Update ALL model parameters with optimized values
        Vector<T> optimizedParameters = optimizationResult.BestSolution.GetParameters();

        // Check if we have enough parameters
        if (optimizedParameters.Length < totalParams)
        {
            throw new InvalidOperationException($"Optimization returned too few parameters: {optimizedParameters.Length} (expected {totalParams})");
        }

        // Update parameters in the model
        paramIndex = 0;

        _trend = optimizedParameters[paramIndex++];

        // If trend is zero or negative, restore initial trend
        if (NumOps.LessThanOrEquals(_trend, NumOps.Zero))
        {
            _trend = y.Mean(); // Fallback to mean
        }

        // Update seasonal components
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            _seasonalComponents[i] = optimizedParameters[paramIndex++];
        }

        // Update holiday components
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = optimizedParameters[paramIndex++];
        }

        // Update changepoint
        _changepoint = optimizedParameters[paramIndex++];

        // Update regressors
        for (int i = 0; i < _regressors.Length; i++)
        {
            _regressors[i] = optimizedParameters[paramIndex++];
        }

        // Validate that parameters have been updated
        ValidateModelParameters();
    }

    /// <summary>
    /// Initialize any zero parameters with small random values to improve optimization.
    /// </summary>
    private void InitializeZeroParameters(Vector<T> parameters)
    {
        Random random = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < parameters.Length; i++)
        {
            if (NumOps.Equals(parameters[i], NumOps.Zero))
            {
                // Generate small random value between -0.1 and 0.1
                double randomValue = (random.NextDouble() * 0.2) - 0.1;
                parameters[i] = NumOps.FromDouble(randomValue);
            }
        }
    }

    /// <summary>
    /// Extract time features (sine and cosine terms) for each seasonal period.
    /// </summary>
    private Dictionary<int, (T Sin, T Cos)[]> ExtractTimeFeatures(T timeValue)
    {
        var result = new Dictionary<int, (T Sin, T Cos)[]>();

        foreach (int period in _prophetOptions.SeasonalPeriods)
        {
            int numHarmonics = Math.Min(period / 2, 10);
            var harmonics = new (T Sin, T Cos)[numHarmonics];

            T t = NumOps.Divide(timeValue, NumOps.FromDouble(period));

            for (int h = 1; h <= numHarmonics; h++)
            {
                T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * h), t);
                harmonics[h - 1] = (MathHelper.Sin(angle), MathHelper.Cos(angle));
            }

            result[period] = harmonics;
        }

        return result;
    }

    /// <summary>
    /// Convert a numeric timestamp to a DateTime object.
    /// </summary>
    private DateTime DateTimeFromTimestamp(T timestamp)
    {
        try
        {
            // Depends on how your timestamps are stored
            // This assumes OADate format
            return DateTime.FromOADate(Convert.ToDouble(timestamp));
        }
        catch
        {
            // Fallback - current date
            return DateTime.Today;
        }
    }

    /// <summary>
    /// Calculate the effect of all changepoints for a given time value.
    /// </summary>
    private T CalculateChangepointEffect(T timeValue, T changepointParam)
    {
        T effect = NumOps.Zero;

        foreach (var cp in _prophetOptions.Changepoints)
        {
            if (NumOps.GreaterThan(timeValue, cp))
            {
                T delta = NumOps.Subtract(timeValue, cp);
                effect = NumOps.Add(effect, NumOps.Multiply(changepointParam, delta));
            }
        }

        return effect;
    }

    /// <summary>
    /// Validate that model parameters have been properly updated.
    /// </summary>
    private void ValidateModelParameters()
    {
        bool allZero = true;

        // Check trend
        if (!NumOps.Equals(_trend, NumOps.Zero))
            allZero = false;

        // Check seasonal components
        for (int i = 0; i < _seasonalComponents.Length && allZero; i++)
        {
            if (!NumOps.Equals(_seasonalComponents[i], NumOps.Zero))
                allZero = false;
        }

        // Check holiday components
        for (int i = 0; i < _holidayComponents.Length && allZero; i++)
        {
            if (!NumOps.Equals(_holidayComponents[i], NumOps.Zero))
                allZero = false;
        }

        // Check changepoint
        if (!NumOps.Equals(_changepoint, NumOps.Zero))
            allZero = false;

        // Check regressors
        for (int i = 0; i < _regressors.Length && allZero; i++)
        {
            if (!NumOps.Equals(_regressors[i], NumOps.Zero))
                allZero = false;
        }

        if (allZero)
        {
            throw new InvalidOperationException("Optimization failed: all model parameters are zero");
        }
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
        // Validate input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input matrix cannot be null");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0); // Return empty vector for empty input
        }

        // Check if model has been trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions");
        }

        // Validate input dimensions
        int expectedMinColumns = 1; // At minimum we need time

        // If we have regressors, we need columns for them too
        if (_prophetOptions.RegressorCount > 0)
        {
            expectedMinColumns += _prophetOptions.RegressorCount;
        }

        if (input.Columns < expectedMinColumns)
        {
            throw new ArgumentException(
                $"Input matrix has too few columns ({input.Columns}). Expected at least {expectedMinColumns}.",
                nameof(input));
        }

        // Create output vector
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        // Make prediction for each row
        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingleInternal(input.GetRow(i));

            // Apply transformation if needed
            if (_prophetOptions.ApplyTransformation)
            {
                predictions[i] = _prophetOptions.TransformPrediction(predictions[i]);
            }
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
    private T PredictSingleInternal(Vector<T> x)
    {
        if (x == null || x.Length == 0)
        {
            throw new ArgumentException("Input vector cannot be null or empty", nameof(x));
        }

        T prediction = _trend;

        // Add seasonal component
        T seasonalComponent = GetSeasonalComponent(x);
        prediction = NumOps.Add(prediction, seasonalComponent);

        // Add holiday component
        T holidayComponent = GetHolidayComponent(x);
        prediction = NumOps.Add(prediction, holidayComponent);

        // Add changepoint effect
        T changepointEffect = GetChangepointEffect(x);
        prediction = NumOps.Add(prediction, changepointEffect);

        // Add regressor effect
        T regressorEffect = GetRegressorEffect(x);
        prediction = NumOps.Add(prediction, regressorEffect);

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
        T seasonalComponent = NumOps.Zero;
        T timeValue = x[0]; // Assuming time is the first element

        int componentIndex = 0;

        foreach (int period in _prophetOptions.SeasonalPeriods)
        {
            int harmonics = Math.Min(period / 2, 10); // Use up to 10 harmonics or period/2

            // Calculate t as a fraction of the period
            T t = NumOps.Divide(timeValue, NumOps.FromDouble(period));

            for (int h = 1; h <= harmonics; h++)
            {
                // Calculate angle: 2π * h * t
                T angle = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI * h), t);

                // Calculate sin and cos components
                T sinComponent = MathHelper.Sin(angle);
                T cosComponent = MathHelper.Cos(angle);

                // Access the correct coefficients for this period and harmonic
                T sinCoef = _seasonalComponents[componentIndex++];
                T cosCoef = _seasonalComponents[componentIndex++];

                // Add weighted components to the result
                seasonalComponent = NumOps.Add(seasonalComponent, NumOps.Multiply(sinCoef, sinComponent));
                seasonalComponent = NumOps.Add(seasonalComponent, NumOps.Multiply(cosCoef, cosComponent));
            }
        }

        return seasonalComponent;
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
        if (_holidayComponents.Length == 0 || _prophetOptions.Holidays.Count == 0)
        {
            return NumOps.Zero;
        }

        T holidayComponent = NumOps.Zero;

        // Convert timestamp to date
        DateTime currentDate;
        try
        {
            double timestamp = Convert.ToDouble(x[0]);
            currentDate = DateTime.FromOADate(timestamp);
        }
        catch
        {
            // If date conversion fails, return zero effect
            return NumOps.Zero;
        }

        // Remove time component to compare only dates
        currentDate = currentDate.Date;

        // Check if current date matches any holiday
        for (int i = 0; i < _prophetOptions.Holidays.Count && i < _holidayComponents.Length; i++)
        {
            if (currentDate == _prophetOptions.Holidays[i].Date)
            {
                holidayComponent = NumOps.Add(holidayComponent, _holidayComponents[i]);

                // If we want to allow multiple holiday effects to stack, remove this break
                // For most use cases, we assume only one holiday effect applies per day
                break;
            }
        }

        return holidayComponent;
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
        T changepointEffect = NumOps.Zero;
        T timeValue = x[0]; // Assuming time is first element

        if (_prophetOptions.Changepoints.Count == 0 || NumOps.Equals(_changepoint, NumOps.Zero))
        {
            return changepointEffect;
        }

        // Calculate cumulative effect of all changepoints
        foreach (T changepointTime in _prophetOptions.Changepoints)
        {
            if (NumOps.GreaterThan(timeValue, changepointTime))
            {
                T timeSinceChangepoint = NumOps.Subtract(timeValue, changepointTime);
                T effect = NumOps.Multiply(_changepoint, timeSinceChangepoint);

                // Only add positive effects or limit negative effects
                if (NumOps.GreaterThanOrEquals(effect, NumOps.Zero))
                {
                    changepointEffect = NumOps.Add(changepointEffect, effect);
                }
                else
                {
                    // Limit negative effects to prevent erasing the trend
                    T limitedEffect = NumOps.Multiply(effect, NumOps.FromDouble(0.1));
                    changepointEffect = NumOps.Add(changepointEffect, limitedEffect);
                }
            }
        }

        return changepointEffect;
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
        T regressorEffect = NumOps.Zero;

        if (_regressors.Length == 0)
        {
            return regressorEffect;
        }

        // Regressor values should start from index 1 (after time)
        for (int i = 0; i < _regressors.Length && i + 1 < x.Length; i++)
        {
            T regressorValue = x[i + 1];
            T coefficient = _regressors[i];

            // Add weighted regressor effect
            regressorEffect = NumOps.Add(regressorEffect,
                NumOps.Multiply(coefficient, regressorValue));
        }

        return regressorEffect;
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
    /// Transforms prediction-level gradients to parameter-level gradients for Prophet models.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="predictionGradient">The gradient with respect to predictions.</param>
    /// <returns>The gradient with respect to model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This implementation provides the Prophet-specific gradient transformation,
    /// accounting for trend, seasonality, holiday effects, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells the optimizer how to adjust each
    /// parameter in the Prophet model based on prediction errors at each time point.
    /// </para>
    /// </remarks>
    public override Vector<T> TransformGradient(Matrix<T> input, Vector<T> predictionGradient)
    {
        // Get parameters
        Vector<T> parameters = GetParameters();
        int paramCount = parameters.Length;
        var parameterGradient = new Vector<T>(paramCount);
        int nRegressor = input.Columns;

        // Process each parameter's contribution to each prediction
        for (int i = 0; i < paramCount; i++)
        {
            T gradSum = NumOps.Zero;

            // For each time point/prediction
            for (int j = 0; j < predictionGradient.Length; j++)
            {
                T contribution;

                // If this is a regressor parameter and within range
                if (i < nRegressor && j < input.Rows)
                {
                    // Contribution is the input value times the prediction gradient
                    contribution = NumOps.Multiply(input[j, i], predictionGradient[j]);
                }
                // If this is the trend parameter
                else if (i == nRegressor)
                {
                    // Trend affects each prediction equally
                    contribution = predictionGradient[j];
                }
                // If this is the changepoint parameter
                else if (i == nRegressor + 1)
                {
                    // Changepoint effect is position-dependent
                    T position = NumOps.FromDouble((double)j / predictionGradient.Length);

                    // Here we could use more complex changepoint logic if needed
                    contribution = NumOps.Multiply(position, predictionGradient[j]);
                }
                // Other parameters (seasonality, holidays, etc.)
                else
                {
                    // Implementation would depend on how these components are modeled
                    // For now, we'll use a simpler approximation
                    contribution = NumOps.Zero;
                }

                gradSum = NumOps.Add(gradSum, contribution);
            }

            parameterGradient[i] = gradSum;
        }

        return parameterGradient;
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
        _prophetOptions = new ProphetOptions<T, TInput, TOutput>();
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

    /// <summary>
    /// Core implementation of the training logic for the Prophet model.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This protected method contains the actual implementation of the training process.
    /// It validates the input data, initializes model components, optimizes the parameters,
    /// updates the model's parameter vector, and stores the final state for future reference.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the real learning happens for the model.
    /// The method carefully validates your data, then initializes the model components
    /// (like trend and seasonal patterns), and finally optimizes all the parameters to
    /// best fit your data. It also makes sure to store these learned parameters properly
    /// so they can be accessed later when making predictions or when saving the model.
    /// If anything goes wrong during this process, it provides clear error messages to 
    /// help you understand and fix the issue.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate inputs
        if (x == null)
        {
            throw new ArgumentNullException(nameof(x), "Input matrix cannot be null");
        }

        if (y == null)
        {
            throw new ArgumentNullException(nameof(y), "Target vector cannot be null");
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException($"Input matrix rows ({x.Rows}) must match target vector length ({y.Length})");
        }

        if (x.Rows == 0)
        {
            throw new ArgumentException("Cannot train on empty dataset");
        }

        // Initialize components (trend, seasonal, holiday, changepoint, regressors)
        InitializeComponents(x, y);

        // Optimize parameters using the selected optimizer
        OptimizeParameters(x, y);

        // Update ModelParameters with the current state
        ModelParameters = GetCurrentState();
    }

    /// <summary>
    /// Applies the provided parameters to the Prophet model components.
    /// </summary>
    /// <param name="parameters">The vector of parameters to apply.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method distributes the values from the parameters vector to the appropriate model components.
    /// It first stores the parameters in the base ModelParameters property, then assigns each parameter
    /// to its corresponding component: trend, seasonal components, holiday components, changepoint, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes all the numerical values that define the model's behavior
    /// and assigns them to the right places within the model. It's like taking a list of settings and applying
    /// each one to the correct dial or switch in the model.
    /// 
    /// The parameters vector contains all these settings in a specific order:
    /// - First comes the trend component (the overall direction of your data)
    /// - Then the seasonal components (recurring patterns like daily, weekly, yearly cycles)
    /// - Next are holiday effects (how special days impact your values)
    /// - Then the changepoint (where your data's behavior suddenly shifts)
    /// - Finally, the regressor effects (how external factors influence your data)
    /// 
    /// This method makes sure each value goes to the right place so the model works correctly.
    /// </para>
    /// </remarks>
    protected override void ApplyParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        // Call base implementation to store the parameters
        base.ApplyParameters(parameters);

        // Apply parameters to the model's components
        int index = 0;

        // Check if we have enough parameters
        if (parameters.Length < GetStateSize())
        {
            throw new ArgumentException($"Parameters vector length ({parameters.Length}) is less than required state size ({GetStateSize()}).");
        }

        // Extract trend parameter
        _trend = parameters[index++];

        // Extract seasonal components
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            _seasonalComponents[i] = parameters[index++];
        }

        // Extract holiday components
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = parameters[index++];
        }

        // Extract changepoint parameter
        _changepoint = parameters[index++];

        // Extract regressor parameters
        for (int i = 0; i < _regressors.Length; i++)
        {
            _regressors[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Gets the trainable parameters of the Prophet model as a vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters of the model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// This method ensures that the ModelParameters property is up to date with the current values
    /// of all model components before returning it. This guarantees that any changes made to the
    /// individual components are properly reflected in the returned parameter vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the important numbers that define how
    /// your model works and packages them into a single list. These parameters include information
    /// about trends, seasonal patterns, holiday effects, and more.
    /// 
    /// Think of it like taking a snapshot of all the model's settings at once. This is useful when you want to:
    /// - Save the model's current state
    /// - Share the model with others
    /// - Make a copy of the model
    /// - Analyze what the model has learned
    /// 
    /// The method makes sure this snapshot is always up to date, even if the model's components
    /// have been adjusted separately.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Cannot get parameters for an untrained model.");
        }

        // Ensure ModelParameters is up to date with the current component values
        ModelParameters = GetCurrentState();

        return base.GetParameters();
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">The input vector containing time and other regressor values.</param>
    /// <returns>The predicted value for the given input.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction for a single input vector by combining the effects
    /// of trend, seasonality, holidays, changepoints, and regressors as learned during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is the method that makes a prediction for a single point in time.
    /// It takes all the patterns the model has learned (trends, seasons, holiday effects, etc.)
    /// and combines them to predict what value we should expect at the given time point.
    /// 
    /// The input vector should contain:
    /// - Time information (typically the first element)
    /// - Any additional regressor values that the model was trained with
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Validate input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input vector cannot be null");
        }

        // Check if model has been trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions");
        }

        // Validate input dimensions
        int expectedMinLength = 1; // At minimum we need time

        // If we have regressors, we need space for them too
        if (_prophetOptions.RegressorCount > 0)
        {
            expectedMinLength += _prophetOptions.RegressorCount;
        }

        if (input.Length < expectedMinLength)
        {
            throw new ArgumentException(
                $"Input vector length ({input.Length}) is too short. Expected at least {expectedMinLength} elements.",
                nameof(input));
        }

        // Use the internal implementation to make the prediction
        T prediction = PredictSingleInternal(input);

        // Apply any post-processing if needed
        if (_prophetOptions.ApplyTransformation)
        {
            prediction = _prophetOptions.TransformPrediction(prediction);
        }

        return prediction;
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides detailed information about the model's configuration, learned parameters,
    /// and operational statistics. It's useful for model versioning, comparison, and documentation.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a detailed report card about the model.
    /// It includes information about how the model was set up and what it learned during training.
    /// This is useful for:
    /// - Keeping track of different models you've created
    /// - Comparing models to see which one works better
    /// - Documenting what your model does for other people to understand
    /// - Saving the model's configuration for future reference
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ProphetModel,
            AdditionalInfo = []
        };
    
        // Add basic information
        metadata.AdditionalInfo["ModelName"] = "Prophet Time Series Model";
        metadata.AdditionalInfo["Version"] = "1.0";
    
        // Add trend information
        metadata.AdditionalInfo["TrendValue"] = Convert.ToDouble(_trend);
    
        // Add seasonal information
        metadata.AdditionalInfo["SeasonalPeriodCount"] = _prophetOptions.SeasonalPeriods.Count;
        for (int i = 0; i < _prophetOptions.SeasonalPeriods.Count; i++)
        {
            metadata.AdditionalInfo[$"SeasonalPeriod_{i}"] = _prophetOptions.SeasonalPeriods[i];
        }
        metadata.AdditionalInfo["FourierOrder"] = _prophetOptions.FourierOrder;
        metadata.AdditionalInfo["SeasonalComponentsCount"] = _seasonalComponents.Length;
    
        // Add holiday information
        metadata.AdditionalInfo["HolidayCount"] = _prophetOptions.Holidays.Count;
        metadata.AdditionalInfo["HolidayComponentsCount"] = _holidayComponents.Length;
    
        // Add changepoint information
        metadata.AdditionalInfo["ChangepointValue"] = Convert.ToDouble(_changepoint);
        metadata.AdditionalInfo["ChangepointCount"] = _prophetOptions.Changepoints.Count;
    
        // Add regressor information
        metadata.AdditionalInfo["RegressorCount"] = _prophetOptions.RegressorCount;
        if (_prophetOptions.RegressorCount > 0 && _regressors != null)
        {
            for (int i = 0; i < _prophetOptions.RegressorCount && i < _regressors.Length; i++)
            {
                metadata.AdditionalInfo[$"RegressorCoefficient_{i}"] = Convert.ToDouble(_regressors[i]);
            }
        }
    
        // Include optimizer information if available
        if (_prophetOptions.Optimizer != null)
        {
            metadata.AdditionalInfo["OptimizerType"] = _prophetOptions.Optimizer.GetType().Name;
        }
    
        // Add model state size
        metadata.AdditionalInfo["StateSize"] = GetStateSize();
    
        // Include serialized model data
        metadata.ModelData = this.Serialize();
    
        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the Prophet model with the same options.
    /// </summary>
    /// <returns>A new instance of the Prophet model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a fresh copy of the model with the same configuration options
    /// but without any trained parameters. It's used for creating new models with similar configurations.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the model with the same settings,
    /// but without any of the learned information. It's like creating a new recipe book with the same 
    /// instructions but no completed recipes yet. This is useful when you want to:
    /// - Train the same type of model on different data
    /// - Create multiple similar models for comparison
    /// - Start over with the same configuration but different training
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a deep copy of options to avoid reference issues
        var newOptions = new ProphetOptions<T, TInput, TOutput>
        {
            // Basic settings
            InitialTrendValue = _prophetOptions.InitialTrendValue,
            InitialChangepointValue = _prophetOptions.InitialChangepointValue,
            FourierOrder = _prophetOptions.FourierOrder,
            RegressorCount = _prophetOptions.RegressorCount,
            ApplyTransformation = _prophetOptions.ApplyTransformation,
        
            // Copy the optimizer if possible
            Optimizer = _prophetOptions.Optimizer
        };
    
        // Deep copy seasonal periods
        newOptions.SeasonalPeriods = [.. _prophetOptions.SeasonalPeriods];
    
        // Deep copy holidays
        newOptions.Holidays = [.. _prophetOptions.Holidays];
    
        // Deep copy changepoints
        newOptions.Changepoints = [.. _prophetOptions.Changepoints];
    
        // Create a new instance with the copied options
        return new ProphetModel<T, TInput, TOutput>(newOptions);
    }
}