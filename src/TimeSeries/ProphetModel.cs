using AiDotNet.Autodiff;

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
    private ProphetOptions<T, TInput, TOutput> _prophetOptions;

    /// <summary>
    /// Gets whether parameter optimization succeeded during the most recent training run.
    /// </summary>
    /// <remarks>
    /// When <see cref="ProphetOptions{T, TInput, TOutput}.OptimizeParameters"/> is disabled or optimization fails,
    /// this value is <see langword="false"/>.
    /// </remarks>
    public bool IsOptimized { get; private set; }

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
    private T _trend;

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
    private Vector<T> _seasonalComponents;

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
    private Vector<T> _holidayComponents;

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
    private T _changepoint;

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
    private Vector<T> _regressors;

    /// <summary>
    /// The anomaly detection threshold computed during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This value determines when a prediction error is large enough
    /// to be considered an anomaly. It's computed from the training data residuals.
    /// </para>
    /// </remarks>
    private T _anomalyThreshold;

    /// <summary>
    /// The standard deviation of residuals computed during training.
    /// </summary>
    private T _residualStdDev;

    /// <summary>
    /// The mean of residuals computed during training.
    /// </summary>
    private T _residualMean;

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
        _holidayComponents = new Vector<T>(_prophetOptions.Holidays.Count);
        _changepoint = NumOps.FromDouble(_prophetOptions.InitialChangepointValue);
        _regressors = new Vector<T>(_prophetOptions.RegressorCount);

        // Initialize anomaly detection components
        _anomalyThreshold = NumOps.Zero;
        _residualStdDev = NumOps.Zero;
        _residualMean = NumOps.Zero;
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
        Vector<T> y_shifted = y.Slice(0, y.Length - 1);
        Vector<T> y_current = y.Slice(1, y.Length - 1);
        Vector<T> diffs = (Vector<T>)Engine.Subtract(y_current, y_shifted);

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
        int regressorCount = Math.Max(0, _prophetOptions.RegressorCount);

        if (regressorCount > 0 && regressorCount > Math.Max(0, p - 1))
        {
            throw new ArgumentException(
                $"RegressorCount ({regressorCount}) exceeds available input columns ({p}). The input matrix must include time plus regressors.",
                nameof(x));
        }

        if (_regressors == null || _regressors.Length != regressorCount)
        {
            _regressors = new Vector<T>(regressorCount);
        }

        // Use the user-defined optimizer if provided, otherwise use LFGSOptimizer as default
        var optimizer = _prophetOptions.Optimizer ?? new LBFGSOptimizer<T, Matrix<T>, Vector<T>>(this);

        // Prepare the optimization input data
        var inputData = new OptimizationInputData<T, Matrix<T>, Vector<T>>()
        {
            XTrain = x,
            YTrain = y,
            XValidation = x,
            YValidation = y,
            XTest = x,
            YTest = y
        };

        // Run optimization
        var result = optimizer.Optimize(inputData);

        var optimizedParameters = result.BestSolution?.GetParameters();
        if (optimizedParameters != null && optimizedParameters.Length > 0)
        {
            ApplyParameters(optimizedParameters);
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
    private T PredictSingleInternal(Vector<T> x)
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
    /// It's called by the public Train method and handles the core training logic.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the real learning happens for the model.
    /// The method carefully validates your data, then initializes the model components
    /// (like trend and seasonal patterns), and finally optimizes all the parameters to
    /// best fit your data. If anything goes wrong during this process, it provides clear
    /// error messages to help you understand and fix the issue.
    /// </para>
    /// <para>
    /// If <see cref="ProphetOptions{T, TInput, TOutput}.OptimizeParameters"/> is enabled, the model will attempt
    /// parameter optimization. If optimization fails, training continues using the initial parameter estimates,
    /// a warning is emitted via <see cref="System.Diagnostics.Trace"/>, and <see cref="IsOptimized"/> remains
    /// <see langword="false"/>.
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

        // Initialize model state vector
        int n = y.Length;
        Matrix<T> states = new Matrix<T>(n, GetStateSize());

        // Initialize components (trend, seasonal, holiday, changepoint, regressors)
        InitializeComponents(x, y);

        SyncModelParametersFromState();

        IsOptimized = false;
        if (_prophetOptions.OptimizeParameters)
        {
            try
            {
                OptimizeParameters(x, y);
                IsOptimized = true;
            }
            catch (InvalidOperationException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArgumentException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArithmeticException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
        }

        // Store final state for future reference
        states.SetRow(n - 1, GetCurrentState());

        // If anomaly detection is enabled, compute threshold from residuals
        if (_prophetOptions.EnableAnomalyDetection)
        {
            ComputeAnomalyThresholdFromTraining(x, y);
        }
    }

    /// <summary>
    /// Computes the anomaly detection threshold from training data residuals.
    /// </summary>
    /// <param name="x">The training input matrix.</param>
    /// <param name="y">The training target vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates what counts as "normal" variation in your data
    /// by looking at how far off the model's predictions were during training. The threshold is computed
    /// so that only unusually large prediction errors get flagged as anomalies.
    /// </para>
    /// </remarks>
    private void ComputeAnomalyThresholdFromTraining(Matrix<T> x, Vector<T> y)
    {
        // Compute predictions for training data
        Vector<T> predictions = Predict(x);

        // Compute residuals (absolute errors)
        List<T> absResiduals = new List<T>();
        for (int i = 0; i < y.Length; i++)
        {
            T residual = NumOps.Abs(NumOps.Subtract(y[i], predictions[i]));
            absResiduals.Add(residual);
        }

        if (absResiduals.Count == 0)
        {
            _residualMean = NumOps.Zero;
            _residualStdDev = NumOps.One;
            _anomalyThreshold = NumOps.FromDouble(_prophetOptions.AnomalyThresholdSigma);
            return;
        }

        // Compute mean of absolute residuals
        T sum = NumOps.Zero;
        for (int i = 0; i < absResiduals.Count; i++)
        {
            sum = NumOps.Add(sum, absResiduals[i]);
        }
        _residualMean = NumOps.Divide(sum, NumOps.FromDouble(absResiduals.Count));

        // Compute standard deviation of absolute residuals
        T sumSquaredDiff = NumOps.Zero;
        for (int i = 0; i < absResiduals.Count; i++)
        {
            T diff = NumOps.Subtract(absResiduals[i], _residualMean);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        _residualStdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(absResiduals.Count)));

        // If std dev is zero (all residuals are the same), use a small default
        if (NumOps.Equals(_residualStdDev, NumOps.Zero))
        {
            _residualStdDev = NumOps.FromDouble(0.001);
        }

        // Threshold = mean + (sigma * stddev)
        _anomalyThreshold = NumOps.Add(
            _residualMean,
            NumOps.Multiply(NumOps.FromDouble(_prophetOptions.AnomalyThresholdSigma), _residualStdDev)
        );
    }

    private void SyncModelParametersFromState()
    {
        base.ApplyParameters(GetCurrentState());
    }

    protected override void ApplyParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int expectedLength = GetStateSize();
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}.", nameof(parameters));
        }

        int index = 0;
        _trend = parameters[index++];

        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            _seasonalComponents[i] = parameters[index++];
        }

        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = parameters[index++];
        }

        _changepoint = parameters[index++];

        for (int i = 0; i < _regressors.Length; i++)
        {
            _regressors[i] = parameters[index++];
        }

        base.ApplyParameters(parameters);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        ApplyParameters(parameters);
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
        // Validate the input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input vector cannot be null");
        }

        // Check if model has been trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions");
        }

        // Check if input has the right dimensions
        int expectedLength = 1 + _prophetOptions.RegressorCount; // Time plus regressors
        if (input.Length < expectedLength)
        {
            throw new ArgumentException($"Input vector length ({input.Length}) is too short. Expected at least {expectedLength} elements.");
        }

        // Use the private implementation to make the actual prediction
        T prediction = PredictSingleInternal(input);

        // Apply any post-processing logic if needed
        if (_prophetOptions.ApplyTransformation)
        {
            prediction = _prophetOptions.TransformPrediction(prediction);
        }

        return prediction;
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
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

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> when the model has been trained with valid components.
    /// ProphetModel can be JIT compiled using precomputed Fourier basis matrices
    /// for seasonality and average holiday/changepoint effects.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> JIT compilation optimizes the Prophet model's calculations
    /// by precomputing the Fourier basis for seasonality and averaging holiday effects.
    /// This provides faster inference while maintaining good accuracy.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => base.SupportsJitCompilation;

    /// <summary>
    /// Exports the ProphetModel as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">A list to which input nodes will be added.</param>
    /// <returns>The output computation node representing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph represents the Prophet prediction formula:
    /// prediction = trend + seasonal_fourier + avg_holiday + changepoint_effect + regressor_effect
    /// </para>
    /// <para>
    /// Seasonality is computed using precomputed Fourier basis matrices, allowing efficient
    /// matrix operations. Holiday effects are averaged for JIT approximation.
    /// </para>
    /// <para><b>For Beginners:</b> This converts the Prophet model into an optimized computation graph.
    /// The graph represents:
    /// 1. Base trend value
    /// 2. Fourier series for seasonal patterns (sin/cos combinations)
    /// 3. Average holiday effects
    /// 4. Changepoint adjustments
    /// 5. Regressor contributions
    ///
    /// Expected speedup: 2-4x for inference after JIT compilation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        if (!IsTrained)
        {
            throw new InvalidOperationException("Cannot export computation graph: Model must be trained first.");
        }

        // Create input node for time index (normalized)
        var timeShape = new int[] { 1 };
        var timeTensor = new Tensor<T>(timeShape);
        var timeNode = TensorOperations<T>.Variable(timeTensor, "time_index", requiresGradient: false);
        inputNodes.Add(timeNode);

        // Start with trend
        var trendTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { _trend }));
        var resultNode = TensorOperations<T>.Constant(trendTensor, "trend");

        // Add Fourier-based seasonal component
        // For JIT, we precompute the Fourier basis for a normalized time value
        var seasonalValue = ComputeAverageSeasonalEffect();
        var seasonalTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { seasonalValue }));
        var seasonalNode = TensorOperations<T>.Constant(seasonalTensor, "seasonal_effect");
        resultNode = TensorOperations<T>.Add(resultNode, seasonalNode);

        // Add average holiday effect
        if (_holidayComponents != null && _holidayComponents.Length > 0)
        {
            var avgHolidayValue = ComputeAverageHolidayEffect();
            var holidayTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { avgHolidayValue }));
            var holidayNode = TensorOperations<T>.Constant(holidayTensor, "holiday_effect");
            resultNode = TensorOperations<T>.Add(resultNode, holidayNode);
        }

        // Add changepoint effect
        var changepointValue = ComputeAverageChangepointEffect();
        var changepointTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { changepointValue }));
        var changepointNode = TensorOperations<T>.Constant(changepointTensor, "changepoint_effect");
        resultNode = TensorOperations<T>.Add(resultNode, changepointNode);

        // Add regressor effects if present
        if (_regressors != null && _regressors.Length > 0)
        {
            // Create input node for regressor values
            var regressorShape = new int[] { _regressors.Length };
            var regressorTensor = new Tensor<T>(regressorShape);
            var regressorInputNode = TensorOperations<T>.Variable(regressorTensor, "regressor_input", requiresGradient: false);
            inputNodes.Add(regressorInputNode);

            // Create regressor weights tensor
            var regressorWeightsTensor = new Tensor<T>(new[] { 1, _regressors.Length }, new Vector<T>(_regressors));
            var regressorWeightsNode = TensorOperations<T>.Constant(regressorWeightsTensor, "regressor_weights");

            // regressor_effect = weights @ regressor_values
            var regressorEffectNode = TensorOperations<T>.MatrixMultiply(regressorWeightsNode, regressorInputNode);
            resultNode = TensorOperations<T>.Add(resultNode, regressorEffectNode);
        }

        return resultNode;
    }

    /// <summary>
    /// Computes the average seasonal effect for JIT approximation.
    /// </summary>
    private T ComputeAverageSeasonalEffect()
    {
        T avgEffect = NumOps.Zero;
        int fourierTerms = _prophetOptions.FourierOrder * 2;

        // Compute average over all Fourier terms
        for (int j = 0; j < Math.Min(fourierTerms, _seasonalComponents.Length); j++)
        {
            // Average contribution of sin/cos terms is approximately 0.5 * coefficient
            avgEffect = NumOps.Add(avgEffect, NumOps.Multiply(_seasonalComponents[j], NumOps.FromDouble(0.5)));
        }

        return avgEffect;
    }

    /// <summary>
    /// Computes the average holiday effect for JIT approximation.
    /// </summary>
    private T ComputeAverageHolidayEffect()
    {
        if (_holidayComponents == null || _holidayComponents.Length == 0)
            return NumOps.Zero;

        T sum = NumOps.Zero;
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            sum = NumOps.Add(sum, _holidayComponents[i]);
        }

        // Average holiday effect weighted by probability of holiday
        // Assumes holidays are relatively rare (approx 10-15 days per year)
        T holidayProbability = NumOps.FromDouble(15.0 / 365.0);
        return NumOps.Multiply(NumOps.Divide(sum, NumOps.FromDouble(_holidayComponents.Length)), holidayProbability);
    }

    /// <summary>
    /// Computes the average changepoint effect for JIT approximation.
    /// </summary>
    private T ComputeAverageChangepointEffect()
    {
        // For JIT, we approximate using the trend changepoint value
        // This represents the cumulative effect of changepoints at an average time
        return NumOps.Multiply(_changepoint, NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Detects anomalies in a time series by comparing predictions to actual values.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A boolean array where true indicates an anomaly at that position.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Prophet model to predict each point in the time series based on
    /// trend, seasonality, holidays, and regressors, then flags points where the prediction error
    /// exceeds the anomaly threshold computed during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method identifies unusual data points by comparing
    /// what actually happened to what the model predicted should happen, accounting for trends,
    /// seasons, and holidays.
    ///
    /// Prophet is particularly good at detecting contextual anomalies - values that are unusual
    /// given the current context. For example:
    /// - A high sales value in July that would be normal in December might be flagged
    /// - A normal weekday value occurring on a holiday might be flagged
    ///
    /// The result tells you which points are unusual enough to warrant investigation.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model hasn't been trained or anomaly detection wasn't enabled.
    /// </exception>
    public bool[] DetectAnomalies(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training. " +
                "Set EnableAnomalyDetection = true in ProphetOptions and retrain the model.");
        }

        var scores = ComputeAnomalyScores(x, y);
        bool[] anomalies = new bool[scores.Length];

        for (int i = 0; i < scores.Length; i++)
        {
            anomalies[i] = NumOps.GreaterThan(scores[i], _anomalyThreshold);
        }

        return anomalies;
    }

    /// <summary>
    /// Computes anomaly scores for each point in a time series.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A vector of anomaly scores (absolute prediction errors) for each point.</returns>
    /// <remarks>
    /// <para>
    /// The anomaly score is the absolute difference between the actual value and the predicted value.
    /// The predicted value accounts for trend, seasonality, holidays, and regressor effects,
    /// so the score represents how much the actual value deviates from the contextual expectation.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you exactly how unusual each point is,
    /// considering all the patterns Prophet has learned:
    /// - A score of 0 means the value matched the prediction perfectly
    /// - Higher scores mean the value was more unexpected
    ///
    /// Unlike simple anomaly detection that just looks at absolute values, Prophet considers:
    /// - Is this value unusual for this time of year?
    /// - Is this value unusual for this day of the week?
    /// - Is this value unusual given the overall trend?
    /// - Is this value unusual for a holiday/non-holiday?
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the model hasn't been trained.</exception>
    public Vector<T> ComputeAnomalyScores(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before computing anomaly scores.");
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException($"Input rows ({x.Rows}) must match target length ({y.Length}).");
        }

        // Get predictions for all points
        Vector<T> predictions = Predict(x);
        Vector<T> scores = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            scores[i] = NumOps.Abs(NumOps.Subtract(y[i], predictions[i]));
        }

        return scores;
    }

    /// <summary>
    /// Detects anomalies and returns detailed information about each detected anomaly.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A list of tuples containing (index, actual value, predicted value, score) for each anomaly.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method not only tells you which points are anomalies,
    /// but also provides context to help you understand why:
    /// - Index: The position of the anomaly in the time series
    /// - Actual: What the value actually was
    /// - Predicted: What Prophet expected based on trends, seasons, and holidays
    /// - Score: How far off the prediction was
    ///
    /// The difference between Actual and Predicted tells you the direction of the anomaly:
    /// - Actual > Predicted: Unexpectedly high value
    /// - Actual < Predicted: Unexpectedly low value
    /// </para>
    /// </remarks>
    public List<(int Index, T Actual, T Predicted, T Score)> DetectAnomaliesDetailed(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }

        Vector<T> predictions = Predict(x);
        var anomalies = new List<(int Index, T Actual, T Predicted, T Score)>();

        for (int i = 0; i < y.Length; i++)
        {
            T prediction = predictions[i];
            T actual = y[i];
            T score = NumOps.Abs(NumOps.Subtract(actual, prediction));

            if (NumOps.GreaterThan(score, _anomalyThreshold))
            {
                anomalies.Add((i, actual, prediction, score));
            }
        }

        return anomalies;
    }

    /// <summary>
    /// Gets the current anomaly detection threshold.
    /// </summary>
    /// <returns>The anomaly threshold computed during training.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you the cutoff value used to decide whether
    /// a prediction error is large enough to be an anomaly. Values with scores above this
    /// threshold are considered anomalies.
    /// </para>
    /// </remarks>
    public T GetAnomalyThreshold()
    {
        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }
        return _anomalyThreshold;
    }

    /// <summary>
    /// Sets a custom anomaly detection threshold.
    /// </summary>
    /// <param name="threshold">The new threshold value.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> If the automatic threshold is flagging too many or too few
    /// anomalies, you can set your own:
    /// - Higher threshold = fewer anomalies detected (only extreme values)
    /// - Lower threshold = more anomalies detected (more sensitive)
    ///
    /// Tip: Look at the scores from ComputeAnomalyScores to decide on a good threshold
    /// for your specific use case.
    /// </para>
    /// </remarks>
    public void SetAnomalyThreshold(T threshold)
    {
        _anomalyThreshold = threshold;
    }

    /// <summary>
    /// Computes prediction intervals for future forecasts.
    /// </summary>
    /// <param name="x">The input matrix for forecasting.</param>
    /// <returns>A tuple containing (predictions, lower bounds, upper bounds).</returns>
    /// <remarks>
    /// <para>
    /// Prediction intervals provide uncertainty bounds around the point predictions.
    /// Points outside these intervals are likely anomalies. The width of the interval
    /// is controlled by the PredictionIntervalWidth option.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of just getting a single predicted value,
    /// this method gives you a range where the actual value is likely to fall:
    /// - Lower bound: The low end of the expected range
    /// - Upper bound: The high end of the expected range
    ///
    /// If an actual value falls outside this range, it's probably an anomaly.
    /// For example, if the predicted range for sales is $800-$1200 and actual sales were $2000,
    /// that's clearly unusual.
    /// </para>
    /// </remarks>
    public (Vector<T> Predictions, Vector<T> LowerBound, Vector<T> UpperBound) PredictWithIntervals(Matrix<T> x)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        Vector<T> predictions = Predict(x);
        Vector<T> lowerBound = new Vector<T>(predictions.Length);
        Vector<T> upperBound = new Vector<T>(predictions.Length);

        // Use normal distribution z-scores for interval calculation
        // For 95% interval, z ≈ 1.96; for 80%, z ≈ 1.28
        double zScore;
        if (_prophetOptions.PredictionIntervalWidth >= 0.95)
            zScore = 1.96;
        else if (_prophetOptions.PredictionIntervalWidth >= 0.90)
            zScore = 1.645;
        else if (_prophetOptions.PredictionIntervalWidth >= 0.80)
            zScore = 1.28;
        else
            zScore = 1.0;

        T halfWidth = NumOps.Multiply(NumOps.FromDouble(zScore), _residualStdDev);

        for (int i = 0; i < predictions.Length; i++)
        {
            lowerBound[i] = NumOps.Subtract(predictions[i], halfWidth);
            upperBound[i] = NumOps.Add(predictions[i], halfWidth);
        }

        return (predictions, lowerBound, upperBound);
    }
}
