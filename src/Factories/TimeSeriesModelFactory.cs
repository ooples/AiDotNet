global using AiDotNet.TimeSeries;

namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates time series models for forecasting and analysis.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series models are specialized algorithms that analyze data collected over time 
/// (like daily temperatures, monthly sales, or yearly population) to identify patterns and make predictions 
/// about future values.
/// </para>
/// <para>
/// This factory helps you create different types of time series models without needing to know their 
/// internal implementation details. Think of it like ordering a specific tool from a catalog - you just 
/// specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public class TimeSeriesModelFactory<T, TInput, TOutput>
{
    private static INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a time series model of the specified type with custom options.
    /// </summary>
    /// <param name="modelType">The type of time series model to create.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <returns>An implementation of ITimeSeriesModel<T> for the specified model type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported model type is specified or when the options don't match the model type.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a specific type of time series model with custom settings. 
    /// Different models are better suited for different types of time series data.
    /// </para>
    /// <para>
    /// Choose the model that best matches the patterns in your data. For example:
    /// - Use ARIMA for data with trends but no seasonality
    /// - Use SARIMA for data with both trends and seasonal patterns
    /// - Use Exponential Smoothing when recent observations are more important than older ones
    /// - Use Prophet for data with multiple seasonal patterns and holiday effects
    /// </para>
    /// <para>
    /// Note that you must provide options that match the model type. For example, if you specify 
    /// TimeSeriesModelType.ARIMA, you must provide ARIMAOptions<T>.
    /// </para>
    /// </remarks>
    public static ITimeSeriesModel<T> CreateModel(TimeSeriesModelType modelType, TimeSeriesRegressionOptions<T> options)
    {
        var inputData = DefaultInputCache.GetDefaultInputData<T, Matrix<T>, Vector<T>>();
        return (modelType, options) switch
        {
            (TimeSeriesModelType.ARIMA, ARIMAOptions<T> arimaOptions) => new ARIMAModel<T>(arimaOptions),
            (TimeSeriesModelType.SARIMA, SARIMAOptions<T> sarimaOptions) => new SARIMAModel<T>(sarimaOptions),
            (TimeSeriesModelType.ARMA, ARMAOptions<T> armaOptions) => new ARMAModel<T>(armaOptions),
            (TimeSeriesModelType.AR, ARModelOptions<T> arOptions) => new ARModel<T>(arOptions),
            (TimeSeriesModelType.MA, MAModelOptions<T> maOptions) => new MAModel<T>(maOptions),
            (TimeSeriesModelType.ExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) => 
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.SimpleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) => 
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.DoubleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) => 
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.TripleExponentialSmoothing, ExponentialSmoothingOptions<T> esOptions) => 
                new ExponentialSmoothingModel<T>(esOptions),
            (TimeSeriesModelType.StateSpace, StateSpaceModelOptions<T> ssOptions) => new StateSpaceModel<T>(ssOptions),
            (TimeSeriesModelType.TBATS, TBATSModelOptions<T> tbatsOptions) => new TBATSModel<T>(tbatsOptions),
            (TimeSeriesModelType.DynamicRegressionWithARIMAErrors, DynamicRegressionWithARIMAErrorsOptions<T> drOptions) => new DynamicRegressionWithARIMAErrors<T>(drOptions),
            (TimeSeriesModelType.ARIMAX, ARIMAXModelOptions<T> arimaxOptions) => new ARIMAXModel<T>(arimaxOptions),
            (TimeSeriesModelType.GARCH, GARCHModelOptions<T> garchOptions) => new GARCHModel<T>(garchOptions),
            (TimeSeriesModelType.VAR, VARModelOptions<T> varOptions) => new VectorAutoRegressionModel<T>(varOptions),
            (TimeSeriesModelType.VARMA, VARMAModelOptions<T> varmaOptions) => new VARMAModel<T>(varmaOptions),
            (TimeSeriesModelType.ProphetModel, ProphetOptions<T, TInput, TOutput> prophetOptions) => CreateProphetModel(inputData.XTrain, inputData.YTrain),
            (TimeSeriesModelType.NeuralNetworkARIMA, NeuralNetworkARIMAOptions<T> nnarimaOptions) => new NeuralNetworkARIMAModel<T>(nnarimaOptions),
            (TimeSeriesModelType.BayesianStructuralTimeSeriesModel, BayesianStructuralTimeSeriesOptions<T> bstsOptions) => new BayesianStructuralTimeSeriesModel<T>(bstsOptions),
            (TimeSeriesModelType.SpectralAnalysis, SpectralAnalysisOptions<T> saOptions) => new SpectralAnalysisModel<T>(saOptions),
            (TimeSeriesModelType.STLDecomposition, STLDecompositionOptions<T> stlOptions) => new STLDecomposition<T>(stlOptions),
            (TimeSeriesModelType.InterventionAnalysis, InterventionAnalysisOptions<T, Matrix<T>, Vector<T>> iaOptions) => new InterventionAnalysisModel<T>(iaOptions),
            (TimeSeriesModelType.TransferFunctionModel, TransferFunctionOptions<T, Matrix<T>, Vector<T>> tfOptions) => new TransferFunctionModel<T>(tfOptions),
            (TimeSeriesModelType.UnobservedComponentsModel, UnobservedComponentsOptions<T, TInput, TOutput> ucOptions) => new UnobservedComponentsModel<T, TInput, TOutput>(ucOptions),
            _ => throw new ArgumentException($"Unsupported model type or invalid options: {modelType}")
        };
    }

    /// <summary>
    /// Creates a simplified ProphetModel with auto-configured settings based on the data.
    /// </summary>
    /// <param name="timeSeriesData">The input time series data matrix (first column is time).</param>
    /// <param name="targetValues">The target values to predict.</param>
    /// <param name="forecastHorizon">The number of time steps to forecast.</param>
    /// <returns>A configured ProphetModel ready for training.</returns>
    public static ITimeSeriesModel<T> CreateProphetModel(
        Matrix<T> timeSeriesData,
        Vector<T> targetValues, int forecastHorizon = 7)
    {
        // Automatically analyze data to find appropriate settings
        var options = new ProphetOptions<T, Matrix<T>, Vector<T>>
        {
            // Auto-detect trend from data
            InitialTrendValue = Convert.ToDouble(DetectInitialTrend(timeSeriesData, targetValues)),

            // Auto-detect seasonal periods
            SeasonalPeriods = DetectSeasonalPeriods(timeSeriesData, targetValues),

            // Auto-generate reasonable changepoints
            Changepoints = GenerateDefaultChangepoints(timeSeriesData),

            // Set reasonable defaults for other options
            FourierOrder = 3,
            InitialChangepointValue = Convert.ToDouble(EstimateChangepointValue(timeSeriesData, targetValues)),
            ForecastHorizon = forecastHorizon
        };

        return (ITimeSeriesModel<T>)new ProphetModel<T, Matrix<T>, Vector<T>>(options);
    }

    /// <summary>
    /// Detects if the data likely has a yearly pattern.
    /// </summary>
    private static bool DetectYearlyPattern(Matrix<T> timeData, Vector<T> values)
    {
        // Need at least 2 years of data
        if (values.Length < 730)
            return false;

        T yearlyCorrelation = TimeSeriesHelper<T>.CalculateAutoCorrelation(values, 365);
        return _numOps.GreaterThan(yearlyCorrelation, _numOps.FromDouble(0.3)); // Threshold for detection
    }

    /// <summary>
    /// Detects if the data likely has a monthly pattern.
    /// </summary>
    private static bool DetectMonthlyPattern(Matrix<T> timeData, Vector<T> values)
    {
        // Need at least 2 months of data
        if (values.Length < 60)
            return false;

        T monthlyCorrelation = TimeSeriesHelper<T>.CalculateAutoCorrelation(values, 30);
        return _numOps.GreaterThan(monthlyCorrelation, _numOps.FromDouble(0.3)); // Threshold for detection
    }

    /// <summary>
    /// Detects if the data likely has a weekly pattern.
    /// </summary>
    private static bool DetectWeeklyPattern(Matrix<T> timeData, Vector<T> values)
    {
        // Simple correlation test for weekly pattern
        if (values.Length < 14) // Need at least 2 weeks of data
            return false;

        T weeklyCorrelation = TimeSeriesHelper<T>.CalculateAutoCorrelation(values, 7);
        return _numOps.GreaterThan(weeklyCorrelation, _numOps.FromDouble(0.3)); // Threshold for detection
    }

    /// <summary>
    /// Detects the initial trend value from the data, making it easy for beginners.
    /// </summary>
    private static T DetectInitialTrend(Matrix<T> timeData, Vector<T> values)
    {
        // Use the first few data points to estimate trend
        T startValue;

        if (values.Length >= 10)
        {
            // Use average of first 10 points for stability
            startValue = _numOps.Zero;
            for (int i = 0; i < 10; i++)
            {
                startValue = _numOps.Add(startValue, values[i]);
            }
            startValue = _numOps.Divide(startValue, _numOps.FromDouble(10));
        }
        else if (values.Length > 0)
        {
            // Use first point if we don't have enough data
            startValue = values[0];
        }
        else
        {
            // Fallback
            startValue = _numOps.Zero;
        }

        Console.WriteLine($"Auto-detected initial trend: {startValue}");
        return startValue;
    }

    /// <summary>
    /// Automatically detects likely seasonal periods in the data.
    /// </summary>
    private static List<int> DetectSeasonalPeriods(Matrix<T> timeData, Vector<T> values)
    {
        var periods = new List<int>();

        // Check if we have enough data for seasonality detection
        if (values.Length < 14)
        {
            // Not enough data for reliable detection
            return periods;
        }

        // Check common periods

        // Daily data often has weekly seasonality (7 days)
        bool weeklyPattern = DetectWeeklyPattern(timeData, values);
        if (weeklyPattern)
        {
            periods.Add(7);
            Console.WriteLine("Auto-detected weekly (7-day) seasonality");
        }

        // Monthly pattern (around 30 days)
        bool monthlyPattern = DetectMonthlyPattern(timeData, values);
        if (monthlyPattern)
        {
            periods.Add(30);
            Console.WriteLine("Auto-detected monthly (30-day) seasonality");
        }

        // Yearly pattern (365/366 days)
        bool yearlyPattern = DetectYearlyPattern(timeData, values);
        if (yearlyPattern)
        {
            periods.Add(365);
            Console.WriteLine("Auto-detected yearly (365-day) seasonality");
        }

        // If no patterns detected, suggest common patterns based on data size
        if (periods.Count == 0)
        {
            if (values.Length >= 14 && values.Length < 60)
            {
                // For short series, suggest weekly pattern
                periods.Add(7);
                Console.WriteLine("Suggesting weekly (7-day) seasonality based on data length");
            }
            else if (values.Length >= 60)
            {
                // For longer series, suggest both weekly and monthly
                periods.Add(7);
                periods.Add(30);
                Console.WriteLine("Suggesting weekly (7-day) and monthly (30-day) seasonality based on data length");
            }
        }

        return periods;
    }

    /// <summary>
    /// Generates reasonable default changepoints spaced throughout the data.
    /// </summary>
    private static List<T> GenerateDefaultChangepoints(Matrix<T> timeData)
    {
        var changepoints = new List<T>();

        // If we don't have enough data, don't add changepoints
        if (timeData.Rows < 20)
        {
            return changepoints;
        }

        // Get the time range
        T startTime = timeData[0, 0];
        T endTime = timeData[timeData.Rows - 1, 0];
        T timeRange = _numOps.Subtract(endTime, startTime);

        // Create changepoints avoiding the very beginning and end
        T padding = _numOps.Multiply(timeRange, _numOps.FromDouble(0.05)); // 5% padding
        T effectiveStart = _numOps.Add(startTime, padding);
        T effectiveEnd = _numOps.Subtract(endTime, padding);
        T effectiveRange = _numOps.Subtract(effectiveEnd, effectiveStart);

        // Determine reasonable number of changepoints based on data size
        int numChangepoints;
        if (timeData.Rows < 50)
            numChangepoints = 3;
        else if (timeData.Rows < 100)
            numChangepoints = 5;
        else if (timeData.Rows < 200)
            numChangepoints = 10;
        else
            numChangepoints = 15;

        // Generate evenly spaced changepoints
        for (int i = 1; i <= numChangepoints; i++)
        {
            T position = _numOps.Add(
                effectiveStart,
                _numOps.Multiply(
                    _numOps.FromDouble(i),
                    _numOps.Divide(effectiveRange, _numOps.FromDouble(numChangepoints + 1))
                )
            );
            changepoints.Add(position);
        }

        Console.WriteLine($"Auto-generated {numChangepoints} changepoints");
        return changepoints;
    }

    /// <summary>
    /// Estimates a reasonable changepoint value based on data trends.
    /// </summary>
    private static T EstimateChangepointValue(Matrix<T> timeData, Vector<T> values)
    {
        if (values.Length < 10)
            return _numOps.FromDouble(0.01); // Default for very short series

        // Calculate average rate of change
        T firstAvg = _numOps.Zero;
        T lastAvg = _numOps.Zero;
        int sampleSize = Math.Min(5, values.Length / 4); // Use 5 points or 1/4 of data

        // Average of first few points
        for (int i = 0; i < sampleSize; i++)
            firstAvg = _numOps.Add(firstAvg, values[i]);
        firstAvg = _numOps.Divide(firstAvg, _numOps.FromDouble(sampleSize));

        // Average of last few points
        for (int i = 0; i < sampleSize; i++)
            lastAvg = _numOps.Add(lastAvg, values[values.Length - 1 - i]);
        lastAvg = _numOps.Divide(lastAvg, _numOps.FromDouble(sampleSize));

        // Time difference
        T timeDiff = _numOps.Subtract(timeData[timeData.Rows - 1, 0], timeData[0, 0]);

        // If time difference is zero or very small, return a small default value
        if (_numOps.LessThan(timeDiff, _numOps.FromDouble(1)))
            return _numOps.FromDouble(0.01);

        // Calculate slope
        T slope = _numOps.Divide(_numOps.Subtract(lastAvg, firstAvg), timeDiff);

        // Scale the slope for changepoint value (typical trend change rate)
        T changeValue = _numOps.Multiply(_numOps.Abs(slope), _numOps.FromDouble(0.1));

        // Bound the value to reasonable limits
        changeValue = _numOps.LessThan(changeValue, _numOps.FromDouble(0.001)) ?
            _numOps.FromDouble(0.001) : changeValue;

        // Then ensure it's at most 0.5
        changeValue = _numOps.GreaterThan(changeValue, _numOps.FromDouble(0.5)) ?
            _numOps.FromDouble(0.5) : changeValue;

        Console.WriteLine($"Auto-estimated changepoint value: {changeValue}");
        return changeValue;
    }
}