namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements an Unobserved Components Model (UCM) for time series decomposition and forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// The Unobserved Components Model decomposes a time series into several distinct components:
/// trend, seasonal, cycle, and irregular components. It uses state-space modeling and Kalman filtering
/// to estimate these components, which can then be used for forecasting or understanding the
/// underlying patterns in the data.
/// </para>
/// 
/// <para>
/// For Beginners:
/// An Unobserved Components Model is like having X-ray vision for your time series data.
/// It helps you see the hidden patterns that make up your data by breaking it down into
/// several meaningful parts:
/// 
/// 1. Trend Component: The long-term direction of your data. Is it generally going up,
///    down, or staying level over time? This is like the "big picture" movement.
/// 
/// 2. Seasonal Component: Regular patterns that repeat at fixed intervals, such as
///    daily, weekly, monthly, or yearly cycles. For example, retail sales might
///    spike every December for holiday shopping.
/// 
/// 3. Cycle Component: Longer-term ups and downs that don't have a fixed period, often
///    related to business or economic cycles. Unlike seasonal patterns, these aren't tied
///    to the calendar and can vary in length and intensity.
/// 
/// 4. Irregular Component: The random "noise" or unexpected fluctuations that don't fit
///    into the other components. This captures events like unusual weather, one-time promotions,
///    or other unpredictable factors.
/// 
/// The model uses a mathematical technique called Kalman filtering (a bit like a sophisticated
/// version of moving averages) to separate these components from your data. Once separated,
/// you can examine each component individually to better understand what's driving your time series,
/// or recombine them to make forecasts.
/// 
/// This approach is particularly valuable because it:
/// - Helps you understand the "why" behind your data's behavior
/// - Allows you to forecast each component separately, improving accuracy
/// - Makes it easier to spot unusual patterns or anomalies
/// - Provides insights that simpler models might miss
/// </para>
/// </remarks>
public class UnobservedComponentsModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the Unobserved Components Model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These options control how the model works, such as whether to include
    /// a cycle component, how many seasonal periods to consider, how many iterations
    /// to run, and whether to optimize the model parameters automatically.
    /// </remarks>
    private readonly UnobservedComponentsOptions<T> _ucOptions;
    
    /// <summary>
    /// The estimated trend component of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This represents the long-term direction or movement in your data.
    /// Think of it as the "backbone" of your time series, showing where
    /// the data is generally headed over the long run, ignoring seasonal
    /// fluctuations and short-term noise.
    /// </remarks>
    private Vector<T> _trend;
    
    /// <summary>
    /// The previous iteration's trend estimates, used to check for convergence.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This keeps track of the trend estimates from the previous step in the
    /// algorithm. By comparing the current trend to this previous trend,
    /// the model can determine if it has "settled down" enough to stop
    /// its calculations (convergence).
    /// </remarks>
    private Vector<T> _previousTrend;
    
    /// <summary>
    /// The original time series data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This stores the actual data you're analyzing, like daily sales numbers
    /// or monthly temperature readings.
    /// </remarks>
    private Vector<T> _y;
    
    /// <summary>
    /// The estimated seasonal component of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This captures the regular, calendar-related patterns in your data.
    /// For example, retail sales might always peak during holidays, or
    /// energy usage might be higher during summer and winter months.
    /// These patterns repeat at fixed intervals (like weekly, monthly, or yearly).
    /// </remarks>
    private Vector<T> _seasonal;
    
    /// <summary>
    /// The estimated cycle component of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This represents longer-term ups and downs that aren't tied to the calendar.
    /// Unlike seasonal patterns, cycles don't have a fixed length and can vary
    /// in duration and intensity. Business cycles or economic booms and busts
    /// are examples of cyclical patterns.
    /// </remarks>
    private Vector<T> _cycle;
    
    /// <summary>
    /// The estimated irregular component of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This captures the random "noise" or unexpected fluctuations in your data
    /// that can't be explained by trend, seasonal, or cyclical patterns.
    /// These might be caused by one-time events, measurement errors, or
    /// truly random variations.
    /// </remarks>
    private Vector<T> _irregular;
    
    /// <summary>
    /// Fast Fourier Transform utility for frequency analysis.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is a mathematical tool that helps identify cyclical patterns in your data
    /// by breaking down the signal into different frequency components. It's like
    /// how a prism breaks white light into different colors - the FFT helps separate
    /// your time series into different cycle lengths.
    /// </remarks>
    private readonly FastFourierTransform<T> _fft;
    
    /// <summary>
    /// The state transition matrix for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This matrix describes how the different components (trend, seasonal, cycle)
    /// are expected to evolve from one time step to the next. It's like a set of
    /// rules that tell the model how each component should normally behave over time
    /// if no new information is observed.
    /// </remarks>
    private Matrix<T> _stateTransition;
    
    /// <summary>
    /// The observation model matrix for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This matrix defines how the hidden components combine to produce the observed
    /// values in your time series. It's like a recipe that tells the model which
    /// ingredients (components) to mix together and in what proportions to get the
    /// final dish (observed value).
    /// </remarks>
    private Matrix<T> _observationModel;
    
    /// <summary>
    /// The process noise covariance matrix for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This represents the uncertainty or randomness in how the components evolve
    /// over time. Higher values mean the model expects more random changes in
    /// the components, making it more responsive to new data but potentially
    /// less smooth.
    /// </remarks>
    private Matrix<T> _processNoise;
    
    /// <summary>
    /// The observation noise variance for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This represents the uncertainty or randomness in your measurements.
    /// Higher values mean the model treats the observed data as noisier or
    /// less reliable, putting more weight on its own internal understanding
    /// of how the components should behave.
    /// </remarks>
    private T _observationNoise;
    
    /// <summary>
    /// The current state vector for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This vector stores the current best estimates of all the components
    /// the model is tracking (like trend, seasonal factors, and cycle).
    /// It's like a snapshot of the model's current understanding of what's
    /// happening in your time series.
    /// </remarks>
    private Vector<T> _state;
    
    /// <summary>
    /// The current state covariance matrix for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This matrix represents how confident the model is about its estimates
    /// of each component, and how these uncertainties might be related.
    /// Lower values mean more confidence. As the model processes more data,
    /// this uncertainty typically decreases as the model becomes more sure
    /// of its estimates.
    /// </remarks>
    private Matrix<T> _stateCovariance;
    
    /// <summary>
    /// The threshold for determining when the model has converged.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This value determines when the model stops its iterations. When the
    /// changes between consecutive iterations become smaller than this threshold,
    /// the model considers its estimates "good enough" and stops further refinement.
    /// </remarks>
    private T _convergenceThreshold;
    
    /// <summary>
    /// Collection of filtered state vectors from the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This stores the model's estimates of the components at each time point,
    /// using only the data available up to that point. It's like making the
    /// best guess possible at each step without peeking into the future.
    /// </remarks>
    private List<Vector<T>> _filteredState;
    
    /// <summary>
    /// Collection of filtered state covariance matrices from the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This stores how confident the model is about its filtered estimates
    /// at each time point. These uncertainty measures are crucial for the
    /// smoother, which will later refine these estimates using future information.
    /// </remarks>
    private List<Matrix<T>> _filteredCovariance;

    /// <summary>
    /// Creates a new Unobserved Components Model with the specified options.
    /// </summary>
    /// <param name="options">Options for configuring the model. If null, default options are used.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor sets up a new Unobserved Components Model with your chosen settings.
    /// It initializes all the internal structures needed for the model to work.
    /// 
    /// You can customize:
    /// - Whether to include cyclical patterns
    /// - How many seasonal periods to use (e.g., 7 for weekly, 12 for monthly, 4 for quarterly)
    /// - How many iterations to run when fitting the model
    /// - Whether to automatically optimize the model parameters
    /// 
    /// If you don't provide options, the model will use reasonable default settings.
    /// </remarks>
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

    /// <summary>
    /// Trains the Unobserved Components Model on the provided data.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for UCM models).</param>
    /// <param name="y">Target vector (the time series values to be decomposed).</param>
    /// <remarks>
    /// For Beginners:
    /// This method is where the model learns from your data. It works through these steps:
    /// 
    /// 1. Initial Estimates: First, it makes educated guesses about the trend, seasonal,
    ///    and cycle components in your data using simple techniques.
    /// 
    /// 2. Kalman Filter: Then it uses a sophisticated statistical technique (Kalman filtering)
    ///    that works through your data point by point, constantly updating its understanding
    ///    of each component based on new observations.
    /// 
    /// 3. Smoothing: Next, it goes back through the data a second time, now using information
    ///    from both past and future to refine its estimates.
    /// 
    /// 4. Parameter Optimization (optional): Finally, it can fine-tune its internal settings
    ///    to better match your specific data patterns.
    /// 
    /// The process repeats until the estimates stabilize (converge) or until it reaches
    /// the maximum number of iterations. After training, the model will have separated
    /// your time series into its component parts, which can be used for analysis or prediction.
    /// </remarks>
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

    /// <summary>
    /// Initializes the trend, seasonal, cycle, and irregular components.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method creates the initial estimates for each component in your time series.
    /// These are educated guesses that give the model a starting point:
    /// 
    /// - For trend, it uses a simple moving average to capture the general direction
    /// - For seasonal components, it looks for patterns that repeat at regular intervals
    /// - For cycles, it identifies longer-term patterns that aren't tied to fixed periods
    /// - The irregular component is whatever remains after accounting for the other components
    /// 
    /// These initial estimates are rough approximations that will be refined during the 
    /// Kalman filtering and smoothing steps. Think of them as a first draft that gives
    /// the model something to work with before it starts its more sophisticated analysis.
    /// </remarks>
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

    /// <summary>
    /// Calculates a simple moving average of the data.
    /// </summary>
    /// <param name="data">The time series data.</param>
    /// <param name="windowSize">The size of the moving average window.</param>
    /// <returns>Vector of moving average values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates a moving average, which is a simple way to smooth out
    /// short-term fluctuations and highlight longer-term trends.
    /// 
    /// For each point in your data, it takes the average of that point and several surrounding
    /// points (determined by the windowSize). For example, with a window size of 7, each value
    /// would be the average of itself and the 6 preceding values.
    /// 
    /// This helps identify the general direction of your data by reducing the impact of
    /// day-to-day or other short-term variations. The result is a smoother line that
    /// approximates the trend component.
    /// </remarks>
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

    /// <summary>
    /// Initializes the seasonal component using averaging across periods.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="trend">The estimated trend component.</param>
    /// <returns>Vector of initial seasonal component values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method estimates the initial seasonal pattern in your data.
    /// 
    /// It works by:
    /// 1. Subtracting the trend from your data to get the "de-trended" series
    /// 2. Grouping these values by their position in the seasonal cycle (e.g., by day of week)
    /// 3. Calculating the average value for each position
    /// 4. Adjusting these averages so they sum to zero (ensuring the seasonal component
    ///    doesn't affect the overall level)
    /// 
    /// For example, if you have weekly seasonality, it might find that Mondays are typically
    /// 5% below average, while Saturdays are 10% above average. These patterns repeat
    /// every period (e.g., every week), creating the seasonal component.
    /// </remarks>
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

    /// <summary>
    /// Initializes the cycle component using filtering techniques.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <param name="trend">The estimated trend component.</param>
    /// <param name="seasonal">The estimated seasonal component.</param>
    /// <returns>Vector of initial cycle component values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method identifies longer-term cyclical patterns in your data after
    /// removing the trend and seasonal components.
    /// 
    /// It uses several steps:
    /// 1. First, it calculates what's left after removing trend and seasonal patterns
    /// 2. Then it applies a special filter (Hodrick-Prescott) to separate smooth changes from noise
    /// 3. Next, it uses a band-pass filter to focus on cycles within a specific length range
    /// 4. Finally, it normalizes the cycles to have a consistent scale
    /// 
    /// Cycles differ from seasonal patterns because they don't have fixed lengths -
    /// they might stretch or shrink over time. Think of business cycles with booms and busts
    /// that can last for varying periods, rather than calendar-based patterns like weekends or holidays.
    /// </remarks>
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

    /// <summary>
    /// Applies a Hodrick-Prescott filter to separate trend and cyclical components.
    /// </summary>
    /// <param name="data">The time series data to filter.</param>
    /// <param name="lambda">The smoothing parameter.</param>
    /// <returns>The filtered data.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method applies the Hodrick-Prescott filter, which is a common technique
    /// for separating a time series into trend and cyclical components.
    /// 
    /// The filter works like a special kind of smoothing that balances two goals:
    /// 1. Making the trend component fit the data reasonably well
    /// 2. Making the trend component smooth (not too wiggly)
    /// 
    /// The lambda parameter controls this balance:
    /// - Higher values make the trend smoother but potentially less accurate
    /// - Lower values make the trend follow the data more closely but potentially less smooth
    /// 
    /// This filter is often used in economics to separate business cycles from long-term trends,
    /// but it's useful for many types of time series data.
    /// </remarks>
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

    /// <summary>
    /// Applies a band-pass filter to isolate specific frequency components.
    /// </summary>
    /// <param name="data">The time series data to filter.</param>
    /// <param name="minPeriod">The minimum period length to include.</param>
    /// <param name="maxPeriod">The maximum period length to include.</param>
    /// <returns>The filtered data.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method acts like a specialized music equalizer for your time series.
    /// Just as an equalizer can isolate certain sound frequencies (like bass or treble),
    /// this filter isolates cyclical patterns within a specific range of lengths.
    /// 
    /// It works by:
    /// 1. Converting your time series from the time domain to the frequency domain using
    ///    a mathematical technique called Fast Fourier Transform (FFT)
    /// 2. Keeping only the frequencies that correspond to cycles between minPeriod and maxPeriod
    /// 3. Converting back to the time domain
    /// 
    /// For example, you might use this to focus on economic cycles that last between 2 and 8 years,
    /// while removing shorter fluctuations and longer trends.
    /// </remarks>
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

    /// <summary>
    /// Initializes the parameters for the Kalman filter.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method sets up the mathematical structures needed for the Kalman filter,
    /// which is the core algorithm that will refine our initial component estimates.
    /// 
    /// It creates:
    /// - A transition matrix that describes how components naturally evolve over time
    /// - An observation model that defines how components combine to form the observed values
    /// - Matrices that represent the uncertainties in the process and observations
    /// - Initial state vector and covariance matrix to start the filtering process
    /// - A convergence threshold that determines when the algorithm has found a stable solution
    /// 
    /// These technical parameters control how the Kalman filter works, balancing its
    /// responsiveness to new data against the smoothness of its component estimates.
    /// </remarks>
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

    /// <summary>
    /// Applies the Kalman filter to estimate state at each time point.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method applies the Kalman filter, which is a powerful algorithm
    /// for estimating hidden states from noisy measurements.
    /// 
    /// For each time point in your data, the Kalman filter:
    /// 1. Predicts what it expects the state to be, based on previous estimates and
    ///    how components naturally evolve (using the state transition matrix)
    /// 2. Compares this prediction to the actual observation
    /// 3. Updates its state estimate by balancing the prediction against the observation,
    ///    giving more weight to whichever it's more confident about
    /// 4. Updates its uncertainty estimates for the next time step
    /// 
    /// The Kalman filter is like a sophisticated GPS system that constantly updates
    /// its position estimate using both its movement model and sensor readings,
    /// accounting for the uncertainty in each.
    /// 
    /// The result is stored in _filteredState and _filteredCovariance for later use
    /// in the smoothing step.
    /// </remarks>
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

    /// <summary>
    /// Applies the Kalman smoother to refine state estimates using future information.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method applies the Kalman smoother, which further refines the estimates
    /// from the Kalman filter by incorporating information from future observations.
    /// 
    /// While the filter only uses data up to the current time point (making it suitable for
    /// real-time applications), the smoother uses the entire dataset to make better estimates.
    /// This is possible when analyzing historical data where future values are already known.
    /// 
    /// The smoother works backward through time:
    /// 1. It starts with the final state estimate from the filter
    /// 2. For each previous time step, it adjusts the filtered estimate based on:
    ///    - How that estimate propagated forward in time
    ///    - What we now know about later states
    /// 
    /// This is like refining your understanding of what happened yesterday based on what
    /// you observed today. The result is smoother, more accurate component estimates
    /// that make better use of all available information.
    /// </remarks>
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

    /// <summary>
    /// Updates the model components based on the smoothed state estimates.
    /// </summary>
    /// <param name="smoothedState">List of smoothed state vectors.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method extracts the individual trend, seasonal, and cycle components
    /// from the smoothed state estimates and updates the model's component vectors.
    /// 
    /// The smoothed state contains all these components bundled together, so this method:
    /// 1. Extracts each component into its own vector
    /// 2. Recalculates the irregular component as whatever is left over after
    ///    accounting for trend, seasonal, and cycle components
    /// 
    /// After this method completes, the model's _trend, _seasonal, _cycle, and _irregular
    /// vectors contain the final, refined estimates of each component based on all available
    /// information in the time series.
    /// </remarks>
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

    /// <summary>
    /// Checks if the model has converged based on changes in the trend component.
    /// </summary>
    /// <returns>True if the model has converged, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method determines when the model has "settled down" enough to stop
    /// the iterative estimation process.
    /// 
    /// It works by:
    /// 1. Calculating the maximum difference between the current trend estimates
    ///    and the trend estimates from the previous iteration
    /// 2. Comparing this maximum difference to a predefined threshold
    /// 3. If the difference is smaller than the threshold, the model is considered converged
    /// 
    /// Convergence means that further iterations wouldn't significantly change the component
    /// estimates, so the algorithm can stop. This saves computation time while still
    /// ensuring accurate results.
    /// 
    /// The method also updates _previousTrend for the next iteration's comparison.
    /// </remarks>
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

    /// <summary>
    /// Optimizes the model parameters to better fit the data.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">The time series data.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fine-tunes the model's parameters to better match your specific
    /// time series data. It's like adjusting the settings on a camera to get the clearest picture.
    /// 
    /// The optimization process:
    /// 1. Uses a mathematical optimization algorithm (by default, L-BFGS) to find the
    ///    parameter values that minimize prediction errors
    /// 2. Considers parameters like how variable the trend, seasonal, and cycle components are
    /// 3. Updates the model's internal parameters with these optimized values
    /// 4. Re-runs the Kalman filter and smoother with the new parameters
    /// 
    /// This optimization can significantly improve model performance, especially for
    /// complex time series with unique characteristics. However, it may take longer to compute.
    /// </remarks>
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

    /// <summary>
    /// Updates the model parameters based on optimized values.
    /// </summary>
    /// <param name="optimizedParameters">Vector of optimized parameter values.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method takes the optimized parameter values found during the parameter
    /// optimization step and applies them to the model's internal structures.
    /// 
    /// The parameters include:
    /// - Trend level and slope variances (how much the trend can change)
    /// - Seasonal variance (how much seasonal patterns can evolve)
    /// - Cycle period and variance (length and variability of cyclical patterns)
    /// - Irregular component variance (how much randomness to expect)
    /// 
    /// These parameters control the behavior of the state transition and noise covariance
    /// matrices, which in turn affect how the Kalman filter and smoother work.
    /// 
    /// After updating the parameters, the method re-runs the Kalman filter and smoother
    /// to generate revised component estimates based on these optimized settings.
    /// </remarks>
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

    /// <summary>
    /// Makes predictions using the trained model.
    /// </summary>
    /// <param name="input">Input matrix containing time indices for prediction.</param>
    /// <returns>Vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method generates predictions for future time points based on the decomposed
    /// components identified during training.
    /// 
    /// For each time point in the input:
    /// 1. It calls the PredictSingle method to generate a single prediction
    /// 2. It combines the trend, seasonal, cycle, and irregular components
    /// 
    /// The predictions incorporate all the patterns identified in your data, including:
    /// - The long-term trend direction
    /// - Seasonal patterns that repeat at fixed intervals
    /// - Cyclical patterns with varying periods
    /// - A reasonable amount of random variation
    /// 
    /// These comprehensive predictions reflect the full structure of your time series.
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
    /// Predicts a single value based on the time index.
    /// </summary>
    /// <param name="x">Vector containing the time index.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method generates a prediction for a single time point by combining
    /// all the decomposed components.
    /// 
    /// It:
    /// 1. Extracts the time index from the input (typically the first value)
    /// 2. Gets the trend, seasonal, cycle, and irregular components for that time index
    /// 3. Adds these components together to form the complete prediction
    /// 
    /// For example, a prediction might combine:
    /// - An upward trend component of 105
    /// - A seasonal component of +10 for this time of year
    /// - A cycle component of -5 for the current phase of the business cycle
    /// - An irregular component of +2
    /// 
    /// Resulting in a total prediction of 112.
    /// </remarks>
    private T PredictSingle(Vector<T> x)
    {
        int timeIndex = Convert.ToInt32(x[0]); // Assume the first column is the time index
        T prediction = NumOps.Add(_trend[timeIndex], _seasonal[timeIndex]);
        prediction = NumOps.Add(prediction, _cycle[timeIndex]);
        prediction = NumOps.Add(prediction, _irregular[timeIndex]);

        return prediction;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Test input matrix containing time indices.</param>
    /// <param name="yTest">Test target vector containing actual values.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// 
    /// - MAE (Mean Absolute Error): The average absolute difference between predictions and actual values.
    ///   Lower is better. If MAE = 5, your predictions are off by 5 units on average.
    /// 
    /// - MSE (Mean Squared Error): The average of squared differences between predictions and actual values.
    ///   Lower is better, but squaring emphasizes large errors. MSE is useful for optimization.
    /// 
    /// - RMSE (Root Mean Squared Error): The square root of MSE, which gives errors in the same units
    ///   as your original data. For example, if forecasting sales in dollars, RMSE is also in dollars.
    /// 
    /// - R² (R-squared): The proportion of variance in the dependent variable explained by the model.
    ///   Values range from 0 to 1, with higher values indicating better fit. An R² of 0.75 means
    ///   the model explains 75% of the variation in the data.
    /// 
    /// These metrics together provide a comprehensive assessment of model performance.
    /// </remarks>
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

    /// <summary>
    /// Serializes the model's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// For Beginners:
    /// This protected method saves the model's internal state to a file or stream.
    /// 
    /// Serialization allows you to:
    /// 1. Save a trained model to disk
    /// 2. Load it later without having to retrain
    /// 3. Share the model with others
    /// 
    /// The method saves the essential components of the model:
    /// - The decomposed trend component
    /// - The seasonal component
    /// - The cycle component
    /// - The irregular component
    /// - Key model options
    /// 
    /// This allows the model to be fully reconstructed later.
    /// </remarks>
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

    /// <summary>
    /// Deserializes the model's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// For Beginners:
    /// This protected method loads a previously saved model from a file or stream.
    /// 
    /// Deserialization allows you to:
    /// 1. Load a previously trained model
    /// 2. Use it immediately without retraining
    /// 3. Apply the exact same model to new data
    /// 
    /// The method loads:
    /// - The decomposed trend component
    /// - The seasonal component
    /// - The cycle component
    /// - The irregular component
    /// - Key model options
    /// 
    /// After deserialization, the model is ready to make predictions as if it had just been trained.
    /// </remarks>
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