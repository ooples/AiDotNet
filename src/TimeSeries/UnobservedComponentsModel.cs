using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Extensions;

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
/// <example>
/// <code>
/// var trainingMatrix = new Matrix&lt;double&gt;(new double[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } });
/// var trainingLabels = new Vector&lt;double&gt;(new double[] { 112, 118, 132, 129 });
///
/// // Decompose time series into unobserved trend, cycle, and seasonal components
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new UnobservedComponentsModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(
///         new UnobservedComponentsOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()))
///     .Build(trainingMatrix, trainingLabels);
///
/// Vector&lt;double&gt; forecast = result.Forecast(horizon: 24);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Forecasting)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Time Series Analysis by State Space Methods", "https://doi.org/10.1093/acprof:oso/9780199641178.001.0001", Year = 2012, Authors = "James Durbin, Siem Jan Koopman")]
public partial class UnobservedComponentsModel<T, TInput, TOutput> : TimeSeriesModelBase<T>
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
    private readonly UnobservedComponentsOptions<T, TInput, TOutput> _ucOptions;

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
    [Buffer]
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
    [Scratch]
    private Vector<T> _previousTrend;

    /// <summary>
    /// The original time series data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This stores the actual data you're analyzing, like daily sales numbers
    /// or monthly temperature readings.
    /// </remarks>
    [Buffer]
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
    [Buffer]
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
    [Buffer]
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
    [Buffer]
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
    [AiDotNet.Attributes.TrainableParameter]
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
    [Buffer]
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
    [AiDotNet.Attributes.TrainableParameter]
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
    [Buffer]
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
    [Buffer]
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
    [Buffer]
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
    [Buffer]
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
    public UnobservedComponentsModel(UnobservedComponentsOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new UnobservedComponentsOptions<T, TInput, TOutput>())
    {
        _ucOptions = options ?? new UnobservedComponentsOptions<T, TInput, TOutput>();

        // Initialize model components
        _trend = new Vector<T>(_ucOptions.MaxIterations);
        _previousTrend = new Vector<T>(_ucOptions.MaxIterations);
        _y = new Vector<T>(_ucOptions.MaxIterations);
        _seasonal = new Vector<T>(_ucOptions.MaxIterations);
        _cycle = new Vector<T>(_ucOptions.MaxIterations);
        _irregular = new Vector<T>(_ucOptions.MaxIterations);
        _fft = new FastFourierTransform<T>();
        _filteredState = new List<Vector<T>>();
        _filteredCovariance = new List<Matrix<T>>();
        _stateCovariance = Matrix<T>.Empty();
        _state = new Vector<T>(_ucOptions.MaxIterations);
        _stateTransition = Matrix<T>.Empty();
        _observationModel = Matrix<T>.Empty();
        _processNoise = Matrix<T>.Empty();
        _observationNoise = NumOps.Zero;
        _convergenceThreshold = NumOps.Zero;
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
        _irregular = Irregular(y);
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
        var engine = AiDotNetEngine.Current;
        int n = data.Length;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Trailing window sums: the series plus its copies delayed by 1 .. windowSize - 1 steps (zeros shifted in);
        // the first windowSize - 1 points average over the i + 1 points they have.
        int width = Math.Min(windowSize, n);
        var values = data.AsTensor();
        var sum = values;
        for (int lag = 1; lag < width; lag++)
        {
            var delayed = engine.TensorConcatenate(new[]
            {
                new Tensor<T>(new[] { lag }),
                engine.TensorSlice(values, new[] { 0 }, new[] { n - lag })
            }, 0);
            sum = engine.TensorAdd(sum, delayed);
        }

        var counts = engine.TensorClamp(
            engine.TensorArange<T>(1, n + 1, 1), NumOps.One, NumOps.FromDouble(width));
        return HeapVector(engine.TensorDivide(sum, counts));
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
    internal Vector<T> InitializeSeasonal(Vector<T> y, Vector<T> trend)
    {
        var engine = AiDotNetEngine.Current;
        int n = y.Length;
        int period = _ucOptions.SeasonalPeriod;
        int cycles = (n + period - 1) / period;
        int padded = cycles * period;
        var shape = new[] { cycles, period };

        // Seasonal index i: the mean detrended value over positions i, i + period, ...; laid out as [cycle, position]
        // with the incomplete last cycle zero-padded. Position i occurs in every cycle but the last unless the last
        // reaches it; the counts are integers (a 0/1 sum in a narrow T stops at 2048 in Half).
        var detrended = engine.TensorSubtract(y.AsTensor(), trend.AsTensor());
        var pad = new Tensor<T>(new[] { padded - n });
        var sums = engine.ReduceSum(engine.Reshape(PadEnd(detrended, pad), shape), new[] { 0 }, false);
        int lastCycle = n - ((cycles - 1) * period);
        var counts = new Tensor<int>(
            Enumerable.Range(0, period).Select(i => i < lastCycle ? cycles : cycles - 1).ToArray(), new[] { period }).Cast<T>();
        var seasonalIndices = engine.TensorDivide(sums, counts);

        // Normalize seasonal indices to sum to zero
        seasonalIndices = engine.TensorSubtractScalar(seasonalIndices, engine.TensorMean(seasonalIndices));

        // Apply seasonal indices to the full series
        var tiled = engine.Reshape(
            engine.TensorBroadcastTo(engine.Reshape(seasonalIndices, new[] { 1, period }), shape), new[] { padded });
        return HeapVector(engine.TensorSlice(tiled, new[] { 0 }, new[] { n }));
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
        var engine = AiDotNetEngine.Current;

        // Step 1: Calculate residuals after removing trend and seasonal components
        var residual = engine.TensorSubtract(y.AsTensor(), engine.TensorAdd(trend.AsTensor(), seasonal.AsTensor()));
        Vector<T> cycle = HeapVector(residual);

        // Step 2: Apply Hodrick-Prescott filter to separate cycle from noise
        cycle = HodrickPrescottFilter(cycle, _ucOptions.CycleLambda);

        // Step 3: Apply band-pass filter to isolate cycle frequencies
        cycle = BandPassFilter(cycle, _ucOptions.CycleMinPeriod, _ucOptions.CycleMaxPeriod);

        // Step 4: Normalize the cycle component
        T cycleMean = cycle.Average();
        T cycleStd = cycle.StandardDeviation();
        return HeapVector(engine.TensorDivideScalar(engine.TensorSubtractScalar(cycle.AsTensor(), cycleMean), cycleStd));
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
    internal Vector<T> HodrickPrescottFilter(Vector<T> data, double lambda)
    {
        var engine = AiDotNetEngine.Current;
        int n = data.Length;

        // Hodrick & Prescott (1997), statsmodels hpfilter: the trend solves (I + lambda D'D) trend = data, with D the
        // (n - 2) x n second-difference operator (rows [1, -2, 1]). Its rows near the ends differ from the interior
        // [lambda, -4 lambda, 1 + 6 lambda, -4 lambda, lambda].
        var identity = Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(n));
        Tensor<T> system = identity;
        if (n > 2)
        {
            int rows = n - 2;
            var secondDifference = engine.TensorAdd(
                engine.TensorSubtract(
                    engine.TensorSlice(identity, new[] { 0, 0 }, new[] { rows, n }),
                    engine.TensorMultiplyScalar(engine.TensorSlice(identity, new[] { 1, 0 }, new[] { rows, n }), NumOps.FromDouble(2.0))),
                engine.TensorSlice(identity, new[] { 2, 0 }, new[] { rows, n }));
            system = engine.TensorAdd(identity, engine.TensorMultiplyScalar(
                engine.TensorMatMul(engine.TensorTranspose(secondDifference), secondDifference), NumOps.FromDouble(lambda)));
        }

        Matrix<T> A = HeapMatrix(system);

        // Solve the system A * trend = data
        var decomposition = _ucOptions.Decomposition ?? new LuDecomposition<T>(A);
        Vector<T> trend = MatrixSolutionHelper.SolveLinearSystem(data, decomposition);

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
    internal Vector<T> BandPassFilter(Vector<T> data, int minPeriod, int maxPeriod)
    {
        int n = data.Length;

        // Apply FFT
        Vector<Complex<T>> spectrum = _fft.Forward(data);

        // Keep the bins whose frequency lies in [1 / maxPeriod, 1 / minPeriod]. Bin i of a length-n transform is
        // frequency i / n for i <= n / 2 and the negative frequency -(n - i) / n above it; a real series needs both
        // halves kept or dropped together, so the test is on min(i, n - i) / n.
        double minFreq = 1.0 / maxPeriod;
        double maxFreq = 1.0 / minPeriod;
        var keep = Enumerable.Range(0, n)
            .Select(i => Math.Min(i, n - i) / (double)n)
            .Select(freq => freq >= minFreq && freq <= maxFreq)
            .ToArray();
        var passed = AiDotNetEngine.Current.TensorWhere(
            keep, spectrum.AsTensor(),
            new Tensor<Complex<T>>(Enumerable.Repeat(new Complex<T>(NumOps.Zero, NumOps.Zero), n).ToArray(), new[] { n })).ToVector();

        // Apply inverse FFT
        return _fft.Inverse(passed);
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
    internal void InitializeKalmanParameters()
    {
        int stateSize = 3; // trend, seasonal, cycle
        if (!_ucOptions.IncludeCycle) stateSize--;
        if (_ucOptions.SeasonalPeriod <= 1) stateSize--;

        _stateTransition = Matrix<T>.CreateIdentity(stateSize);
        _observationModel = HeapMatrix(Filled(new[] { 1, stateSize }, NumOps.One));

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
    internal void KalmanFilter(Vector<T> y)
    {
        var engine = AiDotNetEngine.Current;
        int n = y.Length;
        int stateSize = _stateTransition.Rows;
        var filteredState = new List<Vector<T>>(n);
        var filteredCovariance = new List<Matrix<T>>(n);

        var transition = Tensor<T>.FromMatrix(_stateTransition);
        var transitionT = engine.TensorTranspose(transition);
        var observation = Tensor<T>.FromMatrix(_observationModel);
        var observationT = engine.TensorTranspose(observation);
        var processNoise = Tensor<T>.FromMatrix(_processNoise);
        var observations = engine.Reshape(y.AsTensor(), new[] { n, 1, 1 });
        var state = engine.Reshape(_state.AsTensor(), new[] { stateSize, 1 });
        var covariance = Tensor<T>.FromMatrix(_stateCovariance);

        // The time recursion is sequential; each step is whole-matrix engine work.
        for (int t = 0; t < n; t++)
        {
            // Predict: a = T a, P = T P T' + Q
            var predictedState = engine.TensorMatMul(transition, state);
            var predictedCovariance = engine.TensorAdd(
                engine.TensorMatMul(engine.TensorMatMul(transition, covariance), transitionT), processNoise);

            // Update: v = y - Z a, F = Z P Z' + H, K = P Z' / F
            var observed = engine.Reshape(engine.TensorSlice(observations, new[] { t, 0, 0 }, new[] { 1, 1, 1 }), new[] { 1, 1 });
            var innovation = engine.TensorSubtract(observed, engine.TensorMatMul(observation, predictedState));
            var gainNumerator = engine.TensorMatMul(predictedCovariance, observationT);
            var innovationCovariance = engine.TensorAddScalar(engine.TensorMatMul(observation, gainNumerator), _observationNoise);
            var kalmanGain = engine.TensorDivide(
                gainNumerator, engine.TensorBroadcastTo(innovationCovariance, new[] { stateSize, 1 }));

            // a = a + K v, P = P - K Z P
            state = engine.TensorAdd(predictedState, engine.TensorMatMul(kalmanGain, innovation));
            covariance = engine.TensorSubtract(
                predictedCovariance, engine.TensorMatMul(engine.TensorMatMul(kalmanGain, observation), predictedCovariance));

            filteredState.Add(HeapVector(engine.Reshape(state, new[] { stateSize })));
            filteredCovariance.Add(HeapMatrix(covariance));
        }

        _state = HeapVector(engine.Reshape(state, new[] { stateSize }));
        _stateCovariance = HeapMatrix(covariance);

        // Store filtered results
        _filteredState = filteredState;
        _filteredCovariance = filteredCovariance;
    }

    /// <summary>The filtered state at every step of the last <see cref="KalmanFilter"/> run.</summary>
    internal IReadOnlyList<Vector<T>> FilteredStates => _filteredState;

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

        Vector<T> nextSmoothedState = _filteredState[n - 1];
        Matrix<T> nextSmoothedCovariance = _filteredCovariance[n - 1];

        smoothedState.Add(nextSmoothedState);

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
        var engine = AiDotNetEngine.Current;
        int n = smoothedState.Count;

        // The state at each step is [trend, seasonal (if seasonal), cycle (if cycle)]: [steps, state].
        var states = engine.TensorStack(smoothedState.Select(s => s.AsTensor()).ToArray(), 0);
        int stateIndex = 0;
        _trend = StateColumn(states, stateIndex++);
        _seasonal = _ucOptions.SeasonalPeriod > 1 ? StateColumn(states, stateIndex++) : new Vector<T>(n);
        _cycle = _ucOptions.IncludeCycle ? StateColumn(states, stateIndex) : new Vector<T>(n);

        // Update irregular component
        _irregular = Irregular(_y);
    }

    /// <summary>Column <paramref name="index"/> of the stacked states [steps, state].</summary>
    private Vector<T> StateColumn(Tensor<T> states, int index)
    {
        var engine = AiDotNetEngine.Current;
        int steps = states.Shape[0];
        return HeapVector(engine.Reshape(engine.TensorSlice(states, new[] { 0, index }, new[] { steps, 1 }), new[] { steps }));
    }

    /// <summary>The series less the trend, seasonal and cycle components.</summary>
    private Vector<T> Irregular(Vector<T> y)
    {
        var engine = AiDotNetEngine.Current;
        return HeapVector(engine.TensorSubtract(
            y.AsTensor(), engine.TensorAdd(_trend.AsTensor(), engine.TensorAdd(_seasonal.AsTensor(), _cycle.AsTensor()))));
    }

    private static Tensor<T> PadEnd(Tensor<T> values, Tensor<T> padding)
        => padding.Length == 0 ? values : AiDotNetEngine.Current.TensorConcatenate(new[] { values, padding }, 0);

    private static Tensor<T> Filled(int[] shape, T value)
    {
        var tensor = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>A plain-heap copy of a 1-D engine result (training-arena buffers are recycled when training ends).</summary>
    private static Vector<T> HeapVector(Tensor<T> values)
    {
        var vector = new Vector<T>(values.Length);
        values.CopyTo(vector.AsWritableSpan());
        return vector;
    }

    /// <summary>A plain-heap copy of a [rows, columns] engine result.</summary>
    private static Matrix<T> HeapMatrix(Tensor<T> values)
    {
        var matrix = new Matrix<T>(values.Shape[0], values.Shape[1]);
        values.CopyTo(matrix.AsWritableSpan());
        return matrix;
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
        var engine = AiDotNetEngine.Current;
        T maxDifference = _trend.Length == 0
            ? NumOps.Zero
            : engine.TensorMaxValue(engine.TensorAbs(engine.TensorSubtract(_trend.AsTensor(), _previousTrend.AsTensor())));

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
        var optimizer = _ucOptions.Optimizer ?? new LBFGSOptimizer<T, Matrix<T>, Vector<T>>(this);

        // Prepare the optimization input data
        var inputData = new OptimizationInputData<T, Matrix<T>, Vector<T>>
        {
            XTrain = x,
            YTrain = y
        };

        // Run optimization
        var result = optimizer.Optimize(inputData);

        // Update model parameters with optimized values
        UpdateModelParameters((result.BestSolution as IParameterizable<T, Matrix<T>, Vector<T>>)?.GetParameters() ?? Vector<T>.Empty());
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
            int block = _ucOptions.SeasonalPeriod - 1;
            _stateTransition = WithBlock(_stateTransition, stateSize, block, NumOps.FromDouble(-1));
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
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        if (_trend.Length == 0 || _seasonal.Length == 0 || _cycle.Length == 0 || _irregular.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        if (input.Columns == 0)
        {
            throw new ArgumentException("Input vector must contain at least one element (time index).", nameof(input));
        }

        // Each row's first column is its time index (as PredictSingle).
        var engine = AiDotNetEngine.Current;
        var timeColumn = engine.Reshape(
            engine.TensorSlice(Tensor<T>.FromMatrix(input), new[] { 0, 0 }, new[] { n, 1 }), new[] { n });
        var indices = engine.TensorRound(timeColumn).Cast<int>();
        int first = indices.ToArray().Min();
        int last = indices.ToArray().Max();
        if (first < 0)
        {
            throw new ArgumentException($"Time index {first} must be non-negative.", nameof(input));
        }

        // In-sample: trend + seasonal (if enabled) + cycle (if enabled) + 0.2 x irregular. Beyond the fitted
        // horizon the series continues with the damped-slope forecast from the last fitted point.
        int fitted = _trend.Length;
        var series = engine.TensorAdd(_trend.AsTensor(), engine.TensorMultiplyScalar(_irregular.AsTensor(), NumOps.FromDouble(0.2)));
        if (_ucOptions.SeasonalPeriod > 1) series = engine.TensorAdd(series, _seasonal.AsTensor());
        if (_ucOptions.IncludeCycle) series = engine.TensorAdd(series, _cycle.AsTensor());
        if (last >= fitted)
        {
            series = engine.TensorConcatenate(new[] { series, Forecast(last - (fitted - 1), fitted - 1).AsTensor() }, 0);
        }

        return engine.TensorTake(series, indices).ToVector();
    }

    /// <summary>
    /// <paramref name="matrix"/> with the square block of side <paramref name="size"/> at (<paramref name="offset"/>,
    /// <paramref name="offset"/>) set to <paramref name="value"/>.
    /// </summary>
    private Matrix<T> WithBlock(Matrix<T> matrix, int offset, int size, T value)
    {
        var engine = AiDotNetEngine.Current;
        int rows = matrix.Rows, columns = matrix.Columns;
        if (offset + size > rows || offset + size > columns)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"A {size} x {size} block at {offset} does not fit a {rows} x {columns} matrix.");
        }

        // A mask of ones over the block, zero-padded to the matrix's shape.
        var mask = new Tensor<T>(new[] { size, size });
        engine.TensorFill(mask, NumOps.One);
        var withColumns = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { size, offset }), mask, new Tensor<T>(new[] { size, columns - offset - size })
        }, 1);
        var full = engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new[] { offset, columns }), withColumns, new Tensor<T>(new[] { rows - offset - size, columns })
        }, 0);
        return HeapMatrix(engine.TensorWhere(full, Filled(new[] { rows, columns }, value), Tensor<T>.FromMatrix(matrix)));
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


    /// <summary>
    /// Implements the model-specific training logic for the Unobserved Components Model.
    /// </summary>
    /// <param name="x">The input features matrix (typically time indices).</param>
    /// <param name="y">The target values (time series data).</param>
    /// <exception cref="ArgumentException">Thrown when the input data is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method handles the core training process for the UCM model, orchestrating
    /// the initialization, Kalman filtering, smoothing, and parameter optimization steps.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method coordinates the learning process for the model. It manages the different
    /// steps that transform your raw time series data into separate components (trend, seasonal,
    /// cycle, and irregular).
    /// 
    /// The process follows these main steps:
    /// 1. It makes initial guesses about the components using simple techniques
    /// 2. It sets up the mathematical structures needed for the Kalman filter
    /// 3. It runs the Kalman filter to refine these initial estimates 
    /// 4. It applies the Kalman smoother to further improve the estimates using future information
    /// 5. It checks if the estimates have stabilized (converged)
    /// 6. If requested, it optimizes the model parameters to better fit your specific data
    /// 
    /// When the training completes, your time series will be fully decomposed into its 
    /// component parts, which can then be used for analysis or forecasting.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Save the original time series data
        _y = y.Clone();
        int n = _y.Length;

        if (n < 2)
        {
            throw new ArgumentException("Time series must contain at least 2 observations for training.", nameof(y));
        }

        // Verify that options are appropriate for the data length
        if (_ucOptions.SeasonalPeriod >= n)
        {
            throw new ArgumentException($"Seasonal period ({_ucOptions.SeasonalPeriod}) must be less than the number of observations ({n}).", nameof(_ucOptions));
        }

        // Initialize components with initial estimates
        InitializeComponents(_y);

        // Save initial trend estimates for convergence checking
        _previousTrend = _trend.Clone();

        // Initialize Kalman filter parameters
        InitializeKalmanParameters();

        // Iterative refinement of components
        bool converged = false;
        int iteration = 0;

        while (!converged && iteration < _ucOptions.MaxIterations)
        {
            // Apply Kalman filter to estimate state at each time point
            KalmanFilter(_y);

            // Apply Kalman smoother to refine estimates using future information
            KalmanSmoother(_y);

            // Check for convergence
            converged = HasConverged();

            iteration++;
        }

        // Optimize parameters if requested
        if (_ucOptions.OptimizeParameters)
        {
            try
            {
                OptimizeParameters(x, _y);
            }
            catch (Exception ex)
            {
                // Log warning but continue with unoptimized parameters
                System.Diagnostics.Debug.WriteLine($"Parameter optimization failed: {ex.Message}");
                // In production, you might use a proper logging framework here
            }
        }

        // Final refinement with optimized parameters if optimization was performed
        if (_ucOptions.OptimizeParameters)
        {
            KalmanFilter(_y);
            KalmanSmoother(_y);
        }
    }

    /// <summary>
    /// Generates a prediction for a single input vector.
    /// </summary>
    /// <param name="input">A vector containing the time index and any additional features.</param>
    /// <returns>The predicted value for the specified time point.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when the input index is invalid or out of range.</exception>
    /// <remarks>
    /// <para>
    /// This method combines the trend, seasonal, cycle, and irregular components to produce
    /// a prediction for a single time point.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method makes a prediction for a single point in time by adding together
    /// all the components the model has identified in your data:
    /// 
    /// - Trend component: The long-term direction (going up, down, or flat)
    /// - Seasonal component: Regular patterns tied to the calendar (like weekly or yearly cycles)
    /// - Cycle component: Longer-term patterns not tied to fixed periods (like business cycles)
    /// - Irregular component: Random variations or "noise" in the data
    /// 
    /// For example, if forecasting retail sales, the prediction might combine:
    /// - An upward trend (business growth)
    /// - A seasonal peak (December holiday shopping)
    /// - A downward cycle (current economic recession)
    /// - Some random variation
    /// 
    /// The result gives you a comprehensive forecast that accounts for all identified patterns.
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        if (_trend.Length == 0 || _seasonal.Length == 0 || _cycle.Length == 0 || _irregular.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // Extract time index from input
        int timeIndex;
        if (input.Length > 0)
        {
            timeIndex = Convert.ToInt32(input[0]);
        }
        else
        {
            throw new ArgumentException("Input vector must contain at least one element (time index).", nameof(input));
        }

        // Validate time index
        if (timeIndex < 0)
        {
            throw new ArgumentException($"Time index {timeIndex} must be non-negative.", nameof(input));
        }

        // Predicting beyond the fitted horizon is a legitimate forecast — extrapolate
        // via the existing damped-slope Forecast path rather than rejecting the index.
        // (Forecast uses ForecastTrend/Seasonal/Cycle/Irregular; it does not call back
        // into PredictSingle, so there is no recursion.)
        if (timeIndex >= _trend.Length)
        {
            int horizon = timeIndex - (_trend.Length - 1);
            return Forecast(horizon, _trend.Length - 1)[horizon - 1];
        }

        // Combine components to produce the prediction
        T prediction = NumOps.Zero;

        // Add trend component
        prediction = NumOps.Add(prediction, _trend[timeIndex]);

        // Add seasonal component if enabled
        if (_ucOptions.SeasonalPeriod > 1)
        {
            prediction = NumOps.Add(prediction, _seasonal[timeIndex]);
        }

        // Add cycle component if enabled
        if (_ucOptions.IncludeCycle)
        {
            prediction = NumOps.Add(prediction, _cycle[timeIndex]);
        }

        // Add a portion of the irregular component to maintain some of the time series character
        // This is optional and depends on the application
        T irregularComponent = NumOps.Multiply(_irregular[timeIndex], NumOps.FromDouble(0.2));
        prediction = NumOps.Add(prediction, irregularComponent);

        return prediction;
    }

    /// <summary>
    /// Creates a new instance of the UnobservedComponentsModel class.
    /// </summary>
    /// <returns>A new instance of the UnobservedComponentsModel class.</returns>
    /// <remarks>
    /// <para>
    /// This factory method creates a new instance of the model with the same options
    /// as the current instance. It's used by Clone and DeepCopy methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method creates a fresh copy of the model with the same settings
    /// but no trained components. It's like getting a blank copy of the original
    /// model's template.
    /// 
    /// This is used internally when you clone a model or make a deep copy.
    /// It ensures that the copy is of the exact same type as the original,
    /// with all the same configuration options.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new UnobservedComponentsModel<T, TInput, TOutput>(_ucOptions);
    }

    /// <summary>
    /// Forecasts multiple steps into the future using the decomposed components.
    /// </summary>
    /// <param name="horizon">The number of time steps to forecast.</param>
    /// <param name="startIndex">The time index to start forecasting from (defaults to the end of the training data).</param>
    /// <returns>A vector containing the forecasted values.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <exception cref="ArgumentException">Thrown when horizon is not positive or startIndex is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for multiple future time points by extending each component
    /// (trend, seasonal, cycle) based on its natural pattern and properties.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts values for multiple future time points by:
    /// 
    /// 1. Extending the trend component using its growth pattern
    /// 2. Repeating the seasonal component based on its period
    /// 3. Continuing the cycle component based on its frequency
    /// 4. Adding small random variations to represent the irregular component
    /// 
    /// For example, a 12-month forecast of retail sales might:
    /// - Continue the upward trend line
    /// - Repeat the December sales peak and summer slump
    /// - Account for the current phase of the business cycle
    /// - Include small random variations
    /// 
    /// The result is a comprehensive forecast that captures all the patterns
    /// identified in your historical data.
    /// </para>
    /// </remarks>
    public Vector<T> Forecast(int horizon, int startIndex = -1)
    {
        if (_trend.Length == 0 || _seasonal.Length == 0 || _cycle.Length == 0 || _irregular.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before forecasting.");
        }

        if (horizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be positive.", nameof(horizon));
        }

        // If startIndex is not specified, use the last point in the training data
        if (startIndex < 0)
        {
            startIndex = _trend.Length - 1;
        }

        if (startIndex < 0 || startIndex >= _trend.Length)
        {
            throw new ArgumentException($"Start index {startIndex} is out of range [0, {_trend.Length - 1}].", nameof(startIndex));
        }

        Vector<T> forecast = new Vector<T>(horizon);

        // Calculate trend component forecasts
        Vector<T> trendForecast = ForecastTrend(horizon, startIndex);

        // Calculate seasonal component forecasts
        Vector<T> seasonalForecast = ForecastSeasonal(horizon, startIndex);

        // Calculate cycle component forecasts
        Vector<T> cycleForecast = ForecastCycle(horizon, startIndex);

        // Generate irregular component forecasts (small random variations)
        Vector<T> irregularForecast = ForecastIrregular(horizon);

        // Combine components to produce the final forecast
        // VECTORIZED: Add forecast components using Engine operations
        forecast = (Vector<T>)Engine.Add(trendForecast, seasonalForecast);
        forecast = (Vector<T>)Engine.Add(forecast, cycleForecast);
        forecast = (Vector<T>)Engine.Add(forecast, irregularForecast);

        return forecast;
    }

    /// <summary>
    /// Forecasts the trend component for future time points.
    /// </summary>
    /// <param name="horizon">The number of time steps to forecast.</param>
    /// <param name="startIndex">The time index to start forecasting from.</param>
    /// <returns>A vector containing the forecasted trend values.</returns>
    /// <remarks>
    /// <para>
    /// This method extrapolates the trend component using a linear or damped growth model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts how the long-term direction of your data will continue
    /// into the future. It examines the recent trend and projects it forward,
    /// potentially with some damping (slowing down) over time to avoid unrealistic
    /// growth or decline.
    /// </para>
    /// </remarks>
    private Vector<T> ForecastTrend(int horizon, int startIndex)
    {
        var engine = AiDotNetEngine.Current;

        // Calculate recent slope in the trend
        T slope = NumOps.Zero;
        int lookback = Math.Min(5, startIndex); // Look at the last 5 points or fewer if not available

        if (lookback > 0)
        {
            slope = NumOps.Divide(
                NumOps.Subtract(_trend[startIndex], _trend[startIndex - lookback]),
                NumOps.FromDouble(lookback)
            );
        }

        // Damped growth: step i adds slope * 0.95^i, so the forecast i steps ahead (1-based) is
        // trend + slope * (0.95^0 + ... + 0.95^(i-1)).
        T dampingFactor = NumOps.FromDouble(0.95);
        var dampedSlopes = engine.TensorMultiplyScalar(
            engine.TensorPower(Filled(new[] { horizon }, dampingFactor), engine.TensorArange<T>(0, horizon, 1)), slope);
        return engine.TensorAddScalar(engine.TensorCumSum(dampedSlopes, 0), _trend[startIndex]).ToVector();
    }

    /// <summary>
    /// Forecasts the seasonal component for future time points.
    /// </summary>
    /// <param name="horizon">The number of time steps to forecast.</param>
    /// <param name="startIndex">The time index to start forecasting from.</param>
    /// <returns>A vector containing the forecasted seasonal values.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the seasonal component by repeating the last complete seasonal cycle.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts the seasonal patterns in your future data by repeating
    /// the patterns observed in your historical data. For example, if December always
    /// has higher sales in your historical data, the forecast will show higher sales
    /// in future Decembers too.
    /// </para>
    /// </remarks>
    private Vector<T> ForecastSeasonal(int horizon, int startIndex)
    {
        // If no seasonality is included in the model, return zeros
        if (_ucOptions.SeasonalPeriod <= 1)
        {
            return new Vector<T>(horizon); // All zeros
        }

        int period = _ucOptions.SeasonalPeriod;

        // Use the last complete seasonal cycle as the pattern for forecasting: step i takes the most recent fitted
        // point j <= startIndex with j % period == (seasonStart + i) % period (zero if there is none).
        int seasonStart = Math.Max(0, startIndex - period + 1);
        var source = Enumerable.Range(0, horizon)
            .Select(i => (seasonStart + i) % period)
            .Select(position => startIndex - ((((startIndex - position) % period) + period) % period))
            .ToArray();
        var engine = AiDotNetEngine.Current;
        var found = source.Select(j => j >= 0).ToArray();
        var indices = new Tensor<int>(source.Select(j => Math.Max(0, j)).ToArray(), new[] { horizon });
        return engine.TensorWhere(found, engine.TensorTake(_seasonal.AsTensor(), indices), new Tensor<T>(new[] { horizon })).ToVector();
    }

    /// <summary>
    /// Forecasts the cycle component for future time points.
    /// </summary>
    /// <param name="horizon">The number of time steps to forecast.</param>
    /// <param name="startIndex">The time index to start forecasting from.</param>
    /// <returns>A vector containing the forecasted cycle values.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the cycle component using its auto-regressive properties,
    /// gradually dampening the cycle amplitude over time.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method predicts how cyclical patterns (like business cycles or economic ups and downs)
    /// will continue in the future. Unlike seasonal patterns, these cycles don't have fixed lengths,
    /// so the method uses recent cycle behavior to project forward, while gradually reducing the
    /// cycle strength over time (since cycles are harder to predict far into the future).
    /// </para>
    /// </remarks>
    private Vector<T> ForecastCycle(int horizon, int startIndex)
    {
        // If no cycle is included in the model, return zeros
        if (!_ucOptions.IncludeCycle)
        {
            return new Vector<T>(horizon); // All zeros
        }

        // Estimate cycle properties from the most recent data
        T cycleAmplitude = _cycle[startIndex];

        // Calculate approximate cycle frequency
        T cyclePeriod = NumOps.Divide(NumOps.FromDouble(_ucOptions.CycleMinPeriod + _ucOptions.CycleMaxPeriod), NumOps.FromDouble(2));

        // Find the phase of the cycle at the starting point
        T cyclePhase = NumOps.Zero;
        int lookback = Math.Min(startIndex, Convert.ToInt32(cyclePeriod));

        if (lookback > 0 && !NumOps.Equals(_cycle[startIndex], NumOps.Zero))
        {
            // Estimate phase based on recent movement
            if (NumOps.LessThan(_cycle[startIndex], _cycle[startIndex - 1]))
            {
                // Cycle is decreasing
                if (NumOps.LessThan(_cycle[startIndex], NumOps.Zero))
                {
                    // In the negative half and decreasing (between p and 3p/2)
                    cyclePhase = NumOps.FromDouble(4 * Math.PI / 3);
                }
                else
                {
                    // In the positive half and decreasing (between p/2 and p)
                    cyclePhase = NumOps.FromDouble(3 * Math.PI / 4);
                }
            }
            else
            {
                // Cycle is increasing
                if (NumOps.LessThan(_cycle[startIndex], NumOps.Zero))
                {
                    // In the negative half and increasing (between 3p/2 and 2p)
                    cyclePhase = NumOps.FromDouble(7 * Math.PI / 4);
                }
                else
                {
                    // In the positive half and increasing (between 0 and p/2)
                    cyclePhase = NumOps.FromDouble(Math.PI / 4);
                }
            }
        }

        // Damping factor to reduce cycle amplitude over time
        T dampingFactor = NumOps.FromDouble(0.98);

        // Generate cycle forecasts: amplitude * damping^i * cos(phase + 2 pi i / period)
        var engine = AiDotNetEngine.Current;
        var steps = engine.TensorArange<T>(0, horizon, 1);
        var positions = engine.TensorAddScalar(
            engine.TensorDivideScalar(engine.TensorMultiplyScalar(steps, NumOps.FromDouble(2 * Math.PI)), cyclePeriod), cyclePhase);
        var dampedAmplitudes = engine.TensorMultiplyScalar(
            engine.TensorPower(Filled(new[] { horizon }, dampingFactor), steps), cycleAmplitude);
        return engine.TensorMultiply(dampedAmplitudes, engine.TensorCos(positions)).ToVector();
    }

    /// <summary>
    /// Returns the conditional-mean forecast of the irregular component.
    /// </summary>
    /// <param name="horizon">The number of time steps to forecast.</param>
    /// <returns>A vector containing the forecasted irregular values.</returns>
    /// <remarks>
    /// <para>
    /// The irregular component is zero-mean observation noise. A point forecast therefore uses
    /// its conditional expectation, zero. Probabilistic sample paths belong in an explicit
    /// simulation API with caller-controlled random state; injecting secure randomness here made
    /// repeated predictions and serialized clones disagree for identical inputs.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Unpredictable noise has no directional contribution to the best single forecast, so this
    /// method contributes zeros. Prediction intervals or simulated futures can still use the
    /// fitted irregular variance through a dedicated stochastic API.
    /// </para>
    /// </remarks>
    private Vector<T> ForecastIrregular(int horizon)
    {
        return new Vector<T>(horizon);
    }

    /// <summary>
    /// Gets the decomposed components of the time series.
    /// </summary>
    /// <returns>A dictionary containing the trend, seasonal, cycle, and irregular components.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model has not been trained.</exception>
    /// <remarks>
    /// <para>
    /// This method provides access to the individual decomposed components identified by the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method gives you the separate pieces that make up your time series, like
    /// looking inside a watch to see all the gears and springs. You can examine:
    /// 
    /// - The trend component: The overall direction (going up, down, or flat)
    /// - The seasonal component: Regular patterns tied to the calendar
    /// - The cycle component: Longer-term patterns not tied to fixed periods
    /// - The irregular component: Random variations or "noise"
    /// 
    /// This decomposition helps you understand what's really driving your data and
    /// makes it easier to interpret and forecast.
    /// </para>
    /// </remarks>
    public Dictionary<string, Vector<T>> GetComponents()
    {
        if (_trend.Length == 0 || _seasonal.Length == 0 || _cycle.Length == 0 || _irregular.Length == 0)
        {
            throw new InvalidOperationException("Model must be trained before getting components.");
        }

        Dictionary<string, Vector<T>> components = new Dictionary<string, Vector<T>>
        {
            ["Trend"] = _trend.Clone(),
            ["Seasonal"] = _seasonal.Clone(),
            ["Cycle"] = _cycle.Clone(),
            ["Irregular"] = _irregular.Clone()
        };

        return components;
    }

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all trained parameters and decomposed components,
    /// returning the model to its initial state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method clears all the learning the model has done, basically
    /// giving you a fresh start. It's like erasing the whiteboard so you
    /// can use the same model to analyze a different time series.
    /// 
    /// After calling this method, you'll need to train the model again
    /// before making predictions.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        base.Reset();

        // Reset decomposed components
        _trend = new Vector<T>(0);
        _previousTrend = new Vector<T>(0);
        _seasonal = new Vector<T>(0);
        _cycle = new Vector<T>(0);
        _irregular = new Vector<T>(0);
        _y = new Vector<T>(0);

        // Reset Kalman filter state
        _stateCovariance = Matrix<T>.Empty();
        _state = new Vector<T>(0);
        _stateTransition = Matrix<T>.Empty();
        _observationModel = Matrix<T>.Empty();
        _processNoise = Matrix<T>.Empty();
        _observationNoise = NumOps.Zero;

        // Clear filtered results
        _filteredState = new List<Vector<T>>();
        _filteredCovariance = new List<Matrix<T>>();
    }

    /// <summary>
    /// Gets metadata about the model, including its type, components, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed metadata about the Unobserved Components Model, including its type,
    /// trained components (trend, seasonal, cycle), and configuration options. This metadata can be used
    /// for model selection, comparison, documentation, and serialization purposes.
    /// </para>
    /// <para>
    /// For Beginners:
    /// This provides a summary of your model's settings and what it has learned.
    /// 
    /// The metadata includes:
    /// - The type of model (Unobserved Components Model)
    /// - Information about each component (trend, seasonal, cycle, irregular)
    /// - Configuration settings like seasonal period and whether cycles are included
    /// - A serialized version of the entire model
    /// 
    /// This information is useful for:
    /// - Keeping track of different models you've created
    /// - Comparing model configurations
    /// - Documenting which settings worked best
    /// - Sharing model information with others
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include information about model components
                { "HasTrend", _trend.Length > 0 },
                { "HasSeasonal", _seasonal.Length > 0 && _ucOptions.SeasonalPeriod > 1 },
                { "HasCycle", _cycle.Length > 0 && _ucOptions.IncludeCycle },
                { "TrendLength", _trend.Length },
                { "DataLength", _y.Length },
            
                // Include information about model configuration
                { "SeasonalPeriod", _ucOptions.SeasonalPeriod },
                { "IncludeCycle", _ucOptions.IncludeCycle },
                { "CycleMinPeriod", _ucOptions.CycleMinPeriod },
                { "CycleMaxPeriod", _ucOptions.CycleMaxPeriod },
                { "MaxIterations", _ucOptions.MaxIterations },
                { "OptimizeParameters", _ucOptions.OptimizeParameters },
            
                // Include statistical information if available
                { "TrendMean", _trend.Length > 0 ? Convert.ToDouble(_trend.Average()) : 0.0 },
                { "SeasonalAmplitude", _seasonal.Length > 0 ? Convert.ToDouble(NumOps.Subtract(_seasonal.Max(), _seasonal.Min())) : 0.0 },
                { "IrregularStdDev", _irregular.Length > 0 ? Convert.ToDouble(_irregular.StandardDeviation()) : 0.0 }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };

        return metadata;
    }
}
