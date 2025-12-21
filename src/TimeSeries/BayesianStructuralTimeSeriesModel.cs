using AiDotNet.Autodiff;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Bayesian Structural Time Series model for flexible time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// The Bayesian Structural Time Series (BSTS) model is a powerful and flexible approach for analyzing
/// and forecasting time series data. It decomposes a time series into interpretable components
/// including level, trend, seasonality, and regression effects, using Bayesian methods to handle
/// uncertainty and combine information from different sources.
/// </para>
/// 
/// <para>
/// For Beginners:
/// A Bayesian Structural Time Series model is like having a flexible toolkit for forecasting time series data.
/// Unlike simpler models like AR or ARIMA that use fixed patterns, BSTS breaks down your data into
/// meaningful components:
/// 
/// 1. Level: The current "baseline" value of your series
/// 2. Trend: Whether your data is generally increasing or decreasing over time
/// 3. Seasonal patterns: Regular cycles in your data (daily, weekly, yearly, etc.)
/// 4. Effects of external factors: How other variables influence your data
/// 
/// The "Bayesian" part means the model handles uncertainty well. Instead of making single point predictions,
/// it gives you a range of possible outcomes with probabilities attached. It uses something called a
/// "Kalman filter" to continually update its understanding as new data arrives.
/// 
/// BSTS models are especially powerful because they:
/// - Can handle missing data
/// - Allow you to incorporate external information
/// - Provide predictions with uncertainty ranges
/// - Let you see which components are driving your forecast
/// 
/// This makes them ideal for analyzing complex time series where you want to understand
/// what's driving changes in addition to making predictions.
/// </para>
/// </remarks>
public class BayesianStructuralTimeSeriesModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the Bayesian Structural Time Series model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These options control how the model works, including which components to include
    /// (trend, seasonal patterns, regression effects) and various technical settings
    /// that determine how the model learns from data.
    /// </remarks>
    private readonly BayesianStructuralTimeSeriesOptions<T> _bayesianOptions;

    /// <summary>
    /// The current level (baseline value) of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The level represents the current baseline value of your time series after
    /// removing seasonal effects and other patterns. It's like the "average" value
    /// around which other components fluctuate. In a business context, it might
    /// represent your baseline sales before seasonal peaks and special events.
    /// </remarks>
    private T _level;

    /// <summary>
    /// The current trend (rate of change) of the time series.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The trend indicates whether your time series is generally increasing or decreasing
    /// and by how much. A positive trend means values are typically growing over time,
    /// while a negative trend means they're declining. For example, in sales data,
    /// a positive trend might indicate growing market share or increased demand.
    /// </remarks>
    private T _trend;

    /// <summary>
    /// The seasonal components of the time series, representing cyclical patterns.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Seasonal components capture cyclical patterns that repeat at fixed intervals.
    /// For example:
    /// - Retail sales often peak during holidays
    /// - Energy usage might be higher in summer and winter
    /// - Website traffic could spike on weekends
    /// 
    /// Each seasonal component represents a different cycle length (daily, weekly, yearly, etc.).
    /// The model learns these patterns from your data and uses them to make better predictions.
    /// </remarks>
    private List<Vector<T>> _seasonalComponents;

    /// <summary>
    /// The uncertainty in the state estimates, represented as a covariance matrix.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This tracks how uncertain the model is about its estimates of level, trend, and
    /// seasonal components. Higher values mean more uncertainty. As the model processes
    /// more data, this uncertainty typically decreases as the model becomes more confident.
    /// 
    /// Think of it like error bars around the model's internal estimates - they tend to
    /// get smaller as more data confirms the patterns.
    /// </remarks>
    private Matrix<T> _stateCovariance;

    /// <summary>
    /// The estimated variance (uncertainty) in observations.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This represents the "noise" or random fluctuations in your data that can't be
    /// explained by the model's components. Higher values mean your data has more
    /// unexplained variability or randomness.
    /// 
    /// For example, daily sales might fluctuate randomly even after accounting for
    /// trends and seasonal patterns. This parameter measures the size of those
    /// unexplained fluctuations.
    /// </remarks>
    private T _observationVariance;

    /// <summary>
    /// Coefficients for the regression component (impact of external variables).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients measure how external factors affect your time series.
    /// For example, if you're forecasting ice cream sales, temperature might be
    /// an external factor. A positive coefficient would indicate that higher
    /// temperatures are associated with higher sales.
    /// 
    /// Each coefficient quantifies the effect of one external variable. The model
    /// learns these coefficients from your data to improve predictions.
    /// </remarks>
    private Vector<T>? _regression;

    /// <summary>
    /// Creates a new Bayesian Structural Time Series model with the specified options.
    /// </summary>
    /// <param name="options">Options for configuring the BSTS model. If null, default options are used.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new BSTS model, initializing all its components.
    /// You can provide options to customize the model, such as:
    /// - Whether to include a trend component
    /// - What seasonal patterns to look for (weekly, monthly, etc.)
    /// - Whether to include external factors (regression)
    /// - Technical parameters that control how the model learns
    /// 
    /// If you don't provide options, the model will use default settings, but it's
    /// usually better to configure it specifically for your data.
    /// </remarks>
    public BayesianStructuralTimeSeriesModel(BayesianStructuralTimeSeriesOptions<T>? options = null)
    : base(options ?? new BayesianStructuralTimeSeriesOptions<T>())
    {
        _bayesianOptions = options ?? new BayesianStructuralTimeSeriesOptions<T>();

        // Initialize model components
        _level = NumOps.FromDouble(_bayesianOptions.InitialLevelValue);
        _trend = _bayesianOptions.IncludeTrend ? NumOps.FromDouble(_bayesianOptions.InitialTrendValue) : NumOps.Zero;
        _seasonalComponents = [];
        foreach (int period in _bayesianOptions.SeasonalPeriods)
        {
            _seasonalComponents.Add(new Vector<T>(period));
        }
        _observationVariance = NumOps.FromDouble(_bayesianOptions.InitialObservationVariance);

        int stateSize = GetStateSize();
        _stateCovariance = new Matrix<T>(stateSize, stateSize);

        // Initialize regression component if included
        if (_bayesianOptions.IncludeRegression)
        {
            // We'll initialize the regression vector later when we have the actual input data
            _regression = null;
        }
    }

    /// <summary>
    /// Initializes the regression coefficients using Ordinary Least Squares or Ridge Regression.
    /// </summary>
    /// <param name="x">Feature matrix of external variables.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <exception cref="InvalidOperationException">Thrown when the regression vector is not initialized.</exception>
    /// <remarks>
    /// For Beginners:
    /// This private method sets up the initial coefficients for external factors.
    /// 
    /// For example, if you're forecasting ice cream sales and using temperature as a factor,
    /// this method would estimate how much each degree of temperature affects sales.
    /// 
    /// It tries to use a simple method called "Ordinary Least Squares" first, but if that
    /// doesn't work well (which can happen with highly correlated variables), it switches
    /// to a more robust method called "Ridge Regression." Both methods aim to find the
    /// relationship between external factors and your time series.
    /// 
    /// The "shrinkage factor" prevents the model from overreacting to patterns that might
    /// just be coincidences in your data, making predictions more stable.
    /// </remarks>
    private void InitializeRegressionCoefficients(Matrix<T> x, Vector<T> y)
    {
        if (_regression == null)
        {
            throw new InvalidOperationException("Regression vector is not initialized.");
        }

        // Perform OLS to initialize regression coefficients
        Matrix<T> xTranspose = x.Transpose();
        Matrix<T> xTx = xTranspose.Multiply(x);
        Vector<T> xTy = xTranspose.Multiply(y);

        try
        {
            // Solve the normal equations: (X^T * X) * beta = X^T * y
            Vector<T> olsCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _bayesianOptions.RegressionDecompositionType);

            // Apply shrinkage to prevent overfitting
            T shrinkageFactor = NumOps.FromDouble(0.95); // You might want to make this configurable
            _regression = (Vector<T>)Engine.Multiply(olsCoefficients, shrinkageFactor);
        }
        catch (Exception)
        {
            // If matrix is singular or near-singular, use ridge regression
            T ridgeParameter = NumOps.FromDouble(_bayesianOptions.RidgeParameter);
            Vector<T> ridgeDiagonal = new Vector<T>(x.Columns);
            for (int i = 0; i < x.Columns; i++)
            {
                ridgeDiagonal[i] = ridgeParameter;
            }
            Matrix<T> ridgeMatrix = Matrix<T>.CreateDiagonal(ridgeDiagonal);
            Matrix<T> regularizedXTX = xTx.Add(ridgeMatrix);
            Vector<T> ridgeCoefficients = MatrixSolutionHelper.SolveLinearSystem(regularizedXTX, xTy, _bayesianOptions.RegressionDecompositionType);

            // Apply shrinkage to prevent overfitting
            T shrinkageFactor = NumOps.FromDouble(0.95); // You might want to make this configurable
            _regression = (Vector<T>)Engine.Multiply(ridgeCoefficients, shrinkageFactor);
        }
    }

    /// <summary>
    /// Predicts the next state of the time series using the current model.
    /// </summary>
    /// <param name="x">External variables for the current time point.</param>
    /// <returns>The predicted state vector.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method creates a prediction for the next state of all the model's
    /// components (level, trend, seasonal factors, etc.).
    /// 
    /// The state vector combines all these components into a single mathematical object
    /// that represents the model's current understanding of the time series.
    /// 
    /// For example, if your model includes level, trend, and weekly seasonality,
    /// the state would include the current baseline value, the direction of change,
    /// and the effect of each day of the week.
    /// 
    /// This state prediction is a key part of the Kalman filter algorithm that helps
    /// the model learn from data.
    /// </remarks>
    private Vector<T> PredictState(Vector<T> x)
    {
        int stateSize = GetStateSize();
        Vector<T> predictedState = new Vector<T>(stateSize);
        int index = 0;

        // Level
        predictedState[index] = _level;
        index++;

        // Trend
        if (_bayesianOptions.IncludeTrend)
        {
            predictedState[index] = NumOps.Add(_level, _trend);
            index++;
        }

        // Seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                predictedState[index + i] = seasonalComponent[i];
            }
            index += seasonalComponent.Length;
        }

        // Regression component - store coefficients (not products)
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                predictedState[index + i] = _regression[i];
            }
        }

        return predictedState;
    }

    /// <summary>
    /// Predicts the covariance matrix for the next state.
    /// </summary>
    /// <returns>The predicted covariance matrix.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method predicts how uncertain the model will be about its next state.
    /// 
    /// The covariance matrix captures uncertainty in all the model components and how they
    /// might be related. For example, if the level and trend are uncertain in similar ways,
    /// the covariance matrix would capture that relationship.
    /// 
    /// Higher values in this matrix indicate more uncertainty. As the model learns from data,
    /// these values typically decrease, showing growing confidence in the predictions.
    /// 
    /// This uncertainty prediction is another key part of the Kalman filter algorithm.
    /// </remarks>
    private Matrix<T> PredictCovariance()
    {
        int stateSize = GetStateSize();
        Matrix<T> transitionMatrix = CreateTransitionMatrix();

        return transitionMatrix.Multiply(_stateCovariance).Multiply(transitionMatrix.Transpose()) + CreateProcessNoiseMatrix();
    }

    /// <summary>
    /// Calculates the innovation (difference between observation and prediction).
    /// </summary>
    /// <param name="observation">The actual observed value.</param>
    /// <param name="predictedState">The predicted state vector.</param>
    /// <returns>The innovation value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how surprised the model is by a new observation.
    /// 
    /// The "innovation" is the difference between what the model predicted and what actually
    /// happened. It's like the model's forecast error for that time point.
    /// 
    /// A large innovation means the model's prediction was significantly off, which suggests
    /// it needs to learn more from this data point. A small innovation means the model
    /// predicted well, so less adjustment is needed.
    /// 
    /// These innovations drive the learning process in the Kalman filter algorithm.
    /// </remarks>
    private T CalculateInnovation(T observation, Vector<T> predictedState)
    {
        return NumOps.Subtract(observation, CalculatePrediction(predictedState));
    }

    /// <summary>
    /// Calculates the Kalman gain, which determines how much to adjust the state based on new observations.
    /// </summary>
    /// <param name="predictedCovariance">The predicted covariance matrix.</param>
    /// <returns>The Kalman gain vector.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method determines how much the model should learn from new data.
    /// 
    /// The Kalman gain is like a learning rate for each component of the model. Higher values
    /// mean the model will adjust more strongly in response to prediction errors.
    /// 
    /// The gain is higher when:
    /// - The model is very uncertain about its current estimates (high covariance)
    /// - The observation is likely to be accurate (low observation variance)
    /// 
    /// It's lower when:
    /// - The model is already confident in its estimates
    /// - The observation is likely to be noisy or unreliable
    /// 
    /// This adaptive learning rate is what makes Kalman filtering powerful for time series analysis.
    /// </remarks>
    private Vector<T> CalculateKalmanGain(Matrix<T> predictedCovariance)
    {
        Vector<T> observationVector = CreateObservationVector();
        T denominator = NumOps.Add(
            observationVector.DotProduct(predictedCovariance.Multiply(observationVector)),
            _observationVariance
        );

        return predictedCovariance.Multiply(observationVector).Divide(denominator);
    }

    /// <summary>
    /// Updates the state vector based on the Kalman gain and innovation.
    /// </summary>
    /// <param name="predictedState">The predicted state vector.</param>
    /// <param name="kalmanGain">The Kalman gain vector.</param>
    /// <param name="innovation">The innovation value.</param>
    /// <exception cref="ArgumentException">Thrown when predicted state and Kalman gain vectors have different lengths.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the state update is incomplete.</exception>
    /// <remarks>
    /// For Beginners:
    /// This private method updates the model's understanding of the time series based on
    /// new information. It's like a course correction based on the latest observation.
    /// 
    /// The update combines:
    /// - The previous prediction (where the model thought things were heading)
    /// - The new observation (what actually happened)
    /// - The Kalman gain (how much to trust this new information)
    /// 
    /// Components with higher Kalman gain values will be adjusted more strongly in
    /// response to prediction errors. This lets the model adapt quickly to changes
    /// in some components while maintaining stability in others.
    /// 
    /// After this update, the model has incorporated the latest data point into its understanding.
    /// </remarks>
    private void UpdateState(Vector<T> predictedState, Vector<T> kalmanGain, T innovation)
    {
        if (predictedState.Length != kalmanGain.Length)
        {
            throw new ArgumentException("Predicted state and Kalman gain must have the same length");
        }

        // Calculate the update vector
        Vector<T> update = kalmanGain.Multiply(innovation);

        // Apply the update to the entire state vector
        Vector<T> updatedState = predictedState.Add(update);

        // Update individual components
        int index = 0;

        // Update level
        _level = updatedState[index++];

        // Update trend if included
        if (_bayesianOptions.IncludeTrend)
        {
            _trend = updatedState[index++];
        }

        // Update seasonal components
        for (int i = 0; i < _seasonalComponents.Count; i++)
        {
            for (int j = 0; j < _seasonalComponents[i].Length; j++)
            {
                _seasonalComponents[i][j] = updatedState[index++];
            }
        }

        // Update regression components if included
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = updatedState[index++];
            }
        }

        // Sanity check
        if (index != updatedState.Length)
        {
            throw new InvalidOperationException("State update mismatch: not all components were updated");
        }
    }

    /// <summary>
    /// Updates the state covariance matrix based on the Kalman gain.
    /// </summary>
    /// <param name="predictedCovariance">The predicted covariance matrix.</param>
    /// <param name="kalmanGain">The Kalman gain vector.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method updates the model's uncertainty about its components after
    /// observing new data. It's like adjusting your confidence levels based on whether
    /// your predictions were accurate.
    /// 
    /// The covariance update uses:
    /// - The previous uncertainty (predictedCovariance)
    /// - How much the model learned from the new data (kalmanGain)
    /// 
    /// Generally, this update reduces uncertainty for components that were able to
    /// learn effectively from the new observation. Components with higher Kalman gain
    /// values will see larger reductions in uncertainty.
    /// 
    /// This adaptation of uncertainty is another key feature of Bayesian methods.
    /// </remarks>
    private void UpdateCovariance(Matrix<T> predictedCovariance, Vector<T> kalmanGain)
    {
        Vector<T> observationVector = CreateObservationVector();
        Matrix<T> identity = Matrix<T>.CreateIdentity(predictedCovariance.Rows);
        Matrix<T> kalmanGainMatrix = kalmanGain.ToColumnMatrix();
        _stateCovariance = identity.Subtract(kalmanGainMatrix.Multiply(observationVector.ToRowMatrix())).Multiply(predictedCovariance);
    }

    /// <summary>
    /// Gets the current state vector for all model components.
    /// </summary>
    /// <returns>The current state vector.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method creates a vector that represents the model's current understanding
    /// of all time series components.
    /// 
    /// The state vector combines:
    /// - The current level (baseline value)
    /// - The trend (if included)
    /// - All seasonal components
    /// - Regression coefficients (if included)
    /// 
    /// This comprehensive state representation is what allows the model to track complex
    /// patterns and relationships over time. It's used both during training and when
    /// making predictions.
    /// </remarks>
    private Vector<T> GetCurrentState()
    {
        int stateSize = GetStateSize();
        Vector<T> currentState = new Vector<T>(stateSize);
        int index = 0;

        currentState[index++] = _level;
        if (_bayesianOptions.IncludeTrend) currentState[index++] = _trend;

        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                currentState[index++] = seasonalComponent[i];
            }
        }

        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                currentState[index++] = _regression[i];
            }
        }

        return currentState;
    }

    /// <summary>
    /// Performs backward smoothing to improve state estimates using future information.
    /// </summary>
    /// <param name="states">Matrix of state vectors for all time points.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method refines the model's understanding of the past using information
    /// from the future. It's like going back and reinterpreting earlier data in light of
    /// what happened later.
    /// 
    /// The Kalman filter only looks at data up to the current time point, which means
    /// early estimates don't benefit from later observations. Backward smoothing fixes this
    /// by going back and adjusting those earlier estimates using all available information.
    /// 
    /// For example, if the model initially thought a spike in sales was random noise, but
    /// later saw similar spikes every Friday, smoothing would help it recognize that the
    /// earlier spike was probably also a Friday effect.
    /// 
    /// This typically produces more accurate and stable estimates of historical patterns.
    /// </remarks>
    private void PerformBackwardSmoothing(Matrix<T> states)
    {
        int n = states.Rows;
        int stateSize = states.Columns;
        Matrix<T> smoothedStates = new Matrix<T>(n, stateSize);
        Matrix<T> transitionMatrix = CreateTransitionMatrix();

        // Initialize with the last state
        smoothedStates.SetRow(n - 1, states.GetRow(n - 1));

        for (int t = n - 2; t >= 0; t--)
        {
            Vector<T> currentState = states.GetRow(t);
            Vector<T> nextState = states.GetRow(t + 1);
            Vector<T> smoothedNextState = smoothedStates.GetRow(t + 1);

            Matrix<T> predictedCovariance = PredictCovariance();
            Matrix<T> smoothingGain = _stateCovariance
                .Multiply(transitionMatrix.Transpose())
                .Multiply(predictedCovariance.Inverse());

            Vector<T> stateDiff = smoothedNextState.Subtract(nextState);
            Vector<T> smoothedState = currentState.Add(smoothingGain.Multiply(stateDiff));

            smoothedStates.SetRow(t, smoothedState);
        }

        // Update model components with smoothed states
        UpdateModelComponentsFromSmoothedStates(smoothedStates);
    }

    /// <summary>
    /// Estimates optimal parameters for the model using an EM-like algorithm.
    /// </summary>
    /// <param name="x">Feature matrix of external variables.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <param name="states">Matrix of state vectors for all time points.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fine-tunes the model's parameters to better fit your data.
    /// It's like adjusting the settings on a camera until the picture is in perfect focus.
    /// 
    /// The method uses an approach similar to the "Expectation-Maximization" algorithm, which:
    /// 1. Runs the Kalman filter and smoother to get best estimates for all time points (E-step)
    /// 2. Uses these estimates to update model parameters like variances (M-step)
    /// 3. Repeats until the improvements become very small
    /// 
    /// This process helps the model find the optimal balance between:
    /// - Fitting your historical data well
    /// - Avoiding overfitting (being too sensitive to noise)
    /// - Maintaining appropriate uncertainty estimates
    /// 
    /// These optimized parameters are crucial for making accurate predictions with
    /// appropriate confidence intervals.
    /// </remarks>
    private void EstimateParameters(Matrix<T> x, Vector<T> y, Matrix<T> states)
    {
        T previousLogLikelihood = NumOps.MinValue;

        for (int iteration = 0; iteration < _bayesianOptions.MaxIterations; iteration++)
        {
            // E-step: Run Kalman filter and smoother
            Matrix<T> smoothedStates = RunKalmanFilterAndSmoother(x, y);

            // M-step: Update model parameters
            T currentLogLikelihood = UpdateModelParameters(x, y, smoothedStates);

            // Check for convergence
            if (CheckConvergence(previousLogLikelihood, currentLogLikelihood))
            {
                break;
            }

            previousLogLikelihood = currentLogLikelihood;
        }
    }

    /// <summary>
    /// Runs a single iteration of Kalman filter and smoother.
    /// </summary>
    /// <param name="x">Feature matrix of external variables.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <returns>Matrix of smoothed state vectors.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method performs one complete pass through your data, both forward and backward.
    /// 
    /// The forward pass (Kalman filter):
    /// - Starts at the beginning of your time series
    /// - Makes a prediction for each point, one at a time
    /// - Updates the model based on how accurate each prediction was
    /// 
    /// The backward pass (smoother), if enabled:
    /// - Goes back through the data in reverse
    /// - Refines earlier estimates using information from later observations
    /// 
    /// This combined approach helps the model extract the maximum information from your data.
    /// The result is a matrix containing the best estimate of each model component at each
    /// time point in your data.
    /// </remarks>
    private Matrix<T> RunKalmanFilterAndSmoother(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        int stateSize = GetStateSize();
        Matrix<T> filteredStates = new Matrix<T>(n, stateSize);

        // Forward pass (Kalman filter)
        for (int t = 0; t < n; t++)
        {
            Vector<T> predictedState = PredictState(x.GetRow(t));
            Matrix<T> predictedCovariance = PredictCovariance();
            T innovation = CalculateInnovation(y[t], predictedState);
            var kalmanGain = CalculateKalmanGain(predictedCovariance);
            UpdateState(predictedState, kalmanGain, innovation);
            UpdateCovariance(predictedCovariance, kalmanGain);
            filteredStates.SetRow(t, GetCurrentState());
        }

        // Backward pass (smoother)
        if (_bayesianOptions.PerformBackwardSmoothing)
        {
            PerformBackwardSmoothing(filteredStates);
            return filteredStates;
        }

        return filteredStates;
    }

    /// <summary>
    /// Updates model parameters based on smoothed states.
    /// </summary>
    /// <param name="x">Feature matrix of external variables.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <param name="smoothedStates">Matrix of smoothed state vectors.</param>
    /// <returns>The log-likelihood of the current model.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method adjusts the model's parameters to better match your data.
    /// It's like fine-tuning the model based on what it learned from analyzing the entire time series.
    /// 
    /// The method updates:
    /// - Variances for level, trend, and seasonal components (how much they can change over time)
    /// - Observation variance (how noisy the data is)
    /// 
    /// For each component, it looks at how much that component actually changed throughout
    /// your time series, and sets the variance parameter accordingly. Components that changed
    /// a lot will get higher variance, allowing them to adapt more quickly in the future.
    /// 
    /// The method also calculates how well the model fits your data overall, returning a
    /// log-likelihood score that helps determine when to stop training.
    /// </remarks>
    private T UpdateModelParameters(Matrix<T> x, Vector<T> y, Matrix<T> smoothedStates)
    {
        T logLikelihood = NumOps.Zero;
        int n = y.Length;
        int stateSize = GetStateSize();

        // Update level and trend variances
        T levelVariance = NumOps.Zero;
        T trendVariance = NumOps.Zero;
        for (int t = 1; t < n; t++)
        {
            Vector<T> prevState = smoothedStates.GetRow(t - 1);
            Vector<T> currState = smoothedStates.GetRow(t);
            levelVariance = NumOps.Add(levelVariance, NumOps.Square(NumOps.Subtract(currState[0], prevState[0])));
            if (_bayesianOptions.IncludeTrend)
            {
                trendVariance = NumOps.Add(trendVariance, NumOps.Square(NumOps.Subtract(currState[1], prevState[1])));
            }
        }
        levelVariance = NumOps.Divide(levelVariance, NumOps.FromDouble(n - 1));
        _stateCovariance[0, 0] = levelVariance;
        if (_bayesianOptions.IncludeTrend)
        {
            trendVariance = NumOps.Divide(trendVariance, NumOps.FromDouble(n - 1));
            _stateCovariance[1, 1] = trendVariance;
        }

        // Update seasonal variances
        int seasonalIndex = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            T seasonalVariance = NumOps.Zero;
            for (int t = 1; t < n; t++)
            {
                Vector<T> prevSeasonalSlice = smoothedStates.GetRow(t - 1).Slice(seasonalIndex, seasonalComponent.Length);
                Vector<T> currSeasonalSlice = smoothedStates.GetRow(t).Slice(seasonalIndex, seasonalComponent.Length);
                var diff = (Vector<T>)Engine.Subtract(currSeasonalSlice, prevSeasonalSlice);
                T sumSquares = Engine.DotProduct(diff, diff);
                seasonalVariance = NumOps.Add(seasonalVariance, sumSquares);
            }
            seasonalVariance = NumOps.Divide(seasonalVariance, NumOps.FromDouble((n - 1) * seasonalComponent.Length));
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                _stateCovariance[seasonalIndex + i, seasonalIndex + i] = seasonalVariance;
            }
            seasonalIndex += seasonalComponent.Length;
        }

        // Update observation variance
        T totalVariance = NumOps.Zero;
        for (int t = 0; t < n; t++)
        {
            T prediction = CalculatePrediction(smoothedStates.GetRow(t));
            T error = NumOps.Subtract(y[t], prediction);
            totalVariance = NumOps.Add(totalVariance, NumOps.Square(error));
            logLikelihood = NumOps.Add(logLikelihood, NumOps.Log(NumOps.Abs(error)));
        }
        _observationVariance = NumOps.Divide(totalVariance, NumOps.FromDouble(n));

        return logLikelihood;
    }

    /// <summary>
    /// Checks if the parameter estimation has converged.
    /// </summary>
    /// <param name="previousLogLikelihood">The log-likelihood from the previous iteration.</param>
    /// <param name="currentLogLikelihood">The log-likelihood from the current iteration.</param>
    /// <returns>True if the estimation has converged, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method determines when to stop the parameter estimation process.
    /// 
    /// It compares how much the model improved from one iteration to the next.
    /// If the improvement (measured by log-likelihood) is smaller than the
    /// convergence tolerance, the method returns true, indicating that further
    /// iterations would only provide minimal benefits.
    /// 
    /// This is similar to how you might stop adjusting a TV's picture settings
    /// once additional tweaks stop making noticeable improvements.
    /// </remarks>
    private bool CheckConvergence(T previousLogLikelihood, T currentLogLikelihood)
    {
        T difference = NumOps.Abs(NumOps.Subtract(currentLogLikelihood, previousLogLikelihood));
        T threshold = NumOps.FromDouble(_bayesianOptions.ConvergenceTolerance);

        return NumOps.LessThanOrEquals(difference, threshold);
    }

    /// <summary>
    /// Creates the transition matrix for the state-space model.
    /// </summary>
    /// <returns>The transition matrix.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method creates a matrix that defines how the model components
    /// are expected to evolve from one time point to the next.
    /// 
    /// The transition matrix encodes rules like:
    /// - The level updates based on previous level plus trend
    /// - Seasonal components cycle through their pattern
    /// - Regression coefficients persist with possible changes
    /// 
    /// Think of it as describing the natural evolution of the time series components
    /// if no new information were observed. It's like predicting how a system will
    /// behave if left to follow its current trajectory.
    /// 
    /// This matrix is a key part of the state-space representation used in Kalman filtering.
    /// </remarks>
    private Matrix<T> CreateTransitionMatrix()
    {
        int stateSize = GetStateSize();
        Matrix<T> transitionMatrix = Matrix<T>.CreateIdentity(stateSize);

        // Add trend component if included
        if (_bayesianOptions.IncludeTrend)
        {
            transitionMatrix[0, 1] = NumOps.FromDouble(1.0);
        }

        // Add seasonal components
        int index = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length - 1; i++)
            {
                transitionMatrix[index + i, index + i + 1] = NumOps.FromDouble(1.0);
            }
            transitionMatrix[index + seasonalComponent.Length - 1, index] = NumOps.FromDouble(-1.0);
            index += seasonalComponent.Length;
        }

        return transitionMatrix;
    }

    /// <summary>
    /// Creates the process noise matrix for the state-space model.
    /// </summary>
    /// <returns>The process noise matrix.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method creates a matrix that defines how much random variability
    /// is expected in each model component from one time point to the next.
    /// 
    /// Higher values in this matrix allow components to change more rapidly in response to new data.
    /// For example:
    /// - A high noise value for level means the baseline can shift quickly
    /// - A low noise value for trend means the trend changes gradually
    /// 
    /// The values come from the "smoothing priors" specified in the model options.
    /// These act like settings for how stable vs. responsive each component should be:
    /// - Lower values make components more stable (change slowly)
    /// - Higher values make components more responsive (adapt quickly)
    /// 
    /// Finding the right balance is important for good forecasting performance.
    /// </remarks>
    private Matrix<T> CreateProcessNoiseMatrix()
    {
        int stateSize = GetStateSize();
        Matrix<T> processNoiseMatrix = new Matrix<T>(stateSize, stateSize);

        // Set variances for level and trend
        processNoiseMatrix[0, 0] = NumOps.FromDouble(_bayesianOptions.LevelSmoothingPrior);
        if (_bayesianOptions.IncludeTrend)
        {
            processNoiseMatrix[1, 1] = NumOps.FromDouble(_bayesianOptions.TrendSmoothingPrior);
        }

        // Set variances for seasonal components
        int index = _bayesianOptions.IncludeTrend ? 2 : 1;
        foreach (var seasonalComponent in _seasonalComponents)
        {
            for (int i = 0; i < seasonalComponent.Length; i++)
            {
                processNoiseMatrix[index + i, index + i] = NumOps.FromDouble(_bayesianOptions.SeasonalSmoothingPrior);
            }
            index += seasonalComponent.Length;
        }

        return processNoiseMatrix;
    }

    /// <summary>
    /// Creates the observation vector for the state-space model.
    /// </summary>
    /// <returns>The observation vector.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method creates a vector that defines how each component
    /// of the state contributes to the observed value.
    /// 
    /// The observation vector maps the internal state (which might have many components)
    /// to what we actually observe in the time series. It's like defining how all the
    /// internal components combine to produce the final visible value.
    /// 
    /// For most components, the value is 1.0, meaning they directly contribute
    /// to the observed value. However, some components might have different values
    /// or even zeros if they don't directly affect observations.
    /// 
    /// This vector is used in the Kalman filter to connect the model's internal
    /// representation with the actual data points in your time series.
    /// </remarks>
    private Vector<T> CreateObservationVector()
    {
        int stateSize = GetStateSize();
        Vector<T> observationVector = new Vector<T>(stateSize);
        observationVector[0] = NumOps.FromDouble(1.0); // Level component

        int index = 1;
        if (_bayesianOptions.IncludeTrend)
        {
            observationVector[index] = NumOps.FromDouble(1.0); // Trend component
            index++;
        }

        // Seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            observationVector[index] = NumOps.FromDouble(1.0);
            index += seasonalComponent.Length;
        }

        // Regression components
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                observationVector[index + i] = NumOps.FromDouble(1.0);
            }
        }

        return observationVector;
    }

    /// <summary>
    /// Updates model components from the smoothed states.
    /// </summary>
    /// <param name="smoothedStates">Matrix of smoothed state vectors.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method updates the model's components based on the smoother's results.
    /// 
    /// After running the backward smoother, which refines state estimates using future
    /// information, this method extracts the final refined values for:
    /// - Level (baseline value)
    /// - Trend (if included)
    /// - Seasonal components
    /// - Regression coefficients (if included)
    /// 
    /// It uses the very last smoothed state, which represents the best estimate of
    /// the model's current state incorporating all available information. These values
    /// will be used as the starting point for future predictions.
    /// </remarks>
    private void UpdateModelComponentsFromSmoothedStates(Matrix<T> smoothedStates)
    {
        Vector<T> lastState = smoothedStates.GetRow(smoothedStates.Rows - 1);
        int index = 0;

        _level = lastState[index++];
        if (_bayesianOptions.IncludeTrend) _trend = lastState[index++];

        for (int i = 0; i < _seasonalComponents.Count; i++)
        {
            for (int j = 0; j < _seasonalComponents[i].Length; j++)
            {
                _seasonalComponents[i][j] = lastState[index++];
            }
        }

        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                _regression[i] = lastState[index++];
            }
        }
    }

    /// <summary>
    /// Makes predictions using the trained BSTS model.
    /// </summary>
    /// <param name="input">Matrix of exogenous variables for the periods to predict.</param>
    /// <returns>Vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained model to forecast future values of your time series.
    /// 
    /// For each future time point in your input:
    /// 1. It predicts the state of all model components
    /// 2. Combines these components to generate a prediction
    /// 
    /// The prediction includes:
    /// - The level (baseline value)
    /// - Trend projections (if enabled)
    /// - Seasonal effects appropriate for that time point
    /// - Effects of external variables (if regression is enabled)
    /// 
    /// Unlike simpler models, BSTS accounts for all these components separately,
    /// giving you more accurate and interpretable forecasts.
    /// 
    /// If you need uncertainty intervals around these predictions, you would typically
    /// use additional methods that build on this prediction process.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int horizon = input.Rows;
        Vector<T> predictions = new Vector<T>(horizon);

        for (int t = 0; t < horizon; t++)
        {
            Vector<T> state = PredictState(input.GetRow(t));
            predictions[t] = CalculatePrediction(state);
        }

        return predictions;
    }

    /// <summary>
    /// Calculates a prediction from a state vector.
    /// </summary>
    /// <param name="state">The state vector.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method combines all the model components to produce a single predicted value.
    /// 
    /// It adds together:
    /// - The level component (baseline value)
    /// - The trend component (if included)
    /// - The appropriate seasonal component for this time point
    /// - The effects of external variables (if regression is included)
    /// 
    /// Think of it as assembling the different pieces of the forecast. For example,
    /// a sales forecast might combine:
    /// - Base sales level (100 units)
    /// - Upward trend (+5 units per month)
    /// - December holiday effect (+20 units)
    /// - Effect of ongoing promotion (+15 units)
    /// 
    /// The result would be a prediction of 140 units for December during the promotion.
    /// </remarks>
    private T CalculatePrediction(Vector<T> state)
    {
        // In a Bayesian Structural Time Series model, the prediction is typically
        // the sum of the level, trend, and seasonal components
        T prediction = state[0]; // Level is always the first component

        int index = 1;
        if (_bayesianOptions.IncludeTrend)
        {
            prediction = NumOps.Add(prediction, state[index]);
            index++;
        }

        // Add seasonal components
        foreach (var seasonalComponent in _seasonalComponents)
        {
            prediction = NumOps.Add(prediction, state[index]);
            index += seasonalComponent.Length;
        }

        // Add regression component if present
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            for (int i = 0; i < _regression.Length; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(state[index + i], _regression[i]));
            }
        }

        return prediction;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Matrix of exogenous variables for testing.</param>
    /// <param name="yTest">Vector of actual target values for testing.</param>
    /// <returns>Dictionary of evaluation metrics (MSE, RMSE, MAE, MAPE).</returns>
    /// <remarks>
    /// For Beginners:
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// 
    /// - MSE (Mean Squared Error): Average of squared differences between predictions and actual values.
    ///   Lower is better, but squaring emphasizes large errors. MSE is useful for comparing models
    ///   but harder to interpret directly.
    /// 
    /// - RMSE (Root Mean Squared Error): Square root of MSE, which gives errors in the same units
    ///   as your original data. For example, if forecasting sales in dollars, RMSE is also in dollars.
    /// 
    /// - MAE (Mean Absolute Error): Average of absolute differences between predictions and actual values.
    ///   Easier to interpret than MSE and treats all error sizes equally.
    /// 
    /// - MAPE (Mean Absolute Percentage Error): Average of percentage differences between predictions
    ///   and actual values. Useful for understanding relative size of errors. For example, MAPE = 5%
    ///   means predictions are off by 5% on average.
    /// 
    /// Lower values for all these metrics indicate better performance.
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = [];

        // Calculate Mean Squared Error (MSE)
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);
        metrics["MSE"] = mse;

        // Calculate Root Mean Squared Error (RMSE)
        T rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);
        metrics["RMSE"] = rmse;

        // Calculate Mean Absolute Error (MAE)
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);
        metrics["MAE"] = mae;

        // Calculate Mean Absolute Percentage Error (MAPE)
        T mape = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions);
        metrics["MAPE"] = mape;

        return metrics;
    }

    /// <summary>
    /// Gets the total size of the state vector.
    /// </summary>
    /// <returns>The size of the state vector.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how many components the model is tracking in total.
    /// 
    /// The state size includes:
    /// - Level component (always included)
    /// - Trend component (if enabled)
    /// - All seasonal components (for each period specified)
    /// - Regression coefficients (if included)
    /// 
    /// This total determines the size of vectors and matrices used in the Kalman filter calculations.
    /// A larger state size means the model is tracking more components, which can make it more
    /// flexible but also more complex and potentially slower to train.
    /// </remarks>
    private int GetStateSize()
    {
        int size = 1; // Always include level
        if (_bayesianOptions.IncludeTrend) size++;
        size += _seasonalComponents.Sum(s => s.Length);
        if (_bayesianOptions.IncludeRegression && _regression != null) size += _regression.Length;

        return size;
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
    /// The method saves all essential components:
    /// - The level and trend values
    /// - Seasonal components
    /// - State covariance matrix
    /// - Observation variance
    /// - Key model options
    /// 
    /// This allows the model to be fully reconstructed later.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Serialize model parameters
        writer.Write(Convert.ToDouble(_level));

        if (_bayesianOptions.IncludeTrend)
        {
            writer.Write(Convert.ToDouble(_trend));
        }

        writer.Write(_seasonalComponents.Count);
        foreach (var component in _seasonalComponents)
        {
            writer.Write(component.Length);
            foreach (var val in component) writer.Write(Convert.ToDouble(val));
        }

        writer.Write(_stateCovariance.Rows);
        writer.Write(_stateCovariance.Columns);
        for (int i = 0; i < _stateCovariance.Rows; i++)
            for (int j = 0; j < _stateCovariance.Columns; j++)
                writer.Write(Convert.ToDouble(_stateCovariance[i, j]));

        writer.Write(Convert.ToDouble(_observationVariance));

        // Serialize options
        writer.Write(_bayesianOptions.IncludeTrend);
        writer.Write(_bayesianOptions.IncludeRegression);
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
    /// The method loads all essential components that were saved during serialization:
    /// - The level and trend values
    /// - Seasonal components
    /// - State covariance matrix
    /// - Observation variance
    /// - Key model options
    /// 
    /// After deserialization, the model is ready to make predictions as if it had just been trained.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Deserialize model parameters
        _level = NumOps.FromDouble(reader.ReadDouble());

        if (_bayesianOptions.IncludeTrend)
        {
            _trend = NumOps.FromDouble(reader.ReadDouble());
        }

        int seasonalComponentsCount = reader.ReadInt32();
        _seasonalComponents = new List<Vector<T>>();
        for (int i = 0; i < seasonalComponentsCount; i++)
        {
            int componentLength = reader.ReadInt32();
            Vector<T> component = new Vector<T>(componentLength);
            for (int j = 0; j < componentLength; j++) component[j] = NumOps.FromDouble(reader.ReadDouble());
            _seasonalComponents.Add(component);
        }

        int covarianceRows = reader.ReadInt32();
        int covarianceColumns = reader.ReadInt32();
        _stateCovariance = new Matrix<T>(covarianceRows, covarianceColumns);
        for (int i = 0; i < covarianceRows; i++)
            for (int j = 0; j < covarianceColumns; j++)
                _stateCovariance[i, j] = NumOps.FromDouble(reader.ReadDouble());

        _observationVariance = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize options
        _bayesianOptions.IncludeTrend = reader.ReadBoolean();
        _bayesianOptions.IncludeRegression = reader.ReadBoolean();
    }

    /// <summary>
    /// Creates a new instance of the BSTS model with the same options.
    /// </summary>
    /// <returns>A new instance of the BSTS model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new, uninitialized instance of the BSTS model with the same configuration options.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model with the same settings.
    /// 
    /// Think of this like creating a new blank notebook with the same paper quality, size, and number of pages
    /// as another notebook, but without copying any of the written content.
    /// 
    /// This is used internally by the framework to create new model instances when needed,
    /// such as when cloning a model or creating ensemble models.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new BayesianStructuralTimeSeriesModel<T>((BayesianStructuralTimeSeriesOptions<T>)Options);
    }

    /// <summary>
    /// Gets metadata about the trained model, including its type, components, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive information about the model, including its type, parameters, components,
    /// and serialized state. This metadata can be used for model inspection, selection, or persistence.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns a complete description of your trained model.
    /// 
    /// The metadata includes:
    /// - The type of model (BSTS in this case)
    /// - The current values for level, trend, and seasonal components
    /// - The configuration options you specified when creating the model
    /// - A serialized version of the entire model that can be saved
    /// 
    /// This is useful for:
    /// - Comparing different models to choose the best one
    /// - Documenting what model was used for a particular analysis
    /// - Saving model details for future reference
    /// - Understanding which components are most important in your forecasts
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var bstsOptions = (BayesianStructuralTimeSeriesOptions<T>)Options;
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.BayesianStructuralTimeSeriesModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "Level", Convert.ToDouble(_level) },
                { "Trend", _bayesianOptions.IncludeTrend ? Convert.ToDouble(_trend) : 0.0 },
                { "SeasonalComponents", _seasonalComponents.Select(c => c.ToArray()).ToArray() },
                { "ObservationVariance", Convert.ToDouble(_observationVariance) },
                { "StateCovariance", _stateCovariance.ToArray() },
                { "RegressionCoefficients", _regression?.ToArray() ?? [] },
            
                // Include model configuration as well
                { "IncludeTrend", bstsOptions.IncludeTrend },
                { "IncludeRegression", bstsOptions.IncludeRegression },
                { "SeasonalPeriods", bstsOptions.SeasonalPeriods },
                { "LevelSmoothingPrior", bstsOptions.LevelSmoothingPrior },
                { "TrendSmoothingPrior", bstsOptions.TrendSmoothingPrior },
                { "SeasonalSmoothingPrior", bstsOptions.SeasonalSmoothingPrior },
                { "PerformBackwardSmoothing", bstsOptions.PerformBackwardSmoothing },
                { "MaxIterations", bstsOptions.MaxIterations },
                { "ConvergenceTolerance", bstsOptions.ConvergenceTolerance },
                { "RidgeParameter", bstsOptions.RidgeParameter },
                { "RegressionDecompositionType", bstsOptions.RegressionDecompositionType.ToString() }
            },
            ModelData = this.Serialize()
        };
        return metadata;
    }

    /// <summary>
    /// Forecasts future values based on a history of time series data and optional exogenous variables.
    /// </summary>
    /// <param name="history">The historical time series data.</param>
    /// <param name="horizon">The number of future periods to predict.</param>
    /// <param name="exogenousVariables">Optional matrix of external variables for future periods.</param>
    /// <returns>A vector of predicted values for future periods.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for future time periods based on the trained model, historical data,
    /// and optional external variables.
    /// </para>
    /// <para><b>For Beginners:</b> This method forecasts future values based on past data.
    /// 
    /// Given:
    /// - A series of past observations (the "history")
    /// - The number of future periods to predict (the "horizon")
    /// - Optional external factors that might affect predictions
    /// 
    /// This method projects each component of the model forward:
    /// - The level continues at its current value with possible trend adjustments
    /// - Seasonal patterns repeat as expected
    /// - External variables influence the forecast according to learned relationships
    /// 
    /// For example, if forecasting retail sales, this method might predict:
    /// - Continuing the overall upward trend
    /// - Adding holiday peaks in appropriate months
    /// - Adjusting for planned promotions or price changes (if included as external variables)
    /// </para>
    /// </remarks>
    public Vector<T> Forecast(Vector<T> history, int horizon, Matrix<T>? exogenousVariables = null)
    {
        if (horizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be greater than zero.", nameof(horizon));
        }

        // Create the forecast vector
        Vector<T> forecast = new Vector<T>(horizon);

        // Create default exogenous variables if not provided and needed
        Matrix<T>? futureExog = exogenousVariables;
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            if (futureExog == null || futureExog.Rows != horizon || futureExog.Columns != _regression.Length)
            {
                // Create a zero matrix as fallback
                futureExog = new Matrix<T>(horizon, _regression.Length);
            }
        }

        // Track the seasonal position
        List<int> seasonalPositions = new List<int>();
        foreach (var seasonComponent in _seasonalComponents)
        {
            seasonalPositions.Add(0); // Start at position 0 for each seasonal component
        }

        // Make predictions
        T currentLevel = _level;
        T currentTrend = _bayesianOptions.IncludeTrend ? _trend : NumOps.Zero;

        for (int t = 0; t < horizon; t++)
        {
            // Start with the level
            T prediction = currentLevel;

            // Add trend if included
            if (_bayesianOptions.IncludeTrend)
            {
                prediction = NumOps.Add(prediction, currentTrend);
                // Update level for next period
                currentLevel = NumOps.Add(currentLevel, currentTrend);
            }

            // Add seasonal components
            for (int i = 0; i < _seasonalComponents.Count; i++)
            {
                int seasonLength = _seasonalComponents[i].Length;
                prediction = NumOps.Add(prediction, _seasonalComponents[i][seasonalPositions[i]]);

                // Update seasonal position for next period
                seasonalPositions[i] = (seasonalPositions[i] + 1) % seasonLength;
            }

            // Add regression component if included
            if (_bayesianOptions.IncludeRegression && _regression != null && futureExog != null)
            {
                Vector<T> exogRow = futureExog.GetRow(t);
                for (int i = 0; i < _regression.Length; i++)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(exogRow[i], _regression[i]));
                }
            }

            forecast[t] = prediction;
        }

        return forecast;
    }

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the trained components, effectively resetting the model to its initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method erases all the learned patterns from your model.
    /// 
    /// After calling this method:
    /// - The level returns to its initial value
    /// - The trend (if included) returns to its initial value
    /// - All seasonal components are reset to zero
    /// - Regression coefficients (if included) are cleared
    /// - All variance parameters return to their initial values
    /// 
    /// The model behaves as if it was never trained, and you would need to train it again before
    /// making predictions.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        // Reset level and trend to initial values
        _level = NumOps.FromDouble(_bayesianOptions.InitialLevelValue);
        _trend = _bayesianOptions.IncludeTrend ? NumOps.FromDouble(_bayesianOptions.InitialTrendValue) : NumOps.Zero;

        // Reset seasonal components
        _seasonalComponents.Clear();
        foreach (int period in _bayesianOptions.SeasonalPeriods)
        {
            _seasonalComponents.Add(new Vector<T>(period));
        }

        // Reset observation variance
        _observationVariance = NumOps.FromDouble(_bayesianOptions.InitialObservationVariance);

        // Reset state covariance 
        int stateSize = GetStateSize();
        _stateCovariance = new Matrix<T>(stateSize, stateSize);

        // Reset regression component if included
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            _regression = new Vector<T>(_regression.Length);
        }
    }

    /// <summary>
    /// Creates a deep copy of the current model.
    /// </summary>
    /// <returns>A new instance of the BSTS model with the same state and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the model, including its configuration and trained components.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of your trained model.
    /// 
    /// Unlike CreateInstance(), which creates a blank model with the same settings,
    /// Clone() creates a complete copy including:
    /// - The model configuration (level, trend, seasonal settings, etc.)
    /// - All trained components and their current values
    /// - The current uncertainty estimates
    /// 
    /// This is useful for:
    /// - Creating a backup before experimenting with a model
    /// - Using the same trained model in multiple scenarios
    /// - Creating ensemble models that use variations of the same base model
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new BayesianStructuralTimeSeriesModel<T>((BayesianStructuralTimeSeriesOptions<T>)Options);

        // Copy level and trend
        clone._level = _level;
        if (_bayesianOptions.IncludeTrend)
        {
            clone._trend = _trend;
        }

        // Copy seasonal components
        clone._seasonalComponents.Clear();
        foreach (var component in _seasonalComponents)
        {
            Vector<T> componentCopy = new Vector<T>(component.Length);
            for (int i = 0; i < component.Length; i++)
            {
                componentCopy[i] = component[i];
            }
            clone._seasonalComponents.Add(componentCopy);
        }

        // Copy state covariance
        clone._stateCovariance = new Matrix<T>(_stateCovariance.Rows, _stateCovariance.Columns);
        for (int i = 0; i < _stateCovariance.Rows; i++)
        {
            for (int j = 0; j < _stateCovariance.Columns; j++)
            {
                clone._stateCovariance[i, j] = _stateCovariance[i, j];
            }
        }

        // Copy observation variance
        clone._observationVariance = _observationVariance;

        // Copy regression component if included
        if (_bayesianOptions.IncludeRegression && _regression != null)
        {
            clone._regression = new Vector<T>(_regression.Length);
            for (int i = 0; i < _regression.Length; i++)
            {
                clone._regression[i] = _regression[i];
            }
        }

        return clone;
    }

    /// <summary>
    /// Implements the core training algorithm for the Bayesian Structural Time Series model.
    /// </summary>
    /// <param name="x">Feature matrix of external variables.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <remarks>
    /// <para>
    /// This method contains the implementation details of the training process, handling initialization,
    /// Kalman filtering, smoothing, and parameter estimation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine room of the training process.
    /// 
    /// While the public Train method provides a high-level interface, this method does the actual work:
    /// 1. Initializes regression coefficients for external factors (if included)
    /// 2. Runs the Kalman filter to estimate model components based on historical data
    /// 3. Optionally performs backward smoothing to improve estimates using all available information
    /// 4. Estimates optimal parameters to balance fit and flexibility
    /// 
    /// Think of it as the detailed step-by-step recipe that the chef follows when you order a meal.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;
        Matrix<T> states = new Matrix<T>(n, GetStateSize());

        // Initialize or update regression component if included
        if (_bayesianOptions.IncludeRegression)
        {
            if (_regression == null || _regression.Length != x.Columns)
            {
                _regression = new Vector<T>(x.Columns);
            }

            // Initialize regression coefficients using Ordinary Least Squares (OLS)
            InitializeRegressionCoefficients(x, y);
        }

        // Kalman filter
        for (int t = 0; t < n; t++)
        {
            // Prediction step
            Vector<T> predictedState = PredictState(x.GetRow(t));
            Matrix<T> predictedCovariance = PredictCovariance();

            // Update step
            T innovation = CalculateInnovation(y[t], predictedState);
            var kalmanGain = CalculateKalmanGain(predictedCovariance);
            UpdateState(predictedState, kalmanGain, innovation);
            UpdateCovariance(predictedCovariance, kalmanGain);

            // Store state
            states.SetRow(t, GetCurrentState());
        }

        // Backward smoothing (optional)
        if (_bayesianOptions.PerformBackwardSmoothing)
        {
            PerformBackwardSmoothing(states);
        }

        // Parameter estimation (e.g., EM algorithm or variational inference)
        EstimateParameters(x, y, states);
    }

    /// <summary>
    /// Predicts a single value based on a single input vector of external regressors.
    /// </summary>
    /// <param name="input">Vector of external regressors for a single time point.</param>
    /// <returns>The predicted value for that time point.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a convenient way to get a prediction for a single time point without
    /// having to create a matrix with a single row.
    /// </para>
    /// <para><b>For Beginners:</b> This is a shortcut for getting just one prediction.
    /// 
    /// Instead of providing a table of inputs for multiple time periods, you can provide
    /// just one set of external factors and get back a single prediction.
    /// 
    /// For example, if you want to predict tomorrow's sales based on tomorrow's promotions, 
    /// weather forecast, and other factors, this method lets you do that directly.
    /// 
    /// Under the hood, it:
    /// 1. Takes your single set of inputs
    /// 2. Creates a small table with just one row
    /// 3. Gets a prediction using the main prediction engine
    /// 4. Returns that single prediction to you
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Validate input
        if (_bayesianOptions.IncludeRegression && _regression != null && input.Length != _regression.Length)
        {
            throw new ArgumentException(
                $"Input vector length ({input.Length}) must match the number of regression variables ({_regression.Length}).",
                nameof(input));
        }

        // Create a matrix with a single row
        Matrix<T> singleRowMatrix = new Matrix<T>(1, input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            singleRowMatrix[0, i] = input[i];
        }

        // Use the existing Predict method
        Vector<T> predictions = Predict(singleRowMatrix);

        // Return the single prediction
        return predictions[0];
    }

    /// <summary>
    /// Gets whether this model supports JIT compilation.
    /// </summary>
    /// <value>
    /// Returns <c>true</c> when the model has estimated components.
    /// Prediction uses the point estimates from Bayesian inference.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> While BSTS training uses MCMC sampling,
    /// prediction uses point estimates and can be JIT compiled.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => !NumOps.Equals(_level, NumOps.Zero) || !NumOps.Equals(_trend, NumOps.Zero);

    /// <summary>
    /// Exports the BSTS model as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">A list to which input nodes will be added.</param>
    /// <returns>The output computation node representing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// The computation graph represents: forecast = level + trend + seasonal
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
        {
            throw new ArgumentNullException(nameof(inputNodes), "Input nodes list cannot be null.");
        }

        // Create input node for time index
        var timeIndexTensor = new Tensor<T>(new[] { 1 });
        var timeIndexNode = TensorOperations<T>.Variable(timeIndexTensor, "time_index", requiresGradient: false);
        inputNodes.Add(timeIndexNode);

        // Start with level
        var levelTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { _level }));
        var resultNode = TensorOperations<T>.Constant(levelTensor, "level");

        // Add trend
        var trendTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { _trend }));
        var trendNode = TensorOperations<T>.Constant(trendTensor, "trend");
        resultNode = TensorOperations<T>.Add(resultNode, trendNode);

        // Add average seasonal effect
        if (_seasonalComponents != null && _seasonalComponents.Count > 0)
        {
            T avgSeasonal = NumOps.Zero;
            int count = 0;
            foreach (var component in _seasonalComponents)
            {
                for (int i = 0; i < component.Length; i++)
                {
                    avgSeasonal = NumOps.Add(avgSeasonal, component[i]);
                    count++;
                }
            }
            if (count > 0)
            {
                avgSeasonal = NumOps.Divide(avgSeasonal, NumOps.FromDouble(count));
            }
            var seasonalTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { avgSeasonal }));
            var seasonalNode = TensorOperations<T>.Constant(seasonalTensor, "seasonal");
            resultNode = TensorOperations<T>.Add(resultNode, seasonalNode);
        }

        return resultNode;
    }
}
