using Newtonsoft.Json;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Dynamic Regression model with ARIMA errors for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This model combines regression analysis with ARIMA (AutoRegressive Integrated Moving Average) error modeling.
/// It first models the relationship between the target variable and external predictors using regression,
/// then applies ARIMA modeling to the residuals to capture temporal patterns in the error terms.
/// </para>
/// 
/// <para>
/// For Beginners:
/// Dynamic Regression with ARIMA Errors is like having two powerful forecasting tools working together:
/// 
/// 1. Regression Component: This part captures how external factors (like temperature, price changes,
///    or marketing campaigns) affect what you're trying to predict. For example, if you're forecasting
///    ice cream sales, this component would measure how much each degree of temperature increases sales.
/// 
/// 2. ARIMA Error Component: After accounting for external factors, there are often still patterns
///    in the data that the regression alone can't explain. The ARIMA component captures these patterns
///    by looking at:
///    - Past values (AR - AutoRegressive)
///    - Trends removed through differencing (I - Integrated)
///    - Past prediction errors (MA - Moving Average)
/// 
/// When combined, these components create a powerful forecasting model that can:
/// - Account for the impact of known external factors
/// - Capture complex temporal patterns in the data
/// - Handle both stationary and non-stationary time series
/// - Provide more accurate forecasts than either approach alone
/// 
/// This model is particularly useful when you have both:
/// - External variables that influence your target variable
/// - Temporal patterns that persist in the data after accounting for these external influences
/// </para>
/// </remarks>
public class DynamicRegressionWithARIMAErrors<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Configuration options for the Dynamic Regression with ARIMA Errors model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These options control how the model works, including:
    /// - How many past values to consider (AR order)
    /// - How many past prediction errors to consider (MA order)
    /// - How many times to difference the data (difference order)
    /// - How many external factors to include in the regression
    /// - Which regularization method to use (to prevent overfitting)
    /// - Which matrix decomposition to use for solving linear systems
    /// </remarks>
    private DynamicRegressionWithARIMAErrorsOptions<T> _arimaOptions;

    /// <summary>
    /// Regularization method to prevent overfitting in the regression component.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Regularization helps prevent the model from becoming too complex and overfitting the training data.
    /// It's like putting constraints on the model so it focuses on the most important patterns
    /// rather than trying to explain every little fluctuation (which might just be noise).
    /// 
    /// Common regularization methods include:
    /// - L1 (Lasso): Can set some coefficients to exactly zero, effectively removing less important variables
    /// - L2 (Ridge): Shrinks all coefficients toward zero, but rarely makes them exactly zero
    /// - ElasticNet: A combination of L1 and L2 regularization
    /// </remarks>
    private IRegularization<T, Matrix<T>, Vector<T>> _regularization;

    /// <summary>
    /// Coefficients for the regression component, representing the impact of external variables.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients measure how much each external factor affects your target variable.
    /// 
    /// For example, if you're forecasting sales:
    /// - A coefficient of 5 for advertising spend means every additional $1 spent on ads results in 5 more units sold
    /// - A coefficient of -2 for price means every $1 price increase results in 2 fewer units sold
    /// 
    /// The model learns these coefficients from your historical data to quantify relationships
    /// between external factors and what you're predicting.
    /// </remarks>
    private Vector<T> _regressionCoefficients;

    /// <summary>
    /// Coefficients for the autoregressive (AR) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past value influences the prediction.
    /// 
    /// For example, if the AR coefficient for yesterday's value is 0.7, it means yesterday's
    /// value has a strong influence on today's prediction. A negative coefficient would mean
    /// the value tends to move in the opposite direction.
    /// 
    /// These coefficients are applied to the regression residuals (errors), not to the original time series.
    /// </remarks>
    private Vector<T> _arCoefficients;

    /// <summary>
    /// Coefficients for the moving average (MA) component of the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// These coefficients determine how much each past prediction error influences the forecast.
    /// 
    /// For example, if the MA coefficient for yesterday's error is 0.3, it means if yesterday's
    /// prediction was too high by 10 units, today's prediction will be adjusted downward by 3 units.
    /// 
    /// This helps the model correct for systematic errors in its predictions.
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// Values needed to reverse differencing when making predictions.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Differencing is a technique used to remove trends from data by looking at changes
    /// rather than absolute values.
    /// 
    /// For example, instead of working with temperatures [68, 70, 73, 71], differencing
    /// would transform this to [2, 3, -2] (the differences between consecutive values).
    /// 
    /// This field stores the original values needed to convert predictions back to the original scale.
    /// </remarks>
    private Vector<T> _differenced;

    /// <summary>
    /// The constant term (intercept) in the regression equation.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The intercept represents the baseline value when all other factors are zero.
    /// It's like the starting point before applying the effects of external factors
    /// and time series patterns.
    /// 
    /// For example, if you're predicting ice cream sales, the intercept might represent
    /// the baseline daily sales you'd expect regardless of temperature, promotions, etc.
    /// </remarks>
    private T _intercept;

    /// <summary>
    /// Creates a new Dynamic Regression with ARIMA Errors model with the specified options.
    /// </summary>
    /// <param name="options">Options for configuring the model, including AR order, MA order, 
    /// differencing order, and external regressors.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor creates a new instance of the model with the specified settings.
    /// 
    /// The options include:
    /// - AROrder: How many past values to consider (like using yesterday and the day before to predict today)
    /// - MAOrder: How many past prediction errors to consider
    /// - DifferenceOrder: How many times to difference the data to remove trends
    /// - ExternalRegressors: How many external variables to include in the model
    /// - Regularization: What method to use to prevent the model from becoming too complex
    /// 
    /// These settings should be chosen based on your specific data and forecasting needs.
    /// </remarks>
    public DynamicRegressionWithARIMAErrors(DynamicRegressionWithARIMAErrorsOptions<T> options) : base(options)
    {
        _arimaOptions = options;
        _regressionCoefficients = new Vector<T>(options.ExternalRegressors);
        _arCoefficients = new Vector<T>(options.AROrder);
        _maCoefficients = new Vector<T>(options.MAOrder);
        _differenced = new Vector<T>(0);
        _intercept = NumOps.Zero;
        _regularization = options.Regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
    }

    /// <summary>
    /// Makes predictions using the trained model.
    /// </summary>
    /// <param name="xNew">Matrix of external regressors for the periods to be predicted.</param>
    /// <returns>Vector of predicted values.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method uses the trained model to forecast future values.
    /// 
    /// For each future time point:
    /// 1. It starts with the intercept (baseline value)
    /// 2. Adds the effects of external factors (regression component)
    /// 3. Adds the effects of past observations (AR component)
    /// 4. Adds the effects of past prediction errors (MA component)
    /// 5. If differencing was used in training, it "undoes" the differencing to get predictions in the original scale
    /// 
    /// The combination of regression and ARIMA components allows the model to make forecasts
    /// that account for both external influences and internal time-dependent patterns.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> xNew)
    {
        Vector<T> predictions = new Vector<T>(xNew.Rows);

        for (int t = 0; t < xNew.Rows; t++)
        {
            T prediction = _intercept;

            // Apply regression component
            for (int i = 0; i < xNew.Columns; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(xNew[t, i], _regressionCoefficients[i]));
            }

            // Apply ARIMA component
            for (int p = 0; p < _arimaOptions.AROrder; p++)
            {
                if (t - p - 1 >= 0)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[p], NumOps.Subtract(predictions[t - p - 1], _intercept)));
                }
            }

            for (int q = 0; q < _arimaOptions.MAOrder; q++)
            {
                if (t - q - 1 >= 0)
                {
                    T error = NumOps.Subtract(predictions[t - q - 1], xNew[t - q - 1, 0]); // Assuming the first column is the target variable
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[q], error));
                }
            }

            predictions[t] = prediction;
        }

        // Reverse differencing if necessary
        if (_arimaOptions.DifferenceOrder > 0)
        {
            predictions = InverseDifferenceTimeSeries(predictions, _differenced);
        }

        return predictions;
    }

    /// <summary>
    /// Applies differencing to the time series to make it stationary.
    /// </summary>
    /// <param name="y">The original time series.</param>
    /// <param name="order">The number of times to apply differencing.</param>
    /// <returns>The differenced time series.</returns>
    /// <remarks>
    /// For Beginners:
    /// Differencing is a technique that transforms the data to remove trends. Instead of looking at
    /// absolute values, it looks at changes from one period to the next.
    /// 
    /// For example:
    /// - Original data: [10, 15, 14, 18]
    /// - After first-order differencing: [5, -1, 4] (the differences between consecutive values)
    /// - After second-order differencing: [-6, 5] (the differences of the differences)
    /// 
    /// This helps make the data "stationary," which means its statistical properties don't change over time.
    /// Many time series models work better with stationary data.
    /// 
    /// The "order" parameter tells the method how many times to apply this differencing operation.
    /// </remarks>
    private Vector<T> DifferenceTimeSeries(Vector<T> y, int order)
    {
        Vector<T> diffY = y;
        for (int d = 0; d < order; d++)
        {
            Vector<T> temp = new Vector<T>(diffY.Length - 1);
            for (int i = 0; i < temp.Length; i++)
            {
                temp[i] = NumOps.Subtract(diffY[i + 1], diffY[i]);
            }
            _differenced = new Vector<T>(diffY.Take(order));
            diffY = temp;
        }

        return diffY;
    }

    /// <summary>
    /// Reverses the differencing process to convert predictions back to the original scale.
    /// </summary>
    /// <param name="diffY">The differenced predictions.</param>
    /// <param name="original">The original values needed to reverse the differencing.</param>
    /// <returns>Predictions in the original scale.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method "undoes" the differencing that was applied during training. Since differencing
    /// looks at changes rather than absolute values, we need the original starting values to
    /// convert back.
    /// 
    /// For example:
    /// - If original data was [10, 15, 14, 18]
    /// - And differenced data was [5, -1, 4]
    /// - To predict the next value in the original scale, we need both the prediction in the differenced scale
    ///   (let's say 2) and the last value of the original series (18)
    /// - The prediction in the original scale would be 18 + 2 = 20
    /// 
    /// This method handles this conversion process, potentially through multiple levels of differencing.
    /// </remarks>
    private Vector<T> InverseDifferenceTimeSeries(Vector<T> diffY, Vector<T> original)
    {
        Vector<T> y = diffY;
        for (int d = _arimaOptions.DifferenceOrder - 1; d >= 0; d--)
        {
            Vector<T> temp = new Vector<T>(y.Length + 1);
            temp[0] = original[d];
            for (int i = 1; i < temp.Length; i++)
            {
                temp[i] = NumOps.Add(temp[i - 1], y[i - 1]);
            }
            y = temp;
        }

        return y;
    }

    /// <summary>
    /// Fits the regression component of the model to the data.
    /// </summary>
    /// <param name="x">Feature matrix of external regressors.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fits the regression part of the model, which captures how external
    /// factors affect your target variable.
    /// 
    /// It uses Ordinary Least Squares (OLS) regression, which is a standard statistical technique
    /// for estimating relationships between variables. The method:
    /// 
    /// 1. Calculates how external factors relate to your target variable
    /// 2. Finds the coefficient values that minimize prediction errors
    /// 3. Computes an intercept (baseline value)
    /// 
    /// The result is a set of coefficients that tell you how much each external factor
    /// influences your time series.
    /// </remarks>
    private void FitRegressionModel(Matrix<T> x, Vector<T> y)
    {
        // Use OLS or other regression method to fit the model
        Matrix<T> xT = x.Transpose();
        Matrix<T> xTx = xT * x;
        Vector<T> xTy = xT * y;

        _regressionCoefficients = MatrixSolutionHelper.SolveLinearSystem(xTx, xTy, _arimaOptions.DecompositionType);
        _intercept = NumOps.Divide(Engine.Sum(y), NumOps.FromDouble(y.Length));
    }

    /// <summary>
    /// Calculates the residuals from the regression model.
    /// </summary>
    /// <param name="x">Feature matrix of external regressors.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <returns>Vector of residuals (prediction errors).</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how much of your time series the regression model
    /// couldn't explain. These "leftovers" are called residuals.
    /// 
    /// For each time point, it:
    /// 1. Uses the regression model to make a prediction
    /// 2. Subtracts this prediction from the actual value
    /// 3. The difference is the residual
    /// 
    /// These residuals often still contain time-dependent patterns that the ARIMA
    /// component will try to capture. For example, even after accounting for temperature
    /// effects on ice cream sales, there might still be weekly patterns in the residuals.
    /// </remarks>
    private Vector<T> ExtractResiduals(Matrix<T> x, Vector<T> y)
    {
        Vector<T> predictions = x * _regressionCoefficients;
        return y - predictions;
    }

    /// <summary>
    /// Fits an ARIMA model to the residuals from the regression component.
    /// </summary>
    /// <param name="residuals">Vector of residuals from the regression model.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fits the ARIMA part of the model to capture patterns in what
    /// the regression couldn't explain.
    /// 
    /// The process involves:
    /// 1. Calculating how residuals correlate with past residuals (autocorrelations)
    /// 2. Estimating AR coefficients (how past values affect future values)
    /// 3. Estimating MA coefficients (how past errors affect future values)
    /// 4. Fine-tuning these coefficients to best match the observed patterns
    /// 
    /// This helps the model capture cyclical patterns, momentum effects, and other
    /// time-dependent structures that remain after accounting for external factors.
    /// </remarks>
    private void FitARIMAModel(Vector<T> residuals)
    {
        int p = _arimaOptions.AROrder;
        int q = _arimaOptions.MAOrder;
        int maxLag = Math.Max(p, q);

        // Calculate autocorrelations
        T[] autocorrelations = CalculateAutocorrelations(residuals, maxLag);

        // Estimate AR coefficients using Yule-Walker equations
        EstimateARCoefficients(autocorrelations, p);

        // Estimate MA coefficients using innovation algorithm
        EstimateMACoefficients(residuals, autocorrelations, q);

        // Perform joint optimization of AR and MA coefficients
        OptimizeARMACoefficients(residuals);
    }

    /// <summary>
    /// Estimates the AR coefficients using the Yule-Walker equations.
    /// </summary>
    /// <param name="autocorrelations">Array of autocorrelations.</param>
    /// <param name="p">The AR order.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method estimates how past values influence future values in the time series.
    /// 
    /// It uses the Yule-Walker equations, which are a set of mathematical formulas that:
    /// 1. Use the autocorrelations (how values relate to past values) as input
    /// 2. Solve a system of linear equations to find the optimal AR coefficients
    /// 
    /// The result is a set of coefficients that capture how each past value contributes
    /// to the prediction of future values. For example, an AR(2) model might show that
    /// yesterday's value has a strong positive effect while the day before has a weaker
    /// negative effect.
    /// 
    /// The method also includes error handling for cases where the equations can't be
    /// solved reliably (which can happen with certain data patterns).
    /// </remarks>
    private void EstimateARCoefficients(T[] autocorrelations, int p)
    {
        Matrix<T> R = new Matrix<T>(p, p);
        Vector<T> r = new Vector<T>(p);

        for (int i = 0; i < p; i++)
        {
            r[i] = autocorrelations[i + 1];
            for (int j = 0; j < p; j++)
            {
                R[i, j] = autocorrelations[Math.Abs(i - j)];
            }
        }

        try
        {
            _arCoefficients = MatrixSolutionHelper.SolveLinearSystem(R, r, _arimaOptions.DecompositionType);
        }
        catch (Exception ex)
        {
            // Handle potential numerical instability
            Console.WriteLine($"Error in AR coefficient estimation: {ex.Message}");
            _arCoefficients = new Vector<T>(p); // Initialize with zeros
        }
    }

    /// <summary>
    /// Estimates the MA coefficients using the innovation algorithm.
    /// </summary>
    /// <param name="residuals">Vector of residuals from the regression model.</param>
    /// <param name="autocorrelations">Array of autocorrelations.</param>
    /// <param name="q">The MA order.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method estimates how past prediction errors influence future values.
    /// 
    /// It uses an approach called the "innovation algorithm" that:
    /// 1. Uses both the residuals and their autocorrelations
    /// 2. Iteratively builds up the MA model, starting from simpler to more complex
    /// 3. Calculates coefficients that best explain the pattern of errors
    /// 
    /// The result is a set of coefficients that capture how each past prediction error
    /// contributes to future predictions. For example, an MA(1) model might show that
    /// if yesterday's prediction was too high, today's prediction should be adjusted downward.
    /// 
    /// This helps the model correct for systematic patterns in its errors over time.
    /// </remarks>
    private void EstimateMACoefficients(Vector<T> residuals, T[] autocorrelations, int q)
    {
        _maCoefficients = new Vector<T>(q);
        Vector<T> v = new Vector<T>(q + 1);
        v[0] = autocorrelations[0];

        for (int k = 1; k <= q; k++)
        {
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_maCoefficients[j - 1], v[k - j]));
            }
            _maCoefficients[k - 1] = NumOps.Divide(NumOps.Subtract(autocorrelations[k], sum), v[0]);

            for (int j = 1; j <= k; j++)
            {
                v[j] = NumOps.Subtract(v[j], NumOps.Multiply(_maCoefficients[k - 1], v[k - j]));
            }
            v[k] = NumOps.Multiply(_maCoefficients[k - 1], v[0]);
        }
    }

    /// <summary>
    /// Optimizes the AR and MA coefficients jointly to improve model fit.
    /// </summary>
    /// <param name="residuals">Vector of residuals from the regression model.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method fine-tunes both the AR and MA coefficients together to get
    /// the best possible model fit.
    /// 
    /// While the previous methods estimated AR and MA coefficients separately, this method:
    /// 1. Starts with those initial estimates
    /// 2. Uses an iterative approach to refine them together
    /// 3. Repeatedly calculates model residuals and likelihood
    /// 4. Updates coefficients to improve the model fit
    /// 5. Stops when improvements become small enough
    /// 
    /// This joint optimization is important because AR and MA components interact with each other,
    /// and optimizing them together can produce better results than optimizing them separately.
    /// 
    /// The method uses a form of gradient descent, which is like finding the bottom of a valley
    /// by taking small steps downhill from your current position.
    /// </remarks>
    private void OptimizeARMACoefficients(Vector<T> residuals)
    {
        int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-6);
        T prevLikelihood = NumOps.MinValue;

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Compute model residuals
            Vector<T> modelResiduals = ComputeModelResiduals(residuals);

            // Compute log-likelihood
            T likelihood = ComputeLogLikelihood(modelResiduals);

            // Check for convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(likelihood, prevLikelihood)), tolerance))
            {
                break;
            }

            prevLikelihood = likelihood;

            // Update AR coefficients
            UpdateARCoefficients(modelResiduals);

            // Update MA coefficients
            UpdateMACoefficients(modelResiduals);
        }
    }

    /// <summary>
    /// Computes residuals using the current ARMA model.
    /// </summary>
    /// <param name="residuals">Vector of residuals from the regression model.</param>
    /// <returns>Vector of model residuals.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how well the current ARIMA model explains
    /// the residuals from the regression.
    /// 
    /// For each time point, it:
    /// 1. Uses the AR and MA components to make a prediction
    /// 2. Subtracts this prediction from the actual residual
    /// 3. The difference is the "model residual" (error in explaining the error!)
    /// 
    /// These model residuals help measure how well the ARIMA component is capturing
    /// the time-dependent patterns. Smaller model residuals indicate a better fit.
    /// 
    /// This is used during optimization to assess how changes to the AR and MA
    /// coefficients impact model performance.
    /// </remarks>
    private Vector<T> ComputeModelResiduals(Vector<T> residuals)
    {
        int n = residuals.Length;
        Vector<T> modelResiduals = new Vector<T>(n);

        for (int t = 0; t < n; t++)
        {
            T prediction = NumOps.Zero;

            for (int i = 0; i < _arimaOptions.AROrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i], residuals[t - i - 1]));
            }

            for (int i = 0; i < _arimaOptions.MAOrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], modelResiduals[t - i - 1]));
            }

            modelResiduals[t] = NumOps.Subtract(residuals[t], prediction);
        }

        return modelResiduals;
    }

    /// <summary>
    /// Computes the log-likelihood of the current ARMA model.
    /// </summary>
    /// <param name="modelResiduals">Vector of model residuals.</param>
    /// <returns>The log-likelihood value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates a score that measures how well the current model
    /// fits the data. Higher log-likelihood values indicate better fit.
    /// 
    /// The calculation:
    /// 1. Sums the squares of the model residuals
    /// 2. Computes the variance (average squared residual)
    /// 3. Uses this variance in a formula to compute the log-likelihood
    /// 
    /// Log-likelihood is a standard statistical measure used to compare different models
    /// or different parameter settings within the same model. During optimization,
    /// the goal is to find parameters that maximize this value.
    /// 
    /// Think of it as a score card for your model - the higher the score, the better
    /// your model explains the observed data.
    /// </remarks>
    private T ComputeLogLikelihood(Vector<T> modelResiduals)
    {
        T sumSquaredResiduals = Engine.DotProduct(modelResiduals, modelResiduals);
        T variance = NumOps.Divide(sumSquaredResiduals, NumOps.FromDouble(modelResiduals.Length));
        T logLikelihood = NumOps.Multiply(NumOps.FromDouble(-0.5 * modelResiduals.Length),
            NumOps.Add(NumOps.Log(NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), variance)), NumOps.One));

        return logLikelihood;
    }

    /// <summary>
    /// Updates the AR coefficients using gradient descent.
    /// </summary>
    /// <param name="modelResiduals">Vector of model residuals.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method adjusts the AR coefficients to improve model fit.
    /// 
    /// It uses gradient descent, which is an optimization technique that:
    /// 1. Calculates the direction of steepest improvement (the gradient)
    /// 2. Takes a small step in that direction
    /// 3. This gradually moves toward better coefficient values
    /// 
    /// The learning rate controls the size of these steps:
    /// - Too small: Progress is slow but safe
    /// - Too large: Progress is fast but might overshoot the best solution
    /// 
    /// The method calls another function to calculate the gradients for each AR coefficient,
    /// then updates all coefficients accordingly.
    /// </remarks>
    private void UpdateARCoefficients(Vector<T> modelResiduals)
    {
        // Implement a gradient descent step for AR coefficients
        T learningRate = NumOps.FromDouble(0.01);
        for (int i = 0; i < _arimaOptions.AROrder; i++)
        {
            T gradient = ComputeARGradient(modelResiduals, i);
            _arCoefficients[i] = NumOps.Add(_arCoefficients[i], NumOps.Multiply(learningRate, gradient));
        }
    }

    /// <summary>
    /// Updates the MA coefficients using gradient descent.
    /// </summary>
    /// <param name="modelResiduals">Vector of model residuals.</param>
    /// <remarks>
    /// For Beginners:
    /// This private method adjusts the MA coefficients to improve model fit.
    /// 
    /// Similar to the AR coefficient update, it:
    /// 1. Calculates the gradient for each MA coefficient
    /// 2. Takes a small step in the direction of improvement
    /// 3. This gradually refines the coefficients
    /// 
    /// The MA coefficients capture how past prediction errors influence future predictions.
    /// Optimizing these coefficients helps the model better correct for systematic errors
    /// in its forecasts.
    /// 
    /// Both AR and MA coefficient updates together help the model find the optimal
    /// representation of time-dependent patterns in the data.
    /// </remarks>
    private void UpdateMACoefficients(Vector<T> modelResiduals)
    {
        // Implement a gradient descent step for MA coefficients
        T learningRate = NumOps.FromDouble(0.01);
        for (int i = 0; i < _arimaOptions.MAOrder; i++)
        {
            T gradient = ComputeMAGradient(modelResiduals, i);
            _maCoefficients[i] = NumOps.Add(_maCoefficients[i], NumOps.Multiply(learningRate, gradient));
        }
    }

    /// <summary>
    /// Computes the gradient for an AR coefficient.
    /// </summary>
    /// <param name="modelResiduals">Vector of model residuals.</param>
    /// <param name="lag">The lag index for which to compute the gradient.</param>
    /// <returns>The gradient value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how much and in which direction to adjust a specific
    /// AR coefficient to improve the model.
    /// 
    /// The gradient indicates:
    /// - The direction of adjustment (positive or negative)
    /// - The magnitude of adjustment (large or small)
    /// 
    /// It measures how changing the coefficient would affect the model residuals.
    /// A large gradient means changing the coefficient would have a big impact on model fit.
    /// 
    /// The calculation examines how residuals at each time point relate to residuals at
    /// the specified lag. This helps determine how important that particular lag is for
    /// prediction.
    /// </remarks>
    private T ComputeARGradient(Vector<T> modelResiduals, int lag)
    {
        T gradient = NumOps.Zero;
        for (int t = lag + 1; t < modelResiduals.Length; t++)
        {
            gradient = NumOps.Add(gradient, NumOps.Multiply(modelResiduals[t], modelResiduals[t - lag - 1]));
        }

        return NumOps.Multiply(NumOps.FromDouble(-2), gradient);
    }

    /// <summary>
    /// Computes the gradient for an MA coefficient.
    /// </summary>
    /// <param name="modelResiduals">Vector of model residuals.</param>
    /// <param name="lag">The lag index for which to compute the gradient.</param>
    /// <returns>The gradient value.</returns>
    /// <remarks>
    /// For Beginners:
    /// This private method calculates how much and in which direction to adjust a specific
    /// MA coefficient to improve the model.
    /// 
    /// Similar to the AR gradient, it measures the potential effect of changing the
    /// MA coefficient on model fit. However, the calculation is more complex because:
    /// 
    /// 1. MA terms involve previous model residuals (not observed residuals)
    /// 2. Changes to MA coefficients have cascading effects on all subsequent predictions
    /// 
    /// The gradient weighs how much each residual is influenced by previous errors,
    /// helping determine the optimal MA coefficient values for capturing patterns
    /// in the prediction errors.
    /// </remarks>
    private T ComputeMAGradient(Vector<T> modelResiduals, int lag)
    {
        T gradient = NumOps.Zero;
        for (int t = lag + 1; t < modelResiduals.Length; t++)
        {
            T prevError = NumOps.Zero;
            for (int i = 0; i < _arimaOptions.MAOrder && t - i - 1 >= 0; i++)
            {
                prevError = NumOps.Add(prevError, NumOps.Multiply(_maCoefficients[i], modelResiduals[t - i - 1]));
            }
            gradient = NumOps.Add(gradient, NumOps.Multiply(modelResiduals[t], prevError));
        }

        return NumOps.Multiply(NumOps.FromDouble(-2), gradient);
    }

    /// <summary>
    /// Calculates autocorrelations of a time series up to a specified lag.
    /// </summary>
    /// <param name="y">The time series.</param>
    /// <param name="maxLag">The maximum lag to calculate autocorrelations for.</param>
    /// <returns>Array of autocorrelations from lag 0 to maxLag.</returns>
    /// <remarks>
    /// For Beginners:
    /// Autocorrelation measures how similar a time series is to a delayed version of itself.
    /// It helps identify patterns and dependencies over time.
    /// 
    /// For example:
    /// - An autocorrelation of 0.8 at lag 1 means values tend to be very similar to the previous day's values
    /// - An autocorrelation of -0.4 at lag 7 means values tend to be opposite of values from a week ago
    /// - An autocorrelation near 0 means there's no relationship between values at that time lag
    /// 
    /// The result is an array where:
    /// - Position 0 contains the autocorrelation at lag 0 (always 1.0)
    /// - Position 1 contains the autocorrelation at lag 1 (correlation with previous value)
    /// - Position 2 contains the autocorrelation at lag 2 (correlation with value from 2 periods ago)
    /// And so on...
    /// 
    /// These autocorrelations help determine the appropriate structure for the ARIMA model.
    /// </remarks>
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

    /// <summary>
    /// Updates and optimizes model parameters before making predictions.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method applies several adjustments to ensure the model is stable
    /// and will generate reliable forecasts.
    /// 
    /// The adjustments include:
    /// 
    /// 1. Ensuring AR stationarity - This prevents the AR component from generating
    ///    predictions that explode or oscillate wildly
    /// 
    /// 2. Ensuring MA invertibility - This ensures the MA component behaves in a stable
    ///    and predictable way
    /// 
    /// 3. Normalizing regression coefficients - This helps prevent any single external
    ///    factor from dominating the model
    /// 
    /// 4. Applying regularization - This prevents overfitting by constraining coefficient values
    /// 
    /// 5. Updating the intercept - This adjusts the baseline prediction level based on
    ///    the differencing that was applied
    /// 
    /// These adjustments work together to ensure the model makes reasonable, stable predictions.
    /// </remarks>
    private void UpdateModelParameters()
    {
        // 1. Ensure stationarity of AR process
        EnsureARStationarity();

        // 2. Ensure invertibility of MA process
        EnsureMAInvertibility();

        // 3. Normalize regression coefficients
        NormalizeRegressionCoefficients();

        // 4. Apply regularization to prevent overfitting
        ApplyRegularization();

        // 5. Update intercept
        UpdateIntercept();
    }

    /// <summary>
    /// Applies regularization to the regression coefficients.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method helps prevent overfitting by applying constraints to the
    /// regression coefficients.
    /// 
    /// Overfitting happens when a model learns to explain noise or random fluctuations
    /// in the training data instead of the true underlying patterns. This makes the model
    /// perform well on training data but poorly on new data.
    /// 
    /// Regularization helps by:
    /// - Shrinking coefficient values toward zero
    /// - Potentially eliminating less important variables
    /// - Balancing model complexity against accuracy
    /// 
    /// The specific regularization method used depends on what was specified in the model options
    /// (L1/Lasso, L2/Ridge, ElasticNet, or none).
    /// </remarks>
    private void ApplyRegularization()
    {
        _regressionCoefficients = _regularization.Regularize(_regressionCoefficients);
    }

    /// <summary>
    /// Ensures the AR process is stationary by scaling coefficients if necessary.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method ensures that the AR component will produce stable predictions.
    /// 
    /// A "stationary" AR process means that predictions won't explode to infinity or
    /// oscillate wildly over time. This is important for making reliable forecasts.
    /// 
    /// The method uses a simple but effective approach:
    /// 1. Calculates the sum of absolute values of AR coefficients
    /// 2. If this sum exceeds 1, scales all coefficients down proportionally
    /// 
    /// For example, if coefficients are [0.7, 0.5], their sum is 1.2, which exceeds 1.
    /// They would be scaled down to approximately [0.58, 0.41] to ensure stationarity.
    /// 
    /// This constraint helps ensure the model's long-term behavior is reasonable.
    /// </remarks>
    private void EnsureARStationarity()
    {
        // Use the constraint that the roots of the AR polynomial should lie outside the unit circle
        // A simple approximation: ensure the sum of absolute AR coefficients is less than 1
        T sum = NumOps.Zero;
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(_arCoefficients[i]));
        }

        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            for (int i = 0; i < _arCoefficients.Length; i++)
            {
                _arCoefficients[i] = NumOps.Multiply(_arCoefficients[i], scaleFactor);
            }
        }
    }

    /// <summary>
    /// Ensures the MA process is invertible by scaling coefficients if necessary.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method ensures that the MA component will produce stable and
    /// well-behaved predictions.
    /// 
    /// "Invertibility" is a technical property that ensures the MA process can be
    /// equivalently represented as an infinite AR process. This makes the model more
    /// interpretable and better behaved.
    /// 
    /// Similar to ensuring AR stationarity, the method:
    /// 1. Calculates the sum of absolute values of MA coefficients
    /// 2. If this sum exceeds 1, scales all coefficients down proportionally
    /// 
    /// This constraint helps ensure the model will make sensible predictions and
    /// that coefficient estimates are unique and meaningful.
    /// </remarks>
    private void EnsureMAInvertibility()
    {
        // Similar to AR stationarity, ensure the sum of absolute MA coefficients is less than 1
        T sum = NumOps.Zero;
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Abs(_maCoefficients[i]));
        }

        if (NumOps.GreaterThan(sum, NumOps.One))
        {
            T scaleFactor = NumOps.Divide(NumOps.FromDouble(0.99), sum);
            for (int i = 0; i < _maCoefficients.Length; i++)
            {
                _maCoefficients[i] = NumOps.Multiply(_maCoefficients[i], scaleFactor);
            }
        }
    }

    /// <summary>
    /// Normalizes the regression coefficients to have unit norm.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method adjusts the regression coefficients to prevent any single
    /// external factor from dominating the model.
    /// 
    /// Normalization works by:
    /// 1. Calculating the "norm" of the coefficient vector (a measure of its total size)
    /// 2. Dividing each coefficient by this norm
    /// 
    /// The result is a set of coefficients that maintain their relative importance
    /// to each other, but have a combined "weight" of 1.
    /// 
    /// This can help with numerical stability and prevent one variable with a large
    /// scale from overwhelming the influence of other variables.
    /// </remarks>
    private void NormalizeRegressionCoefficients()
    {
        // Normalize regression coefficients to have unit norm
        T norm = NumOps.Sqrt(_regressionCoefficients.DotProduct(_regressionCoefficients));
        if (!NumOps.Equals(norm, NumOps.Zero))
        {
            for (int i = 0; i < _regressionCoefficients.Length; i++)
            {
                _regressionCoefficients[i] = NumOps.Divide(_regressionCoefficients[i], norm);
            }
        }
    }

    /// <summary>
    /// Updates the intercept based on differencing.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This private method adjusts the intercept (baseline prediction) to account for
    /// any differencing that was applied to the data.
    /// 
    /// Since differencing changes the scale and interpretation of the data, the intercept
    /// needs to be adjusted when converting predictions back to the original scale.
    /// 
    /// The method adds the mean of the differenced values to the intercept, which helps
    /// ensure that predictions are properly calibrated when they're converted back to
    /// the original scale.
    /// 
    /// This adjustment is particularly important when working with data that has strong
    /// trends or seasonal patterns that were removed through differencing.
    /// </remarks>
    private void UpdateIntercept()
    {
        // Adjust intercept based on the mean of the differenced series
        if (_arimaOptions.DifferenceOrder > 0 && _differenced.Length > 0)
        {
            T diffMean = StatisticsHelper<T>.CalculateMean(_differenced);
            _intercept = NumOps.Add(_intercept, diffMean);
        }
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Matrix of external regressors for testing.</param>
    /// <param name="yTest">Actual target values for testing.</param>
    /// <returns>A dictionary of evaluation metrics (MSE, RMSE, MAE, MAPE).</returns>
    /// <remarks>
    /// For Beginners:
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// 
    /// - MSE (Mean Squared Error): Average of squared differences between predictions and actual values.
    ///   Lower is better, but squaring emphasizes large errors.
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
        Dictionary<string, T> metrics = new Dictionary<string, T>
        {
            ["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions),
            ["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions),
            ["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions),
            ["MAPE"] = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions)
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
    /// The method saves all essential components of the model:
    /// - Regression coefficients
    /// - AR coefficients
    /// - MA coefficients
    /// - Differencing information
    /// - Intercept value
    /// - Model options
    /// 
    /// This allows the model to be fully reconstructed later.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        writer.Write(_regressionCoefficients.Length);
        for (int i = 0; i < _regressionCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_regressionCoefficients[i]));

        writer.Write(_arCoefficients.Length);
        for (int i = 0; i < _arCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_arCoefficients[i]));

        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
            writer.Write(Convert.ToDouble(_maCoefficients[i]));

        writer.Write(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
            writer.Write(Convert.ToDouble(_differenced[i]));

        writer.Write(Convert.ToDouble(_intercept));

        writer.Write(JsonConvert.SerializeObject(Options));
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
    /// - Regression coefficients
    /// - AR coefficients
    /// - MA coefficients
    /// - Differencing information
    /// - Intercept value
    /// - Model options
    /// 
    /// After deserialization, the model is ready to make predictions as if it had just been trained.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        int regressionCoefficientsLength = reader.ReadInt32();
        _regressionCoefficients = new Vector<T>(regressionCoefficientsLength);
        for (int i = 0; i < regressionCoefficientsLength; i++)
            _regressionCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int arCoefficientsLength = reader.ReadInt32();
        _arCoefficients = new Vector<T>(arCoefficientsLength);
        for (int i = 0; i < arCoefficientsLength; i++)
            _arCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int maCoeffientsLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maCoeffientsLength);
        for (int i = 0; i < maCoeffientsLength; i++)
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());

        int differencedLength = reader.ReadInt32();
        _differenced = new Vector<T>(differencedLength);
        for (int i = 0; i < differencedLength; i++)
            _differenced[i] = NumOps.FromDouble(reader.ReadDouble());

        _intercept = NumOps.FromDouble(reader.ReadDouble());

        string optionsJson = reader.ReadString();
    }

    /// <summary>
    /// Resets the model to its untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all trained parameters, effectively resetting the model to its initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method erases all the learned patterns from your model.
    /// 
    /// After calling this method:
    /// - All regression coefficients are reset to zero
    /// - All AR coefficients are reset to zero
    /// - All MA coefficients are reset to zero
    /// - The differencing information is cleared
    /// - The intercept is reset to zero
    /// 
    /// The model behaves as if it was never trained, and you would need to train it again before
    /// making predictions. This is useful when you want to:
    /// - Experiment with different training data on the same model
    /// - Retrain a model from scratch with new parameters
    /// - Reset a model that might have been trained incorrectly
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        // Reset regression coefficients
        _regressionCoefficients = new Vector<T>(_arimaOptions.ExternalRegressors);

        // Reset AR coefficients
        _arCoefficients = new Vector<T>(_arimaOptions.AROrder);

        // Reset MA coefficients
        _maCoefficients = new Vector<T>(_arimaOptions.MAOrder);

        // Reset differencing information
        _differenced = new Vector<T>(0);

        // Reset intercept
        _intercept = NumOps.Zero;
    }

    /// <summary>
    /// Creates a new instance of the model with the same options.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration options.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new, uninitialized instance of the model with the same options.
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
        // Create a clone of the options
        var optionsClone = new DynamicRegressionWithARIMAErrorsOptions<T>
        {
            AROrder = _arimaOptions.AROrder,
            MAOrder = _arimaOptions.MAOrder,
            DifferenceOrder = _arimaOptions.DifferenceOrder,
            ExternalRegressors = _arimaOptions.ExternalRegressors,
            Regularization = _arimaOptions.Regularization,
            DecompositionType = _arimaOptions.DecompositionType
        };

        return new DynamicRegressionWithARIMAErrors<T>(optionsClone);
    }

    /// <summary>
    /// Creates a deep copy of the current model.
    /// </summary>
    /// <returns>A new instance of the model with the same state and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the model, including its configuration and trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of your trained model.
    /// 
    /// Unlike CreateInstance(), which creates a blank model with the same settings,
    /// Clone() creates a complete copy including:
    /// - The model configuration (AR order, MA order, etc.)
    /// - All trained regression coefficients
    /// - All trained AR and MA coefficients
    /// - Differencing information and intercept value
    /// 
    /// This is useful for:
    /// - Creating a backup before experimenting with a model
    /// - Using the same trained model in multiple scenarios
    /// - Creating ensemble models that use variations of the same base model
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (DynamicRegressionWithARIMAErrors<T>)CreateInstance();

        // Copy regression coefficients
        for (int i = 0; i < _regressionCoefficients.Length; i++)
        {
            clone._regressionCoefficients[i] = _regressionCoefficients[i];
        }

        // Copy AR coefficients
        for (int i = 0; i < _arCoefficients.Length; i++)
        {
            clone._arCoefficients[i] = _arCoefficients[i];
        }

        // Copy MA coefficients
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            clone._maCoefficients[i] = _maCoefficients[i];
        }

        // Copy differenced values
        clone._differenced = new Vector<T>(_differenced.Length);
        for (int i = 0; i < _differenced.Length; i++)
        {
            clone._differenced[i] = _differenced[i];
        }

        // Copy intercept
        clone._intercept = _intercept;

        return clone;
    }

    /// <summary>
    /// Forecasts future values based on a history of time series data and exogenous variables.
    /// </summary>
    /// <param name="history">The historical time series data.</param>
    /// <param name="horizon">The number of future periods to predict.</param>
    /// <param name="exogenousVariables">Matrix of external regressors for future periods.</param>
    /// <returns>A vector of predicted values for future periods.</returns>
    /// <remarks>
    /// <para>
    /// This method generates forecasts for future time periods based on the trained model, historical data,
    /// and external variables.
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts future values based on past data.
    /// 
    /// Given:
    /// - A series of past observations (the "history")
    /// - The number of future periods to predict (the "horizon")
    /// - External factors that might affect predictions (exogenous variables)
    /// 
    /// This method:
    /// 1. Uses the regression component to capture the effects of external factors
    /// 2. Uses the ARIMA component to capture time-dependent patterns
    /// 3. Combines these components to generate comprehensive forecasts
    /// 4. If differencing was used, it properly undoes the differencing to get results in the original scale
    /// 
    /// For example, if you have sales data for Jan-Jun and want to predict Jul-Sep,
    /// this method will use both the external factors (like promotions, holidays) and 
    /// the historical patterns to create accurate forecasts.
    /// </para>
    /// </remarks>
    public Vector<T> Forecast(Vector<T> history, int horizon, Matrix<T> exogenousVariables)
    {
        if (horizon <= 0)
        {
            throw new ArgumentException("Forecast horizon must be greater than zero.", nameof(horizon));
        }

        if (exogenousVariables == null || exogenousVariables.Rows != horizon ||
            exogenousVariables.Columns != _regressionCoefficients.Length)
        {
            throw new ArgumentException($"Exogenous variables matrix must have {horizon} rows and {_regressionCoefficients.Length} columns.");
        }

        // Step 1: Apply differencing to history if needed
        Vector<T> diffHistory = history;
        if (_arimaOptions.DifferenceOrder > 0)
        {
            diffHistory = DifferenceTimeSeries(history, _arimaOptions.DifferenceOrder);
        }

        // Step 2: Generate forecasts in the differenced space
        Vector<T> diffForecasts = new Vector<T>(horizon);

        // Combined history and forecasts to simplify access to lags
        Vector<T> combinedDiff = new Vector<T>(diffHistory.Length + horizon);
        for (int i = 0; i < diffHistory.Length; i++)
        {
            combinedDiff[i] = diffHistory[i];
        }

        // Compute past prediction errors for MA component
        Vector<T> errors = new Vector<T>(diffHistory.Length + horizon);
        for (int t = Math.Max(_arimaOptions.AROrder, _arimaOptions.MAOrder); t < diffHistory.Length; t++)
        {
            // Use available history to compute errors
            T prediction = _intercept;

            // Regression component for historical period (simple approximation)
            // In a real implementation, you would need historical exogenous variables

            // AR component
            for (int i = 0; i < _arimaOptions.AROrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i],
                    NumOps.Subtract(diffHistory[t - i - 1], _intercept)));
            }

            // MA component
            for (int i = 0; i < _arimaOptions.MAOrder && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], errors[t - i - 1]));
            }

            // Compute error
            errors[t] = NumOps.Subtract(diffHistory[t], prediction);
        }

        // Generate forecasts
        for (int t = 0; t < horizon; t++)
        {
            // Start with intercept
            T prediction = _intercept;

            // Add regression component
            for (int i = 0; i < _regressionCoefficients.Length; i++)
            {
                prediction = NumOps.Add(prediction, NumOps.Multiply(exogenousVariables[t, i], _regressionCoefficients[i]));
            }

            // Add AR component
            for (int i = 0; i < _arimaOptions.AROrder; i++)
            {
                int histIndex = diffHistory.Length - i - 1 + t;
                if (histIndex >= 0 && histIndex < combinedDiff.Length)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_arCoefficients[i],
                        NumOps.Subtract(combinedDiff[histIndex], _intercept)));
                }
            }

            // Add MA component
            for (int i = 0; i < _arimaOptions.MAOrder; i++)
            {
                int errorIndex = diffHistory.Length - i - 1 + t;
                if (errorIndex >= 0 && errorIndex < errors.Length)
                {
                    prediction = NumOps.Add(prediction, NumOps.Multiply(_maCoefficients[i], errors[errorIndex]));
                }
            }

            // Store forecast
            diffForecasts[t] = prediction;

            // Update combined series and errors for next steps
            int currentIndex = diffHistory.Length + t;
            if (currentIndex < combinedDiff.Length)
            {
                combinedDiff[currentIndex] = prediction;
                errors[currentIndex] = NumOps.Zero; // No error for forecasts
            }
        }

        // Step 3: Convert forecasts back to original scale if needed
        Vector<T> finalForecasts;
        if (_arimaOptions.DifferenceOrder > 0)
        {
            finalForecasts = InverseDifferenceTimeSeries(diffForecasts, _differenced);
        }
        else
        {
            finalForecasts = diffForecasts;
        }

        return finalForecasts;
    }

    /// <summary>
    /// Gets metadata about the trained model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive information about the model, including its type, parameters, and configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns a complete description of your trained model.
    /// 
    /// The metadata includes:
    /// - The type of model (Dynamic Regression with ARIMA Errors)
    /// - The coefficients for external factors (regression component)
    /// - The coefficients for time-dependent patterns (AR and MA components)
    /// - The configuration options you specified when creating the model
    /// - A serialized version of the entire model that can be saved
    /// 
    /// This is useful for:
    /// - Understanding what patterns the model has learned
    /// - Documenting how the model was configured
    /// - Saving the model's state for later analysis
    /// - Comparing different models to choose the best one
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var options = (DynamicRegressionWithARIMAErrorsOptions<T>)Options;
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.DynamicRegressionWithARIMAErrors,
            AdditionalInfo = new Dictionary<string, object>
            {
                // Include the actual model state variables
                { "RegressionCoefficients", _regressionCoefficients },
                { "ARCoefficients", _arCoefficients },
                { "MACoefficients", _maCoefficients },
                { "Differenced", _differenced },
                { "Intercept", Convert.ToDouble(_intercept) },
        
                // Include model configuration as well
                { "AROrder", options.AROrder },
                { "MAOrder", options.MAOrder },
                { "DifferenceOrder", options.DifferenceOrder },
                { "ExternalRegressors", options.ExternalRegressors },
                { "DecompositionType", options.DecompositionType },
                { "Regularization", options.Regularization?.GetType().Name ?? "None" }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Implements the core training algorithm for the Dynamic Regression with ARIMA Errors model.
    /// </summary>
    /// <param name="x">Feature matrix of external regressors.</param>
    /// <param name="y">Target vector of time series values.</param>
    /// <remarks>
    /// <para>
    /// This method contains the implementation details of the training process, handling differencing,
    /// regression modeling, and ARIMA parameter estimation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the engine room of the training process.
    /// 
    /// While the public Train method provides a high-level interface, this method does the actual work:
    /// 1. Performs differencing to remove trends (if specified)
    /// 2. Fits the regression model to capture external factor effects
    /// 3. Calculates residuals (what the regression couldn't explain)
    /// 4. Fits the ARIMA model to these residuals
    /// 5. Finalizes all model parameters
    /// 
    /// Think of it as the detailed step-by-step recipe that the chef follows when you order a meal.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Step 1: Perform differencing if necessary
        Vector<T> diffY = DifferenceTimeSeries(y, _arimaOptions.DifferenceOrder);

        // Step 2: Fit regression model
        FitRegressionModel(x, diffY);

        // Step 3: Extract residuals
        Vector<T> residuals = ExtractResiduals(x, diffY);

        // Step 4: Fit ARIMA model to residuals
        FitARIMAModel(residuals);

        // Step 5: Update model parameters
        UpdateModelParameters();
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
        // Validate input dimensions
        if (input.Length != _regressionCoefficients.Length)
        {
            throw new ArgumentException(
                $"Input vector length ({input.Length}) must match the number of external regressors ({_regressionCoefficients.Length}).",
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
}
