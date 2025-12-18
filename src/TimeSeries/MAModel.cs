namespace AiDotNet.TimeSeries;

/// <summary>
/// Implements a Moving Average (MA) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// MA models predict future values based on past prediction errors (residuals). 
/// The model is defined as: Yt = μ + et + ?1et-1 + ?2et-2 + ... + ?qet-q
/// where Yt is the value at time t, μ is the mean, et is the error term at time t,
/// and ?i are the MA coefficients.
/// </para>
/// 
/// <para>
/// <b>For Beginners:</b>
/// A Moving Average (MA) model predicts future values based on past prediction errors.
/// 
/// Think of it like this: If you've been consistently underestimating or overestimating 
/// values in the past, the MA model learns from these mistakes and adjusts future predictions.
/// 
/// For example, if a weather forecast has been consistently underestimating temperatures 
/// by 2 degrees for several days, an MA model would learn this pattern and adjust its 
/// future predictions upward.
/// 
/// The key parameter of an MA model is 'q', which determines how many past prediction 
/// errors to consider. For instance, with q=3, the model looks at errors from the last 
/// three periods when making a new prediction.
/// </para>
/// </remarks>
public class MAModel<T> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Options specific to the MA model, including the order (q) parameter.
    /// </summary>
    private MAModelOptions<T> _maOptions;

    /// <summary>
    /// Coefficients for the moving average component of the model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// These coefficients determine how much each past prediction error influences the current prediction.
    /// For example, if the coefficient for yesterday's error is 0.7, it means yesterday's
    /// error strongly influences today's prediction adjustment.
    /// </remarks>
    private Vector<T> _maCoefficients;

    /// <summary>
    /// The mean of the time series, used as a baseline for predictions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This is the average value of your time series. In an MA model, predictions
    /// start with this average value and then get adjusted based on past errors.
    /// </remarks>
    private T _mean;

    /// <summary>
    /// The most recent errors (residuals) from the model's predictions.
    /// Used for generating future predictions.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// These are the differences between the model's recent predictions and the actual values.
    /// The model uses these errors to adjust future predictions.
    /// </remarks>
    private Vector<T> _recentErrors;

    /// <summary>
    /// The variance of the white noise process.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This represents how much random noise is in your data. A higher value indicates
    /// more randomness and less predictability in the time series.
    /// </remarks>
    private T _noiseVariance;

    /// <summary>
    /// Flag indicating whether the model has been trained.
    /// </summary>
    private bool _isTrained;

    /// <summary>
    /// Maximum number of iterations for optimization algorithms.
    /// </summary>
    private readonly int _maxIterations = 100;

    /// <summary>
    /// Convergence tolerance for optimization algorithms.
    /// </summary>
    private readonly T _convergenceTolerance;

    /// <summary>
    /// Creates a new MA model with the specified options.
    /// </summary>
    /// <param name="options">Options for the MA model, including the order (q) parameter. If null, default options are used.</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This constructor creates a new MA model. You can customize the model by providing options:
    /// - q: How many past prediction errors to consider
    /// 
    /// If you don't provide options, default values will be used, but it's usually best
    /// to choose values that make sense for your specific data.
    /// </remarks>
    public MAModel(MAModelOptions<T>? options = null) : base(options ?? new MAModelOptions<T>())
    {
        _maOptions = options ?? new MAModelOptions<T>();
        _mean = NumOps.Zero;
        _maCoefficients = new Vector<T>(_maOptions.MAOrder);
        _recentErrors = new Vector<T>(_maOptions.MAOrder);
        _noiseVariance = NumOps.One;
        _isTrained = false;
        _convergenceTolerance = NumOps.FromDouble(1e-6);
    }

    /// <summary>
    /// Core implementation of the training logic for the MA model.
    /// </summary>
    /// <param name="x">Feature matrix (typically just time indices for MA models).</param>
    /// <param name="y">Target vector (the time series values to be modeled).</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method contains the core implementation of the training process. It:
    /// 
    /// 1. Calculates the mean of the time series to use as a baseline
    /// 2. Estimates the MA coefficients using a robust maximum likelihood estimation
    /// 3. Calculates the variance of the white noise process
    /// 4. Initializes the recent errors vector for making future predictions
    /// 
    /// Training an MA model is complex because we can't directly observe the past errors.
    /// The method uses sophisticated statistical techniques to estimate the most likely
    /// values for the MA coefficients given the observed data.
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate input
        if (y == null || y.Length == 0)
        {
            throw new ArgumentException("Time series data cannot be null or empty", nameof(y));
        }

        if (y.Length <= _maOptions.MAOrder)
        {
            throw new ArgumentException($"Time series length ({y.Length}) must be greater than the MA order ({_maOptions.MAOrder})", nameof(y));
        }

        // Step 1: Calculate the mean of the time series
        _mean = StatisticsHelper<T>.CalculateMean(y);

        // Step 2: Center the time series (subtract mean)
        // VECTORIZED: Use Engine.Subtract with scalar broadcasting
        var meanVec = new Vector<T>(y.Length);
        for (int i = 0; i < meanVec.Length; i++) meanVec[i] = _mean;
        Vector<T> centeredY = (Vector<T>)Engine.Subtract(y, meanVec);

        // Step 3: Estimate MA coefficients
        _maCoefficients = EstimateMACoefficients(centeredY, _maOptions.MAOrder);

        // Step 4: Calculate the variance of the white noise process
        _noiseVariance = EstimateNoiseVariance(centeredY, _maCoefficients);

        // Step 5: Calculate the most recent errors for making future predictions
        _recentErrors = CalculateRecentErrors(y);

        _isTrained = true;
    }

    /// <summary>
    /// Estimates the Moving Average coefficients using maximum likelihood estimation.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="q">The order of the MA model.</param>
    /// <returns>The estimated MA coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method estimates how much each past error influences the current prediction.
    /// 
    /// It uses a sophisticated statistical approach called Maximum Likelihood Estimation (MLE),
    /// which finds the coefficient values that make the observed data most probable.
    /// 
    /// The method combines several techniques to ensure robustness:
    /// 1. Initial estimates based on autocorrelation
    /// 2. Iterative refinement using numerical optimization
    /// 3. Boundary checking to ensure coefficients stay within valid ranges
    /// 
    /// This approach helps find the best possible coefficients even for complex time series.
    /// </remarks>
    private Vector<T> EstimateMACoefficients(Vector<T> y, int q)
    {
        if (q == 0)
        {
            return new Vector<T>(0);
        }

        // Initial estimates using method of moments (via autocorrelation)
        Vector<T> initialTheta = InitialMACoefficientsEstimate(y, q);

        // Refine estimates using maximum likelihood estimation
        return OptimizeMACoefficients(y, initialTheta);
    }

    /// <summary>
    /// Provides initial estimates of MA coefficients based on autocorrelation.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="q">The order of the MA model.</param>
    /// <returns>Initial estimates of MA coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method creates a starting point for the optimization process by analyzing
    /// how values in the time series relate to each other over time (autocorrelation).
    /// 
    /// Good initial estimates are important because they help the optimization process
    /// find the best coefficients more quickly and reliably.
    /// </remarks>
    private Vector<T> InitialMACoefficientsEstimate(Vector<T> y, int q)
    {
        // Calculate the autocorrelation function (ACF) up to lag q+1
        var acf = TimeSeriesHelper<T>.CalculateMultipleAutoCorrelation(y, q + 1);

        // For a pure MA(q) process, we use the innovations algorithm
        // to get initial estimates of the MA coefficients
        Vector<T> theta = new Vector<T>(q);
        Vector<T> v = new Vector<T>(q + 1);

        // Set initial innovation variance to variance of the series
        v[0] = acf[0];

        for (int k = 1; k <= q; k++)
        {
            // Calculate the sum term
            T sum = NumOps.Zero;
            for (int j = 1; j < k; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(theta[j - 1], NumOps.Multiply(v[k - j], v[0])));
            }

            // Calculate the k-th MA coefficient
            if (NumOps.GreaterThan(v[0], NumOps.Zero))
            {
                theta[k - 1] = NumOps.Divide(NumOps.Subtract(acf[k], sum), v[0]);
            }

            // Ensure the coefficient is within the invertibility bounds (-1, 1)
            if (NumOps.GreaterThan(NumOps.Abs(theta[k - 1]), NumOps.FromDouble(0.98)))
            {
                T sign = NumOps.GreaterThan(theta[k - 1], NumOps.Zero) ?
                    NumOps.FromDouble(0.98) : NumOps.FromDouble(-0.98);
                theta[k - 1] = sign;
            }

            // Update the innovation variance for the next iteration
            if (k < q)
            {
                v[k] = NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, NumOps.Multiply(theta[k - 1], theta[k - 1])),
                    v[k - 1]
                );
            }
        }

        return theta;
    }

    /// <summary>
    /// Optimizes MA coefficients using numerical optimization with the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="initialTheta">Initial estimates of MA coefficients.</param>
    /// <returns>Optimized MA coefficients.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method refines the initial coefficient estimates using a sophisticated
    /// mathematical technique called the BFGS algorithm.
    /// 
    /// Think of it like fine-tuning the dial on a radio to get the clearest signal.
    /// The algorithm systematically adjusts the coefficients to maximize how well
    /// the model explains the observed data.
    /// 
    /// Each iteration improves the coefficients until they converge to the best possible values
    /// or reach the maximum number of iterations.
    /// </remarks>
    private Vector<T> OptimizeMACoefficients(Vector<T> y, Vector<T> initialTheta)
    {
        int q = initialTheta.Length;
        if (q == 0)
        {
            return new Vector<T>(0);
        }

        // Copy initial estimates
        Vector<T> theta = new Vector<T>(initialTheta);

        // Initialize optimization variables
        Vector<T> gradient = new Vector<T>(q);
        Matrix<T> hessianApprox = Matrix<T>.CreateIdentity(q);
        T prevLogLikelihood = CalculateNegativeLogLikelihood(y, theta);

        // BFGS optimization
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Calculate gradient
            CalculateGradient(y, theta, gradient);

            // Calculate search direction
            Vector<T> searchDir = CalculateSearchDirection(hessianApprox, gradient);

            // Line search to find step size
            T alpha = LineSearch(y, theta, searchDir, prevLogLikelihood);

            // Update parameters
            // VECTORIZED: Compute new theta using Engine operations
            var alphaScaled = (Vector<T>)Engine.Multiply(searchDir, alpha);
            Vector<T> newTheta = (Vector<T>)Engine.Add(theta, alphaScaled);

            // Apply invertibility constraints element-wise
            for (int i = 0; i < q; i++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(newTheta[i]), NumOps.FromDouble(0.99)))
                {
                    T sign = NumOps.GreaterThan(newTheta[i], NumOps.Zero) ?
                        NumOps.FromDouble(0.99) : NumOps.FromDouble(-0.99);
                    newTheta[i] = sign;
                }
            }

            // Calculate new log-likelihood
            T newLogLikelihood = CalculateNegativeLogLikelihood(y, newTheta);

            // Check convergence
            T improvement = NumOps.Abs(NumOps.Subtract(prevLogLikelihood, newLogLikelihood));
            if (NumOps.LessThan(improvement, _convergenceTolerance))
            {
                theta = newTheta;
                break;
            }

            // Update BFGS approximation of the Hessian
            UpdateHessianApproximation(y, hessianApprox, theta, newTheta, gradient);

            // Update for next iteration
            theta = newTheta;
            prevLogLikelihood = newLogLikelihood;
        }

        return theta;
    }

    /// <summary>
    /// Calculates the negative log-likelihood of the MA model given the parameters.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="theta">The MA coefficients.</param>
    /// <returns>The negative log-likelihood value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method calculates how well the model with specific coefficients explains the observed data.
    /// 
    /// Think of it as a "score" for the model - the lower the negative log-likelihood,
    /// the better the model fits the data. During optimization, we try to minimize this value.
    /// 
    /// The calculation is based on probability theory and measures how probable the
    /// observed data is under the current model parameters.
    /// </remarks>
    private T CalculateNegativeLogLikelihood(Vector<T> y, Vector<T> theta)
    {
        int n = y.Length;
        int q = theta.Length;

        // Initialize errors and their variance
        Vector<T> errors = new Vector<T>(n);
        T variance = NumOps.Zero;

        // Calculate residuals using Kalman filter-like approach
        for (int t = 0; t < n; t++)
        {
            T prediction = NumOps.Zero;

            // MA component (previous errors)
            for (int i = 0; i < q && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction,
                    NumOps.Multiply(theta[i], errors[t - i - 1]));
            }

            // Calculate error
            errors[t] = NumOps.Subtract(y[t], prediction);

            // Accumulate squared errors for variance estimation
            variance = NumOps.Add(variance, NumOps.Multiply(errors[t], errors[t]));
        }

        // Calculate log-likelihood (ignoring constant terms)
        if (NumOps.GreaterThan(variance, NumOps.Zero))
        {
            variance = NumOps.Divide(variance, NumOps.FromDouble(n));

            // log-likelihood = -n/2 * log(2p) - n/2 * log(variance) - 1/(2*variance) * sum(errors²)
            // We ignore the constant terms and return negative log-likelihood
            T logVariance = NumOps.Log(variance);
            T scaledVariance = NumOps.Multiply(NumOps.FromDouble(n), logVariance);

            return scaledVariance;
        }

        // If variance is zero (highly unlikely), return a large number
        return NumOps.FromDouble(1e10);
    }

    /// <summary>
    /// Calculates the gradient of the negative log-likelihood with respect to the MA coefficients.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="theta">The current MA coefficients.</param>
    /// <param name="gradient">The output vector to store the gradient.</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method calculates how sensitive the model's performance is to small changes
    /// in each coefficient. It's like finding which direction to adjust each knob
    /// to improve the model.
    /// 
    /// The gradient points in the direction of steepest increase, so during optimization,
    /// we move in the opposite direction to minimize the negative log-likelihood.
    /// </remarks>
    private void CalculateGradient(Vector<T> y, Vector<T> theta, Vector<T> gradient)
    {
        int n = y.Length;
        int q = theta.Length;

        // Use finite differences to approximate the gradient
        T h = NumOps.FromDouble(1e-5); // Small step size

        for (int i = 0; i < q; i++)
        {
            // Create parameter vectors with slight perturbations
            Vector<T> thetaPlus = new Vector<T>(theta);
            Vector<T> thetaMinus = new Vector<T>(theta);

            thetaPlus[i] = NumOps.Add(theta[i], h);
            thetaMinus[i] = NumOps.Subtract(theta[i], h);

            // Calculate log-likelihood at perturbed points
            T logLikePlus = CalculateNegativeLogLikelihood(y, thetaPlus);
            T logLikeMinus = CalculateNegativeLogLikelihood(y, thetaMinus);

            // Central difference approximation of derivative
            gradient[i] = NumOps.Divide(
                NumOps.Subtract(logLikePlus, logLikeMinus),
                NumOps.Multiply(NumOps.FromDouble(2.0), h)
            );
        }
    }

    /// <summary>
    /// Calculates the search direction for optimization using the approximated Hessian.
    /// </summary>
    /// <param name="hessianApprox">The approximation of the Hessian matrix.</param>
    /// <param name="gradient">The gradient vector.</param>
    /// <returns>The search direction vector.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method determines which way to adjust the coefficients during optimization.
    /// 
    /// The search direction combines information from the gradient (which way to go)
    /// and the Hessian approximation (how the gradients change), to make more efficient
    /// progress toward the optimal values.
    /// 
    /// It's like planning a route that considers both the current direction and the
    /// terrain ahead to find the fastest path to the destination.
    /// </remarks>
    private Vector<T> CalculateSearchDirection(Matrix<T> hessianApprox, Vector<T> gradient)
    {
        // Solve the system H * d = -g for the search direction d
        // where H is the Hessian approximation and g is the gradient
        // VECTORIZED: Use Engine.Multiply to negate vector
        var negOneVec = new Vector<T>(gradient.Length);
        for (int i = 0; i < negOneVec.Length; i++) negOneVec[i] = NumOps.FromDouble(-1.0);
        Vector<T> negGradient = (Vector<T>)Engine.Multiply(gradient, negOneVec);

        try
        {
            // Solve the system using a stable matrix decomposition
            return MatrixSolutionHelper.SolveLinearSystem(
                hessianApprox,
                negGradient,
                MatrixDecompositionType.Qr
            );
        }
        catch (Exception)
        {
            // If the matrix is ill-conditioned, fall back to negative gradient
            return negGradient;
        }
    }

    /// <summary>
    /// Performs line search to find an appropriate step size for the optimization algorithm.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="theta">The current MA coefficients.</param>
    /// <param name="searchDir">The search direction.</param>
    /// <param name="currentLogLikelihood">The current negative log-likelihood value.</param>
    /// <returns>The step size to use.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method determines how big of a step to take in the chosen direction during optimization.
    /// 
    /// It's like deciding how far to walk in a particular direction. Too small a step
    /// means slow progress; too large a step might overshoot the target.
    /// 
    /// The method tries different step sizes and chooses one that gives a good improvement
    /// in the model's performance.
    /// </remarks>
    private T LineSearch(Vector<T> y, Vector<T> theta, Vector<T> searchDir, T currentLogLikelihood)
    {
        int q = theta.Length;

        // Backtracking line search with Armijo condition
        T alpha = NumOps.One; // Initial step size
        T c = NumOps.FromDouble(0.1); // Armijo parameter
        T rho = NumOps.FromDouble(0.5); // Reduction factor

        // VECTORIZED: Calculate directional derivative using Engine.DotProduct
        Vector<T> gradient = new Vector<T>(q);
        CalculateGradient(y, theta, gradient);
        T directionalDerivative = Engine.DotProduct(gradient, searchDir);

        // If directional derivative is non-negative, use steepest descent direction
        if (!NumOps.LessThan(directionalDerivative, NumOps.Zero))
        {
            // VECTORIZED: Calculate gradient norm squared using Engine.DotProduct
            T gradientDotProduct = Engine.DotProduct(gradient, gradient);

            for (int i = 0; i < q; i++)
            {
                searchDir[i] = NumOps.Negate(gradient[i]);
            }
            directionalDerivative = NumOps.Negate(gradientDotProduct);
        }

        // Rest of method unchanged...
        // Perform backtracking line search
        int maxBacktracks = 10;
        for (int i = 0; i < maxBacktracks; i++)
        {
            // Try the current step size
            // VECTORIZED: Compute new theta using Engine operations
            var alphaScaledLS = (Vector<T>)Engine.Multiply(searchDir, alpha);
            Vector<T> newTheta = (Vector<T>)Engine.Add(theta, alphaScaledLS);

            // Apply invertibility constraints element-wise
            for (int j = 0; j < q; j++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(newTheta[j]), NumOps.FromDouble(0.99)))
                {
                    T sign = NumOps.GreaterThan(newTheta[j], NumOps.Zero) ?
                        NumOps.FromDouble(0.99) : NumOps.FromDouble(-0.99);
                    newTheta[j] = sign;
                }
            }

            // Calculate new log-likelihood
            T newLogLikelihood = CalculateNegativeLogLikelihood(y, newTheta);

            // Check Armijo condition
            T armijo = NumOps.Add(currentLogLikelihood,
                NumOps.Multiply(NumOps.Multiply(c, alpha), directionalDerivative));

            if (NumOps.LessThanOrEquals(newLogLikelihood, armijo))
            {
                return alpha;
            }

            // Reduce step size
            alpha = NumOps.Multiply(alpha, rho);
        }

        // If backtracking failed, return a small step size
        return NumOps.FromDouble(0.01);
    }

    /// <summary>
    /// Updates the approximation of the Hessian matrix using the BFGS update formula.
    /// </summary>
    /// <param name="hessianApprox">The current Hessian approximation matrix to update.</param>
    /// <param name="oldTheta">The previous MA coefficients.</param>
    /// <param name="newTheta">The updated MA coefficients.</param>
    /// <param name="oldGradient">The gradient at the previous point.</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method updates the Hessian approximation, which helps the optimization algorithm
    /// understand how the gradient changes as the coefficients change.
    /// 
    /// Think of it as building a map of the optimization landscape. The Hessian approximation
    /// helps the algorithm understand the curvature of the landscape, allowing it to
    /// take more efficient steps toward the optimal solution.
    /// </remarks>
    private void UpdateHessianApproximation(Vector<T> data, Matrix<T> hessianApprox, Vector<T> oldTheta,
                                            Vector<T> newTheta, Vector<T> oldGradient)
    {
        int q = oldTheta.Length;

        // Calculate new gradient
        Vector<T> newGradient = new Vector<T>(q);
        CalculateGradient(data, newTheta, newGradient);

        // Calculate s = newTheta - oldTheta
        Vector<T> s = new Vector<T>(q);
        for (int i = 0; i < q; i++)
        {
            s[i] = NumOps.Subtract(newTheta[i], oldTheta[i]);
        }

        // Calculate y = newGradient - oldGradient
        Vector<T> y = new Vector<T>(q);
        for (int i = 0; i < q; i++)
        {
            y[i] = NumOps.Subtract(newGradient[i], oldGradient[i]);
        }

        // VECTORIZED: Calculate ? = 1 / (y^T * s) using dot product
        T dotProduct = y.DotProduct(s);

        // Skip update if dot product is too small (numerical stability)
        if (NumOps.LessThan(NumOps.Abs(dotProduct), NumOps.FromDouble(1e-10)))
        {
            return;
        }

        T rho = NumOps.Divide(NumOps.One, dotProduct);

        // BFGS update formula:
        // H_{k+1} = (I - ?*s*y^T) * H_k * (I - ?*y*s^T) + ?*s*s^T

        // VECTORIZED: Calculate H_k * y using matrix-vector multiplication
        Vector<T> Hy = hessianApprox.Multiply(y);

        // Calculate intermediate terms
        Matrix<T> term1 = new Matrix<T>(q, q);
        Matrix<T> term2 = new Matrix<T>(q, q);

        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                T factor1 = NumOps.Multiply(rho, NumOps.Multiply(s[i], y[j]));
                T factor2 = NumOps.Multiply(rho, NumOps.Multiply(y[i], s[j]));

                term1[i, j] = NumOps.Negate(factor1);
                term2[i, j] = NumOps.Negate(factor2);
            }
        }

        // Add identity to both terms
        for (int i = 0; i < q; i++)
        {
            term1[i, i] = NumOps.Add(term1[i, i], NumOps.One);
            term2[i, i] = NumOps.Add(term2[i, i], NumOps.One);
        }

        // Calculate term1 * H_k
        Matrix<T> term1H = new Matrix<T>(q, q);
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                for (int k = 0; k < q; k++)
                {
                    term1H[i, j] = NumOps.Add(term1H[i, j],
                        NumOps.Multiply(term1[i, k], hessianApprox[k, j]));
                }
            }
        }

        // Calculate term1 * H_k * term2
        Matrix<T> term1Hterm2 = new Matrix<T>(q, q);
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                for (int k = 0; k < q; k++)
                {
                    term1Hterm2[i, j] = NumOps.Add(term1Hterm2[i, j],
                        NumOps.Multiply(term1H[i, k], term2[k, j]));
                }
            }
        }

        // Calculate ?*s*s^T
        Matrix<T> rhoss = new Matrix<T>(q, q);
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                rhoss[i, j] = NumOps.Multiply(rho, NumOps.Multiply(s[i], s[j]));
            }
        }

        // Update Hessian approximation: H_{k+1} = term1Hterm2 + rhoss
        for (int i = 0; i < q; i++)
        {
            for (int j = 0; j < q; j++)
            {
                hessianApprox[i, j] = NumOps.Add(term1Hterm2[i, j], rhoss[i, j]);
            }
        }
    }

    /// <summary>
    /// Estimates the variance of the white noise process in the MA model.
    /// </summary>
    /// <param name="y">The centered time series data.</param>
    /// <param name="maCoefficients">The MA coefficients.</param>
    /// <returns>The estimated noise variance.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method estimates how much random noise is in your time series data.
    /// 
    /// Think of it as measuring the "unpredictable" part of your data. Even with
    /// the best model, there will always be some randomness that can't be predicted.
    /// This method quantifies that randomness.
    /// 
    /// The variance is used in calculating prediction intervals and understanding
    /// the uncertainty in the model's forecasts.
    /// </remarks>
    private T EstimateNoiseVariance(Vector<T> y, Vector<T> maCoefficients)
    {
        int n = y.Length;
        int q = maCoefficients.Length;

        // Calculate residuals
        Vector<T> residuals = new Vector<T>(n);
        Vector<T> errors = new Vector<T>(n);

        for (int t = 0; t < n; t++)
        {
            T prediction = NumOps.Zero;

            // Add MA component
            for (int i = 0; i < q && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction,
                    NumOps.Multiply(maCoefficients[i], errors[t - i - 1]));
            }

            // Calculate error
            residuals[t] = NumOps.Subtract(y[t], prediction);
            errors[t] = residuals[t];
        }

        // Calculate variance starting after the initial q values
        T sumSquaredResiduals = NumOps.Zero;
        int effectiveN = n - q;

        for (int t = q; t < n; t++)
        {
            sumSquaredResiduals = NumOps.Add(sumSquaredResiduals,
                NumOps.Multiply(residuals[t], residuals[t]));
        }

        // Adjust for degrees of freedom
        return NumOps.Divide(sumSquaredResiduals, NumOps.FromDouble(effectiveN));
    }

    /// <summary>
    /// Calculates the most recent errors for making future predictions.
    /// </summary>
    /// <param name="y">The time series data.</param>
    /// <returns>A vector of the most recent errors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method calculates the differences between the actual values and what the model
    /// would have predicted. These "errors" or "residuals" are essential for the MA model
    /// to make future predictions.
    /// 
    /// The method keeps track of the most recent errors, which will be used to adjust
    /// future predictions. For example, if the model has consistently underestimated
    /// recent values, it will use this information to adjust future forecasts upward.
    /// </remarks>
    private Vector<T> CalculateRecentErrors(Vector<T> y)
    {
        int n = y.Length;
        int q = _maOptions.MAOrder;

        // Calculate all errors
        Vector<T> allErrors = new Vector<T>(n);

        for (int t = 0; t < n; t++)
        {
            T prediction = _mean;

            // Add MA component
            for (int i = 0; i < q && t - i - 1 >= 0; i++)
            {
                prediction = NumOps.Add(prediction,
                    NumOps.Multiply(_maCoefficients[i], allErrors[t - i - 1]));
            }

            // Calculate error
            allErrors[t] = NumOps.Subtract(y[t], prediction);
        }

        // Extract the most recent q errors
        Vector<T> recentErrors = new Vector<T>(q);
        for (int i = 0; i < q; i++)
        {
            if (n - q + i >= 0)
            {
                recentErrors[i] = allErrors[n - q + i];
            }
        }

        return recentErrors;
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">Input vector containing features for prediction (not used in MA models).</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method generates a single prediction for the next time period.
    /// 
    /// In an MA model, predictions start with the mean of the series and then
    /// get adjusted based on recent prediction errors. The input parameter is typically
    /// not used in pure MA models since predictions depend only on past errors.
    /// 
    /// For example, if the average temperature is 70—F but we've been consistently
    /// underestimating by 2—F recently, the model might predict 72—F for tomorrow.
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Check if model has been trained
        if (!_isTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // Start with the mean as the baseline prediction
        T prediction = _mean;

        // VECTORIZED: Add MA component using dot product
        if (_maCoefficients.Length > 0)
        {
            prediction = NumOps.Add(prediction, _maCoefficients.DotProduct(_recentErrors));
        }

        return prediction;
    }

    /// <summary>
    /// Makes predictions using the trained MA model.
    /// </summary>
    /// <param name="input">Input matrix for prediction (typically just time indices for future periods).</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method uses the trained MA model to forecast future values.
    /// 
    /// The prediction process:
    /// 1. Starts with the mean of the time series as a base value
    /// 2. Adds the effects of recent prediction errors (MA component)
    /// 3. For each prediction, updates the error history used for the next prediction
    /// 
    /// Since this is a pure MA model, predictions will gradually converge to the mean
    /// as we forecast further into the future, because we assume zero errors for future
    /// periods that we haven't observed yet.
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // Check if model has been trained
        if (!_isTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        int horizon = input.Rows;
        Vector<T> predictions = new Vector<T>(horizon);

        // Make a copy of recent errors that we'll update during prediction
        Vector<T> workingErrors = new Vector<T>(_recentErrors.Length);
        for (int i = 0; i < _recentErrors.Length; i++)
        {
            workingErrors[i] = _recentErrors[i];
        }

        // Generate predictions for each time step
        for (int t = 0; t < horizon; t++)
        {
            // Start with the mean
            T prediction = _mean;

            // VECTORIZED: Add MA component using dot product
            if (_maCoefficients.Length > 0)
            {
                prediction = NumOps.Add(prediction, _maCoefficients.DotProduct(workingErrors));
            }

            predictions[t] = prediction;

            // Shift the working errors vector and add a zero error for the newly predicted value
            // (since we don't know the actual error for future predictions)
            for (int i = workingErrors.Length - 1; i > 0; i--)
            {
                workingErrors[i] = workingErrors[i - 1];
            }
            workingErrors[0] = NumOps.Zero;
        }

        return predictions;
    }

    /// <summary>
    /// Evaluates the model's performance on test data.
    /// </summary>
    /// <param name="xTest">Feature matrix for testing.</param>
    /// <param name="yTest">Actual target values for testing.</param>
    /// <returns>A dictionary of evaluation metrics (MSE, RMSE, MAE, MAPE).</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method measures how well the model performs by comparing its predictions
    /// against actual values from a test dataset.
    /// 
    /// It calculates several common error metrics:
    /// - MSE (Mean Squared Error): The average of squared differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): The square root of MSE, which is in the same units as the original data
    /// - MAE (Mean Absolute Error): The average of absolute differences between predictions and actual values
    /// - MAPE (Mean Absolute Percentage Error): The average percentage difference between predictions and actual values
    /// 
    /// Lower values for all these metrics indicate better performance.
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        // Check if model has been trained
        if (!_isTrained)
        {
            throw new InvalidOperationException("Model must be trained before evaluation.");
        }

        // Generate predictions for the test data
        Vector<T> predictions = Predict(xTest);

        // Calculate various error metrics
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Squared Error (MSE)
        T mse = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);
        metrics["MSE"] = mse;

        // Root Mean Squared Error (RMSE)
        T rmse = NumOps.Sqrt(mse);
        metrics["RMSE"] = rmse;

        // Mean Absolute Error (MAE)
        T mae = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);
        metrics["MAE"] = mae;

        // Mean Absolute Percentage Error (MAPE)
        T mape = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(yTest, predictions);
        metrics["MAPE"] = mape;

        return metrics;
    }

    /// <summary>
    /// Serializes the model's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This private method saves the model's internal state to a file or stream.
    /// 
    /// Serialization allows you to:
    /// 1. Save a trained model to disk
    /// 2. Load it later without having to retrain
    /// 3. Share the model with others
    /// 
    /// The method saves all the essential parameters: the order (q) value,
    /// the mean of the series, the MA coefficients, and the recent errors.
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Write model status
        writer.Write(_isTrained);

        // Write MA-specific options
        writer.Write(_maOptions.MAOrder);

        // Write mean
        writer.Write(Convert.ToDouble(_mean));

        // Write noise variance
        writer.Write(Convert.ToDouble(_noiseVariance));

        // Write MA coefficients
        writer.Write(_maCoefficients.Length);
        for (int i = 0; i < _maCoefficients.Length; i++)
        {
            writer.Write(Convert.ToDouble(_maCoefficients[i]));
        }

        // Write recent errors
        writer.Write(_recentErrors.Length);
        for (int i = 0; i < _recentErrors.Length; i++)
        {
            writer.Write(Convert.ToDouble(_recentErrors[i]));
        }
    }

    /// <summary>
    /// Deserializes the model's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This private method loads a previously saved model from a file or stream.
    /// 
    /// Deserialization allows you to:
    /// 1. Load a previously trained model
    /// 2. Use it immediately without retraining
    /// 3. Apply the exact same model to new data
    /// 
    /// The method loads all the parameters that were saved during serialization:
    /// the order (q) value, the mean of the series, the MA coefficients, and the recent errors.
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Read model status
        _isTrained = reader.ReadBoolean();

        // Read MA-specific options
        int q = reader.ReadInt32();
        _maOptions = new MAModelOptions<T> { MAOrder = q };

        // Read mean
        _mean = NumOps.FromDouble(reader.ReadDouble());

        // Read noise variance
        _noiseVariance = NumOps.FromDouble(reader.ReadDouble());

        // Read MA coefficients
        int maLength = reader.ReadInt32();
        _maCoefficients = new Vector<T>(maLength);
        for (int i = 0; i < maLength; i++)
        {
            _maCoefficients[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read recent errors
        int errorsLength = reader.ReadInt32();
        _recentErrors = new Vector<T>(errorsLength);
        for (int i = 0; i < errorsLength; i++)
        {
            _recentErrors[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method provides a summary of your model's settings and what it has learned.
    /// 
    /// The metadata includes:
    /// - The type of model (MA)
    /// - The q parameter that defines the model structure
    /// - The MA coefficients that were learned during training
    /// - The mean value that serves as the baseline prediction
    /// 
    /// This information is useful for:
    /// - Documenting your model for future reference
    /// - Comparing different models to see which performs best
    /// - Understanding what patterns the model has identified in your data
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.MAModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                // MA-specific parameters
                { "Q", _maOptions.MAOrder },
                { "IsTrained", _isTrained },
                
                // Model parameters
                { "MACoefficientsCount", _maCoefficients.Length },
                { "Mean", Convert.ToDouble(_mean) },
                { "NoiseVariance", Convert.ToDouble(_noiseVariance) },
                
                // Add specific coefficient values for inspection
                { "MACoefficients", _maCoefficients.Select(c => Convert.ToDouble(c)).ToArray() }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the MA model with the same options.
    /// </summary>
    /// <returns>A new instance of the MA model.</returns>
    /// <remarks>
    /// <b>For Beginners:</b>
    /// This method creates a fresh copy of the model with the same settings.
    /// 
    /// The new copy:
    /// - Has the same q parameter as the original model
    /// - Has the same configuration options
    /// - Is untrained (doesn't have coefficients yet)
    /// 
    /// This is useful when you want to:
    /// - Train multiple versions of the same model on different data
    /// - Create ensemble models that combine predictions from multiple similar models
    /// - Reset a model to start fresh while keeping the same structure
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a new instance with the same options
        return new MAModel<T>(_maOptions);
    }
}
