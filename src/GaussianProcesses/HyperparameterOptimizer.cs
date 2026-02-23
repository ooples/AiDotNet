namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Provides hyperparameter optimization for Gaussian Processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gaussian Processes have hyperparameters (like kernel length scale,
/// signal variance, and noise variance) that greatly affect performance. This class helps
/// find good values for these hyperparameters automatically.
///
/// Methods available:
/// - Grid Search: Try all combinations from a predefined grid
/// - Random Search: Try random combinations (often more efficient than grid)
/// - Gradient Descent: Follow gradients of log marginal likelihood
/// - Bayesian Optimization: Use a GP to model the objective (meta!)
///
/// The optimization target is typically the log marginal likelihood (LML):
/// - Higher LML = better fit to data (accounting for model complexity)
/// - LML naturally balances fit quality with model simplicity
/// </para>
/// </remarks>
public class HyperparameterOptimizer<T>
{
    /// <summary>
    /// The optimization method to use.
    /// </summary>
    public enum OptimizationMethod
    {
        /// <summary>
        /// Grid search over specified parameter values.
        /// </summary>
        GridSearch,

        /// <summary>
        /// Random search over parameter ranges.
        /// </summary>
        RandomSearch,

        /// <summary>
        /// Gradient-based optimization (requires differentiable kernel).
        /// </summary>
        GradientDescent,

        /// <summary>
        /// Bayesian optimization using GP surrogate.
        /// </summary>
        BayesianOptimization
    }

    /// <summary>
    /// Represents a hyperparameter configuration and its score.
    /// </summary>
    public class HyperparameterResult
    {
        /// <summary>
        /// The hyperparameter values.
        /// </summary>
        public Dictionary<string, double> Parameters { get; set; } = new();

        /// <summary>
        /// The log marginal likelihood score.
        /// </summary>
        public double LogMarginalLikelihood { get; set; }

        /// <summary>
        /// Additional metrics (e.g., cross-validation scores).
        /// </summary>
        public Dictionary<string, double> Metrics { get; set; } = new();
    }

    private readonly INumericOperations<T> _numOps;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly int _randomSeed;

    /// <summary>
    /// Initializes a new hyperparameter optimizer.
    /// </summary>
    /// <param name="maxIterations">Maximum number of iterations/evaluations. Default is 100.</param>
    /// <param name="tolerance">Convergence tolerance for gradient methods. Default is 1e-5.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a hyperparameter optimizer.
    ///
    /// Tips:
    /// - maxIterations: Start with 50-100, increase if results aren't satisfactory
    /// - tolerance: Lower values = more precise but slower optimization
    /// - randomSeed: Set for reproducible results
    /// </para>
    /// </remarks>
    public HyperparameterOptimizer(
        int maxIterations = 100,
        double tolerance = 1e-5,
        int randomSeed = 42)
    {
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));
        if (tolerance <= 0)
            throw new ArgumentException("Tolerance must be positive.", nameof(tolerance));

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _randomSeed = randomSeed;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Performs grid search over hyperparameter values.
    /// </summary>
    /// <param name="parameterGrid">Dictionary mapping parameter names to arrays of values to try.</param>
    /// <param name="evaluateFunction">Function that evaluates a parameter configuration.</param>
    /// <returns>The best hyperparameter configuration found.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Grid search tries every combination of specified parameter values.
    ///
    /// Example:
    /// var grid = new Dictionary&lt;string, double[]&gt;
    /// {
    ///     ["lengthScale"] = new[] { 0.1, 0.5, 1.0, 2.0 },
    ///     ["noiseVariance"] = new[] { 0.01, 0.1, 1.0 }
    /// };
    ///
    /// This would try 4 × 3 = 12 combinations.
    ///
    /// Pros: Thorough, guaranteed to find best in grid
    /// Cons: Exponentially expensive with more parameters
    ///
    /// Use when: Few parameters, discrete choices, need to understand parameter landscape.
    /// </para>
    /// </remarks>
    public HyperparameterResult GridSearch(
        Dictionary<string, double[]> parameterGrid,
        Func<Dictionary<string, double>, double> evaluateFunction)
    {
        if (parameterGrid is null) throw new ArgumentNullException(nameof(parameterGrid));
        if (evaluateFunction is null) throw new ArgumentNullException(nameof(evaluateFunction));

        if (parameterGrid.Count == 0)
            throw new ArgumentException("Parameter grid cannot be empty.", nameof(parameterGrid));

        var paramNames = parameterGrid.Keys.ToList();
        var paramValues = parameterGrid.Values.ToList();

        var bestResult = new HyperparameterResult { LogMarginalLikelihood = double.NegativeInfinity };
        var allResults = new List<HyperparameterResult>();

        // Generate all combinations
        var combinations = GenerateGridCombinations(paramNames, paramValues);

        foreach (var combination in combinations)
        {
            try
            {
                double lml = evaluateFunction(combination);

                var result = new HyperparameterResult
                {
                    Parameters = new Dictionary<string, double>(combination),
                    LogMarginalLikelihood = lml
                };

                allResults.Add(result);

                if (lml > bestResult.LogMarginalLikelihood)
                {
                    bestResult = result;
                }
            }
            catch
            {
                // Skip invalid parameter combinations
            }
        }

        bestResult.Metrics["num_evaluations"] = allResults.Count;
        return bestResult;
    }

    /// <summary>
    /// Generates all combinations from a parameter grid.
    /// </summary>
    private IEnumerable<Dictionary<string, double>> GenerateGridCombinations(
        List<string> paramNames,
        List<double[]> paramValues)
    {
        if (paramNames.Count == 0)
        {
            yield return new Dictionary<string, double>();
            yield break;
        }

        var indices = new int[paramNames.Count];
        var sizes = paramValues.Select(v => v.Length).ToArray();

        while (true)
        {
            // Generate current combination
            var combination = new Dictionary<string, double>();
            for (int i = 0; i < paramNames.Count; i++)
            {
                combination[paramNames[i]] = paramValues[i][indices[i]];
            }
            yield return combination;

            // Increment indices
            int dim = paramNames.Count - 1;
            while (dim >= 0)
            {
                indices[dim]++;
                if (indices[dim] < sizes[dim])
                    break;
                indices[dim] = 0;
                dim--;
            }

            if (dim < 0)
                break;
        }
    }

    /// <summary>
    /// Performs random search over hyperparameter ranges.
    /// </summary>
    /// <param name="parameterRanges">Dictionary mapping parameter names to (min, max) ranges.</param>
    /// <param name="evaluateFunction">Function that evaluates a parameter configuration.</param>
    /// <param name="numSamples">Number of random configurations to try. Default is 50.</param>
    /// <param name="logScale">Parameters to sample in log scale. Default is empty.</param>
    /// <returns>The best hyperparameter configuration found.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random search samples random parameter combinations from specified ranges.
    ///
    /// Example:
    /// var ranges = new Dictionary&lt;string, (double, double)&gt;
    /// {
    ///     ["lengthScale"] = (0.01, 10.0),
    ///     ["noiseVariance"] = (0.001, 1.0)
    /// };
    /// var result = optimizer.RandomSearch(ranges, evaluate, numSamples: 100,
    ///     logScale: new[] { "lengthScale", "noiseVariance" });
    ///
    /// Using log scale is important for parameters that span orders of magnitude.
    ///
    /// Pros: More efficient than grid search for many parameters
    /// Cons: May miss good values by chance
    ///
    /// Research shows random search often beats grid search because it explores
    /// more unique values of each parameter.
    /// </para>
    /// </remarks>
    public HyperparameterResult RandomSearch(
        Dictionary<string, (double min, double max)> parameterRanges,
        Func<Dictionary<string, double>, double> evaluateFunction,
        int numSamples = 50,
        string[]? logScale = null)
    {
        if (parameterRanges is null) throw new ArgumentNullException(nameof(parameterRanges));
        if (evaluateFunction is null) throw new ArgumentNullException(nameof(evaluateFunction));

        if (parameterRanges.Count == 0)
            throw new ArgumentException("Parameter ranges cannot be empty.", nameof(parameterRanges));

        var logScaleSet = new HashSet<string>(logScale ?? Array.Empty<string>());
        var rand = RandomHelper.CreateSeededRandom(_randomSeed);

        var bestResult = new HyperparameterResult { LogMarginalLikelihood = double.NegativeInfinity };
        int successfulEvals = 0;

        for (int i = 0; i < numSamples && successfulEvals < _maxIterations; i++)
        {
            // Sample random parameters
            var parameters = new Dictionary<string, double>();
            foreach (var (name, (min, max)) in parameterRanges)
            {
                double value;
                if (logScaleSet.Contains(name))
                {
                    // Log-uniform sampling
                    double logMin = Math.Log(Math.Max(min, 1e-10));
                    double logMax = Math.Log(max);
                    value = Math.Exp(logMin + rand.NextDouble() * (logMax - logMin));
                }
                else
                {
                    // Uniform sampling
                    value = min + rand.NextDouble() * (max - min);
                }
                parameters[name] = value;
            }

            try
            {
                double lml = evaluateFunction(parameters);
                successfulEvals++;

                if (lml > bestResult.LogMarginalLikelihood)
                {
                    bestResult = new HyperparameterResult
                    {
                        Parameters = new Dictionary<string, double>(parameters),
                        LogMarginalLikelihood = lml
                    };
                }
            }
            catch
            {
                // Skip invalid parameter combinations
            }
        }

        bestResult.Metrics["num_evaluations"] = successfulEvals;
        return bestResult;
    }

    /// <summary>
    /// Performs gradient-based optimization of hyperparameters.
    /// </summary>
    /// <param name="initialParameters">Starting parameter values.</param>
    /// <param name="evaluateWithGradient">Function that returns (value, gradient) for given parameters.</param>
    /// <param name="learningRate">Initial learning rate. Default is 0.1.</param>
    /// <param name="parameterBounds">Optional bounds for parameters.</param>
    /// <returns>The optimized hyperparameter configuration.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient descent follows the gradient of the log marginal likelihood
    /// to find optimal hyperparameters.
    ///
    /// This is the most efficient method when gradients are available:
    /// - Standard GP with RBF kernel: gradients available analytically
    /// - Requires the kernel to be differentiable
    ///
    /// The optimization finds local optima, so:
    /// - Multiple restarts from different initial values is recommended
    /// - Combine with random search for initialization
    ///
    /// Uses Adam optimizer internally for robust convergence.
    /// </para>
    /// </remarks>
    public HyperparameterResult GradientDescent(
        Dictionary<string, double> initialParameters,
        Func<Dictionary<string, double>, (double value, Dictionary<string, double> gradient)> evaluateWithGradient,
        double learningRate = 0.1,
        Dictionary<string, (double min, double max)>? parameterBounds = null)
    {
        if (initialParameters is null) throw new ArgumentNullException(nameof(initialParameters));
        if (evaluateWithGradient is null) throw new ArgumentNullException(nameof(evaluateWithGradient));

        if (initialParameters.Count == 0)
            throw new ArgumentException("Initial parameters cannot be empty.", nameof(initialParameters));

        var parameters = new Dictionary<string, double>(initialParameters);

        // Adam optimizer state
        var m = new Dictionary<string, double>();
        var v = new Dictionary<string, double>();
        foreach (var name in parameters.Keys)
        {
            m[name] = 0;
            v[name] = 0;
        }

        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        double bestLml = double.NegativeInfinity;
        var bestParameters = new Dictionary<string, double>(parameters);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            try
            {
                var (lml, gradient) = evaluateWithGradient(parameters);

                if (lml > bestLml)
                {
                    bestLml = lml;
                    bestParameters = new Dictionary<string, double>(parameters);
                }

                // Check convergence
                double gradNorm = 0;
                foreach (var g in gradient.Values)
                {
                    gradNorm += g * g;
                }
                gradNorm = Math.Sqrt(gradNorm);

                if (gradNorm < _tolerance)
                    break;

                // Adam update
                int t = iter + 1;
                foreach (var name in parameters.Keys.ToList())
                {
                    if (!gradient.TryGetValue(name, out double g))
                        continue;

                    // We're maximizing, so follow positive gradient
                    m[name] = beta1 * m[name] + (1 - beta1) * g;
                    v[name] = beta2 * v[name] + (1 - beta2) * g * g;

                    double mHat = m[name] / (1 - Math.Pow(beta1, t));
                    double vHat = v[name] / (1 - Math.Pow(beta2, t));

                    double update = learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                    parameters[name] += update;

                    // Apply bounds
                    if (parameterBounds is not null && parameterBounds.TryGetValue(name, out var bounds))
                    {
                        parameters[name] = Math.Max(bounds.min, Math.Min(bounds.max, parameters[name]));
                    }

                    // Ensure positivity for common parameters
                    if (name.Contains("variance", StringComparison.OrdinalIgnoreCase) ||
                        name.Contains("length", StringComparison.OrdinalIgnoreCase) ||
                        name.Contains("scale", StringComparison.OrdinalIgnoreCase))
                    {
                        parameters[name] = Math.Max(1e-6, parameters[name]);
                    }
                }
            }
            catch
            {
                // Reset to best parameters on error
                parameters = new Dictionary<string, double>(bestParameters);
                learningRate *= 0.5; // Reduce learning rate
            }
        }

        return new HyperparameterResult
        {
            Parameters = bestParameters,
            LogMarginalLikelihood = bestLml,
            Metrics = new Dictionary<string, double>
            {
                ["num_iterations"] = _maxIterations
            }
        };
    }

    /// <summary>
    /// Performs cross-validation to estimate generalization performance.
    /// </summary>
    /// <param name="X">Training inputs.</param>
    /// <param name="y">Training targets.</param>
    /// <param name="createModel">Function that creates a GP model with given parameters.</param>
    /// <param name="parameters">Hyperparameters to evaluate.</param>
    /// <param name="numFolds">Number of cross-validation folds. Default is 5.</param>
    /// <returns>Mean and standard deviation of test log-likelihood.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-validation estimates how well the GP will generalize
    /// to new data by:
    /// 1. Splitting data into K folds
    /// 2. For each fold: train on K-1 folds, test on remaining fold
    /// 3. Average test performance across all folds
    ///
    /// This helps prevent overfitting to the training data when selecting hyperparameters.
    ///
    /// Use when:
    /// - You have limited data
    /// - You want to compare different model configurations
    /// - Log marginal likelihood alone isn't reliable (small datasets)
    /// </para>
    /// </remarks>
    public (double mean, double std) CrossValidate(
        Matrix<T> X,
        Vector<T> y,
        Func<Dictionary<string, double>, (Func<Matrix<T>, Vector<T>, (Vector<T> predictions, Vector<T> variances)> predictFunc, Action<Matrix<T>, Vector<T>> fitFunc)> createModel,
        Dictionary<string, double> parameters,
        int numFolds = 5)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (createModel is null) throw new ArgumentNullException(nameof(createModel));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        if (X.Rows != y.Length)
            throw new ArgumentException("X and y must have the same number of samples.");
        if (numFolds < 2)
            throw new ArgumentException("Number of folds must be at least 2.", nameof(numFolds));
        if (numFolds > X.Rows)
            throw new ArgumentException("Number of folds cannot exceed number of samples.");

        int n = X.Rows;
        int d = X.Columns;
        var rand = RandomHelper.CreateSeededRandom(_randomSeed);

        // Shuffle indices
        var indices = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Compute fold sizes
        int baseFoldSize = n / numFolds;
        int remainder = n % numFolds;

        var foldScores = new List<double>();
        int startIdx = 0;

        for (int fold = 0; fold < numFolds; fold++)
        {
            int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);
            int endIdx = startIdx + foldSize;

            // Create test indices for this fold
            var testIndices = new HashSet<int>();
            for (int i = startIdx; i < endIdx; i++)
            {
                testIndices.Add(indices[i]);
            }

            // Split data
            int trainSize = n - foldSize;
            var XTrain = new Matrix<T>(trainSize, d);
            var yTrain = new Vector<T>(trainSize);
            var XTest = new Matrix<T>(foldSize, d);
            var yTest = new Vector<T>(foldSize);

            int trainIdx = 0, testIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (testIndices.Contains(i))
                {
                    for (int j = 0; j < d; j++)
                    {
                        XTest[testIdx, j] = X[i, j];
                    }
                    yTest[testIdx] = y[i];
                    testIdx++;
                }
                else
                {
                    for (int j = 0; j < d; j++)
                    {
                        XTrain[trainIdx, j] = X[i, j];
                    }
                    yTrain[trainIdx] = y[i];
                    trainIdx++;
                }
            }

            try
            {
                // Create and train model
                var (predictFunc, fitFunc) = createModel(parameters);
                fitFunc(XTrain, yTrain);

                // Predict on test set
                var (predictions, variances) = predictFunc(XTest, yTest);

                // Compute test log-likelihood
                double testLl = 0;
                for (int i = 0; i < foldSize; i++)
                {
                    double pred = _numOps.ToDouble(predictions[i]);
                    double var = Math.Max(_numOps.ToDouble(variances[i]), 1e-10);
                    double actual = _numOps.ToDouble(yTest[i]);

                    // Gaussian log-likelihood
                    double diff = actual - pred;
                    testLl += -0.5 * (Math.Log(2 * Math.PI * var) + diff * diff / var);
                }

                foldScores.Add(testLl / foldSize);
            }
            catch
            {
                // Skip this fold on error
            }

            startIdx = endIdx;
        }

        if (foldScores.Count == 0)
        {
            return (double.NegativeInfinity, 0);
        }

        double mean = foldScores.Average();
        double variance = foldScores.Select(x => (x - mean) * (x - mean)).Average();
        double std = Math.Sqrt(variance);

        return (mean, std);
    }

    /// <summary>
    /// Performs multi-start optimization with random initializations.
    /// </summary>
    /// <param name="parameterRanges">Ranges for each parameter.</param>
    /// <param name="evaluateWithGradient">Evaluation function with gradient.</param>
    /// <param name="numRestarts">Number of random restarts. Default is 10.</param>
    /// <param name="learningRate">Learning rate for gradient descent. Default is 0.1.</param>
    /// <returns>The best result across all restarts.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-start optimization combines random search with gradient descent:
    /// 1. Sample numRestarts random starting points
    /// 2. Run gradient descent from each starting point
    /// 3. Return the best result found
    ///
    /// This helps find global optima (or at least good local optima) by exploring
    /// multiple regions of the hyperparameter space.
    ///
    /// Recommended when:
    /// - The objective has multiple local optima
    /// - You want robust results
    /// - Computational budget allows multiple optimizations
    /// </para>
    /// </remarks>
    public HyperparameterResult MultiStartOptimization(
        Dictionary<string, (double min, double max)> parameterRanges,
        Func<Dictionary<string, double>, (double value, Dictionary<string, double> gradient)> evaluateWithGradient,
        int numRestarts = 10,
        double learningRate = 0.1)
    {
        if (parameterRanges is null) throw new ArgumentNullException(nameof(parameterRanges));
        if (evaluateWithGradient is null) throw new ArgumentNullException(nameof(evaluateWithGradient));

        var rand = RandomHelper.CreateSeededRandom(_randomSeed);
        var bestResult = new HyperparameterResult { LogMarginalLikelihood = double.NegativeInfinity };

        for (int restart = 0; restart < numRestarts; restart++)
        {
            // Sample random initial parameters
            var initialParams = new Dictionary<string, double>();
            foreach (var (name, (min, max)) in parameterRanges)
            {
                // Use log-uniform for scale parameters
                if (name.Contains("variance", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("length", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("scale", StringComparison.OrdinalIgnoreCase))
                {
                    double logMin = Math.Log(Math.Max(min, 1e-10));
                    double logMax = Math.Log(max);
                    initialParams[name] = Math.Exp(logMin + rand.NextDouble() * (logMax - logMin));
                }
                else
                {
                    initialParams[name] = min + rand.NextDouble() * (max - min);
                }
            }

            try
            {
                var result = GradientDescent(initialParams, evaluateWithGradient, learningRate, parameterRanges);

                if (result.LogMarginalLikelihood > bestResult.LogMarginalLikelihood)
                {
                    bestResult = result;
                }
            }
            catch
            {
                // Skip failed restarts
            }
        }

        bestResult.Metrics["num_restarts"] = numRestarts;
        return bestResult;
    }
}
