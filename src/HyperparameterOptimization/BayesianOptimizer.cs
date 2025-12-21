using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements Bayesian optimization for hyperparameter tuning using Gaussian Process regression.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Bayesian optimization is a smart search strategy that learns from
/// previous trials to decide what to try next. Unlike grid or random search, it:
/// - Builds a model of how hyperparameters affect performance
/// - Uses this model to focus on promising regions
/// - Balances exploration (trying new areas) with exploitation (refining good areas)
///
/// This makes it much more efficient than random search, often finding good hyperparameters
/// in 10-20 trials instead of 100s.
///
/// Key components:
/// - Surrogate Model: A Gaussian Process that predicts performance for any hyperparameter combination
/// - Acquisition Function: Decides where to sample next based on predicted mean and uncertainty
/// - Sequential Optimization: Each trial informs the next, unlike parallel random search
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class BayesianOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    private readonly Random _random;
    private readonly AcquisitionFunctionType _acquisitionFunction;
    private readonly int _nInitialPoints;
    private readonly double _explorationWeight;
    private readonly INumericOperations<T> _numOps;

    // Gaussian Process state
    private List<double[]> _observedPoints;
    private List<double> _observedValues;
    private double[,]? _covarianceMatrix;
    private double[,]? _covarianceMatrixInverse;
    private double _lengthScale;
    private double _signalVariance;
    private double _noiseVariance;

    /// <summary>
    /// Initializes a new instance of the BayesianOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="acquisitionFunction">The acquisition function to use for selecting next points.</param>
    /// <param name="nInitialPoints">Number of random points to sample before using Bayesian optimization.</param>
    /// <param name="explorationWeight">Weight for exploration vs exploitation (higher = more exploration).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public BayesianOptimizer(
        bool maximize = true,
        AcquisitionFunctionType acquisitionFunction = AcquisitionFunctionType.ExpectedImprovement,
        int nInitialPoints = 5,
        double explorationWeight = 2.0,
        int? seed = null) : base(maximize)
    {
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        _acquisitionFunction = acquisitionFunction;
        _nInitialPoints = Math.Max(2, nInitialPoints);
        _explorationWeight = explorationWeight;
        _numOps = MathHelper.GetNumericOperations<T>();

        _observedPoints = new List<double[]>();
        _observedValues = new List<double>();
        _lengthScale = 1.0;
        _signalVariance = 1.0;
        _noiseVariance = 0.01;
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration using Bayesian optimization.
    /// </summary>
    public override HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials)
    {
        ValidateOptimizationInputs(objectiveFunction, searchSpace, nTrials);

        SearchSpace = searchSpace;
        Trials.Clear();
        _observedPoints.Clear();
        _observedValues.Clear();

        var startTime = DateTime.UtcNow;
        var parameterNames = searchSpace.Parameters.Keys.ToList();

        lock (SyncLock)
        {
            for (int i = 0; i < nTrials; i++)
            {
                var trial = new HyperparameterTrial<T>(i);
                Dictionary<string, object> parameters;

                if (i < _nInitialPoints)
                {
                    // Initial random sampling phase
                    parameters = SampleRandomPoint(searchSpace);
                }
                else
                {
                    // Bayesian optimization phase
                    parameters = SuggestNext(trial);
                }

                // Evaluate the trial
                EvaluateTrialSafely(trial, objectiveFunction, parameters);

                // Update Gaussian Process with new observation
                if (trial.Status == TrialStatus.Complete && trial.ObjectiveValue != null)
                {
                    var point = ParametersToArray(parameters, parameterNames, searchSpace);
                    var value = _numOps.ToDouble(trial.ObjectiveValue);

                    // Negate if minimizing (GP always maximizes)
                    if (!Maximize)
                        value = -value;

                    _observedPoints.Add(point);
                    _observedValues.Add(value);

                    // Update GP hyperparameters periodically
                    if (_observedPoints.Count >= _nInitialPoints && _observedPoints.Count % 5 == 0)
                    {
                        OptimizeGPHyperparameters();
                    }

                    UpdateCovarianceMatrix();
                }

                Trials.Add(trial);
            }
        }

        var endTime = DateTime.UtcNow;
        return CreateOptimizationResult(searchSpace, startTime, endTime, nTrials);
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration using the acquisition function.
    /// </summary>
    public override Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        if (SearchSpace == null)
            throw new InvalidOperationException("Search space not initialized. Call Optimize() first.");

        if (_observedPoints.Count < _nInitialPoints)
        {
            return SampleRandomPoint(SearchSpace);
        }

        // Optimize acquisition function to find next point
        var parameterNames = SearchSpace.Parameters.Keys.ToList();
        var bestPoint = OptimizeAcquisitionFunction(parameterNames);

        return ArrayToParameters(bestPoint, parameterNames, SearchSpace);
    }

    #region Gaussian Process Methods

    /// <summary>
    /// Computes the RBF (Radial Basis Function) kernel between two points.
    /// </summary>
    private double RBFKernel(double[] x1, double[] x2)
    {
        double sumSquaredDiff = 0;
        for (int i = 0; i < x1.Length; i++)
        {
            double diff = x1[i] - x2[i];
            sumSquaredDiff += diff * diff;
        }
        return _signalVariance * Math.Exp(-sumSquaredDiff / (2 * _lengthScale * _lengthScale));
    }

    /// <summary>
    /// Updates the covariance matrix with current observations.
    /// </summary>
    private void UpdateCovarianceMatrix()
    {
        int n = _observedPoints.Count;
        _covarianceMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _covarianceMatrix[i, j] = RBFKernel(_observedPoints[i], _observedPoints[j]);
                if (i == j)
                    _covarianceMatrix[i, j] += _noiseVariance;
            }
        }

        // Compute inverse using Cholesky decomposition for numerical stability
        _covarianceMatrixInverse = InvertMatrixCholesky(_covarianceMatrix);
    }

    /// <summary>
    /// Predicts the mean and variance at a new point using the Gaussian Process.
    /// </summary>
    private (double mean, double variance) PredictGP(double[] x)
    {
        if (_observedPoints.Count == 0 || _covarianceMatrixInverse == null)
        {
            return (0.0, _signalVariance);
        }

        int n = _observedPoints.Count;
        var kStar = new double[n];

        for (int i = 0; i < n; i++)
        {
            kStar[i] = RBFKernel(x, _observedPoints[i]);
        }

        // Mean prediction: k* @ K^-1 @ y
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * _observedValues[j];
            }
            mean += kStar[i] * sum;
        }

        // Variance prediction: k** - k* @ K^-1 @ k*^T
        double kStarStar = RBFKernel(x, x);
        double variance = kStarStar;

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * kStar[j];
            }
            variance -= kStar[i] * sum;
        }

        // Ensure variance is non-negative
        variance = Math.Max(0, variance);

        return (mean, variance);
    }

    /// <summary>
    /// Optimizes GP hyperparameters using marginal likelihood.
    /// </summary>
    private void OptimizeGPHyperparameters()
    {
        // Simple grid search for length scale optimization
        double bestLengthScale = _lengthScale;
        double bestLogLikelihood = double.NegativeInfinity;

        foreach (double ls in new[] { 0.1, 0.5, 1.0, 2.0, 5.0 })
        {
            _lengthScale = ls;
            UpdateCovarianceMatrix();
            double ll = ComputeLogMarginalLikelihood();

            if (ll > bestLogLikelihood)
            {
                bestLogLikelihood = ll;
                bestLengthScale = ls;
            }
        }

        _lengthScale = bestLengthScale;
        UpdateCovarianceMatrix();
    }

    /// <summary>
    /// Computes the log marginal likelihood of the GP.
    /// </summary>
    private double ComputeLogMarginalLikelihood()
    {
        if (_covarianceMatrix == null || _covarianceMatrixInverse == null)
            return double.NegativeInfinity;

        int n = _observedPoints.Count;
        double[] y = _observedValues.ToArray();

        // -0.5 * (y^T K^-1 y + log|K| + n*log(2*pi))
        double dataFit = 0;
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                sum += _covarianceMatrixInverse[i, j] * y[j];
            }
            dataFit += y[i] * sum;
        }

        double logDet = LogDeterminant(_covarianceMatrix);

        return -0.5 * (dataFit + logDet + n * Math.Log(2 * Math.PI));
    }

    #endregion

    #region Acquisition Functions

    /// <summary>
    /// Computes the acquisition function value at a point.
    /// </summary>
    private double ComputeAcquisition(double[] x)
    {
        var (mean, variance) = PredictGP(x);
        double std = Math.Sqrt(variance + 1e-9);

        return _acquisitionFunction switch
        {
            AcquisitionFunctionType.ExpectedImprovement => ComputeExpectedImprovement(mean, std),
            AcquisitionFunctionType.ProbabilityOfImprovement => ComputeProbabilityOfImprovement(mean, std),
            AcquisitionFunctionType.UpperConfidenceBound => ComputeUpperConfidenceBound(mean, std),
            AcquisitionFunctionType.LowerConfidenceBound => ComputeLowerConfidenceBound(mean, std),
            _ => ComputeExpectedImprovement(mean, std)
        };
    }

    /// <summary>
    /// Computes Expected Improvement (EI) acquisition function.
    /// </summary>
    private double ComputeExpectedImprovement(double mean, double std)
    {
        if (std < 1e-9)
            return 0;

        double bestValue = _observedValues.Count > 0 ? _observedValues.Max() : 0;
        double z = (mean - bestValue) / std;

        // EI = std * (z * Phi(z) + phi(z))
        double phi = NormalPdf(z);
        double Phi = NormalCdf(z);

        return std * (z * Phi + phi);
    }

    /// <summary>
    /// Computes Probability of Improvement (PI) acquisition function.
    /// </summary>
    private double ComputeProbabilityOfImprovement(double mean, double std)
    {
        if (std < 1e-9)
            return 0;

        double bestValue = _observedValues.Count > 0 ? _observedValues.Max() : 0;
        double z = (mean - bestValue) / std;

        return NormalCdf(z);
    }

    /// <summary>
    /// Computes Upper Confidence Bound (UCB) acquisition function.
    /// </summary>
    private double ComputeUpperConfidenceBound(double mean, double std)
    {
        return mean + _explorationWeight * std;
    }

    /// <summary>
    /// Computes Lower Confidence Bound (LCB) for minimization problems.
    /// </summary>
    private double ComputeLowerConfidenceBound(double mean, double std)
    {
        return mean - _explorationWeight * std;
    }

    /// <summary>
    /// Optimizes the acquisition function to find the next sampling point.
    /// </summary>
    private double[] OptimizeAcquisitionFunction(List<string> parameterNames)
    {
        if (SearchSpace == null)
            throw new InvalidOperationException("Search space not initialized.");

        int nDims = parameterNames.Count;
        double[] bestPoint = new double[nDims];
        double bestAcquisition = double.NegativeInfinity;

        // Multi-start optimization: try random restarts
        int nRestarts = 20;
        int nLocalSteps = 50;

        for (int restart = 0; restart < nRestarts; restart++)
        {
            // Random starting point
            var candidatePoint = new double[nDims];
            for (int i = 0; i < nDims; i++)
            {
                candidatePoint[i] = _random.NextDouble();
            }

            // Local optimization (simple gradient-free hill climbing)
            for (int step = 0; step < nLocalSteps; step++)
            {
                double currentAcq = ComputeAcquisition(candidatePoint);

                // Try small perturbations
                var perturbation = new double[nDims];
                for (int i = 0; i < nDims; i++)
                {
                    perturbation[i] = (_random.NextDouble() - 0.5) * 0.1 * Math.Exp(-step / 20.0);
                }

                var newPoint = new double[nDims];
                for (int i = 0; i < nDims; i++)
                {
                    newPoint[i] = Math.Max(0, Math.Min(1, candidatePoint[i] + perturbation[i]));
                }

                double newAcq = ComputeAcquisition(newPoint);

                if (newAcq > currentAcq)
                {
                    Array.Copy(newPoint, candidatePoint, nDims);
                }
            }

            double finalAcq = ComputeAcquisition(candidatePoint);
            if (finalAcq > bestAcquisition)
            {
                bestAcquisition = finalAcq;
                Array.Copy(candidatePoint, bestPoint, nDims);
            }
        }

        return bestPoint;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Samples a random point from the search space.
    /// </summary>
    private Dictionary<string, object> SampleRandomPoint(HyperparameterSearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>();
        foreach (var param in searchSpace.Parameters)
        {
            parameters[param.Key] = param.Value.Sample(_random);
        }
        return parameters;
    }

    /// <summary>
    /// Converts parameters to a normalized array [0, 1].
    /// </summary>
    private double[] ParametersToArray(Dictionary<string, object> parameters, List<string> parameterNames, HyperparameterSearchSpace searchSpace)
    {
        var array = new double[parameterNames.Count];

        for (int i = 0; i < parameterNames.Count; i++)
        {
            var name = parameterNames[i];
            var value = parameters[name];
            var distribution = searchSpace.Parameters[name];

            array[i] = NormalizeParameter(value, distribution);
        }

        return array;
    }

    /// <summary>
    /// Converts a normalized array back to parameters.
    /// </summary>
    private Dictionary<string, object> ArrayToParameters(double[] array, List<string> parameterNames, HyperparameterSearchSpace searchSpace)
    {
        var parameters = new Dictionary<string, object>();

        for (int i = 0; i < parameterNames.Count; i++)
        {
            var name = parameterNames[i];
            var distribution = searchSpace.Parameters[name];

            parameters[name] = DenormalizeParameter(array[i], distribution);
        }

        return parameters;
    }

    /// <summary>
    /// Normalizes a parameter value to [0, 1].
    /// </summary>
    private double NormalizeParameter(object value, ParameterDistribution distribution)
    {
        return distribution switch
        {
            ContinuousDistribution cont => NormalizeContinuous(Convert.ToDouble(value), cont),
            IntegerDistribution intDist => NormalizeInteger(Convert.ToInt32(value), intDist),
            CategoricalDistribution cat => NormalizeCategorical(value, cat),
            _ => 0.5
        };
    }

    /// <summary>
    /// Denormalizes a [0, 1] value back to parameter space.
    /// </summary>
    private object DenormalizeParameter(double normalized, ParameterDistribution distribution)
    {
        return distribution switch
        {
            ContinuousDistribution cont => DenormalizeContinuous(normalized, cont),
            IntegerDistribution intDist => DenormalizeInteger(normalized, intDist),
            CategoricalDistribution cat => DenormalizeCategorical(normalized, cat),
            _ => normalized
        };
    }

    private double NormalizeContinuous(double value, ContinuousDistribution dist)
    {
        // Handle degenerate case where min == max to prevent division by zero
        if (Math.Abs(dist.Max - dist.Min) < double.Epsilon)
            return 0.5; // Return middle of normalized range

        if (dist.LogScale)
        {
            double logMin = Math.Log(dist.Min);
            double logMax = Math.Log(dist.Max);
            // Handle degenerate log case
            if (Math.Abs(logMax - logMin) < double.Epsilon)
                return 0.5;
            double logValue = Math.Log(value);
            return (logValue - logMin) / (logMax - logMin);
        }
        return (value - dist.Min) / (dist.Max - dist.Min);
    }

    private double DenormalizeContinuous(double normalized, ContinuousDistribution dist)
    {
        normalized = Math.Max(0, Math.Min(1, normalized));

        if (dist.LogScale)
        {
            double logMin = Math.Log(dist.Min);
            double logMax = Math.Log(dist.Max);
            return Math.Exp(logMin + normalized * (logMax - logMin));
        }
        return dist.Min + normalized * (dist.Max - dist.Min);
    }

    private double NormalizeInteger(int value, IntegerDistribution dist)
    {
        // Handle degenerate case where min == max to prevent division by zero
        if (dist.Max == dist.Min)
            return 0.5; // Return middle of normalized range
        return (double)(value - dist.Min) / (dist.Max - dist.Min);
    }

    private int DenormalizeInteger(double normalized, IntegerDistribution dist)
    {
        normalized = Math.Max(0, Math.Min(1, normalized));
        int value = (int)Math.Round(dist.Min + normalized * (dist.Max - dist.Min));
        return Math.Max(dist.Min, Math.Min(dist.Max, value));
    }

    private double NormalizeCategorical(object value, CategoricalDistribution dist)
    {
        int index = dist.Choices.IndexOf(value);
        if (index < 0) index = 0;
        return dist.Choices.Count > 1 ? (double)index / (dist.Choices.Count - 1) : 0.5;
    }

    private object DenormalizeCategorical(double normalized, CategoricalDistribution dist)
    {
        normalized = Math.Max(0, Math.Min(1, normalized));
        int index = (int)Math.Round(normalized * (dist.Choices.Count - 1));
        return dist.Choices[Math.Max(0, Math.Min(dist.Choices.Count - 1, index))];
    }

    /// <summary>
    /// Standard normal probability density function.
    /// </summary>
    private static double NormalPdf(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    /// <summary>
    /// Standard normal cumulative distribution function.
    /// </summary>
    private static double NormalCdf(double x)
    {
        // Approximation using error function
        return 0.5 * (1 + Erf(x / Math.Sqrt(2)));
    }

    /// <summary>
    /// Error function approximation.
    /// </summary>
    private static double Erf(double x)
    {
        // Abramowitz and Stegun approximation
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);

        return sign * y;
    }

    /// <summary>
    /// Inverts a positive definite matrix using Cholesky decomposition.
    /// </summary>
    private static double[,] InvertMatrixCholesky(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];
        var inverse = new double[n, n];

        // Cholesky decomposition: A = L * L^T
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        // Invert L
        var Linv = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            Linv[i, i] = 1.0 / L[i, i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum += L[i, k] * Linv[k, j];
                }
                Linv[i, j] = -sum / L[i, i];
            }
        }

        // A^-1 = L^-T * L^-1
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = i; k < n; k++)
                {
                    sum += Linv[k, i] * Linv[k, j];
                }
                inverse[i, j] = sum;
                inverse[j, i] = sum;
            }
        }

        return inverse;
    }

    /// <summary>
    /// Computes the log determinant of a matrix using Cholesky decomposition.
    /// </summary>
    private static double LogDeterminant(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var L = new double[n, n];

        // Cholesky decomposition
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = matrix[i, i] - sum;
                    L[i, j] = Math.Sqrt(Math.Max(1e-10, diag));
                }
                else
                {
                    L[i, j] = (matrix[i, j] - sum) / L[j, j];
                }
            }
        }

        // log|A| = 2 * sum(log(diag(L)))
        double logDet = 0;
        for (int i = 0; i < n; i++)
        {
            logDet += Math.Log(Math.Max(1e-10, L[i, i]));
        }

        return 2 * logDet;
    }

    #endregion
}

/// <summary>
/// Types of acquisition functions for Bayesian optimization.
/// </summary>
public enum AcquisitionFunctionType
{
    /// <summary>
    /// Expected Improvement: Balances exploration and exploitation.
    /// </summary>
    ExpectedImprovement,

    /// <summary>
    /// Probability of Improvement: Focuses on likely improvements.
    /// </summary>
    ProbabilityOfImprovement,

    /// <summary>
    /// Upper Confidence Bound: Optimistic approach for maximization.
    /// </summary>
    UpperConfidenceBound,

    /// <summary>
    /// Lower Confidence Bound: Optimistic approach for minimization.
    /// </summary>
    LowerConfidenceBound
}
