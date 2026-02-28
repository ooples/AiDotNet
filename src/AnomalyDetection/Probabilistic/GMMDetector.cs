using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Gaussian Mixture Models (GMM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GMM models data as a mixture of several Gaussian distributions.
/// Points with low probability under this model are anomalies - they don't fit well
/// into any of the learned Gaussian clusters.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Fit a mixture of k Gaussians using Expectation-Maximization
/// 2. For each point, compute its probability under the mixture model
/// 3. Points with low probability are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data is a mixture of Gaussian clusters
/// - Different clusters have different shapes/sizes
/// - You need probabilistic anomaly scores
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Components: 3 (typical range 2-10)
/// - Covariance type: diagonal (uses variance per feature for stability)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class GMMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nComponents;
    private readonly int _maxIterations;
    private Vector<T>[]? _means;
    private Matrix<T>[]? _covariances;
    private Vector<T>? _weights;
    private int _nFeatures;
    private double[]? _globalVariance;

    /// <summary>
    /// Gets the number of Gaussian components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Creates a new GMM anomaly detector.
    /// </summary>
    /// <param name="nComponents">Number of Gaussian components. Default is 3.</param>
    /// <param name="maxIterations">Maximum EM iterations. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GMMDetector(int nComponents = 3, int maxIterations = 100,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nComponents < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nComponents),
                "NComponents must be at least 1. Recommended is 3.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _nComponents = nComponents;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;

        if (n < _nComponents)
        {
            throw new ArgumentException(
                $"Number of samples ({n}) must be at least nComponents ({_nComponents}).",
                nameof(X));
        }

        // Initialize GMM using k-means
        InitializeParameters(X);

        // Run EM algorithm
        RunEM(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.Columns);
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = X[i, j];
            }

            // Compute log-likelihood under mixture model
            double logLikelihood = ComputeLogLikelihood(point);

            // Convert to anomaly score (negative log-likelihood)
            double score = -logLikelihood;
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private void InitializeParameters(Matrix<T> X)
    {
        int n = X.Rows;
        int d = X.Columns;
        var random = new Random(_randomSeed);

        // Compute global variance per feature for use as a covariance floor.
        // This prevents components with very few points (e.g., a singleton outlier)
        // from collapsing to a tiny covariance that produces astronomically high density.
        _globalVariance = new double[d];
        var featureMeans = new double[d];
        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                featureMeans[j] += NumOps.ToDouble(X[i, j]);
            }
            featureMeans[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(X[i, j]) - featureMeans[j];
                _globalVariance[j] += diff * diff;
            }
            _globalVariance[j] = Math.Max(_globalVariance[j] / n, 1e-6);
        }

        // Initialize means using k-means++ style
        _means = new Vector<T>[_nComponents];
        _covariances = new Matrix<T>[_nComponents];
        _weights = new Vector<T>(_nComponents);

        // Select initial means randomly
        var selectedIndices = new HashSet<int>();
        for (int c = 0; c < _nComponents; c++)
        {
            int idx;
            do
            {
                idx = random.Next(n);
            } while (selectedIndices.Contains(idx));
            selectedIndices.Add(idx);

            _means[c] = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                _means[c][j] = X[idx, j];
            }

            // Initialize covariance as identity
            _covariances[c] = new Matrix<T>(d, d);
            for (int j = 0; j < d; j++)
            {
                _covariances[c][j, j] = NumOps.FromDouble(1.0);
            }

            // Initialize equal weights
            _weights[c] = NumOps.FromDouble(1.0 / _nComponents);
        }
    }

    private void RunEM(Matrix<T> X)
    {
        int n = X.Rows;
        int d = X.Columns;

        // Responsibilities matrix (n x k)
        var responsibilities = new double[n, _nComponents];

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // E-step: Compute responsibilities
            for (int i = 0; i < n; i++)
            {
                var point = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    point[j] = X[i, j];
                }

                double totalProb = 0;
                var probs = new double[_nComponents];

                for (int c = 0; c < _nComponents; c++)
                {
                    double weight = NumOps.ToDouble(_weights![c]);
                    double density = GaussianDensity(point, _means![c], _covariances![c]);
                    probs[c] = weight * density;
                    totalProb += probs[c];
                }

                for (int c = 0; c < _nComponents; c++)
                {
                    responsibilities[i, c] = totalProb > 0 ? probs[c] / totalProb : 1.0 / _nComponents;
                }
            }

            // M-step: Update parameters
            for (int c = 0; c < _nComponents; c++)
            {
                double Nc = 0;
                for (int i = 0; i < n; i++)
                {
                    Nc += responsibilities[i, c];
                }

                // Update weight
                _weights![c] = NumOps.FromDouble(Nc / n);

                // Update mean
                var newMean = new Vector<T>(d);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < d; j++)
                    {
                        newMean[j] = NumOps.Add(newMean[j],
                            NumOps.Multiply(NumOps.FromDouble(responsibilities[i, c]), X[i, j]));
                    }
                }
                for (int j = 0; j < d; j++)
                {
                    _means![c][j] = Nc > 0 ? NumOps.Divide(newMean[j], NumOps.FromDouble(Nc)) : _means[c][j];
                }

                // Update covariance
                var newCov = new Matrix<T>(d, d);
                for (int i = 0; i < n; i++)
                {
                    for (int j1 = 0; j1 < d; j1++)
                    {
                        T diff1 = NumOps.Subtract(X[i, j1], _means![c][j1]);
                        for (int j2 = j1; j2 < d; j2++)
                        {
                            T diff2 = NumOps.Subtract(X[i, j2], _means[c][j2]);
                            T contrib = NumOps.Multiply(NumOps.FromDouble(responsibilities[i, c]),
                                NumOps.Multiply(diff1, diff2));
                            newCov[j1, j2] = NumOps.Add(newCov[j1, j2], contrib);
                            if (j1 != j2)
                            {
                                newCov[j2, j1] = newCov[j1, j2];
                            }
                        }
                    }
                }

                // Normalize covariance first, then apply regularization
                if (Nc > 1e-6)
                {
                    for (int j1 = 0; j1 < d; j1++)
                    {
                        for (int j2 = 0; j2 < d; j2++)
                        {
                            _covariances![c][j1, j2] = NumOps.Divide(newCov[j1, j2], NumOps.FromDouble(Nc));
                        }
                    }

                    // Apply regularization: floor the diagonal at a fraction of the global variance.
                    // This prevents components with very few points from having extremely small
                    // variance, which would cause astronomically high density and break anomaly scoring.
                    for (int j = 0; j < d; j++)
                    {
                        double currentVar = NumOps.ToDouble(_covariances![c][j, j]);
                        double minVar = Math.Max(_globalVariance![j] * 0.01, 1e-6);
                        if (currentVar < minVar)
                        {
                            _covariances[c][j, j] = NumOps.FromDouble(minVar);
                        }
                    }
                }
            }
        }
    }

    private double ComputeLogLikelihood(Vector<T> point)
    {
        double likelihood = 0;

        for (int c = 0; c < _nComponents; c++)
        {
            double weight = NumOps.ToDouble(_weights![c]);
            double density = GaussianDensity(point, _means![c], _covariances![c]);
            likelihood += weight * density;
        }

        return likelihood > 0 ? Math.Log(likelihood) : -1000;
    }

    private double GaussianDensity(Vector<T> point, Vector<T> mean, Matrix<T> covariance)
    {
        int d = point.Length;

        // Compute (x - mu)
        var diff = new double[d];
        for (int i = 0; i < d; i++)
        {
            diff[i] = NumOps.ToDouble(NumOps.Subtract(point[i], mean[i]));
        }

        // Simplified: Use diagonal approximation for stability
        double mahalanobis = 0;
        double logDet = 0;

        for (int i = 0; i < d; i++)
        {
            double variance = NumOps.ToDouble(covariance[i, i]);
            // Use global variance floor to prevent density explosion for singleton components
            double minVar = _globalVariance != null ? _globalVariance[i] * 0.01 : 1e-6;
            variance = Math.Max(variance, minVar);
            mahalanobis += diff[i] * diff[i] / variance;
            logDet += Math.Log(variance);
        }

        double logDensity = -0.5 * (d * Math.Log(2 * Math.PI) + logDet + mahalanobis);

        return Math.Exp(logDensity);
    }
}
