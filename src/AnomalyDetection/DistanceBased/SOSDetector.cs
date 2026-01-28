using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using SOS (Stochastic Outlier Selection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SOS is based on the concept of affinity - how likely a point is to be
/// selected as a neighbor by other points. If a point has low affinity (other points rarely
/// select it as a neighbor), it is likely an outlier.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute pairwise affinities using Gaussian kernel
/// 2. Normalize affinities (like t-SNE probabilities)
/// 3. Compute binding probability for each point
/// 4. Points with low binding probability are outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you want a probabilistic interpretation of outlierness
/// - Medium-sized datasets
/// - When local density variations exist
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Perplexity: 30 (similar to t-SNE)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Janssens, J.H.M., et al. (2012). "Stochastic Outlier Selection." Technical Report.
/// </para>
/// </remarks>
public class SOSDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _perplexity;
    private double[][]? _trainingData;
    private double[]? _trainingBindingProbs;

    /// <summary>
    /// Gets the perplexity parameter.
    /// </summary>
    public double Perplexity => _perplexity;

    /// <summary>
    /// Creates a new SOS anomaly detector.
    /// </summary>
    /// <param name="perplexity">
    /// Perplexity parameter (similar to t-SNE). Controls the effective number of neighbors.
    /// Default is 30.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SOSDetector(double perplexity = 30, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (perplexity < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(perplexity),
                "Perplexity must be at least 1. Recommended is 30.");
        }

        _perplexity = perplexity;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        // Convert to double array
        _trainingData = new double[n][];
        for (int i = 0; i < n; i++)
        {
            _trainingData[i] = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                _trainingData[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Compute binding probabilities for training data
        _trainingBindingProbs = ComputeBindingProbabilities(_trainingData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[] ComputeBindingProbabilities(double[][] data)
    {
        int n = data.Length;

        // Compute squared distances
        var sqDistances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int d = 0; d < data[0].Length; d++)
                {
                    double diff = data[i][d] - data[j][d];
                    dist += diff * diff;
                }
                sqDistances[i, j] = dist;
                sqDistances[j, i] = dist;
            }
        }

        // Compute affinities with binary search for sigma (perplexity-based)
        var affinities = new double[n, n];
        double targetEntropy = Math.Log(_perplexity);

        for (int i = 0; i < n; i++)
        {
            // Binary search for sigma that gives target perplexity
            double sigmaMin = 1e-20;
            double sigmaMax = 1e20;
            double sigma = 1.0;

            for (int iter = 0; iter < 50; iter++)
            {
                // Compute affinities with current sigma
                double sumAff = 0;
                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        affinities[i, j] = Math.Exp(-sqDistances[i, j] / (2 * sigma * sigma));
                        sumAff += affinities[i, j];
                    }
                }

                // Normalize and compute entropy
                double entropy = 0;
                if (sumAff > 1e-10)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (i != j)
                        {
                            double p = affinities[i, j] / sumAff;
                            if (p > 1e-10)
                            {
                                entropy -= p * Math.Log(p);
                            }
                        }
                    }
                }

                // Binary search update
                if (Math.Abs(entropy - targetEntropy) < 1e-5) break;

                if (entropy < targetEntropy)
                {
                    sigmaMin = sigma;
                    sigma = (sigmaMax == 1e20) ? sigma * 2 : (sigma + sigmaMax) / 2;
                }
                else
                {
                    sigmaMax = sigma;
                    sigma = (sigma + sigmaMin) / 2;
                }
            }

            // Normalize affinities
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sum += affinities[i, j];
                }
            }

            if (sum > 1e-10)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        affinities[i, j] /= sum;
                    }
                }
            }
        }

        // Compute binding probability for each point
        // p_j = product over i of (1 - a_ij)
        // = probability that j is NOT selected by ANY point
        var bindingProbs = new double[n];

        for (int j = 0; j < n; j++)
        {
            double logProd = 0;
            for (int i = 0; i < n; i++)
            {
                if (i != j)
                {
                    logProd += Math.Log(1 - affinities[i, j] + 1e-10);
                }
            }
            // Outlier score = probability of NOT being selected
            bindingProbs[j] = Math.Exp(logProd);
        }

        return bindingProbs;
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

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // For scoring, we need to include the new points in the affinity computation
        int nTrain = trainingData.Length;
        int nTest = X.Rows;
        int nTotal = nTrain + nTest;

        // Combine training and test data
        var allData = new double[nTotal][];
        for (int i = 0; i < nTrain; i++)
        {
            allData[i] = trainingData[i];
        }

        for (int i = 0; i < nTest; i++)
        {
            allData[nTrain + i] = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                allData[nTrain + i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Compute binding probabilities for all data
        var allBindingProbs = ComputeBindingProbabilities(allData);

        // Extract scores for test points
        var scores = new Vector<T>(nTest);
        for (int i = 0; i < nTest; i++)
        {
            // Binding probability = probability of being an outlier
            scores[i] = NumOps.FromDouble(allBindingProbs[nTrain + i]);
        }

        return scores;
    }
}
