using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
public class SOSDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _perplexity;
    private Matrix<T>? _trainingData;
    private Vector<T>? _trainingBindingProbs;

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

        _trainingData = X;

        // Compute binding probabilities for training data
        _trainingBindingProbs = ComputeBindingProbabilities(X);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Vector<T> ComputeBindingProbabilities(Matrix<T> data)
    {
        int n = data.Rows;

        // Compute squared distances using Engine vectorization
        var sqDistances = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var pointI = new Vector<T>(data.GetRowReadOnlySpan(i).ToArray());
            for (int j = i + 1; j < n; j++)
            {
                var pointJ = new Vector<T>(data.GetRowReadOnlySpan(j).ToArray());
                var diff = Engine.Subtract(pointI, pointJ);
                T distSq = Engine.DotProduct(diff, diff);
                sqDistances[i, j] = distSq;
                sqDistances[j, i] = distSq;
            }
        }

        // Compute affinities with binary search for sigma (perplexity-based)
        var affinities = new Matrix<T>(n, n);
        T targetEntropy = NumOps.Log(NumOps.FromDouble(_perplexity));
        T eps10 = NumOps.FromDouble(1e-10);
        T eps5 = NumOps.FromDouble(1e-5);
        T two = NumOps.FromDouble(2);

        for (int i = 0; i < n; i++)
        {
            // Binary search for sigma that gives target perplexity
            T sigmaMin = NumOps.FromDouble(1e-20);
            T sigmaMax = NumOps.FromDouble(1e20);
            T sigma = NumOps.One;
            T sigmaMaxInit = sigmaMax;

            for (int iter = 0; iter < 50; iter++)
            {
                // Compute affinities with current sigma
                T sumAff = NumOps.Zero;
                T twoSigmaSq = NumOps.Multiply(two, NumOps.Multiply(sigma, sigma));
                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        affinities[i, j] = NumOps.Exp(NumOps.Negate(NumOps.Divide(sqDistances[i, j], twoSigmaSq)));
                        sumAff = NumOps.Add(sumAff, affinities[i, j]);
                    }
                }

                // Normalize and compute entropy
                T entropy = NumOps.Zero;
                if (NumOps.GreaterThan(sumAff, eps10))
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (i != j)
                        {
                            T p = NumOps.Divide(affinities[i, j], sumAff);
                            if (NumOps.GreaterThan(p, eps10))
                            {
                                entropy = NumOps.Subtract(entropy, NumOps.Multiply(p, NumOps.Log(p)));
                            }
                        }
                    }
                }

                // Binary search update
                if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(entropy, targetEntropy)), eps5)) break;

                if (NumOps.LessThan(entropy, targetEntropy))
                {
                    sigmaMin = sigma;
                    sigma = NumOps.Equals(sigmaMax, sigmaMaxInit)
                        ? NumOps.Multiply(sigma, two)
                        : NumOps.Divide(NumOps.Add(sigma, sigmaMax), two);
                }
                else
                {
                    sigmaMax = sigma;
                    sigma = NumOps.Divide(NumOps.Add(sigma, sigmaMin), two);
                }
            }

            // Normalize affinities
            T sum = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                if (i != j)
                {
                    sum = NumOps.Add(sum, affinities[i, j]);
                }
            }

            if (NumOps.GreaterThan(sum, eps10))
            {
                for (int j = 0; j < n; j++)
                {
                    if (i != j)
                    {
                        affinities[i, j] = NumOps.Divide(affinities[i, j], sum);
                    }
                }
            }
        }

        // Compute binding probability for each point
        // p_j = product over i of (1 - a_ij)
        // = probability that j is NOT selected by ANY point
        var bindingProbs = new Vector<T>(n);

        for (int j = 0; j < n; j++)
        {
            T logProd = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                if (i != j)
                {
                    logProd = NumOps.Add(logProd,
                        NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, affinities[i, j]), eps10)));
                }
            }
            // Outlier score = probability of NOT being selected
            bindingProbs[j] = NumOps.Exp(logProd);
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
        int nTrain = trainingData.Rows;
        int nTest = X.Rows;
        int nTotal = nTrain + nTest;
        int d = X.Columns;

        if (trainingData.Columns != d)
        {
            throw new ArgumentException($"Training data has {trainingData.Columns} features but test data has {d}.", nameof(X));
        }

        // Combine training and test data into a single Matrix<T>
        var allData = new Matrix<T>(nTotal, d);
        for (int i = 0; i < nTrain; i++)
        {
            for (int j = 0; j < d; j++)
            {
                allData[i, j] = trainingData[i, j];
            }
        }

        for (int i = 0; i < nTest; i++)
        {
            for (int j = 0; j < d; j++)
            {
                allData[nTrain + i, j] = X[i, j];
            }
        }

        // Compute binding probabilities for all data
        var allBindingProbs = ComputeBindingProbabilities(allData);

        // Extract scores for test points
        var scores = new Vector<T>(nTest);
        for (int i = 0; i < nTest; i++)
        {
            scores[i] = allBindingProbs[nTrain + i];
        }

        return scores;
    }
}
