using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Bayesian probability estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector uses Bayesian probability to model normal data distribution.
/// It assumes data follows a multivariate Gaussian distribution with prior beliefs about
/// the parameters. Points with low likelihood under this model are considered anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Estimate mean and covariance with Bayesian priors
/// 2. Compute posterior predictive probability for each point
/// 3. Low probability points are flagged as anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have prior knowledge about the data distribution
/// - For probabilistic anomaly scoring
/// - When you want uncertainty estimates
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Prior strength (kappa0): 0.01 (weak prior)
/// - Prior degrees of freedom: n_features + 2
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective."
/// Chapter on Bayesian inference for MVN.
/// </para>
/// </remarks>
public class BayesianDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _priorStrength;
    private double[]? _posteriorMean;
    private double[,]? _posteriorCovariance;
    private double[,]? _posteriorPrecision;
    private int _nFeatures;
    private double _logNormalization;

    /// <summary>
    /// Gets the prior strength parameter.
    /// </summary>
    public double PriorStrength => _priorStrength;

    /// <summary>
    /// Creates a new Bayesian anomaly detector.
    /// </summary>
    /// <param name="priorStrength">
    /// Strength of the prior (kappa_0). Smaller values = weaker prior. Default is 0.01.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public BayesianDetector(double priorStrength = 0.01, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (priorStrength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(priorStrength),
                "PriorStrength must be positive. Recommended is 0.01.");
        }

        _priorStrength = priorStrength;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_nFeatures];
            for (int j = 0; j < _nFeatures; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Compute sample mean
        var sampleMean = new double[_nFeatures];
        for (int j = 0; j < _nFeatures; j++)
        {
            for (int i = 0; i < n; i++)
            {
                sampleMean[j] += data[i][j];
            }
            sampleMean[j] /= n;
        }

        // Compute sample covariance
        var sampleCov = new double[_nFeatures, _nFeatures];
        for (int i = 0; i < n; i++)
        {
            for (int j1 = 0; j1 < _nFeatures; j1++)
            {
                for (int j2 = 0; j2 < _nFeatures; j2++)
                {
                    sampleCov[j1, j2] += (data[i][j1] - sampleMean[j1]) * (data[i][j2] - sampleMean[j2]);
                }
            }
        }

        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                sampleCov[j1, j2] /= n;
            }
        }

        // Prior parameters (vague/weakly informative)
        double kappa0 = _priorStrength;
        double nu0 = _nFeatures + 2; // Prior degrees of freedom
        var mu0 = sampleMean; // Prior mean = sample mean (data-driven)
        var Lambda0 = new double[_nFeatures, _nFeatures]; // Prior precision (identity)
        for (int j = 0; j < _nFeatures; j++)
        {
            Lambda0[j, j] = 1.0;
        }

        // Posterior parameters (Normal-Inverse-Wishart)
        double kappaN = kappa0 + n;
        double nuN = nu0 + n;

        // Posterior mean
        _posteriorMean = new double[_nFeatures];
        for (int j = 0; j < _nFeatures; j++)
        {
            _posteriorMean[j] = (kappa0 * mu0[j] + n * sampleMean[j]) / kappaN;
        }

        // Posterior covariance (for predictive distribution)
        _posteriorCovariance = new double[_nFeatures, _nFeatures];
        double scale = (kappaN + 1) / (kappaN * (nuN - _nFeatures + 1));

        // Simplified: use regularized sample covariance
        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                _posteriorCovariance[j1, j2] = scale * sampleCov[j1, j2];
                if (j1 == j2)
                {
                    _posteriorCovariance[j1, j2] += 1e-6; // Regularization
                }
            }
        }

        // Compute precision (inverse covariance)
        _posteriorPrecision = InvertMatrix(_posteriorCovariance);

        // Compute log normalization for MVN
        double logDet = LogDeterminant(_posteriorCovariance);
        _logNormalization = -0.5 * (_nFeatures * Math.Log(2 * Math.PI) + logDet);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[,] InvertMatrix(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var result = new double[n, n];
        var temp = new double[n, n];

        // Copy matrix and create identity
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                result[i, j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(temp[row, col]) > Math.Abs(temp[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < n; j++)
                {
                    double tmpVal = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmpVal;

                    tmpVal = result[col, j];
                    result[col, j] = result[maxRow, j];
                    result[maxRow, j] = tmpVal;
                }
            }

            // Scale pivot row
            double pivot = temp[col, col];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int j = 0; j < n; j++)
            {
                temp[col, j] /= pivot;
                result[col, j] /= pivot;
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = temp[row, col];
                    for (int j = 0; j < n; j++)
                    {
                        temp[row, j] -= factor * temp[col, j];
                        result[row, j] -= factor * result[col, j];
                    }
                }
            }
        }

        return result;
    }

    private double LogDeterminant(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var temp = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
            }
        }

        double logDet = 0;
        int sign = 1;

        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(temp[row, col]) > Math.Abs(temp[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            if (maxRow != col)
            {
                sign *= -1;
                for (int j = col; j < n; j++)
                {
                    double tmpVal = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmpVal;
                }
            }

            if (Math.Abs(temp[col, col]) < 1e-10)
            {
                return double.NegativeInfinity;
            }

            logDet += Math.Log(Math.Abs(temp[col, col]));

            for (int row = col + 1; row < n; row++)
            {
                double factor = temp[row, col] / temp[col, col];
                for (int j = col + 1; j < n; j++)
                {
                    temp[row, j] -= factor * temp[col, j];
                }
            }
        }

        return logDet;
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[_nFeatures];
            for (int j = 0; j < _nFeatures; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Compute Mahalanobis distance squared
            double mahalSq = 0;
            for (int j1 = 0; j1 < _nFeatures; j1++)
            {
                double diff1 = point[j1] - _posteriorMean![j1];
                for (int j2 = 0; j2 < _nFeatures; j2++)
                {
                    double diff2 = point[j2] - _posteriorMean[j2];
                    mahalSq += diff1 * _posteriorPrecision![j1, j2] * diff2;
                }
            }

            // Negative log likelihood (higher = more anomalous)
            double negLogLikelihood = -(_logNormalization - 0.5 * mahalSq);

            scores[i] = NumOps.FromDouble(negLogLikelihood);
        }

        return scores;
    }
}
