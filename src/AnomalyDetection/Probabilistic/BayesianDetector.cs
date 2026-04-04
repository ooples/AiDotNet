using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
public class BayesianDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _priorStrength;
    private Vector<T>? _posteriorMean;
    private Matrix<T>? _posteriorCovariance;
    private Matrix<T>? _posteriorPrecision;
    private int _nFeatures;

    private T _logNormalization;


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
        _logNormalization = NumOps.Zero;
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
        T nT = NumOps.FromDouble(n);

        // Compute sample mean
        var sampleMean = new Vector<T>(_nFeatures);
        for (int j = 0; j < _nFeatures; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, X[i, j]);
            }
            sampleMean[j] = NumOps.Divide(sum, nT);
        }

        // Compute sample covariance
        var sampleCov = new Matrix<T>(_nFeatures, _nFeatures);
        for (int i = 0; i < n; i++)
        {
            for (int j1 = 0; j1 < _nFeatures; j1++)
            {
                T diff1 = NumOps.Subtract(X[i, j1], sampleMean[j1]);
                for (int j2 = 0; j2 < _nFeatures; j2++)
                {
                    T diff2 = NumOps.Subtract(X[i, j2], sampleMean[j2]);
                    sampleCov[j1, j2] = NumOps.Add(sampleCov[j1, j2], NumOps.Multiply(diff1, diff2));
                }
            }
        }

        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                sampleCov[j1, j2] = NumOps.Divide(sampleCov[j1, j2], nT);
            }
        }

        // Prior parameters (vague/weakly informative)
        T kappa0 = NumOps.FromDouble(_priorStrength);
        T nu0 = NumOps.FromDouble(_nFeatures + 2); // Prior degrees of freedom

        // Posterior parameters (Normal-Inverse-Wishart)
        T kappaN = NumOps.Add(kappa0, nT);
        T nuN = NumOps.Add(nu0, nT);

        // Posterior mean
        _posteriorMean = new Vector<T>(_nFeatures);
        for (int j = 0; j < _nFeatures; j++)
        {
            _posteriorMean[j] = NumOps.Divide(
                NumOps.Add(NumOps.Multiply(kappa0, sampleMean[j]), NumOps.Multiply(nT, sampleMean[j])),
                kappaN);
        }

        // Posterior covariance (for predictive distribution)
        _posteriorCovariance = new Matrix<T>(_nFeatures, _nFeatures);
        T scale = NumOps.Divide(
            NumOps.Add(kappaN, NumOps.One),
            NumOps.Multiply(kappaN, NumOps.Subtract(nuN, NumOps.FromDouble(_nFeatures - 1))));
        T regularization = NumOps.FromDouble(1e-6);

        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                _posteriorCovariance[j1, j2] = NumOps.Multiply(scale, sampleCov[j1, j2]);
                if (j1 == j2)
                {
                    _posteriorCovariance[j1, j2] = NumOps.Add(_posteriorCovariance[j1, j2], regularization);
                }
            }
        }

        // Compute precision (inverse covariance)
        _posteriorPrecision = InvertMatrix(_posteriorCovariance);

        // Compute log normalization for MVN
        T logDet = LogDeterminant(_posteriorCovariance);
        T half = NumOps.FromDouble(0.5);
        _logNormalization = NumOps.Negate(NumOps.Multiply(half,
            NumOps.Add(NumOps.FromDouble(_nFeatures * Math.Log(2 * Math.PI)), logDet)));

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var result = new Matrix<T>(n, n);
        var temp = new Matrix<T>(n, n);
        T eps = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                result[i, j] = (i == j) ? NumOps.One : NumOps.Zero;
            }
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(temp[row, col]), NumOps.Abs(temp[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            if (maxRow != col)
            {
                for (int j = 0; j < n; j++)
                {
                    T tmpVal = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmpVal;

                    tmpVal = result[col, j];
                    result[col, j] = result[maxRow, j];
                    result[maxRow, j] = tmpVal;
                }
            }

            T pivot = temp[col, col];
            if (NumOps.LessThan(NumOps.Abs(pivot), eps))
            {
                pivot = eps;
            }

            for (int j = 0; j < n; j++)
            {
                temp[col, j] = NumOps.Divide(temp[col, j], pivot);
                result[col, j] = NumOps.Divide(result[col, j], pivot);
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = temp[row, col];
                    for (int j = 0; j < n; j++)
                    {
                        temp[row, j] = NumOps.Subtract(temp[row, j], NumOps.Multiply(factor, temp[col, j]));
                        result[row, j] = NumOps.Subtract(result[row, j], NumOps.Multiply(factor, result[col, j]));
                    }
                }
            }
        }

        return result;
    }

    private T LogDeterminant(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var temp = new Matrix<T>(n, n);
        T eps = NumOps.FromDouble(1e-10);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
            }
        }

        T logDet = NumOps.Zero;

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(temp[row, col]), NumOps.Abs(temp[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            if (maxRow != col)
            {
                for (int j = col; j < n; j++)
                {
                    T tmpVal = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmpVal;
                }
            }

            if (NumOps.LessThan(NumOps.Abs(temp[col, col]), eps))
            {
                return NumOps.FromDouble(-1e30); // Approximate negative infinity
            }

            logDet = NumOps.Add(logDet, NumOps.Log(NumOps.Abs(temp[col, col])));

            for (int row = col + 1; row < n; row++)
            {
                T factor = NumOps.Divide(temp[row, col], temp[col, col]);
                for (int j = col + 1; j < n; j++)
                {
                    temp[row, j] = NumOps.Subtract(temp[row, j], NumOps.Multiply(factor, temp[col, j]));
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

        var posteriorMean = _posteriorMean ?? throw new InvalidOperationException("_posteriorMean has not been initialized.");
        var posteriorPrecision = _posteriorPrecision ?? throw new InvalidOperationException("_posteriorPrecision has not been initialized.");

        var scores = new Vector<T>(X.Rows);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute Mahalanobis distance squared
            T mahalSq = NumOps.Zero;
            for (int j1 = 0; j1 < _nFeatures; j1++)
            {
                T diff1 = NumOps.Subtract(X[i, j1], posteriorMean[j1]);
                for (int j2 = 0; j2 < _nFeatures; j2++)
                {
                    T diff2 = NumOps.Subtract(X[i, j2], posteriorMean[j2]);
                    mahalSq = NumOps.Add(mahalSq, NumOps.Multiply(NumOps.Multiply(diff1, posteriorPrecision[j1, j2]), diff2));
                }
            }

            // Negative log likelihood (higher = more anomalous)
            T negLogLikelihood = NumOps.Negate(NumOps.Subtract(_logNormalization, NumOps.Multiply(half, mahalSq)));

            scores[i] = negLogLikelihood;
        }

        return scores;
    }
}
