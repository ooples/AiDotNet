using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using One-Class SVM with simplified SGD training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> One-Class SVM finds a boundary that encompasses the normal data.
/// It learns the shape of normal data in a high-dimensional kernel space and flags points
/// outside this region as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Map data to kernel space using RBF kernel
/// 2. Find hyperplane that separates data from origin with maximum margin
/// 3. Points on the negative side of the hyperplane are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have only normal examples
/// - High-dimensional data
/// - When you need a decision boundary
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Nu: 0.1 (roughly proportion of outliers)
/// - Gamma: auto (1/n_features)
/// - Kernel: RBF
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Estimating the Support of a High-Dimensional Distribution", "https://doi.org/10.1162/089976601750264965", Year = 2001, Authors = "Bernhard Scholkopf, John C. Platt, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson")]
public class OCSVMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _nu;
    private readonly double _gamma;
    private readonly int _maxIterations;
    private Matrix<T>? _supportVectors;
    private Vector<T>? _alphas;

    private T _rho;


    /// <summary>
    /// Gets the nu parameter.
    /// </summary>
    public double Nu => _nu;

    /// <summary>
    /// Gets the gamma parameter.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Creates a new OCSVM anomaly detector.
    /// </summary>
    /// <param name="nu">Upper bound on outlier fraction. Default is 0.1.</param>
    /// <param name="gamma">RBF kernel parameter. -1 means auto (1/n_features). Default is -1.</param>
    /// <param name="maxIterations">Maximum training iterations. Default is 1000.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public OCSVMDetector(double nu = 0.1, double gamma = -1, int maxIterations = 1000,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _rho = NumOps.Zero;
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nu),
                "Nu must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.1.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _nu = nu;
        _gamma = gamma;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int d = X.Columns;
        T effectiveGamma = _gamma > 0
            ? NumOps.FromDouble(_gamma)
            : NumOps.Divide(NumOps.One, NumOps.FromDouble(d));

        // Train using simplified SMO-like algorithm
        TrainOCSVM(X, effectiveGamma);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void TrainOCSVM(Matrix<T> data, T gamma)
    {
        int n = data.Rows;

        // Initialize alphas
        var alphas = new Vector<T>(n);
        T maxAlpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(_nu * n));
        T initAlpha = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));

        for (int i = 0; i < n; i++)
        {
            alphas[i] = initAlpha;
        }

        // Compute kernel matrix (cache for efficiency)
        var K = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var pointI = new Vector<T>(data.GetRowReadOnlySpan(i).ToArray());
            for (int j = i; j < n; j++)
            {
                var pointJ = new Vector<T>(data.GetRowReadOnlySpan(j).ToArray());
                T kval = RBFKernel(pointI, pointJ, gamma);
                K[i, j] = kval;
                K[j, i] = kval;
            }
        }

        T stepSize = NumOps.FromDouble(0.1);
        T eps8 = NumOps.FromDouble(1e-8);
        T eps10 = NumOps.FromDouble(1e-10);

        // SMO-like optimization
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                // Compute decision function value
                T fi = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    fi = NumOps.Add(fi, NumOps.Multiply(alphas[k], K[k, i]));
                }

                // If violates KKT, update
                bool violates = (NumOps.LessThan(alphas[i], maxAlpha) && NumOps.LessThan(fi, NumOps.Zero))
                    || (NumOps.GreaterThan(alphas[i], NumOps.Zero) && NumOps.GreaterThan(fi, NumOps.Zero));

                if (violates)
                {
                    // Select random j != i
                    int j = _random.Next(n);
                    while (j == i) j = _random.Next(n);

                    T fj = NumOps.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        fj = NumOps.Add(fj, NumOps.Multiply(alphas[k], K[k, j]));
                    }

                    // Compute step
                    T eta = NumOps.Subtract(NumOps.Add(K[i, i], K[j, j]), NumOps.Multiply(NumOps.FromDouble(2), K[i, j]));
                    if (NumOps.GreaterThan(eta, eps10))
                    {
                        T oldAlphaI = alphas[i];

                        // Update (simplified)
                        T delta = NumOps.Divide(NumOps.Subtract(fi, fj), eta);
                        alphas[i] = NumOps.Subtract(alphas[i], NumOps.Multiply(delta, stepSize));
                        alphas[j] = NumOps.Add(alphas[j], NumOps.Multiply(delta, stepSize));

                        // Clip
                        if (NumOps.LessThan(alphas[i], NumOps.Zero)) alphas[i] = NumOps.Zero;
                        if (NumOps.GreaterThan(alphas[i], maxAlpha)) alphas[i] = maxAlpha;
                        if (NumOps.LessThan(alphas[j], NumOps.Zero)) alphas[j] = NumOps.Zero;
                        if (NumOps.GreaterThan(alphas[j], maxAlpha)) alphas[j] = maxAlpha;

                        if (NumOps.GreaterThan(NumOps.Abs(NumOps.Subtract(alphas[i], oldAlphaI)), eps8))
                        {
                            changed = true;
                        }
                    }
                }
            }

            if (!changed) break;
        }

        // Extract support vectors
        var svIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.GreaterThan(alphas[i], eps8))
            {
                svIndices.Add(i);
            }
        }

        int svCount = svIndices.Count;
        int d = data.Columns;
        _supportVectors = new Matrix<T>(svCount, d);
        _alphas = new Vector<T>(svCount);
        for (int si = 0; si < svCount; si++)
        {
            _alphas[si] = alphas[svIndices[si]];
            for (int j = 0; j < d; j++)
            {
                _supportVectors[si, j] = data[svIndices[si], j];
            }
        }

        // Compute rho (threshold)
        _rho = NumOps.Zero;
        int count = 0;
        T upperBound = NumOps.Subtract(maxAlpha, eps8);
        for (int i = 0; i < svCount; i++)
        {
            if (NumOps.GreaterThan(_alphas[i], eps8) && NumOps.LessThan(_alphas[i], upperBound))
            {
                T f = NumOps.Zero;
                for (int j = 0; j < svCount; j++)
                {
                    var svI = new Vector<T>(_supportVectors.GetRowReadOnlySpan(i).ToArray());
                    var svJ = new Vector<T>(_supportVectors.GetRowReadOnlySpan(j).ToArray());
                    f = NumOps.Add(f, NumOps.Multiply(_alphas[j], RBFKernel(svJ, svI, gamma)));
                }
                _rho = NumOps.Add(_rho, f);
                count++;
            }
        }

        if (count > 0) _rho = NumOps.Divide(_rho, NumOps.FromDouble(count));
    }

    private T RBFKernel(Vector<T> a, Vector<T> b, T gamma)
    {
        var diff = Engine.Subtract(a, b);
        T sqDist = Engine.DotProduct(diff, diff);
        return NumOps.Exp(NumOps.Negate(NumOps.Multiply(gamma, sqDist)));
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

        var supportVectors = _supportVectors;
        var alphas = _alphas;

        if (supportVectors == null || alphas == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        T effectiveGamma = _gamma > 0
            ? NumOps.FromDouble(_gamma)
            : NumOps.Divide(NumOps.One, NumOps.FromDouble(X.Columns));
        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Decision function: sum(alpha_i * K(x_i, x)) - rho
            T f = NumOps.Zero;
            for (int k = 0; k < supportVectors.Rows; k++)
            {
                var sv = new Vector<T>(supportVectors.GetRowReadOnlySpan(k).ToArray());
                f = NumOps.Add(f, NumOps.Multiply(alphas[k], RBFKernel(sv, point, effectiveGamma)));
            }
            f = NumOps.Subtract(f, _rho);

            // Higher (more positive) = more normal
            // We want higher score = more anomalous, so negate
            scores[i] = NumOps.Negate(f);
        }

        return scores;
    }
}
