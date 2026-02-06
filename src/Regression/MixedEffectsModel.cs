using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Mixed-Effects (Hierarchical/Multilevel) Linear Model for clustered and hierarchical data.
/// </summary>
/// <remarks>
/// <para>
/// Mixed-effects models combine fixed effects (population-level patterns) with random effects
/// (group-level variations). They properly account for the correlation structure in hierarchical
/// data, providing correct inference when observations are not independent.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when your data has groups or clusters:
///
/// <b>Model structure:</b>
/// y_ij = X_ij * β + Z_ij * u_j + ε_ij
///
/// Where:
/// - y_ij: Outcome for observation i in group j
/// - X_ij * β: Fixed effects (same for everyone)
/// - Z_ij * u_j: Random effects (vary by group)
/// - ε_ij: Residual error
///
/// <b>Example - Student test scores:</b>
/// score = study_hours * β + (school_intercept + study_hours * school_slope) + error
///
/// Fixed effects tell you: "On average, each study hour adds β points"
/// Random effects tell you: "School 1 starts 5 points higher, School 2 has steeper study benefit"
///
/// <b>Key outputs:</b>
/// - Fixed effect coefficients and their standard errors
/// - Random effect predictions (BLUPs) for each group
/// - Variance components showing how much groups vary
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MixedEffectsModel<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Fixed effect coefficients.
    /// </summary>
    private T[]? _fixedEffects;

    /// <summary>
    /// Random effect predictions (BLUPs) for each group.
    /// </summary>
    private Dictionary<int, T[]>? _randomEffects;

    /// <summary>
    /// Variance of random effects.
    /// </summary>
    private T[,]? _randomEffectVariance;

    /// <summary>
    /// Residual variance.
    /// </summary>
    private T _residualVariance;

    /// <summary>
    /// Standard errors of fixed effects.
    /// </summary>
    private T[]? _fixedEffectStdErrors;

    /// <summary>
    /// Group indices for training data.
    /// </summary>
    private int[]? _groupIndices;

    /// <summary>
    /// Feature means for centering.
    /// </summary>
    private T[]? _featureMeans;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Number of random effect dimensions.
    /// </summary>
    private int _numRandomEffects;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly MixedEffectsModelOptions _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of Mixed-Effects Model.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public MixedEffectsModel(MixedEffectsModelOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(null, regularization)
    {
        _options = options ?? new MixedEffectsModelOptions();
        _residualVariance = NumOps.One;
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the mixed-effects model with explicit group indicators.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Target values.</param>
    /// <param name="groupIndices">Group indicator for each observation.</param>
    public void Train(Matrix<T> x, Vector<T> y, int[] groupIndices)
    {
        _numFeatures = x.Columns;
        _groupIndices = groupIndices;

        // Center features if requested
        if (_options.CenterFeatures)
        {
            x = CenterFeatures(x);
        }

        // Determine number of random effects
        _numRandomEffects = _options.IncludeRandomIntercept ? 1 : 0;
        if (_options.IncludeRandomSlopes)
        {
            var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
            _numRandomEffects += slopeFeatures.Length;
        }

        // Convert to double arrays
        var xData = ConvertToDoubleArray(x);
        var yData = y.Select(yi => NumOps.ToDouble(yi)).ToArray();

        // Get unique groups
        var uniqueGroups = groupIndices.Distinct().OrderBy(g => g).ToArray();
        int numGroups = uniqueGroups.Length;

        // Initialize parameters
        var beta = new double[_numFeatures + 1];  // +1 for intercept
        var sigma2 = 1.0;  // Residual variance
        var D = InitializeRandomEffectVariance(_numRandomEffects);  // Random effect variance

        // EM-style optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: Compute conditional modes of random effects (BLUPs)
            var u = ComputeBLUPs(xData, yData, groupIndices, uniqueGroups, beta, sigma2, D);

            // M-step: Update fixed effects and variance components
            var oldBeta = (double[])beta.Clone();
            var oldSigma2 = sigma2;
            var oldD = (double[,])D.Clone();

            // Update fixed effects
            beta = UpdateFixedEffects(xData, yData, groupIndices, u, sigma2, D);

            // Update variance components
            (sigma2, D) = UpdateVarianceComponents(xData, yData, groupIndices, uniqueGroups, beta, u);

            // Check convergence
            double betaChange = 0;
            for (int i = 0; i < beta.Length; i++)
            {
                betaChange += Math.Abs(beta[i] - oldBeta[i]);
            }

            if (betaChange < _options.Tolerance && Math.Abs(sigma2 - oldSigma2) < _options.Tolerance)
            {
                break;
            }
        }

        // Compute final random effects
        var finalU = ComputeBLUPs(xData, yData, groupIndices, uniqueGroups, beta, sigma2, D);

        // Store results
        _fixedEffects = beta.Select(b => NumOps.FromDouble(b)).ToArray();
        _residualVariance = NumOps.FromDouble(sigma2);
        _randomEffectVariance = ConvertToTMatrix2D(D);

        _randomEffects = new Dictionary<int, T[]>();
        foreach (var group in uniqueGroups)
        {
            if (finalU.TryGetValue(group, out var uGroup))
            {
                _randomEffects[group] = uGroup.Select(ui => NumOps.FromDouble(ui)).ToArray();
            }
        }

        // Compute standard errors
        _fixedEffectStdErrors = ComputeStandardErrors(xData, yData, groupIndices, uniqueGroups, beta, sigma2, D);
    }

    /// <inheritdoc/>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Without explicit groups, treat each observation as its own group (no random effects)
        // Or use a simple clustering heuristic
        var groupIndices = Enumerable.Range(0, x.Rows).Select(i => i % Math.Max(1, x.Rows / 10)).ToArray();
        Train(x, y, groupIndices);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        return Predict(input, null);
    }

    /// <summary>
    /// Predicts values for input samples with known group memberships.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="groupIndices">Group indicator for each observation (null for population-level prediction).</param>
    /// <returns>Vector of predictions.</returns>
    public Vector<T> Predict(Matrix<T> input, int[]? groupIndices)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        var result = new Vector<T>(input.Rows);

        // Center features if we centered during training
        if (_options.CenterFeatures && _featureMeans != null)
        {
            input = CenterFeaturesForPrediction(input);
        }

        for (int i = 0; i < input.Rows; i++)
        {
            // Fixed effects prediction
            double pred = NumOps.ToDouble(_fixedEffects[0]);  // Intercept
            for (int j = 0; j < _numFeatures; j++)
            {
                pred += NumOps.ToDouble(input[i, j]) * NumOps.ToDouble(_fixedEffects[j + 1]);
            }

            // Add random effects if group is known
            if (groupIndices != null && _randomEffects != null)
            {
                int group = groupIndices[i];
                if (_randomEffects.TryGetValue(group, out var u))
                {
                    pred += GetRandomEffectContribution(input, i, u);
                }
            }

            result[i] = NumOps.FromDouble(pred);
        }

        return result;
    }

    /// <summary>
    /// Gets the fixed effect coefficients.
    /// </summary>
    /// <returns>Array of fixed effect coefficients (intercept first).</returns>
    public T[] GetFixedEffects()
    {
        return _fixedEffects ?? Array.Empty<T>();
    }

    /// <summary>
    /// Gets the standard errors of fixed effects.
    /// </summary>
    /// <returns>Array of standard errors.</returns>
    public T[] GetFixedEffectStandardErrors()
    {
        return _fixedEffectStdErrors ?? Array.Empty<T>();
    }

    /// <summary>
    /// Gets the random effects (BLUPs) for each group.
    /// </summary>
    /// <returns>Dictionary mapping group ID to random effect values.</returns>
    public Dictionary<int, T[]> GetRandomEffects()
    {
        return _randomEffects ?? new Dictionary<int, T[]>();
    }

    /// <summary>
    /// Gets the residual variance.
    /// </summary>
    public T ResidualVariance => _residualVariance;

    /// <summary>
    /// Gets the random effect variance-covariance matrix.
    /// </summary>
    /// <returns>Variance-covariance matrix of random effects.</returns>
    public T[,]? GetRandomEffectVariance()
    {
        return _randomEffectVariance;
    }

    /// <summary>
    /// Computes the intraclass correlation coefficient (ICC).
    /// </summary>
    /// <returns>ICC value between 0 and 1.</returns>
    /// <remarks>
    /// ICC measures what proportion of total variance is attributable to between-group differences.
    /// Higher ICC means groups are more different from each other.
    /// </remarks>
    public double ComputeICC()
    {
        if (_randomEffectVariance == null)
        {
            return 0;
        }

        // For simple random intercept model: ICC = var(intercept) / (var(intercept) + var(residual))
        double interceptVar = NumOps.ToDouble(_randomEffectVariance[0, 0]);
        double residVar = NumOps.ToDouble(_residualVariance);

        return interceptVar / (interceptVar + residVar);
    }

    /// <summary>
    /// Gets the log-likelihood of the model.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Target values.</param>
    /// <param name="groupIndices">Group indicators.</param>
    /// <returns>Log-likelihood value.</returns>
    public double GetLogLikelihood(Matrix<T> x, Vector<T> y, int[] groupIndices)
    {
        if (_fixedEffects == null)
        {
            return double.NegativeInfinity;
        }

        var predictions = Predict(x, groupIndices);
        double logLik = 0;
        double sigma2 = NumOps.ToDouble(_residualVariance);

        for (int i = 0; i < y.Length; i++)
        {
            double residual = NumOps.ToDouble(y[i]) - NumOps.ToDouble(predictions[i]);
            logLik -= 0.5 * (Math.Log(2 * Math.PI * sigma2) + residual * residual / sigma2);
        }

        return logLik;
    }

    /// <summary>
    /// Centers features and stores means.
    /// </summary>
    private Matrix<T> CenterFeatures(Matrix<T> x)
    {
        _featureMeans = new T[x.Columns];
        var centered = new Matrix<T>(x.Rows, x.Columns);

        for (int j = 0; j < x.Columns; j++)
        {
            double sum = 0;
            for (int i = 0; i < x.Rows; i++)
            {
                sum += NumOps.ToDouble(x[i, j]);
            }
            double mean = sum / x.Rows;
            _featureMeans[j] = NumOps.FromDouble(mean);

            for (int i = 0; i < x.Rows; i++)
            {
                centered[i, j] = NumOps.FromDouble(NumOps.ToDouble(x[i, j]) - mean);
            }
        }

        return centered;
    }

    /// <summary>
    /// Centers features using stored means.
    /// </summary>
    private Matrix<T> CenterFeaturesForPrediction(Matrix<T> x)
    {
        var centered = new Matrix<T>(x.Rows, x.Columns);

        for (int j = 0; j < x.Columns; j++)
        {
            double mean = NumOps.ToDouble(_featureMeans![j]);
            for (int i = 0; i < x.Rows; i++)
            {
                centered[i, j] = NumOps.FromDouble(NumOps.ToDouble(x[i, j]) - mean);
            }
        }

        return centered;
    }

    /// <summary>
    /// Initializes the random effect variance matrix.
    /// </summary>
    private double[,] InitializeRandomEffectVariance(int dim)
    {
        var D = new double[dim, dim];

        return _options.CovarianceStructure switch
        {
            MixedEffectsCovarianceStructure.Identity => InitializeIdentityCovariance(D, dim),
            MixedEffectsCovarianceStructure.Diagonal => InitializeDiagonalCovariance(D, dim),
            MixedEffectsCovarianceStructure.CompoundSymmetry => InitializeCompoundSymmetryCovariance(D, dim),
            _ => InitializeUnstructuredCovariance(D, dim)
        };
    }

    private double[,] InitializeIdentityCovariance(double[,] D, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = 1.0;
        }
        return D;
    }

    private double[,] InitializeDiagonalCovariance(double[,] D, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = 1.0;
        }
        return D;
    }

    private double[,] InitializeCompoundSymmetryCovariance(double[,] D, int dim)
    {
        double rho = 0.5;
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                D[i, j] = i == j ? 1.0 : rho;
            }
        }
        return D;
    }

    private double[,] InitializeUnstructuredCovariance(double[,] D, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = 1.0;
            for (int j = 0; j < i; j++)
            {
                D[i, j] = D[j, i] = 0.1;
            }
        }
        return D;
    }

    /// <summary>
    /// Computes Best Linear Unbiased Predictors (BLUPs) for random effects.
    /// </summary>
    private Dictionary<int, double[]> ComputeBLUPs(
        double[][] xData, double[] yData, int[] groupIndices,
        int[] uniqueGroups, double[] beta, double sigma2, double[,] D)
    {
        var blups = new Dictionary<int, double[]>();

        foreach (var group in uniqueGroups)
        {
            var groupObs = GetGroupObservations(groupIndices, group);

            if (groupObs.Length < _options.MinObservationsPerGroup)
            {
                blups[group] = new double[_numRandomEffects];
                continue;
            }

            // Design matrix for random effects (Z)
            var Z = BuildRandomEffectDesignMatrix(xData, groupObs);

            // Compute residuals
            var residuals = new double[groupObs.Length];
            for (int i = 0; i < groupObs.Length; i++)
            {
                int idx = groupObs[i];
                residuals[i] = yData[idx] - ComputeFixedPrediction(xData[idx], beta);
            }

            // BLUP = D * Z' * V^(-1) * residuals
            // Simplified: D * Z' * (Z * D * Z' + sigma2 * I)^(-1) * residuals
            var u = ComputeGroupBLUP(Z, residuals, D, sigma2);
            blups[group] = u;
        }

        return blups;
    }

    /// <summary>
    /// Computes BLUP for a single group.
    /// </summary>
    private double[] ComputeGroupBLUP(double[][] Z, double[] residuals, double[,] D, double sigma2)
    {
        int n = residuals.Length;
        int q = _numRandomEffects;

        // V = Z * D * Z' + sigma2 * I
        var V = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < q; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        sum += Z[i][k] * D[k, l] * Z[j][l];
                    }
                }
                V[i, j] = sum + (i == j ? sigma2 : 0);
            }
        }

        // Invert V (simplified - use Cholesky or iterative for large matrices)
        var Vinv = InvertMatrix(V);

        // D * Z' * V^(-1) * residuals
        var u = new double[q];
        for (int k = 0; k < q; k++)
        {
            for (int i = 0; i < n; i++)
            {
                double vInvRes = 0;
                for (int j = 0; j < n; j++)
                {
                    vInvRes += Vinv[i, j] * residuals[j];
                }

                for (int l = 0; l < q; l++)
                {
                    u[k] += D[k, l] * Z[i][l] * vInvRes;
                }
            }
        }

        return u;
    }

    /// <summary>
    /// Updates fixed effects using weighted least squares.
    /// </summary>
    private double[] UpdateFixedEffects(
        double[][] xData, double[] yData, int[] groupIndices,
        Dictionary<int, double[]> u, double sigma2, double[,] D)
    {
        int n = xData.Length;
        int p = _numFeatures + 1;  // +1 for intercept

        // X'X
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int i = 0; i < n; i++)
        {
            int group = groupIndices[i];
            double yi = yData[i];

            // Subtract random effect contribution
            if (u.TryGetValue(group, out var uGroup))
            {
                yi -= GetRandomEffectContribution(xData, i, uGroup);
            }

            // Intercept
            XtX[0, 0] += 1;
            Xty[0] += yi;

            for (int j = 0; j < _numFeatures; j++)
            {
                XtX[0, j + 1] += xData[i][j];
                XtX[j + 1, 0] += xData[i][j];
                Xty[j + 1] += xData[i][j] * yi;

                for (int k = 0; k < _numFeatures; k++)
                {
                    XtX[j + 1, k + 1] += xData[i][j] * xData[i][k];
                }
            }
        }

        // Solve (X'X)^(-1) * X'y
        var XtXinv = InvertMatrix(XtX);
        var beta = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < p; k++)
            {
                beta[j] += XtXinv[j, k] * Xty[k];
            }
        }

        return beta;
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private (double sigma2, double[,] D) UpdateVarianceComponents(
        double[][] xData, double[] yData, int[] groupIndices,
        int[] uniqueGroups, double[] beta, Dictionary<int, double[]> u)
    {
        int n = xData.Length;

        // Update residual variance
        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            int group = groupIndices[i];
            double pred = ComputeFixedPrediction(xData[i], beta);

            if (u.TryGetValue(group, out var uGroup))
            {
                pred += GetRandomEffectContribution(xData, i, uGroup);
            }

            double residual = yData[i] - pred;
            rss += residual * residual;
        }

        double sigma2;
        if (_options.OptimizationMethod == MixedEffectsOptimization.REML)
        {
            // REML adjustment
            sigma2 = rss / (n - _numFeatures - 1);
        }
        else
        {
            sigma2 = rss / n;
        }

        sigma2 = Math.Max(1e-10, sigma2);

        // Update random effect variance
        var D = new double[_numRandomEffects, _numRandomEffects];
        int numGroups = 0;

        foreach (var group in uniqueGroups)
        {
            if (u.TryGetValue(group, out var uGroup) && uGroup.Any(ui => Math.Abs(ui) > 1e-10))
            {
                numGroups++;
                for (int i = 0; i < _numRandomEffects; i++)
                {
                    for (int j = 0; j < _numRandomEffects; j++)
                    {
                        D[i, j] += uGroup[i] * uGroup[j];
                    }
                }
            }
        }

        if (numGroups > 0)
        {
            for (int i = 0; i < _numRandomEffects; i++)
            {
                for (int j = 0; j < _numRandomEffects; j++)
                {
                    D[i, j] /= numGroups;
                    if (i == j)
                    {
                        D[i, j] = Math.Max(1e-10, D[i, j]);
                    }
                }
            }
        }
        else
        {
            // Initialize with small positive values
            for (int i = 0; i < _numRandomEffects; i++)
            {
                D[i, i] = 0.1;
            }
        }

        return (sigma2, D);
    }

    /// <summary>
    /// Computes standard errors of fixed effects.
    /// </summary>
    private T[] ComputeStandardErrors(
        double[][] xData, double[] yData, int[] groupIndices,
        int[] uniqueGroups, double[] beta, double sigma2, double[,] D)
    {
        int n = xData.Length;
        int p = _numFeatures + 1;

        // X'X
        var XtX = new double[p, p];

        for (int i = 0; i < n; i++)
        {
            XtX[0, 0] += 1;
            for (int j = 0; j < _numFeatures; j++)
            {
                XtX[0, j + 1] += xData[i][j];
                XtX[j + 1, 0] += xData[i][j];
                for (int k = 0; k < _numFeatures; k++)
                {
                    XtX[j + 1, k + 1] += xData[i][j] * xData[i][k];
                }
            }
        }

        // Standard errors = sqrt(diag(sigma2 * (X'X)^(-1)))
        var XtXinv = InvertMatrix(XtX);
        var stdErrors = new T[p];

        for (int j = 0; j < p; j++)
        {
            stdErrors[j] = NumOps.FromDouble(Math.Sqrt(sigma2 * XtXinv[j, j]));
        }

        return stdErrors;
    }

    /// <summary>
    /// Builds the random effect design matrix for a group.
    /// </summary>
    private double[][] BuildRandomEffectDesignMatrix(double[][] xData, int[] groupObs)
    {
        var Z = new double[groupObs.Length][];

        for (int i = 0; i < groupObs.Length; i++)
        {
            Z[i] = new double[_numRandomEffects];
            int idx = groupObs[i];

            int col = 0;

            // Random intercept
            if (_options.IncludeRandomIntercept)
            {
                Z[i][col++] = 1.0;
            }

            // Random slopes
            if (_options.IncludeRandomSlopes)
            {
                var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
                foreach (int f in slopeFeatures)
                {
                    Z[i][col++] = xData[idx][f];
                }
            }
        }

        return Z;
    }

    /// <summary>
    /// Computes the fixed effects prediction.
    /// </summary>
    private double ComputeFixedPrediction(double[] x, double[] beta)
    {
        double pred = beta[0];  // Intercept
        for (int j = 0; j < _numFeatures; j++)
        {
            pred += x[j] * beta[j + 1];
        }
        return pred;
    }

    /// <summary>
    /// Gets the random effect contribution for an observation.
    /// </summary>
    private double GetRandomEffectContribution(double[][] xData, int obsIdx, double[] u)
    {
        double contrib = 0;
        int col = 0;

        if (_options.IncludeRandomIntercept)
        {
            contrib += u[col++];
        }

        if (_options.IncludeRandomSlopes)
        {
            var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
            foreach (int f in slopeFeatures)
            {
                contrib += u[col++] * xData[obsIdx][f];
            }
        }

        return contrib;
    }

    /// <summary>
    /// Gets the random effect contribution for a Matrix input row.
    /// </summary>
    private double GetRandomEffectContribution(Matrix<T> x, int obsIdx, T[] u)
    {
        double contrib = 0;
        int col = 0;

        if (_options.IncludeRandomIntercept)
        {
            contrib += NumOps.ToDouble(u[col++]);
        }

        if (_options.IncludeRandomSlopes)
        {
            var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
            foreach (int f in slopeFeatures)
            {
                contrib += NumOps.ToDouble(u[col++]) * NumOps.ToDouble(x[obsIdx, f]);
            }
        }

        return contrib;
    }

    /// <summary>
    /// Gets observation indices for a group.
    /// </summary>
    private int[] GetGroupObservations(int[] groupIndices, int group)
    {
        return Enumerable.Range(0, groupIndices.Length)
            .Where(i => groupIndices[i] == group)
            .ToArray();
    }

    /// <summary>
    /// Simple matrix inversion using Gaussian elimination.
    /// </summary>
    private double[,] InvertMatrix(double[,] A)
    {
        int n = A.GetLength(0);
        var augmented = new double[n, 2 * n];

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n + i] = 1;
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            // Make pivot 1
            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;  // Prevent division by zero
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] /= pivot;
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        // Extract inverse
        var inverse = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Converts double array to matrix type.
    /// </summary>
    private double[][] ConvertToDoubleArray(Matrix<T> x)
    {
        var result = new double[x.Rows][];
        for (int i = 0; i < x.Rows; i++)
        {
            result[i] = new double[x.Columns];
            for (int j = 0; j < x.Columns; j++)
            {
                result[i][j] = NumOps.ToDouble(x[i, j]);
            }
        }
        return result;
    }

    /// <summary>
    /// Converts double matrix to T matrix.
    /// </summary>
    private T[,] ConvertToTMatrix2D(double[,] d)
    {
        int rows = d.GetLength(0);
        int cols = d.GetLength(1);
        var result = new T[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = NumOps.FromDouble(d[i, j]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    protected override T PredictSingle(Vector<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        // Fixed effects prediction only (no group known)
        double pred = NumOps.ToDouble(_fixedEffects[0]);
        for (int j = 0; j < _numFeatures; j++)
        {
            double x = NumOps.ToDouble(input[j]);
            if (_options.CenterFeatures && _featureMeans != null)
            {
                x -= NumOps.ToDouble(_featureMeans[j]);
            }
            pred += x * NumOps.ToDouble(_fixedEffects[j + 1]);
        }

        return NumOps.FromDouble(pred);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.MixedEffectsModel,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumFeatures", _numFeatures },
                { "NumRandomEffects", _numRandomEffects },
                { "IncludeRandomIntercept", _options.IncludeRandomIntercept },
                { "IncludeRandomSlopes", _options.IncludeRandomSlopes },
                { "OptimizationMethod", _options.OptimizationMethod.ToString() },
                { "ICC", ComputeICC() }
            }
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options and dimensions
        writer.Write(_numFeatures);
        writer.Write(_numRandomEffects);
        writer.Write(_options.IncludeRandomIntercept);
        writer.Write(_options.IncludeRandomSlopes);
        writer.Write(_options.CenterFeatures);

        // Fixed effects
        writer.Write(_fixedEffects?.Length ?? 0);
        if (_fixedEffects != null)
        {
            foreach (var fe in _fixedEffects)
            {
                writer.Write(NumOps.ToDouble(fe));
            }
        }

        // Residual variance
        writer.Write(NumOps.ToDouble(_residualVariance));

        // Feature means
        writer.Write(_featureMeans?.Length ?? 0);
        if (_featureMeans != null)
        {
            foreach (var mean in _featureMeans)
            {
                writer.Write(NumOps.ToDouble(mean));
            }
        }

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        base.Deserialize(reader.ReadBytes(baseLen));

        _numFeatures = reader.ReadInt32();
        _numRandomEffects = reader.ReadInt32();
        _options.IncludeRandomIntercept = reader.ReadBoolean();
        _options.IncludeRandomSlopes = reader.ReadBoolean();
        _options.CenterFeatures = reader.ReadBoolean();

        int numFE = reader.ReadInt32();
        if (numFE > 0)
        {
            _fixedEffects = new T[numFE];
            for (int i = 0; i < numFE; i++)
            {
                _fixedEffects[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        _residualVariance = NumOps.FromDouble(reader.ReadDouble());

        int numMeans = reader.ReadInt32();
        if (numMeans > 0)
        {
            _featureMeans = new T[numMeans];
            for (int i = 0; i < numMeans; i++)
            {
                _featureMeans[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new MixedEffectsModel<T>(_options, Regularization);
    }

    /// <inheritdoc/>
    protected override ModelType GetModelType()
    {
        return ModelType.MixedEffectsModel;
    }

    /// <inheritdoc/>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Use default grouping when explicit groups aren't provided
        var groupIndices = Enumerable.Range(0, x.Rows).Select(i => i % Math.Max(1, x.Rows / 10)).ToArray();
        Train(x, y, groupIndices);
    }
}
