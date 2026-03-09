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
    private Vector<T>? _fixedEffects;

    /// <summary>
    /// Random effect predictions (BLUPs) for each group.
    /// </summary>
    private Dictionary<int, Vector<T>>? _randomEffects;

    /// <summary>
    /// Variance of random effects.
    /// </summary>
    private Matrix<T>? _randomEffectVariance;

    /// <summary>
    /// Residual variance.
    /// </summary>
    private T _residualVariance;

    /// <summary>
    /// Standard errors of fixed effects.
    /// </summary>
    private Vector<T>? _fixedEffectStdErrors;

    /// <summary>
    /// Group indices for training data.
    /// </summary>
    private int[]? _groupIndices;

    /// <summary>
    /// Feature means for centering.
    /// </summary>
    private Vector<T>? _featureMeans;

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

        // Get unique groups
        var uniqueGroups = groupIndices.Distinct().OrderBy(g => g).ToArray();
        int numGroups = uniqueGroups.Length;
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        // Initialize parameters
        var beta = new Vector<T>(_numFeatures + 1);  // +1 for intercept
        T sigma2 = NumOps.One;  // Residual variance
        var D = InitializeRandomEffectVariance(_numRandomEffects);  // Random effect variance

        // EM-style optimization
        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: Compute conditional modes of random effects (BLUPs)
            var u = ComputeBLUPs(x, y, groupIndices, uniqueGroups, beta, sigma2, D);

            // M-step: Update fixed effects and variance components
            var oldBeta = new Vector<T>(beta.Length);
            for (int i = 0; i < beta.Length; i++) oldBeta[i] = beta[i];
            T oldSigma2 = sigma2;

            // Update fixed effects
            beta = UpdateFixedEffects(x, y, groupIndices, u, sigma2, D);

            // Update variance components
            (sigma2, D) = UpdateVarianceComponents(x, y, groupIndices, uniqueGroups, beta, u);

            // Check convergence
            T betaChange = NumOps.Zero;
            for (int i = 0; i < beta.Length; i++)
            {
                betaChange = NumOps.Add(betaChange, NumOps.Abs(NumOps.Subtract(beta[i], oldBeta[i])));
            }

            if (NumOps.LessThan(betaChange, tolerance) &&
                NumOps.LessThan(NumOps.Abs(NumOps.Subtract(sigma2, oldSigma2)), tolerance))
            {
                break;
            }
        }

        // Compute final random effects
        var finalU = ComputeBLUPs(x, y, groupIndices, uniqueGroups, beta, sigma2, D);

        // Store results
        _fixedEffects = new Vector<T>(beta);
        _residualVariance = sigma2;
        _randomEffectVariance = new Matrix<T>(D.Rows, D.Columns);
        for (int i = 0; i < D.Rows; i++)
        {
            for (int j = 0; j < D.Columns; j++)
            {
                _randomEffectVariance[i, j] = D[i, j];
            }
        }

        _randomEffects = new Dictionary<int, Vector<T>>();
        foreach (var group in uniqueGroups)
        {
            if (finalU.TryGetValue(group, out var uGroup))
            {
                _randomEffects[group] = new Vector<T>(uGroup);
            }
        }

        // Compute standard errors
        _fixedEffectStdErrors = ComputeStandardErrors(x, y, groupIndices, uniqueGroups, beta, sigma2, D);
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
            T pred = _fixedEffects[0];  // Intercept
            for (int j = 0; j < _numFeatures; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(input[i, j], _fixedEffects[j + 1]));
            }

            // Add random effects if group is known
            if (groupIndices != null && _randomEffects != null)
            {
                int group = groupIndices[i];
                if (_randomEffects.TryGetValue(group, out var u))
                {
                    pred = NumOps.Add(pred, GetRandomEffectContribution(input, i, u));
                }
            }

            result[i] = pred;
        }

        return result;
    }

    /// <summary>
    /// Gets the fixed effect coefficients.
    /// </summary>
    /// <returns>Array of fixed effect coefficients (intercept first).</returns>
    public Vector<T> GetFixedEffects()
    {
        return _fixedEffects ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the standard errors of fixed effects.
    /// </summary>
    /// <returns>Vector of standard errors.</returns>
    public Vector<T> GetFixedEffectStandardErrors()
    {
        return _fixedEffectStdErrors ?? new Vector<T>(0);
    }

    /// <summary>
    /// Gets the random effects (BLUPs) for each group.
    /// </summary>
    /// <returns>Dictionary mapping group ID to random effect values.</returns>
    public Dictionary<int, Vector<T>> GetRandomEffects()
    {
        return _randomEffects ?? new Dictionary<int, Vector<T>>();
    }

    /// <summary>
    /// Gets the residual variance.
    /// </summary>
    public T ResidualVariance => _residualVariance;

    /// <summary>
    /// Gets the random effect variance-covariance matrix.
    /// </summary>
    /// <returns>Variance-covariance matrix of random effects.</returns>
    public Matrix<T>? GetRandomEffectVariance()
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
    public T ComputeICC()
    {
        if (_randomEffectVariance == null || _randomEffectVariance.Rows == 0)
        {
            return NumOps.Zero;
        }

        // For simple random intercept model: ICC = var(intercept) / (var(intercept) + var(residual))
        T interceptVar = _randomEffectVariance[0, 0];
        return NumOps.Divide(interceptVar, NumOps.Add(interceptVar, _residualVariance));
    }

    /// <summary>
    /// Gets the log-likelihood of the model.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Target values.</param>
    /// <param name="groupIndices">Group indicators.</param>
    /// <returns>Log-likelihood value.</returns>
    public T GetLogLikelihood(Matrix<T> x, Vector<T> y, int[] groupIndices)
    {
        if (_fixedEffects == null)
        {
            return NumOps.MinValue;
        }

        var predictions = Predict(x, groupIndices);
        T logLik = NumOps.Zero;
        T half = NumOps.FromDouble(0.5);
        T twoPi = NumOps.FromDouble(2.0 * Math.PI);
        T logTerm = NumOps.Log(NumOps.Multiply(twoPi, _residualVariance));

        for (int i = 0; i < y.Length; i++)
        {
            T residual = NumOps.Subtract(y[i], predictions[i]);
            T residualSq = NumOps.Divide(NumOps.Multiply(residual, residual), _residualVariance);
            logLik = NumOps.Subtract(logLik, NumOps.Multiply(half, NumOps.Add(logTerm, residualSq)));
        }

        return logLik;
    }

    /// <summary>
    /// Centers features and stores means.
    /// </summary>
    private Matrix<T> CenterFeatures(Matrix<T> x)
    {
        _featureMeans = new Vector<T>(x.Columns);
        var centered = new Matrix<T>(x.Rows, x.Columns);
        T nT = NumOps.FromDouble(x.Rows);

        for (int j = 0; j < x.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < x.Rows; i++)
            {
                sum = NumOps.Add(sum, x[i, j]);
            }
            _featureMeans[j] = NumOps.Divide(sum, nT);

            for (int i = 0; i < x.Rows; i++)
            {
                centered[i, j] = NumOps.Subtract(x[i, j], _featureMeans[j]);
            }
        }

        return centered;
    }

    /// <summary>
    /// Centers features using stored means.
    /// </summary>
    private Matrix<T> CenterFeaturesForPrediction(Matrix<T> x)
    {
        if (_featureMeans is null)
        {
            throw new InvalidOperationException("Feature means not computed. Model must be trained first.");
        }

        var centered = new Matrix<T>(x.Rows, x.Columns);

        for (int j = 0; j < x.Columns; j++)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                centered[i, j] = NumOps.Subtract(x[i, j], _featureMeans[j]);
            }
        }

        return centered;
    }

    /// <summary>
    /// Initializes the random effect variance matrix.
    /// </summary>
    private Matrix<T> InitializeRandomEffectVariance(int dim)
    {
        var D = new Matrix<T>(dim, dim);

        return _options.CovarianceStructure switch
        {
            MixedEffectsCovarianceStructure.Identity => InitializeIdentityCovariance(D, dim),
            MixedEffectsCovarianceStructure.Diagonal => InitializeDiagonalCovariance(D, dim),
            MixedEffectsCovarianceStructure.CompoundSymmetry => InitializeCompoundSymmetryCovariance(D, dim),
            _ => InitializeUnstructuredCovariance(D, dim)
        };
    }

    private Matrix<T> InitializeIdentityCovariance(Matrix<T> D, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = NumOps.One;
        }
        return D;
    }

    private Matrix<T> InitializeDiagonalCovariance(Matrix<T> D, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = NumOps.One;
        }
        return D;
    }

    private Matrix<T> InitializeCompoundSymmetryCovariance(Matrix<T> D, int dim)
    {
        T rho = NumOps.FromDouble(0.5);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                D[i, j] = i == j ? NumOps.One : rho;
            }
        }
        return D;
    }

    private Matrix<T> InitializeUnstructuredCovariance(Matrix<T> D, int dim)
    {
        T offDiag = NumOps.FromDouble(0.1);
        for (int i = 0; i < dim; i++)
        {
            D[i, i] = NumOps.One;
            for (int j = 0; j < i; j++)
            {
                D[i, j] = D[j, i] = offDiag;
            }
        }
        return D;
    }

    /// <summary>
    /// Computes Best Linear Unbiased Predictors (BLUPs) for random effects.
    /// </summary>
    private Dictionary<int, Vector<T>> ComputeBLUPs(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        int[] uniqueGroups, Vector<T> beta, T sigma2, Matrix<T> D)
    {
        var blups = new Dictionary<int, Vector<T>>();

        foreach (var group in uniqueGroups)
        {
            var groupObs = GetGroupObservations(groupIndices, group);

            if (groupObs.Length < _options.MinObservationsPerGroup)
            {
                blups[group] = new Vector<T>(_numRandomEffects);
                continue;
            }

            // Design matrix for random effects (Z)
            var Z = BuildRandomEffectDesignMatrix(xData, groupObs);

            // Compute residuals
            var residuals = new Vector<T>(groupObs.Length);
            for (int i = 0; i < groupObs.Length; i++)
            {
                int idx = groupObs[i];
                residuals[i] = NumOps.Subtract(yData[idx], ComputeFixedPrediction(xData, idx, beta));
            }

            // BLUP = D * Z' * V^(-1) * residuals
            var u = ComputeGroupBLUP(Z, residuals, D, sigma2);
            blups[group] = u;
        }

        return blups;
    }

    /// <summary>
    /// Computes BLUP for a single group.
    /// </summary>
    private Vector<T> ComputeGroupBLUP(Matrix<T> Z, Vector<T> residuals, Matrix<T> D, T sigma2)
    {
        int n = residuals.Length;
        int q = _numRandomEffects;

        // V = Z * D * Z' + sigma2 * I
        var V = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < q; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(Z[i, k], D[k, l]), Z[j, l]));
                    }
                }
                V[i, j] = i == j ? NumOps.Add(sum, sigma2) : sum;
            }
        }

        // Invert V
        var Vinv = InvertMatrix(V);

        // D * Z' * V^(-1) * residuals
        var u = new Vector<T>(q);
        for (int k = 0; k < q; k++)
        {
            for (int i = 0; i < n; i++)
            {
                T vInvRes = NumOps.Zero;
                for (int j = 0; j < n; j++)
                {
                    vInvRes = NumOps.Add(vInvRes, NumOps.Multiply(Vinv[i, j], residuals[j]));
                }

                for (int l = 0; l < q; l++)
                {
                    u[k] = NumOps.Add(u[k], NumOps.Multiply(NumOps.Multiply(D[k, l], Z[i, l]), vInvRes));
                }
            }
        }

        return u;
    }

    /// <summary>
    /// Updates fixed effects using weighted least squares.
    /// </summary>
    private Vector<T> UpdateFixedEffects(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        Dictionary<int, Vector<T>> u, T sigma2, Matrix<T> D)
    {
        int n = xData.Rows;
        int p = _numFeatures + 1;  // +1 for intercept

        // X'X
        var XtX = new Matrix<T>(p, p);
        var Xty = new Vector<T>(p);

        for (int i = 0; i < n; i++)
        {
            int group = groupIndices[i];
            T yi = yData[i];

            // Subtract random effect contribution
            if (u.TryGetValue(group, out var uGroup))
            {
                yi = NumOps.Subtract(yi, GetRandomEffectContribution(xData, i, uGroup));
            }

            // Intercept
            XtX[0, 0] = NumOps.Add(XtX[0, 0], NumOps.One);
            Xty[0] = NumOps.Add(Xty[0], yi);

            for (int j = 0; j < _numFeatures; j++)
            {
                XtX[0, j + 1] = NumOps.Add(XtX[0, j + 1], xData[i, j]);
                XtX[j + 1, 0] = NumOps.Add(XtX[j + 1, 0], xData[i, j]);
                Xty[j + 1] = NumOps.Add(Xty[j + 1], NumOps.Multiply(xData[i, j], yi));

                for (int k = 0; k < _numFeatures; k++)
                {
                    XtX[j + 1, k + 1] = NumOps.Add(XtX[j + 1, k + 1], NumOps.Multiply(xData[i, j], xData[i, k]));
                }
            }
        }

        // Solve (X'X)^(-1) * X'y
        var XtXinv = InvertMatrix(XtX);
        var beta = new Vector<T>(p);

        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < p; k++)
            {
                beta[j] = NumOps.Add(beta[j], NumOps.Multiply(XtXinv[j, k], Xty[k]));
            }
        }

        return beta;
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private (T sigma2, Matrix<T> D) UpdateVarianceComponents(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        int[] uniqueGroups, Vector<T> beta, Dictionary<int, Vector<T>> u)
    {
        int n = xData.Rows;
        T epsilon = NumOps.FromDouble(1e-10);

        // Update residual variance
        T rss = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            int group = groupIndices[i];
            T pred = ComputeFixedPrediction(xData, i, beta);

            if (u.TryGetValue(group, out var uGroup))
            {
                pred = NumOps.Add(pred, GetRandomEffectContribution(xData, i, uGroup));
            }

            T residual = NumOps.Subtract(yData[i], pred);
            rss = NumOps.Add(rss, NumOps.Multiply(residual, residual));
        }

        T sigma2;
        if (_options.OptimizationMethod == MixedEffectsOptimization.REML)
        {
            // REML adjustment
            sigma2 = NumOps.Divide(rss, NumOps.FromDouble(n - _numFeatures - 1));
        }
        else
        {
            sigma2 = NumOps.Divide(rss, NumOps.FromDouble(n));
        }

        if (NumOps.LessThan(sigma2, epsilon))
        {
            sigma2 = epsilon;
        }

        // Update random effect variance
        var D = new Matrix<T>(_numRandomEffects, _numRandomEffects);
        int numGroups = 0;

        foreach (var group in uniqueGroups)
        {
            if (u.TryGetValue(group, out var uGroup))
            {
                bool hasNonZero = false;
                for (int i = 0; i < uGroup.Length; i++)
                {
                    if (NumOps.GreaterThan(NumOps.Abs(uGroup[i]), epsilon)) { hasNonZero = true; break; }
                }

                if (hasNonZero)
                {
                    numGroups++;
                    for (int i = 0; i < _numRandomEffects; i++)
                    {
                        for (int j = 0; j < _numRandomEffects; j++)
                        {
                            D[i, j] = NumOps.Add(D[i, j], NumOps.Multiply(uGroup[i], uGroup[j]));
                        }
                    }
                }
            }
        }

        if (numGroups > 0)
        {
            T numGroupsT = NumOps.FromDouble(numGroups);
            for (int i = 0; i < _numRandomEffects; i++)
            {
                for (int j = 0; j < _numRandomEffects; j++)
                {
                    D[i, j] = NumOps.Divide(D[i, j], numGroupsT);
                    if (i == j && NumOps.LessThan(D[i, j], epsilon))
                    {
                        D[i, j] = epsilon;
                    }
                }
            }
        }
        else
        {
            // Initialize with small positive values
            T smallVal = NumOps.FromDouble(0.1);
            for (int i = 0; i < _numRandomEffects; i++)
            {
                D[i, i] = smallVal;
            }
        }

        return (sigma2, D);
    }

    /// <summary>
    /// Computes standard errors of fixed effects.
    /// </summary>
    private Vector<T> ComputeStandardErrors(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        int[] uniqueGroups, Vector<T> beta, T sigma2, Matrix<T> D)
    {
        int n = xData.Rows;
        int p = _numFeatures + 1;

        // X'X
        var XtX = new Matrix<T>(p, p);

        for (int i = 0; i < n; i++)
        {
            XtX[0, 0] = NumOps.Add(XtX[0, 0], NumOps.One);
            for (int j = 0; j < _numFeatures; j++)
            {
                XtX[0, j + 1] = NumOps.Add(XtX[0, j + 1], xData[i, j]);
                XtX[j + 1, 0] = NumOps.Add(XtX[j + 1, 0], xData[i, j]);
                for (int k = 0; k < _numFeatures; k++)
                {
                    XtX[j + 1, k + 1] = NumOps.Add(XtX[j + 1, k + 1], NumOps.Multiply(xData[i, j], xData[i, k]));
                }
            }
        }

        // Standard errors = sqrt(diag(sigma2 * (X'X)^(-1)))
        var XtXinv = InvertMatrix(XtX);
        var stdErrors = new Vector<T>(p);

        for (int j = 0; j < p; j++)
        {
            stdErrors[j] = NumOps.Sqrt(NumOps.Multiply(sigma2, XtXinv[j, j]));
        }

        return stdErrors;
    }

    /// <summary>
    /// Builds the random effect design matrix for a group.
    /// </summary>
    private Matrix<T> BuildRandomEffectDesignMatrix(Matrix<T> xData, int[] groupObs)
    {
        var Z = new Matrix<T>(groupObs.Length, _numRandomEffects);

        for (int i = 0; i < groupObs.Length; i++)
        {
            int idx = groupObs[i];
            int col = 0;

            // Random intercept
            if (_options.IncludeRandomIntercept)
            {
                Z[i, col++] = NumOps.One;
            }

            // Random slopes
            if (_options.IncludeRandomSlopes)
            {
                var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
                foreach (int f in slopeFeatures)
                {
                    Z[i, col++] = xData[idx, f];
                }
            }
        }

        return Z;
    }

    /// <summary>
    /// Computes the fixed effects prediction for a row of the data matrix.
    /// </summary>
    private T ComputeFixedPrediction(Matrix<T> xData, int rowIdx, Vector<T> beta)
    {
        T pred = beta[0];  // Intercept
        for (int j = 0; j < _numFeatures; j++)
        {
            pred = NumOps.Add(pred, NumOps.Multiply(xData[rowIdx, j], beta[j + 1]));
        }
        return pred;
    }

    /// <summary>
    /// Gets the random effect contribution for an observation.
    /// </summary>
    private T GetRandomEffectContribution(Matrix<T> xData, int obsIdx, Vector<T> u)
    {
        T contrib = NumOps.Zero;
        int col = 0;

        if (_options.IncludeRandomIntercept)
        {
            contrib = NumOps.Add(contrib, u[col++]);
        }

        if (_options.IncludeRandomSlopes)
        {
            var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
            foreach (int f in slopeFeatures)
            {
                contrib = NumOps.Add(contrib, NumOps.Multiply(u[col++], xData[obsIdx, f]));
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
    private Matrix<T> InvertMatrix(Matrix<T> A)
    {
        int n = A.Rows;
        var augmented = new Matrix<T>(n, 2 * n);
        T epsilon = NumOps.FromDouble(1e-10);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n + i] = NumOps.One;
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])))
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
            T pivot = augmented[col, col];
            if (NumOps.LessThan(NumOps.Abs(pivot), epsilon))
            {
                pivot = epsilon;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j], NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        // Extract inverse
        var inverse = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
    }

    /// <inheritdoc/>
    protected override T PredictSingle(Vector<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        // Fixed effects prediction only (no group known)
        T pred = _fixedEffects[0];
        for (int j = 0; j < _numFeatures; j++)
        {
            T x = input[j];
            if (_options.CenterFeatures && _featureMeans != null)
            {
                x = NumOps.Subtract(x, _featureMeans[j]);
            }
            pred = NumOps.Add(pred, NumOps.Multiply(x, _fixedEffects[j + 1]));
        }

        return pred;
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
                { "ICC", (object)NumOps.ToDouble(ComputeICC()) }
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
            _fixedEffects = new Vector<T>(numFE);
            for (int i = 0; i < numFE; i++)
            {
                _fixedEffects[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        _residualVariance = NumOps.FromDouble(reader.ReadDouble());

        int numMeans = reader.ReadInt32();
        if (numMeans > 0)
        {
            _featureMeans = new Vector<T>(numMeans);
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
