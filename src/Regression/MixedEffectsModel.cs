using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// // Create a mixed-effects model for hierarchical/clustered data
/// var options = new MixedEffectsModelOptions();
///
/// // Prepare training data: 6 samples with 2 fixed-effect features
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with fixed and random effects estimation
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new MixedEffectsModel&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Linear Mixed Models for Longitudinal Data", "https://doi.org/10.1007/b98969")]
public partial class MixedEffectsModel<T> : NonLinearRegressionBase<T>
{
    /// <summary>
    /// Fixed effect coefficients.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _fixedEffects;

    /// <summary>
    /// Random effect predictions (BLUPs) for each group.
    /// </summary>
    private Dictionary<int, Vector<T>>? _randomEffects;

    /// <summary>
    /// Variance of random effects.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T>? _randomEffectVariance;

    /// <summary>
    /// Residual variance.
    /// </summary>
    private T _residualVariance;

    /// <summary>
    /// Standard errors of fixed effects.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
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
            var oldBeta = new Vector<T>(beta);
            T oldSigma2 = sigma2;

            // Update fixed effects
            beta = UpdateFixedEffects(x, y, groupIndices, u, sigma2, D);

            // Update variance components
            (sigma2, D) = UpdateVarianceComponents(x, y, groupIndices, uniqueGroups, beta, u);

            // Check convergence
            if (HasConverged(beta, oldBeta, sigma2, oldSigma2, tolerance))
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
        D.AsSpan().CopyTo(_randomEffectVariance.AsWritableSpan());

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
        // Use the same default group assignment as Train so random effects are included
        var groupIndices = Enumerable.Range(0, input.Rows)
            .Select(i => i % Math.Max(1, input.Rows / 10)).ToArray();
        return Predict(input, groupIndices);
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

        // Center features if we centered during training
        if (_options.CenterFeatures && _featureMeans != null)
        {
            input = CenterFeaturesForPrediction(input);
        }

        // Fixed effects prediction (intercept first), plus random effects where the group is known
        var result = FixedPredictions(input, _fixedEffects);
        if (groupIndices != null && _randomEffects != null)
        {
            result = Engine.Add(result, RandomEffectContributions(input, groupIndices, _randomEffects));
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

        if (y.Length > 0)
        {
            // logLik = -sum_i 0.5 * (log(2 pi sigma2) + r_i^2 / sigma2)
            var residual = Engine.Subtract(y, predictions);
            var residualSq = Engine.Divide(Engine.Multiply(residual, residual), _residualVariance);
            var terms = Engine.Multiply(Engine.Add(Engine.Fill(y.Length, logTerm), residualSq), half);
            logLik = NumOps.Subtract(logLik, Engine.Sum(terms));
        }

        return logLik;
    }

    /// <summary>
    /// Centers features and stores means.
    /// </summary>
    private Matrix<T> CenterFeatures(Matrix<T> x)
    {
        if (x.Rows == 0 || x.Columns == 0)
        {
            _featureMeans = new Vector<T>(x.Columns);
            return new Matrix<T>(x.Rows, x.Columns);
        }

        var xTensor = Tensor<T>.FromMatrix(x);
        var meanRow = Engine.TensorDivideScalar(
            Engine.ReduceSum(xTensor, new[] { 0 }, keepDims: true), NumOps.FromDouble(x.Rows));
        _featureMeans = meanRow.Reshape(new[] { x.Columns }).ToVector();
        return Engine.TensorSubtract(xTensor, Engine.TensorBroadcastTo(meanRow, new[] { x.Rows, x.Columns })).ToMatrix();
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

        if (x.Rows == 0 || x.Columns == 0)
        {
            return new Matrix<T>(x.Rows, x.Columns);
        }

        var meanRow = new Tensor<T>(new[] { 1, x.Columns }, _featureMeans.GetSubVector(0, x.Columns));
        return Engine.TensorSubtract(
            Tensor<T>.FromMatrix(x), Engine.TensorBroadcastTo(meanRow, new[] { x.Rows, x.Columns })).ToMatrix();
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
        return Engine.MatrixAdd(D, Matrix<T>.CreateDiagonal(Engine.Fill(dim, NumOps.One)));
    }

    private Matrix<T> InitializeDiagonalCovariance(Matrix<T> D, int dim)
    {
        return Engine.MatrixAdd(D, Matrix<T>.CreateDiagonal(Engine.Fill(dim, NumOps.One)));
    }

    private Matrix<T> InitializeCompoundSymmetryCovariance(Matrix<T> D, int dim)
    {
        return DiagonalWithConstantOffDiagonal(dim, NumOps.One, NumOps.FromDouble(0.5));
    }

    private Matrix<T> InitializeUnstructuredCovariance(Matrix<T> D, int dim)
    {
        return DiagonalWithConstantOffDiagonal(dim, NumOps.One, NumOps.FromDouble(0.1));
    }

    /// <summary>
    /// A [dim, dim] matrix with <paramref name="diagonal"/> on the diagonal and <paramref name="offDiagonal"/> elsewhere.
    /// </summary>
    private Matrix<T> DiagonalWithConstantOffDiagonal(int dim, T diagonal, T offDiagonal)
    {
        if (dim == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var identity = Tensor<T>.FromMatrix(Matrix<T>.CreateDiagonal(Engine.Fill(dim, NumOps.One)));
        return Engine.TensorWhere(
            identity,
            new Tensor<T>(new[] { dim, dim }, Engine.Fill(dim * dim, diagonal)),
            new Tensor<T>(new[] { dim, dim }, Engine.Fill(dim * dim, offDiagonal))).ToMatrix();
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

            var Z = BuildRandomEffectDesignMatrix(xData, groupObs);

            // Residuals after the fixed effects
            var residuals = Engine.Subtract(
                RegressionEngineOps.Gather(yData, groupObs), FixedPredictions(xData.GetRows(groupObs), beta));

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
        if (n == 0 || q == 0)
        {
            return new Vector<T>(q);
        }

        // V = Z D Z' + sigma2 I
        var zTranspose = Engine.MatrixTranspose(Z);
        var V = Engine.MatrixMultiply(Engine.MatrixMultiply(Z, D), zTranspose);
        V = Engine.MatrixAdd(V, Matrix<T>.CreateDiagonal(Engine.Fill(n, sigma2)));

        var Vinv = InvertMatrix(V);

        // u = D Z' V^-1 r
        var vInvRes = Engine.MatrixVectorMultiply(Vinv, residuals);
        return Engine.MatrixVectorMultiply(D, Engine.MatrixVectorMultiply(zTranspose, vInvRes));
    }

    /// <summary>
    /// Updates fixed effects using weighted least squares.
    /// </summary>
    private Vector<T> UpdateFixedEffects(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        Dictionary<int, Vector<T>> u, T sigma2, Matrix<T> D)
    {
        // Response with each observation's random-effect contribution removed
        var adjustedY = Engine.Subtract(yData, RandomEffectContributions(xData, groupIndices, u));

        // beta = (X'X)^-1 X'y over the design with the intercept column first
        var design = FixedDesign(xData);
        var designTranspose = Engine.MatrixTranspose(design);
        var XtXinv = InvertMatrix(Engine.MatrixMultiply(designTranspose, design));
        return Engine.MatrixVectorMultiply(XtXinv, Engine.MatrixVectorMultiply(designTranspose, adjustedY));
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

        // Residual variance from the fixed + random fit
        var predictions = Engine.Add(FixedPredictions(xData, beta), RandomEffectContributions(xData, groupIndices, u));
        var residual = Engine.Subtract(yData, predictions);
        T rss = n == 0 ? NumOps.Zero : Engine.DotProduct(residual, residual);

        T sigma2;
        if (_options.OptimizationMethod == MixedEffectsOptimization.REML)
        {
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

        // Random effect covariance: mean outer product of the non-zero group effects
        int q = _numRandomEffects;
        var D = new Matrix<T>(q, q);
        int numGroups = 0;
        foreach (var group in uniqueGroups)
        {
            if (u.TryGetValue(group, out var uGroup) && HasEffectAbove(uGroup, epsilon))
            {
                numGroups++;
                var effects = uGroup.GetSubVector(0, q);
                D = Engine.MatrixAdd(D, Engine.MatrixMultiply(
                    new Tensor<T>(new[] { q, 1 }, effects).ToMatrix(),
                    new Tensor<T>(new[] { 1, q }, effects).ToMatrix()));
            }
        }

        if (numGroups > 0)
        {
            var averaged = Engine.TensorDivideScalar(Tensor<T>.FromMatrix(D), NumOps.FromDouble(numGroups));

            // Diagonal entries below epsilon are raised to epsilon
            var identity = Tensor<T>.FromMatrix(Matrix<T>.CreateDiagonal(Engine.Fill(q, NumOps.One)));
            var lowDiagonal = Engine.TensorMultiply(identity, Engine.TensorLessThan(averaged, epsilon));
            D = Engine.TensorWhere(
                lowDiagonal, new Tensor<T>(new[] { q, q }, Engine.Fill(q * q, epsilon)), averaged).ToMatrix();
        }
        else
        {
            T smallVal = NumOps.FromDouble(0.1);
            D = Engine.MatrixAdd(D, Matrix<T>.CreateDiagonal(Engine.Fill(q, smallVal)));
        }

        return (sigma2, D);
    }

    /// <summary>
    /// Whether any |effect| exceeds <paramref name="threshold"/>.
    /// </summary>
    private bool HasEffectAbove(Vector<T> effects, T threshold)
    {
        if (effects.Length == 0)
        {
            return false;
        }

        var above = Engine.TensorGreaterThan(Engine.TensorAbs(Tensor<T>.FromVector(effects)), threshold);
        return RegressionEngineOps.NonzeroIndices(above).Length > 0;
    }

    /// <summary>
    /// Whether both the summed |beta change| and the |sigma2 change| are below the tolerance.
    /// </summary>
    private bool HasConverged(Vector<T> beta, Vector<T> oldBeta, T sigma2, T oldSigma2, T tolerance)
    {
        T betaChange = beta.Length == 0 ? NumOps.Zero : Engine.Sum(Engine.Abs(Engine.Subtract(beta, oldBeta)));
        return NumOps.LessThan(betaChange, tolerance) &&
            NumOps.LessThan(NumOps.Abs(NumOps.Subtract(sigma2, oldSigma2)), tolerance);
    }

    /// <summary>
    /// Computes standard errors of fixed effects.
    /// </summary>
    private Vector<T> ComputeStandardErrors(
        Matrix<T> xData, Vector<T> yData, int[] groupIndices,
        int[] uniqueGroups, Vector<T> beta, T sigma2, Matrix<T> D)
    {
        // se_j = sqrt(sigma2 * [(X'X)^-1]_jj) over the design with the intercept column first
        var design = FixedDesign(xData);
        var XtXinv = InvertMatrix(Engine.MatrixMultiply(Engine.MatrixTranspose(design), design));
        return Engine.Sqrt(Engine.Multiply(XtXinv.Diagonal(), sigma2));
    }

    /// <summary>
    /// Builds the random effect design matrix for a group.
    /// </summary>
    private Matrix<T> BuildRandomEffectDesignMatrix(Matrix<T> xData, int[] groupObs)
    {
        int rows = groupObs.Length;
        var columns = new List<Vector<T>>(_numRandomEffects);
        var groupRows = xData.GetRows(groupObs);

        if (_options.IncludeRandomIntercept)
        {
            columns.Add(Engine.Fill(rows, NumOps.One));
        }

        if (_options.IncludeRandomSlopes)
        {
            var slopeFeatures = _options.RandomSlopeFeatures ?? Enumerable.Range(0, _numFeatures).ToArray();
            foreach (int f in slopeFeatures)
            {
                columns.Add(Engine.GetColumn(groupRows, f));
            }
        }

        // Any columns past the configured effects stay zero
        while (columns.Count < _numRandomEffects)
        {
            columns.Add(new Vector<T>(rows));
        }

        if (rows == 0 || columns.Count == 0)
        {
            return new Matrix<T>(rows, _numRandomEffects);
        }

        return Engine.TensorConcatenate(
            columns.Take(_numRandomEffects).Select(column => new Tensor<T>(new[] { rows, 1 }, column)).ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// The fixed-effects design: an intercept column of ones followed by the first numFeatures columns.
    /// </summary>
    private Matrix<T> FixedDesign(Matrix<T> xData)
    {
        var features = xData.Columns == _numFeatures ? xData : xData.GetSubMatrix(0, 0, xData.Rows, _numFeatures);
        return features.AddConstantColumn(NumOps.One);
    }

    /// <summary>
    /// beta[0] + x[i, :numFeatures] * beta[1..] for every row.
    /// </summary>
    private Vector<T> FixedPredictions(Matrix<T> xData, Vector<T> beta)
    {
        if (xData.Rows == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.MatrixVectorMultiply(FixedDesign(xData), beta.GetSubVector(0, _numFeatures + 1));
    }

    /// <summary>
    /// Z_g u_g for every observation whose group has random effects in <paramref name="u"/> (0 elsewhere).
    /// </summary>
    private Vector<T> RandomEffectContributions(Matrix<T> xData, int[] groupIndices, Dictionary<int, Vector<T>> u)
    {
        int n = xData.Rows;
        var rowsByGroup = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int group = groupIndices[i];
            if (!u.ContainsKey(group))
            {
                continue;
            }

            if (!rowsByGroup.TryGetValue(group, out var rows))
            {
                rows = new List<int>();
                rowsByGroup[group] = rows;
            }

            rows.Add(i);
        }

        var allRows = new List<int>(n);
        var contributions = new List<Vector<T>>(rowsByGroup.Count);
        foreach (var entry in rowsByGroup)
        {
            int[] rows = entry.Value.ToArray();
            var Z = BuildRandomEffectDesignMatrix(xData, rows);
            contributions.Add(_numRandomEffects == 0
                ? new Vector<T>(rows.Length)
                : Engine.MatrixVectorMultiply(Z, u[entry.Key].GetSubVector(0, _numRandomEffects)));
            allRows.AddRange(rows);
        }

        var result = Engine.FillZero<T>(n);
        if (allRows.Count == 0)
        {
            return result;
        }

        return Engine.TensorScatterAdd(
            Tensor<T>.FromVector(result),
            new Tensor<int>(allRows.ToArray(), new[] { allRows.Count }),
            Tensor<T>.FromVector(Vector<T>.Concatenate(contributions)),
            0).ToVector();
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
        T epsilon = NumOps.FromDouble(1e-10);
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // Gauss-Jordan on [A | I] with partial pivoting
        var augmented = Engine.TensorConcatenate(
            new[]
            {
                Tensor<T>.FromMatrix(A),
                Tensor<T>.FromMatrix(Matrix<T>.CreateDiagonal(Engine.Fill(n, NumOps.One)))
            }, 1).ToMatrix();

        for (int col = 0; col < n; col++)
        {
            augmented = RegressionEngineOps.GaussJordanStep(augmented, col, n, epsilon);
        }

        return augmented.GetSubMatrix(0, n, n, n);
    }

    /// <inheritdoc/>
    protected override T PredictSingle(Vector<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        T pred = _fixedEffects[0];
        if (_numFeatures == 0)
        {
            return pred;
        }

        var x = input.GetSubVector(0, _numFeatures);
        if (_options.CenterFeatures && _featureMeans != null)
        {
            x = Engine.Subtract(x, _featureMeans.GetSubVector(0, _numFeatures));
        }

        return NumOps.Add(pred, Engine.DotProduct(x, RegressionEngineOps2.Slice(_fixedEffects, 1, _numFeatures)));
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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

    /// <summary>
    /// MixedEffects doesn't support optimizer parameter injection.
    /// </summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Returns all features used by the model.
    /// </summary>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        return Enumerable.Range(0, _numFeatures > 0 ? _numFeatures : 0);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        return new MixedEffectsModel<T>(_options, Regularization);
    }

    /// <inheritdoc/>
    protected override void OptimizeModel(Matrix<T> x, Vector<T> y)
    {
        // Use default grouping when explicit groups aren't provided
        var groupIndices = Enumerable.Range(0, x.Rows).Select(i => i % Math.Max(1, x.Rows / 10)).ToArray();
        Train(x, y, groupIndices);
    }
}
