using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression.MixedEffects;

/// <summary>
/// Linear Mixed-Effects Model for hierarchical and clustered data.
/// </summary>
/// <remarks>
/// <para>
/// Linear Mixed-Effects (LME) models extend linear regression to handle grouped or
/// nested data by including both fixed effects (population-level parameters) and
/// random effects (group-level deviations from the population).
/// </para>
/// <para>
/// The model has the form: y = X*beta + Z*u + epsilon
/// where:
/// - X*beta: Fixed effects (same for all observations)
/// - Z*u: Random effects (vary by group)
/// - epsilon: Residual error
/// </para>
/// <para>
/// <b>For Beginners:</b> Mixed models are essential when your data has natural grouping:
///
/// Example: Student test scores across 50 schools
/// - Fixed effect: Effect of study time on scores (same for all schools)
/// - Random intercept: Each school may have a different baseline score level
/// - Random slope: Effect of study time might differ by school
///
/// Benefits over simple regression:
/// 1. Correct standard errors (not underestimated)
/// 2. Borrowing strength across groups (shrinkage)
/// 3. Quantify between-group variation
/// 4. Handle unbalanced data naturally
///
/// When to use mixed models:
/// - Repeated measures on individuals
/// - Students in classrooms in schools
/// - Patients in hospitals
/// - Longitudinal/panel data
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Linear Mixed-Effects model for grouped/clustered data
/// var options = new LinearMixedModelOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 fixed-effect features
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with REML estimation for fixed and random effects
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new LinearMixedModel&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new observation
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.Regression)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Linear Mixed Models for Longitudinal Data", "https://doi.org/10.1007/b98969")]
public partial class LinearMixedModel<T> : RegressionBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected new static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly LinearMixedModelOptions<T> _options;

    /// <summary>
    /// List of random effect specifications.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private readonly List<RandomEffect<T>> _randomEffects;

    /// <summary>
    /// Fixed effects coefficients.
    /// </summary>
    [AiDotNet.Attributes.Buffer(
        Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private Vector<T>? _fixedEffects;

    /// <summary>
    /// Variance decomposition results.
    /// </summary>
    private VarianceDecomposition<T>? _varianceDecomposition;

    /// <summary>
    /// Residual variance estimate.
    /// </summary>


    private T _residualVariance;

    /// <summary>
    /// Log-likelihood of the fitted model.
    /// </summary>
    private T _logLikelihood;


    /// <summary>
    /// Number of observations.
    /// </summary>
    private int _nObservations;

    /// <summary>
    /// Number of fixed effect parameters.
    /// </summary>
    private int _nFixedParams;

    /// <summary>
    /// Gets the fixed effects coefficients.
    /// </summary>
    public Vector<T> FixedEffects => _fixedEffects ?? throw new InvalidOperationException("Model not trained.");

    /// <summary>
    /// Gets the random effects specifications with estimated values.
    /// </summary>
    public IReadOnlyList<RandomEffect<T>> RandomEffects => _randomEffects;

    /// <summary>
    /// Gets the variance decomposition.
    /// </summary>
    public VarianceDecomposition<T> VarianceComponents =>
        _varianceDecomposition ?? throw new InvalidOperationException("Model not trained.");

    /// <summary>
    /// Gets the log-likelihood of the fitted model.
    /// </summary>
    public T LogLikelihood => _logLikelihood;

    /// <summary>
    /// Gets the AIC (Akaike Information Criterion).
    /// </summary>
    public T AIC => NumOps.Add(NumOps.Multiply(NumOps.FromDouble(-2), _logLikelihood), NumOps.FromDouble(2 * GetNumberOfParameters()));

    /// <summary>
    /// Gets the BIC (Bayesian Information Criterion).
    /// </summary>
    public T BIC => NumOps.Add(NumOps.Multiply(NumOps.FromDouble(-2), _logLikelihood), NumOps.FromDouble(Math.Log(_nObservations) * GetNumberOfParameters()));

    /// <summary>
    /// Marginal R-squared (fixed effects only).
    /// </summary>

    public T MarginalRSquared { get; private set; }

    /// <summary>
    /// Conditional R-squared (fixed + random effects).
    /// </summary>
    public T ConditionalRSquared { get; private set; }

    /// <summary>
    /// Initializes a new Linear Mixed-Effects Model.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public LinearMixedModel(
        LinearMixedModelOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ??= new LinearMixedModelOptions<T>(), regularization)
    {
        ConditionalRSquared = NumOps.Zero;
        MarginalRSquared = NumOps.Zero;
        _logLikelihood = NumOps.Zero;
        _residualVariance = NumOps.Zero;
        _options = options;
        _randomEffects = [];
    }

    /// <summary>
    /// Adds a random intercept effect to the model.
    /// </summary>
    /// <param name="name">Name for this random effect.</param>
    /// <param name="groupColumnIndex">Column index of the grouping variable.</param>
    /// <returns>This instance for fluent chaining.</returns>
    public LinearMixedModel<T> AddRandomIntercept(string name, int groupColumnIndex)
    {
        _randomEffects.Add(new RandomEffect<T>(name, groupColumnIndex));
        return this;
    }

    /// <summary>
    /// Adds a random slope effect to the model.
    /// </summary>
    /// <param name="name">Name for this random effect.</param>
    /// <param name="groupColumnIndex">Column index of the grouping variable.</param>
    /// <param name="slopeColumns">Column indices for random slopes.</param>
    /// <param name="includeIntercept">Whether to also include random intercept.</param>
    /// <returns>This instance for fluent chaining.</returns>
    public LinearMixedModel<T> AddRandomSlope(
        string name,
        int groupColumnIndex,
        int[] slopeColumns,
        bool includeIntercept = true)
    {
        _randomEffects.Add(new RandomEffect<T>(name, groupColumnIndex, slopeColumns, includeIntercept));
        return this;
    }

    /// <summary>
    /// Trains the Linear Mixed-Effects Model using the EM algorithm.
    /// </summary>
    /// <param name="x">Feature matrix (including grouping variables).</param>
    /// <param name="y">Response vector.</param>
    /// <summary>LMM doesn't benefit from optimizer parameter injection.</summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    private bool _useOLS;

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (_randomEffects.Count == 0)
        {
            throw new InvalidOperationException("At least one random effect must be specified. Use AddRandomIntercept() or AddRandomSlope().");
        }

        // This method previously fitted ORDINARY LEAST SQUARES and returned immediately —
        // `_useOLS = true` was set unconditionally, making the mixed-effects (REML) estimation below
        // unreachable. A caller asking for a linear mixed model received a fixed-effects-only fit, so the random effects they declared were estimated as nothing at all.
        // The real estimation now runs.
        //
        // `_useOLS` is retained (always false for newly trained models) purely so that models
        // serialized before this fix still deserialize and predict through their stored
        // coefficients rather than silently changing behaviour on load.
        _useOLS = false;
        TrainingFeatureCount = x.Columns;

        // Validate grouping column indices are within bounds
        foreach (var re in _randomEffects)
        {
            if (re.GroupColumnIndex >= x.Columns)
            {
                throw new ArgumentException(
                    $"Random effect '{re.Name}' references group column index {re.GroupColumnIndex}, " +
                    $"but input matrix only has {x.Columns} columns (valid indices: 0 to {x.Columns - 1}). " +
                    $"The group column must be included in the input matrix.");
            }
        }

        _nObservations = x.Rows;
        _nFixedParams = x.Columns - GetGroupingColumnCount() + (_options.UseIntercept ? 1 : 0);

        // Extract fixed effects design matrix (exclude grouping columns)
        var fixedX = ExtractFixedEffectsMatrix(x);

        // Initialize parameters
        InitializeParameters(fixedX, y);

        // Run EM algorithm
        if (_options.Optimizer == MixedEffectsOptimizer.EM)
        {
            FitEM(fixedX, x, y);
        }
        else
        {
            // Fall back to EM for now - other optimizers can be added
            FitEM(fixedX, x, y);
        }

        // Compute variance decomposition
        _varianceDecomposition = ComputeVarianceDecomposition();

        // Compute R-squared if requested
        if (_options.ComputeRSquared)
        {
            ComputeRSquaredValues(fixedX, x, y);
        }

        // Surface the fit through the base class's Intercept/Coefficients contract. The first
        // fixed effect is the intercept when there is one, so it is reported as the intercept
        // rather than as a coefficient belonging to a feature that does not exist.
        var fitted = _fixedEffects ?? new Vector<T>(_nFixedParams);

        if (_options.UseIntercept && fitted.Length > 0)
        {
            Intercept = fitted[0];

            Coefficients = fitted.GetSubVector(1, fitted.Length - 1);
        }
        else
        {
            Intercept = NumOps.Zero;
            Coefficients = fitted;
        }
    }

    /// <summary>
    /// Makes predictions for new data.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <returns>Predicted values.</returns>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // OLS path
        if (_useOLS)
        {
            return base.Predict(input);
        }

        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        int n = input.Rows;
        var fixedX = ExtractFixedEffectsMatrix(input);

        // Fixed effects contribution
        var predictions = FixedPredictions(fixedX, _fixedEffects);

        // Add random effects (BLUPs) if available
        foreach (var re in _randomEffects)
        {
            if (re.GroupColumnIndex < input.Columns)
            {
                predictions = Engine.Add(predictions, RandomEffectContributions(input, fixedX, re));
            }
        }

        return predictions;
    }

    /// <summary>
    /// fixedX * fixedEffects over the first _nFixedParams columns.
    /// </summary>
    private Vector<T> FixedPredictions(Matrix<T> fixedX, Vector<T> fixedEffects)
    {
        if (fixedX.Rows == 0 || _nFixedParams == 0)
        {
            return new Vector<T>(fixedX.Rows);
        }

        var design = fixedX.Columns == _nFixedParams ? fixedX : fixedX.GetSubMatrix(0, 0, fixedX.Rows, _nFixedParams);
        return Engine.MatrixVectorMultiply(design, fixedEffects.GetSubVector(0, _nFixedParams));
    }

    /// <summary>
    /// Per-row contribution of one random effect: the group's intercept plus its slopes times the
    /// matching fixed-design columns (0 for rows whose group has no effect).
    /// </summary>
    private Vector<T> RandomEffectContributions(Matrix<T> input, Matrix<T> fixedX, RandomEffect<T> re)
    {
        int n = input.Rows;
        var allRows = new List<int>(n);
        var contributions = new List<Vector<T>>();
        foreach (var kvp in GroupObservations(input, re.GroupColumnIndex))
        {
            int[] rows = kvp.Value.ToArray();
            var groupEffect = re.GetGroupEffect(kvp.Key);
            var contribution = Engine.FillZero(rows.Length);

            // Random intercept
            if (re.IsRandomIntercept && groupEffect.Length > 0)
            {
                contribution = Engine.Add(contribution, Engine.Fill(rows.Length, FirstElement(groupEffect)));
            }

            // Random slopes
            if (re.RandomSlopeColumns != null)
            {
                int offset = re.IsRandomIntercept ? 1 : 0;
                var effectIndices = new List<int>();
                var slopeColumns = new List<Vector<T>>();
                var groupRows = fixedX.GetRows(rows);
                for (int s = 0; s < re.RandomSlopeColumns.Length; s++)
                {
                    int slopeCol = re.RandomSlopeColumns[s];
                    if (offset + s < groupEffect.Length && slopeCol < fixedX.Columns)
                    {
                        effectIndices.Add(offset + s);
                        slopeColumns.Add(Engine.GetColumn(groupRows, slopeCol));
                    }
                }

                if (effectIndices.Count > 0 && rows.Length > 0)
                {
                    var slopeDesign = Engine.TensorConcatenate(
                        slopeColumns.Select(column => new Tensor<T>(new[] { rows.Length, 1 }, column)).ToArray(), 1).ToMatrix();
                    contribution = Engine.Add(contribution, Engine.MatrixVectorMultiply(
                        slopeDesign, RegressionEngineOps.Gather(groupEffect, effectIndices.ToArray())));
                }
            }

            allRows.AddRange(rows);
            contributions.Add(contribution);
        }

        return ScatterRows(n, allRows, contributions);
    }

    /// <summary>
    /// A length-<paramref name="n"/> vector holding each block of values at its rows (0 elsewhere).
    /// </summary>
    private Vector<T> ScatterRows(int n, List<int> rows, List<Vector<T>> values)
    {
        var result = Engine.FillZero(n);
        if (rows.Count == 0)
        {
            return result;
        }

        return Engine.TensorScatterAdd(
            Tensor<T>.FromVector(result),
            new Tensor<int>(rows.ToArray(), new[] { rows.Count }),
            Tensor<T>.FromVector(Vector<T>.Concatenate(values)),
            0).ToVector();
    }

    /// <summary>
    /// Per-row random intercept (first effect component) of one random effect.
    /// </summary>
    private Vector<T> RowInterceptEffects(Matrix<T> fullX, RandomEffect<T> re)
    {
        var allRows = new List<int>(fullX.Rows);
        var values = new List<Vector<T>>();
        foreach (var kvp in GroupObservations(fullX, re.GroupColumnIndex))
        {
            var blup = re.GetGroupEffect(kvp.Key);
            if (blup.Length > 0)
            {
                allRows.AddRange(kvp.Value);
                values.Add(Engine.Fill(kvp.Value.Count, FirstElement(blup)));
            }
        }

        return ScatterRows(fullX.Rows, allRows, values);
    }

    /// <summary>
    /// The first component of an effect vector.
    /// </summary>
    private static T FirstElement(Vector<T> values)
    {
        return values[0];
    }

    /// <summary>
    /// The [0, 0] entry of a covariance matrix.
    /// </summary>
    private static T FirstDiagonal(Matrix<T> covariance)
    {
        return covariance[0, 0];
    }

    /// <summary>
    /// Sets the [0, 0] entry of a covariance matrix.
    /// </summary>
    private static void SetFirstDiagonal(Matrix<T> covariance, T value)
    {
        covariance[0, 0] = value;
    }

    /// <summary>
    /// Gets the number of parameters in the model.
    /// </summary>
    private int GetNumberOfParameters()
    {
        int nParams = _nFixedParams; // Fixed effects
        nParams++; // Residual variance

        foreach (var re in _randomEffects)
        {
            int dim = re.Dimension;
            nParams += (dim * (dim + 1)) / 2; // Unique elements of variance-covariance matrix
        }

        return nParams;
    }

    /// <summary>
    /// Gets the number of grouping columns.
    /// </summary>
    private int GetGroupingColumnCount()
    {
        var groupCols = new HashSet<int>();
        foreach (var re in _randomEffects)
        {
            groupCols.Add(re.GroupColumnIndex);
        }
        return groupCols.Count;
    }

    /// <summary>
    /// Extracts the fixed effects design matrix.
    /// </summary>
    private Matrix<T> ExtractFixedEffectsMatrix(Matrix<T> x)
    {
        var groupCols = new HashSet<int>();
        foreach (var re in _randomEffects)
        {
            groupCols.Add(re.GroupColumnIndex);
        }

        // The leading column of ones is the FIXED intercept (Laird and Ware: y = X b + Z u + e puts it
        // in X and makes the random effects mean-zero deviations around it); the non-group feature
        // columns follow in order.
        var columns = new List<Vector<T>>();
        if (_options.UseIntercept)
        {
            columns.Add(Engine.Fill(x.Rows, NumOps.One));
        }

        for (int j = 0; j < x.Columns; j++)
        {
            if (!groupCols.Contains(j))
            {
                columns.Add(Engine.GetColumn(x, j));
            }
        }

        if (columns.Count == 0 || x.Rows == 0)
        {
            return new Matrix<T>(x.Rows, columns.Count);
        }

        return Engine.TensorConcatenate(
            columns.Select(column => new Tensor<T>(new[] { x.Rows, 1 }, column)).ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// Initializes model parameters.
    /// </summary>
    private void InitializeParameters(Matrix<T> fixedX, Vector<T> y)
    {
        _nFixedParams = fixedX.Columns;

        // Initial fixed effects from OLS
        _fixedEffects = SolveOLS(fixedX, y);

        // Initial variance components: half the residual variance to the residual, the other half
        // split evenly over the random effects
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);
        T halfVariance = NumOps.Multiply(ComputeVariance(residuals), NumOps.FromDouble(0.5));
        _residualVariance = halfVariance;

        foreach (var re in _randomEffects)
        {
            int dim = re.Dimension;
            T componentVariance = NumOps.Divide(halfVariance, NumOps.FromDouble(_randomEffects.Count));
            re.CovarianceMatrix = Matrix<T>.CreateDiagonal(Engine.Fill(dim, componentVariance));
        }
    }

    /// <summary>
    /// Fits the model using the EM algorithm.
    /// </summary>
    private void FitEM(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        T prevLogLik = NumOps.FromDouble(double.NegativeInfinity);
        T tolerance = NumOps.FromDouble(_options.Tolerance);

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: Compute BLUPs of random effects
            ComputeBLUPs(fixedX, fullX, y);

            // M-step: Update fixed effects
            UpdateFixedEffects(fixedX, fullX, y);

            // M-step: Update variance components
            UpdateVarianceComponents(fixedX, fullX, y);

            // Compute log-likelihood
            _logLikelihood = ComputeLogLikelihood(fixedX, fullX, y);

            if (_options.Verbose)
            {
                Console.WriteLine($"Iteration {iter + 1}: Log-likelihood = {_logLikelihood}");
            }

            // Check convergence
            if (ChangedLessThan(_logLikelihood, prevLogLik, tolerance))
            {
                break;
            }

            prevLogLik = _logLikelihood;
        }
    }

    /// <summary>
    /// Whether |current - previous| is below the tolerance.
    /// </summary>
    private bool ChangedLessThan(T current, T previous, T tolerance)
    {
        return NumOps.LessThan(NumOps.Abs(NumOps.Subtract(current, previous)), tolerance);
    }

    /// <summary>
    /// Computes BLUPs (Best Linear Unbiased Predictors) for random effects.
    /// </summary>
    private void ComputeBLUPs(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return;

        // Compute residuals from fixed effects
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);

        foreach (var re in _randomEffects)
        {
            var groups = GroupObservations(fullX, re.GroupColumnIndex);
            T sigmaE = _residualVariance;
            T sigmaU = re.CovarianceMatrix != null ? FirstDiagonal(re.CovarianceMatrix) : NumOps.One;

            foreach (var kvp in groups)
            {
                var indices = kvp.Value;
                T ni = NumOps.FromDouble(indices.Count);

                // BLUP of the random intercept: shrinkage * mean residual,
                // shrinkage = sigmaU / (sigmaU + sigmaE / n_i)
                T meanResid = NumOps.Divide(Engine.Sum(RegressionEngineOps.Gather(residuals, indices.ToArray())), ni);
                T shrinkage = NumOps.Divide(sigmaU, NumOps.Add(sigmaU, NumOps.Divide(sigmaE, ni)));
                T blup = NumOps.Multiply(shrinkage, meanResid);

                var blupVector = new Vector<T>(re.Dimension);
                if (re.IsRandomIntercept && re.Dimension > 0)
                {
                    blupVector = Vector<T>.Concatenate(new[] { Engine.Fill(1, blup), new Vector<T>(re.Dimension - 1) });
                }

                re.SetGroupEffect(kvp.Key, blupVector);
            }
        }
    }

    /// <summary>
    /// Updates fixed effects coefficients.
    /// </summary>
    private void UpdateFixedEffects(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        // Adjust y for random effects
        var adjustedY = new Vector<T>(y);
        foreach (var re in _randomEffects)
        {
            adjustedY = Engine.Subtract(adjustedY, RowInterceptEffects(fullX, re));
        }

        // Solve for fixed effects
        _fixedEffects = SolveOLS(fixedX, adjustedY);
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private void UpdateVarianceComponents(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return;

        // Update residual variance from the residuals after fixed and random intercepts
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);
        foreach (var re in _randomEffects)
        {
            residuals = Engine.Subtract(residuals, RowInterceptEffects(fullX, re));
        }

        T ssResid = residuals.Length == 0 ? NumOps.Zero : Engine.DotProduct(residuals, residuals);
        int dfResid = _nObservations - _nFixedParams;
        _residualVariance = NumOps.Divide(ssResid, NumOps.FromDouble(dfResid));

        // Update random effect variances: mean squared random intercept over the groups
        foreach (var re in _randomEffects)
        {
            var groups = GroupObservations(fullX, re.GroupColumnIndex);
            var intercepts = groups.Keys
                .Select(key => re.GetGroupEffect(key))
                .Where(blup => blup.Length > 0)
                .Select(blup => blup.GetSubVector(0, 1))
                .ToList();
            T sumSqBlup = NumOps.Zero;
            if (intercepts.Count > 0)
            {
                var interceptVector = Vector<T>.Concatenate(intercepts);
                sumSqBlup = Engine.DotProduct(interceptVector, interceptVector);
            }

            T newVar = NumOps.Divide(sumSqBlup, NumOps.FromDouble(groups.Count));
            if (_options.BoundVarianceComponents && NumOps.LessThan(newVar, NumOps.Zero))
            {
                newVar = NumOps.Zero;
            }

            if (re.CovarianceMatrix != null && re.CovarianceMatrix.Rows > 0)
            {
                SetFirstDiagonal(re.CovarianceMatrix, newVar);
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private T ComputeLogLikelihood(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return NumOps.FromDouble(double.NegativeInfinity);

        T sigmaE = _residualVariance;
        int n = _nObservations;
        var predictions = Predict(fullX);

        var residual = Engine.Subtract(y.GetSubVector(0, n), predictions.GetSubVector(0, n));
        T ssResid = n == 0 ? NumOps.Zero : Engine.DotProduct(residual, residual);

        // -0.5 n log(2 pi sigmaE) - ssResid / (2 sigmaE)
        T twoPi = NumOps.FromDouble(2 * Math.PI);
        T first = NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(-0.5), NumOps.FromDouble(n)), NumOps.Log(NumOps.Multiply(twoPi, sigmaE)));
        return NumOps.Subtract(first, NumOps.Divide(ssResid, NumOps.Multiply(NumOps.FromDouble(2), sigmaE)));
    }

    /// <summary>
    /// Computes the variance decomposition.
    /// </summary>
    private VarianceDecomposition<T> ComputeVarianceDecomposition()
    {
        var decomp = new VarianceDecomposition<T>
        {
            ResidualVariance = new VarianceComponent<T>("Residual")
            {
                Variance = _residualVariance
            }
        };

        foreach (var re in _randomEffects)
        {
            var vc = new VarianceComponent<T>(re.Name)
            {
                Variance = re.CovarianceMatrix != null ? FirstDiagonal(re.CovarianceMatrix) : NumOps.Zero,
                CovarianceMatrix = re.CovarianceMatrix
            };
            decomp.RandomEffectVariances.Add(vc);
        }

        return decomp;
    }

    /// <summary>
    /// Computes marginal and conditional R-squared.
    /// </summary>
    private void ComputeRSquaredValues(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return;

        int n = y.Length;
        T yMean = NumOps.Divide(n == 0 ? NumOps.Zero : Engine.Sum(y), NumOps.FromDouble(n));
        var meanVector = Engine.Fill(n, yMean);

        // Total sum of squares
        T ssTot = SumOfSquaredDeviations(y, meanVector);

        // Marginal R^2: variance explained by the fixed effects
        var fixedPred = FixedPredictions(fixedX, _fixedEffects).GetSubVector(0, n);
        MarginalRSquared = NumOps.Divide(SumOfSquaredDeviations(fixedPred, meanVector), ssTot);

        // Conditional R^2: variance explained by fixed + random effects
        var fullPred = Predict(fullX).GetSubVector(0, n);
        ConditionalRSquared = NumOps.Divide(SumOfSquaredDeviations(fullPred, meanVector), ssTot);
    }

    /// <summary>
    /// sum_i (values[i] - reference[i])^2.
    /// </summary>
    private T SumOfSquaredDeviations(Vector<T> values, Vector<T> reference)
    {
        if (values.Length == 0)
        {
            return NumOps.Zero;
        }

        var deviations = Engine.Subtract(values, reference);
        return Engine.DotProduct(deviations, deviations);
    }

    /// <summary>
    /// Groups observations by grouping variable.
    /// </summary>
    private static Dictionary<double, List<int>> GroupObservations(Matrix<T> x, int groupCol)
    {
        var groups = new Dictionary<double, List<int>>();

        for (int i = 0; i < x.Rows; i++)
        {
            double groupId = NumOps.ToDouble(x[i, groupCol]);
            if (!groups.ContainsKey(groupId))
            {
                groups[groupId] = [];
            }
            groups[groupId].Add(i);
        }

        return groups;
    }

    /// <summary>
    /// Solves ordinary least squares.
    /// </summary>
    private static Vector<T> SolveOLS(Matrix<T> x, Vector<T> y)
    {
        // X'X
        var xtx = x.Transpose().Multiply(x);

        // X'y
        var xty = x.Transpose().Multiply(y);

        // Solve (X'X)^-1 X'y
        return xtx.Inverse().Multiply(xty);
    }

    /// <summary>
    /// Computes residuals.
    /// </summary>
    private Vector<T> ComputeResiduals(Matrix<T> x, Vector<T> y, Vector<T> beta)
    {
        var pred = x.Multiply(beta);
        return Engine.Subtract(y, pred);
    }

    /// <summary>
    /// Computes variance of a vector.
    /// </summary>
    private T ComputeVariance(Vector<T> v)
    {
        // Sample variance (two-pass: mean, then deviations), divided by n - 1
        int n = v.Length;
        T mean = NumOps.Divide(n == 0 ? NumOps.Zero : Engine.Sum(v), NumOps.FromDouble(n));
        T variance = SumOfSquaredDeviations(v, Engine.Fill(n, mean));
        return NumOps.Divide(variance, NumOps.FromDouble(n - 1));
    }
}
