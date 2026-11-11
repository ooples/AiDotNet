using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression.MixedEffects;



/// <summary>
/// Generalized Linear Mixed-Effects Model (GLMM) for non-Gaussian hierarchical data.
/// </summary>
/// <remarks>
/// <para>
/// GLMMs extend linear mixed models to handle non-Gaussian responses (binary, count, etc.)
/// by incorporating a link function and response distribution from the exponential family.
/// </para>
/// <para>
/// The model has the form: g(E[y|u]) = X*beta + Z*u
/// where:
/// - g() is the link function (logit, log, identity, etc.)
/// - X*beta: Fixed effects on the linear predictor scale
/// - Z*u: Random effects on the linear predictor scale
/// - u ~ N(0, D) where D is the variance-covariance matrix
/// </para>
/// <para>
/// <b>For Beginners:</b> GLMMs are like mixed models but for outcomes that aren't continuous/normal:
///
/// Common use cases:
/// - Binary outcomes (yes/no): Use logistic GLMM with logit link
/// - Count data: Use Poisson GLMM with log link
/// - Overdispersed counts: Use Negative Binomial GLMM
/// - Proportions: Use Binomial GLMM
///
/// Example: Student pass/fail across schools
/// - Fixed effect: Effect of study time on probability of passing
/// - Random intercept: Schools have different baseline pass rates
/// - Random slope: Effect of study time might differ by school
///
/// The model estimates effects on the log-odds (or log-rate) scale,
/// which can then be converted to probabilities or rates.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GLMM for non-Gaussian hierarchical data (e.g., binary outcomes)
/// var options = new GLMMOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 fixed-effect features
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 0, 1, 0, 1, 1, 1 });
///
/// // Train with link function and random effects estimation
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new GeneralizedLinearMixedModel&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new observation
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 6, 7 } });
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
[ResearchPaper("Generalized Linear Mixed Models", "https://doi.org/10.1201/b13459")]
public partial class GeneralizedLinearMixedModel<T> : RegressionBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected new static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly GLMMOptions<T> _options;

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
    /// Dispersion parameter (for overdispersed models).
    /// </summary>

    private T _dispersion;


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
    /// Gets the dispersion parameter.
    /// </summary>
    public T Dispersion => _dispersion;

    /// <summary>
    /// Gets the response distribution family.
    /// </summary>
    public GLMMFamily Family => _options.Family;

    /// <summary>
    /// Gets the link function.
    /// </summary>
    public GLMMLinkFunction LinkFunction => _options.LinkFunction;

    /// <summary>
    /// Initializes a new Generalized Linear Mixed-Effects Model.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public GeneralizedLinearMixedModel(
        GLMMOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ??= new GLMMOptions<T>(), regularization)
    {
        _logLikelihood = NumOps.Zero;
        _options = options;
        _randomEffects = [];
        _dispersion = NumOps.One;
    }

    /// <summary>
    /// Adds a random intercept effect to the model.
    /// </summary>
    /// <param name="name">Name for this random effect.</param>
    /// <param name="groupColumnIndex">Column index of the grouping variable.</param>
    /// <returns>This instance for fluent chaining.</returns>
    public GeneralizedLinearMixedModel<T> AddRandomIntercept(string name, int groupColumnIndex)
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
    public GeneralizedLinearMixedModel<T> AddRandomSlope(
        string name,
        int groupColumnIndex,
        int[] slopeColumns,
        bool includeIntercept = true)
    {
        _randomEffects.Add(new RandomEffect<T>(name, groupColumnIndex, slopeColumns, includeIntercept));
        return this;
    }

    /// <summary>
    /// Trains the GLMM using Penalized Quasi-Likelihood (PQL) or Laplace approximation.
    /// </summary>
    /// <param name="x">Feature matrix (including grouping variables).</param>
    /// <param name="y">Response vector.</param>
    /// <summary>GLMM doesn't benefit from optimizer parameter injection.</summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (_randomEffects.Count == 0)
        {
            throw new InvalidOperationException("At least one random effect must be specified. Use AddRandomIntercept() or AddRandomSlope().");
        }

        // This method previously fitted ORDINARY LEAST SQUARES and returned immediately —
        // `_useOLS = true` was set unconditionally, making the generalized mixed-effects estimation below
        // unreachable. A caller asking for a generalized linear mixed model received a plain linear fit, so neither the link function nor the random effects had any effect.
        // The real estimation now runs.
        TrainingFeatureCount = x.Columns;

        _nObservations = x.Rows;
        _nFixedParams = x.Columns - GetGroupingColumnCount();

        // Extract fixed effects design matrix (exclude grouping columns)
        var fixedX = ExtractFixedEffectsMatrix(x);

        // Initialize parameters
        InitializeParameters(fixedX, y);

        // Fit using PQL (Penalized Quasi-Likelihood)
        if (_options.EstimationMethod == GLMMEstimationMethod.PQL)
        {
            FitPQL(fixedX, x, y);
        }
        else
        {
            // Laplace approximation
            FitLaplace(fixedX, x, y);
        }

        // Compute variance decomposition
        _varianceDecomposition = ComputeVarianceDecomposition();

        // Surface the fitted parameters on the base class. The fixed-effects vector carries the
        // intercept as its LAST entry (see ExtractFixedEffectsMatrix), so split it out rather than
        // reporting an intercept of zero alongside a coefficient vector that secretly contains one.
        if (_fixedEffects is not null && _fixedEffects.Length > 0)
        {
            int featureCount = _fixedEffects.Length - 1;
            Coefficients = _fixedEffects.GetSubVector(0, featureCount);
            Intercept = _fixedEffects[featureCount];
        }
        else
        {
            Coefficients = new Vector<T>(Math.Max(0, _nFixedParams - 1));
            Intercept = NumOps.Zero;
        }
    }

    /// <summary>
    /// Makes predictions for new data on the response scale.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <returns>Predicted values on response scale.</returns>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var linearPredictor = PredictLinearPredictor(input);
        return ApplyInverseLink(linearPredictor);
    }

    /// <summary>
    /// Gets predictions on the linear predictor scale (before applying inverse link).
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <returns>Linear predictor values.</returns>
    public Vector<T> PredictLinearPredictor(Matrix<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var fixedX = ExtractFixedEffectsMatrix(input);

        // Fixed effects contribution
        var linearPred = FixedPredictions(fixedX, _fixedEffects);

        // Add random effects (BLUPs) if available
        foreach (var re in _randomEffects)
        {
            if (re.GroupColumnIndex < input.Columns)
            {
                linearPred = Engine.Add(linearPred, RandomEffectContributions(input, fixedX, re));
            }
        }

        return linearPred;
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
    /// Gets predicted probabilities for binary classification (logistic GLMM).
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <returns>Predicted probabilities.</returns>
    public Vector<T> PredictProbability(Matrix<T> input)
    {
        if (_options.Family != GLMMFamily.Binomial)
        {
            throw new InvalidOperationException("PredictProbability is only available for Binomial family.");
        }
        return Predict(input);
    }

    /// <summary>
    /// Applies the inverse link function to transform from linear predictor to mean.
    /// </summary>
    private Vector<T> ApplyInverseLink(Vector<T> eta)
    {
        int n = eta.Length;
        var ones = Engine.Fill(n, NumOps.One);
        T minusOne = NumOps.Negate(NumOps.One);
        return _options.LinkFunction switch
        {
            GLMMLinkFunction.Identity => new Vector<T>(eta),
            GLMMLinkFunction.Logit => Engine.Reciprocal(Engine.Add(ones, Engine.Exp(Engine.Multiply(eta, minusOne)))),
            GLMMLinkFunction.Log => Engine.Exp(eta),
            GLMMLinkFunction.Probit => NormalCDF(eta),
            GLMMLinkFunction.CLogLog => Engine.Subtract(ones, Engine.Exp(Engine.Multiply(Engine.Exp(eta), minusOne))),
            GLMMLinkFunction.Inverse => Engine.Reciprocal(eta),
            GLMMLinkFunction.Sqrt => Engine.Multiply(eta, eta),
            _ => new Vector<T>(eta)
        };
    }

    /// <summary>
    /// Applies the link function to transform from mean to linear predictor.
    /// </summary>
    private Vector<T> ApplyLink(Vector<T> mu)
    {
        // Prevent extreme values
        mu = ClampUnit(mu);
        var ones = Engine.Fill(mu.Length, NumOps.One);
        return _options.LinkFunction switch
        {
            GLMMLinkFunction.Identity => mu,
            GLMMLinkFunction.Logit => Engine.Log(Engine.Divide(mu, Engine.Subtract(ones, mu))),
            GLMMLinkFunction.Log => Engine.Log(mu),
            GLMMLinkFunction.Probit => NormalQuantile(mu),
            GLMMLinkFunction.CLogLog => Engine.Log(Engine.Multiply(Engine.Log(Engine.Subtract(ones, mu)), NumOps.Negate(NumOps.One))),
            GLMMLinkFunction.Inverse => Engine.Reciprocal(mu),
            GLMMLinkFunction.Sqrt => Engine.Sqrt(mu),
            _ => mu
        };
    }

    /// <summary>
    /// Clamps every value into [1e-10, 1 - 1e-10].
    /// </summary>
    private Vector<T> ClampUnit(Vector<T> values)
    {
        return Engine.Clamp(values, NumOps.FromDouble(1e-10), NumOps.FromDouble(1 - 1e-10));
    }

    /// <summary>
    /// Computes the derivative of the link function.
    /// </summary>
    private Vector<T> LinkDerivative(Vector<T> mu)
    {
        // Prevent extreme values
        mu = ClampUnit(mu);
        int n = mu.Length;
        var ones = Engine.Fill(n, NumOps.One);
        var oneMinusMu = Engine.Subtract(ones, mu);
        return _options.LinkFunction switch
        {
            GLMMLinkFunction.Identity => ones,
            GLMMLinkFunction.Logit => Engine.Reciprocal(Engine.Multiply(mu, oneMinusMu)),
            GLMMLinkFunction.Log => Engine.Reciprocal(mu),
            GLMMLinkFunction.Probit => Engine.Reciprocal(NormalPDF(NormalQuantile(mu))),
            GLMMLinkFunction.CLogLog => Engine.Reciprocal(Engine.Multiply(oneMinusMu, Engine.Log(oneMinusMu))),
            GLMMLinkFunction.Inverse => Engine.Divide(Engine.Fill(n, NumOps.Negate(NumOps.One)), Engine.Multiply(mu, mu)),
            GLMMLinkFunction.Sqrt => Engine.Divide(Engine.Fill(n, NumOps.FromDouble(0.5)), Engine.Sqrt(mu)),
            _ => ones
        };
    }

    /// <summary>
    /// Computes the variance function for the specified family.
    /// </summary>
    private Vector<T> VarianceFunction(Vector<T> mu)
    {
        // Prevent extreme values
        mu = ClampUnit(mu);
        var ones = Engine.Fill(mu.Length, NumOps.One);
        return _options.Family switch
        {
            GLMMFamily.Gaussian => ones,
            GLMMFamily.Binomial => Engine.Multiply(mu, Engine.Subtract(ones, mu)),
            GLMMFamily.Poisson => mu,
            GLMMFamily.Gamma => Engine.Multiply(mu, mu),
            GLMMFamily.InverseGaussian => Engine.Multiply(Engine.Multiply(mu, mu), mu),
            GLMMFamily.NegativeBinomial => Engine.Add(mu, Engine.Divide(Engine.Multiply(mu, mu), _options.NegBinomialTheta)),
            _ => ones
        };
    }

    /// <summary>
    /// Fits the model using Penalized Quasi-Likelihood.
    /// </summary>
    private void FitPQL(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        // PQL iterates between:
        // 1. Computing working response and weights
        // 2. Fitting a linear mixed model to the working response
        T tolerance = NumOps.FromDouble(_options.Tolerance);
        var observedY = y.GetSubVector(0, _nObservations);

        for (int outerIter = 0; outerIter < _options.MaxIterations; outerIter++)
        {
            // Compute linear predictor: fixed effects plus each random intercept
            var eta = new Vector<T>(_nObservations);
            if (_fixedEffects != null)
            {
                eta = FixedPredictions(fixedX, _fixedEffects).GetSubVector(0, _nObservations);
                foreach (var re in _randomEffects)
                {
                    eta = Engine.Add(eta, RowInterceptEffects(fullX, re).GetSubVector(0, _nObservations));
                }
            }

            // Compute fitted values
            var mu = ApplyInverseLink(eta);

            // Working response z = eta + (y - mu) g'(mu) and weight w = 1 / (g'(mu)^2 V(mu))
            var gPrime = LinkDerivative(mu);
            var v = VarianceFunction(mu);
            var workingY = Engine.Add(eta, Engine.Multiply(Engine.Subtract(observedY, mu), gPrime));
            var weights = Engine.Reciprocal(Engine.Multiply(Engine.Multiply(gPrime, gPrime), v));

            // Fit weighted linear mixed model
            FitWeightedLME(fixedX, fullX, workingY, weights);

            // Compute log-likelihood
            T newLogLik = ComputeLogLikelihood(y, mu);

            // Check convergence
            if (outerIter > 0 && ChangedLessThan(newLogLik, _logLikelihood, tolerance))
            {
                _logLikelihood = newLogLik;
                break;
            }

            _logLikelihood = newLogLik;
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
    /// Fits the model using Laplace approximation.
    /// </summary>
    private void FitLaplace(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        // Laplace approximation is similar to PQL but with better log-likelihood approximation
        // For simplicity, we use PQL as the core algorithm
        FitPQL(fixedX, fullX, y);
    }

    /// <summary>
    /// Fits a weighted linear mixed effects model (inner loop of PQL).
    /// </summary>
    private void FitWeightedLME(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> workingY, Vector<T> weights)
    {
        // Apply weights by transforming the problem: rows of X and y scaled by sqrt(w)
        var sqrtW = Engine.Sqrt(weights);

        // Weight-transform the design matrix and response
        var wX = fixedX.Rows == 0 || fixedX.Columns == 0
            ? new Matrix<T>(fixedX.Rows, fixedX.Columns)
            : Engine.TensorMultiply(
                Tensor<T>.FromMatrix(fixedX),
                Engine.TensorBroadcastTo(new Tensor<T>(new[] { fixedX.Rows, 1 }, sqrtW.GetSubVector(0, fixedX.Rows)),
                    new[] { fixedX.Rows, fixedX.Columns })).ToMatrix();
        var wY = Engine.Multiply(workingY, sqrtW);

        // Compute BLUPs
        ComputeBLUPs(wX, fullX, wY, weights);

        // Update fixed effects
        UpdateFixedEffects(wX, fullX, wY, weights);

        // Update variance components
        UpdateVarianceComponents(wX, fullX, wY, weights);
    }

    /// <summary>
    /// Computes BLUPs for random effects.
    /// </summary>
    private void ComputeBLUPs(Matrix<T> wX, Matrix<T> fullX, Vector<T> wY, Vector<T> weights)
    {
        if (_fixedEffects == null) return;

        // Compute weighted residuals from fixed effects
        var residuals = Engine.Subtract(
            wY.GetSubVector(0, _nObservations), FixedPredictions(wX, _fixedEffects).GetSubVector(0, _nObservations));

        foreach (var re in _randomEffects)
        {
            var groups = GroupObservations(fullX, re.GroupColumnIndex);
            T sigmaU = re.CovarianceMatrix != null ? FirstDiagonal(re.CovarianceMatrix) : NumOps.One;

            foreach (var kvp in groups)
            {
                int[] indices = kvp.Value.ToArray();
                var groupWeights = RegressionEngineOps.Gather(weights, indices);

                // Weighted BLUP: shrinkage * sum(w r) / sum(w), shrinkage = sigmaU / (sigmaU + 1 / sum(w))
                T sumWResid = Engine.DotProduct(groupWeights, RegressionEngineOps.Gather(residuals, indices));
                T sumW = Engine.Sum(groupWeights);
                T shrinkage = NumOps.Divide(sigmaU, NumOps.Add(sigmaU, NumOps.Divide(NumOps.One, sumW)));
                T blup = NumOps.Divide(NumOps.Multiply(shrinkage, sumWResid), sumW);

                var blupVector = new Vector<T>(re.Dimension);
                if (re.IsRandomIntercept && re.Dimension > 0)
                {
                    blupVector = Vector<T>.Concatenate(new[] { Engine.Fill(1, blup), new Vector<T>(re.Dimension - 1) });
                }

                re.SetGroupEffect(kvp.Key, blupVector);
            }

            CentreRandomIntercepts(re, groups);
        }
    }

    /// <summary>
    /// Recentres a random intercept's BLUPs so they average zero across groups.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A random effect is DEFINED as a mean-zero deviation around the fixed intercept. The
    /// alternating scheme here - estimate BLUPs from the residuals, then refit the fixed effects
    /// against the response minus those BLUPs - has no term that enforces that, so it can settle at
    /// a fixed point where the overall level is split between the two.
    /// </para>
    /// <para>
    /// Measured, that is exactly what happened: shifting every response by 1000 moved the fixed
    /// intercept by only 499.99, leaving the other half in per-group intercepts. Those do not
    /// transfer to a group the model has not seen, so predictions for new groups lost half the
    /// shift. Subtracting the mean here puts the level back where it belongs; the next
    /// UpdateFixedEffects picks it up, because it fits against the response minus the recentred
    /// BLUPs.
    /// </para>
    /// </remarks>
    private void CentreRandomIntercepts(RandomEffect<T> re, Dictionary<double, List<int>> groups)
    {
        if (!re.IsRandomIntercept || groups.Count == 0) return;

        // Mean random intercept over the groups that have one
        var effects = groups.Keys
            .Select(groupId => (groupId, blup: re.GetGroupEffect(groupId)))
            .Where(entry => entry.blup.Length > 0)
            .ToList();
        if (effects.Count == 0) return;

        var intercepts = Vector<T>.Concatenate(effects.Select(entry => entry.blup.GetSubVector(0, 1)).ToList());
        T mean = NumOps.Divide(Engine.Sum(intercepts), NumOps.FromDouble(effects.Count));
        if (NumOps.Equals(mean, NumOps.Zero)) return;

        // Subtract the mean from every intercept (the slopes are left as they are)
        foreach (var (groupId, blup) in effects)
        {
            var shift = Vector<T>.Concatenate(new[] { Engine.Fill(1, mean), new Vector<T>(blup.Length - 1) });
            re.SetGroupEffect(groupId, Engine.Subtract(blup, shift));
        }
    }

    /// <summary>
    /// Updates fixed effects coefficients.
    /// </summary>
    private void UpdateFixedEffects(Matrix<T> wX, Matrix<T> fullX, Vector<T> wY, Vector<T> weights)
    {
        // Adjust y for random effects (on the sqrt(w) scale of the working response)
        var sqrtW = Engine.Sqrt(weights);
        var adjustedY = new Vector<T>(wY);
        foreach (var re in _randomEffects)
        {
            adjustedY = Engine.Subtract(adjustedY, Engine.Multiply(RowInterceptEffects(fullX, re), sqrtW));
        }

        _fixedEffects = SolveOLS(wX, adjustedY);
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private void UpdateVarianceComponents(Matrix<T> wX, Matrix<T> fullX, Vector<T> wY, Vector<T> weights)
    {
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
    private T ComputeLogLikelihood(Vector<T> y, Vector<T> mu)
    {
        int n = _nObservations;
        if (n == 0)
        {
            return NumOps.Zero;
        }

        var yVal = y.GetSubVector(0, n);
        var muVal = ClampUnit(mu.GetSubVector(0, n));
        var ones = Engine.Fill(n, NumOps.One);
        var diff = Engine.Subtract(yVal, muVal);
        var gaussian = Engine.Multiply(Engine.Multiply(diff, NumOps.FromDouble(-0.5)), diff);

        var terms = _options.Family switch
        {
            GLMMFamily.Gaussian => gaussian,
            GLMMFamily.Binomial => Engine.Add(
                Engine.Multiply(yVal, Engine.Log(muVal)),
                Engine.Multiply(Engine.Subtract(ones, yVal), Engine.Log(Engine.Subtract(ones, muVal)))),
            GLMMFamily.Poisson => Engine.Subtract(
                Engine.Subtract(Engine.Multiply(yVal, Engine.Log(muVal)), muVal),
                LogGamma(Engine.Add(yVal, ones))),
            _ => gaussian
        };

        return Engine.Sum(terms);
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
                Variance = _dispersion
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
    /// Gets the number of parameters in the model.
    /// </summary>
    private int GetNumberOfParameters()
    {
        int nParams = _nFixedParams; // Fixed effects
        nParams++; // Dispersion parameter

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

        // The non-group feature columns in order, then the intercept column of ones LAST
        var columns = new List<Vector<T>>();
        for (int j = 0; j < x.Columns; j++)
        {
            if (!groupCols.Contains(j))
            {
                columns.Add(Engine.GetColumn(x, j));
            }
        }

        columns.Add(Engine.Fill(x.Rows, NumOps.One));

        if (x.Rows == 0)
        {
            return new Matrix<T>(0, columns.Count);
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

        // Initial fixed effects: OLS on the link-transformed response
        var etaY = ApplyLink(y);
        _fixedEffects = SolveOLS(fixedX, etaY);

        // Initial random effect covariances: identity
        foreach (var re in _randomEffects)
        {
            re.CovarianceMatrix = Matrix<T>.CreateDiagonal(Engine.Fill(re.Dimension, NumOps.One));
        }
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
            if (!groups.TryGetValue(groupId, out List<int>? value))
            {
                value = [];
                groups[groupId] = value;
            }

            value.Add(i);
        }

        return groups;
    }

    /// <summary>
    /// Solves ordinary least squares.
    /// </summary>
    private Vector<T> SolveOLS(Matrix<T> x, Vector<T> y)
    {
        var xtx = x.Transpose().Multiply(x);
        var xty = x.Transpose().Multiply(y);

        // Small ridge on the diagonal for stability
        xtx = Engine.MatrixAdd(xtx, Matrix<T>.CreateDiagonal(Engine.Fill(xtx.Rows, NumOps.FromDouble(1e-10))));

        return xtx.Inverse().Multiply(xty);
    }

    /// <summary>
    /// Standard normal CDF approximation.
    /// </summary>
    private Vector<T> NormalCDF(Vector<T> x)
    {
        // 0.5 * (1 + erf(x / sqrt(2)))
        var erf = Erf(Engine.Divide(x, NumOps.FromDouble(Math.Sqrt(2))));
        return Engine.Multiply(Engine.Add(Engine.Fill(x.Length, NumOps.One), erf), NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Standard normal PDF.
    /// </summary>
    private Vector<T> NormalPDF(Vector<T> x)
    {
        // exp(-0.5 x x) / sqrt(2 pi)
        var exponent = Engine.Multiply(Engine.Multiply(x, NumOps.FromDouble(-0.5)), x);
        return Engine.Divide(Engine.Exp(exponent), NumOps.FromDouble(Math.Sqrt(2 * Math.PI)));
    }

    /// <summary>
    /// Standard normal quantile (inverse CDF) approximation.
    /// </summary>
    private Vector<T> NormalQuantile(Vector<T> p)
    {
        // Abramowitz and Stegun approximation; -inf at p <= 0, +inf at p >= 1
        int n = p.Length;
        var ones = Engine.Fill(n, NumOps.One);
        var pTensor = Tensor<T>.FromVector(p);
        var lowerTail = Engine.TensorLessThan(pTensor, NumOps.FromDouble(0.5));
        var tail = Engine.TensorWhere(lowerTail, pTensor, Tensor<T>.FromVector(Engine.Subtract(ones, p))).ToVector();
        var t = Engine.Sqrt(Engine.Multiply(Engine.Log(tail), NumOps.FromDouble(-2.0)));

        T c0 = NumOps.FromDouble(2.515517), c1 = NumOps.FromDouble(0.802853), c2 = NumOps.FromDouble(0.010328);
        T d1 = NumOps.FromDouble(1.432788), d2 = NumOps.FromDouble(0.189269), d3 = NumOps.FromDouble(0.001308);

        var numerator = Engine.Add(
            Engine.Add(Engine.Fill(n, c0), Engine.Multiply(t, c1)),
            Engine.Multiply(Engine.Multiply(t, c2), t));
        var denominator = Engine.Add(
            Engine.Add(Engine.Add(ones, Engine.Multiply(t, d1)), Engine.Multiply(Engine.Multiply(t, d2), t)),
            Engine.Multiply(Engine.Multiply(Engine.Multiply(t, d3), t), t));

        var result = Tensor<T>.FromVector(Engine.Subtract(t, Engine.Divide(numerator, denominator)));
        var signed = Engine.TensorWhere(lowerTail, Engine.TensorNegate(result), result);

        var negativeInfinity = Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.NegativeInfinity)));
        var positiveInfinity = Tensor<T>.FromVector(Engine.Fill(n, NumOps.FromDouble(double.PositiveInfinity)));
        signed = Engine.TensorWhere(Engine.TensorGreaterEqual(pTensor, Tensor<T>.FromVector(ones)), positiveInfinity, signed);
        signed = Engine.TensorWhere(
            Engine.TensorLessEqual(pTensor, Tensor<T>.FromVector(Engine.FillZero(n))), negativeInfinity, signed);
        return signed.ToVector();
    }

    /// <summary>
    /// Error function approximation.
    /// </summary>
    private Vector<T> Erf(Vector<T> x)
    {
        // Abramowitz and Stegun / Numerical Recipes erfc approximation
        int n = x.Length;
        var t = Engine.Reciprocal(Engine.Add(Engine.Fill(n, NumOps.One), Engine.Multiply(Engine.Abs(x), NumOps.FromDouble(0.5))));

        // Horner from the innermost coefficient outwards: c_k + t * (...)
        double[] coefficients =
        {
            -0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418, 0.37409196, 1.00002368
        };
        var horner = Engine.Multiply(t, NumOps.FromDouble(0.17087277));
        for (int k = 0; k < coefficients.Length; k++)
        {
            var level = Engine.Add(Engine.Fill(n, NumOps.FromDouble(coefficients[k])), horner);
            horner = k == coefficients.Length - 1 ? level : Engine.Multiply(t, level);
        }

        // tau = t * exp(-x x - 1.26551223 + t * horner)
        var exponent = Engine.Subtract(
            Engine.Multiply(Engine.Multiply(x, NumOps.Negate(NumOps.One)), x), Engine.Fill(n, NumOps.FromDouble(1.26551223)));
        exponent = Engine.Add(exponent, Engine.Multiply(t, horner));
        var tau = Engine.Multiply(t, Engine.Exp(exponent));

        // x >= 0 ? 1 - tau : tau - 1
        var ones = Engine.Fill(n, NumOps.One);
        var nonNegative = Engine.TensorGreaterEqual(Tensor<T>.FromVector(x), Tensor<T>.FromVector(Engine.FillZero(n)));
        return Engine.TensorWhere(
            nonNegative,
            Tensor<T>.FromVector(Engine.Subtract(ones, tau)),
            Tensor<T>.FromVector(Engine.Subtract(tau, ones))).ToVector();
    }

    /// <summary>
    /// Log-gamma function approximation.
    /// </summary>
    private Vector<T> LogGamma(Vector<T> x)
    {
        // Stirling's approximation; 0 for x <= 0
        int n = x.Length;
        var series = Engine.Subtract(
            Engine.Multiply(Engine.Subtract(x, Engine.Fill(n, NumOps.FromDouble(0.5))), Engine.Log(x)), x);
        series = Engine.Add(series, Engine.Fill(n, NumOps.FromDouble(0.5 * Math.Log(2 * Math.PI))));
        series = Engine.Add(series, Engine.Reciprocal(Engine.Multiply(x, NumOps.FromDouble(12.0))));
        series = Engine.Subtract(series, Engine.Reciprocal(
            Engine.Multiply(Engine.Multiply(Engine.Multiply(x, NumOps.FromDouble(360.0)), x), x)));

        return Engine.TensorWhere(
            RegressionEngineOps.LessOrEqualMask(x, NumOps.Zero),
            Tensor<T>.FromVector(Engine.FillZero(n)),
            Tensor<T>.FromVector(series)).ToVector();
    }
}
