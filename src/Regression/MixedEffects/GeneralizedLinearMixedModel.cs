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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GeneralizedLinearMixedModel<T> : RegressionBase<T>
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
    private readonly List<RandomEffect<T>> _randomEffects;

    /// <summary>
    /// Fixed effects coefficients.
    /// </summary>
    private Vector<T>? _fixedEffects;

    /// <summary>
    /// Variance decomposition results.
    /// </summary>
    private VarianceDecomposition<T>? _varianceDecomposition;

    /// <summary>
    /// Dispersion parameter (for overdispersed models).
    /// </summary>
    private T _dispersion = default!;

    /// <summary>
    /// Log-likelihood of the fitted model.
    /// </summary>
    private double _logLikelihood;

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
    public double LogLikelihood => _logLikelihood;

    /// <summary>
    /// Gets the AIC (Akaike Information Criterion).
    /// </summary>
    public double AIC => -2 * _logLikelihood + 2 * GetNumberOfParameters();

    /// <summary>
    /// Gets the BIC (Bayesian Information Criterion).
    /// </summary>
    public double BIC => -2 * _logLikelihood + Math.Log(_nObservations) * GetNumberOfParameters();

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
        : base(options ?? new GLMMOptions<T>(), regularization)
    {
        _options = options ?? new GLMMOptions<T>();
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
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (_randomEffects.Count == 0)
        {
            throw new InvalidOperationException("At least one random effect must be specified. Use AddRandomIntercept() or AddRandomSlope().");
        }

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

        // Set base class coefficients for prediction
        Coefficients = _fixedEffects ?? new Vector<T>(_nFixedParams);
        Intercept = NumOps.Zero;
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

        int n = input.Rows;
        var linearPred = new Vector<T>(n);
        var fixedX = ExtractFixedEffectsMatrix(input);

        for (int i = 0; i < n; i++)
        {
            // Fixed effects contribution
            T eta = NumOps.Zero;
            for (int j = 0; j < _nFixedParams; j++)
            {
                eta = NumOps.Add(eta, NumOps.Multiply(fixedX[i, j], _fixedEffects[j]));
            }

            // Add random effects (BLUPs) if available
            foreach (var re in _randomEffects)
            {
                int groupCol = re.GroupColumnIndex;
                if (groupCol < input.Columns)
                {
                    double groupId = NumOps.ToDouble(input[i, groupCol]);
                    var groupEffect = re.GetGroupEffect(groupId);

                    // Random intercept
                    if (re.IsRandomIntercept && groupEffect.Length > 0)
                    {
                        eta = NumOps.Add(eta, groupEffect[0]);
                    }

                    // Random slopes
                    if (re.RandomSlopeColumns != null)
                    {
                        int offset = re.IsRandomIntercept ? 1 : 0;
                        for (int s = 0; s < re.RandomSlopeColumns.Length; s++)
                        {
                            if (offset + s < groupEffect.Length)
                            {
                                int slopeCol = re.RandomSlopeColumns[s];
                                if (slopeCol < fixedX.Columns)
                                {
                                    eta = NumOps.Add(eta,
                                        NumOps.Multiply(fixedX[i, slopeCol], groupEffect[offset + s]));
                                }
                            }
                        }
                    }
                }
            }

            linearPred[i] = eta;
        }

        return linearPred;
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
        var result = new Vector<T>(eta.Length);

        for (int i = 0; i < eta.Length; i++)
        {
            double etaVal = NumOps.ToDouble(eta[i]);
            double mu = _options.LinkFunction switch
            {
                GLMMLinkFunction.Identity => etaVal,
                GLMMLinkFunction.Logit => 1.0 / (1.0 + Math.Exp(-etaVal)),
                GLMMLinkFunction.Log => Math.Exp(etaVal),
                GLMMLinkFunction.Probit => NormalCDF(etaVal),
                GLMMLinkFunction.CLogLog => 1.0 - Math.Exp(-Math.Exp(etaVal)),
                GLMMLinkFunction.Inverse => 1.0 / etaVal,
                GLMMLinkFunction.Sqrt => etaVal * etaVal,
                _ => etaVal
            };
            result[i] = NumOps.FromDouble(mu);
        }

        return result;
    }

    /// <summary>
    /// Applies the link function to transform from mean to linear predictor.
    /// </summary>
    private T ApplyLink(T mu)
    {
        double muVal = NumOps.ToDouble(mu);

        // Prevent extreme values
        muVal = Math.Max(1e-10, Math.Min(1 - 1e-10, muVal));

        double eta = _options.LinkFunction switch
        {
            GLMMLinkFunction.Identity => muVal,
            GLMMLinkFunction.Logit => Math.Log(muVal / (1.0 - muVal)),
            GLMMLinkFunction.Log => Math.Log(muVal),
            GLMMLinkFunction.Probit => NormalQuantile(muVal),
            GLMMLinkFunction.CLogLog => Math.Log(-Math.Log(1.0 - muVal)),
            GLMMLinkFunction.Inverse => 1.0 / muVal,
            GLMMLinkFunction.Sqrt => Math.Sqrt(muVal),
            _ => muVal
        };

        return NumOps.FromDouble(eta);
    }

    /// <summary>
    /// Computes the derivative of the link function.
    /// </summary>
    private double LinkDerivative(double mu)
    {
        // Prevent extreme values
        mu = Math.Max(1e-10, Math.Min(1 - 1e-10, mu));

        return _options.LinkFunction switch
        {
            GLMMLinkFunction.Identity => 1.0,
            GLMMLinkFunction.Logit => 1.0 / (mu * (1.0 - mu)),
            GLMMLinkFunction.Log => 1.0 / mu,
            GLMMLinkFunction.Probit => 1.0 / NormalPDF(NormalQuantile(mu)),
            GLMMLinkFunction.CLogLog => 1.0 / ((1.0 - mu) * Math.Log(1.0 - mu)),
            GLMMLinkFunction.Inverse => -1.0 / (mu * mu),
            GLMMLinkFunction.Sqrt => 0.5 / Math.Sqrt(mu),
            _ => 1.0
        };
    }

    /// <summary>
    /// Computes the variance function for the specified family.
    /// </summary>
    private double VarianceFunction(double mu)
    {
        // Prevent extreme values
        mu = Math.Max(1e-10, Math.Min(1 - 1e-10, mu));

        return _options.Family switch
        {
            GLMMFamily.Gaussian => 1.0,
            GLMMFamily.Binomial => mu * (1.0 - mu),
            GLMMFamily.Poisson => mu,
            GLMMFamily.Gamma => mu * mu,
            GLMMFamily.InverseGaussian => mu * mu * mu,
            GLMMFamily.NegativeBinomial => mu + mu * mu / NumOps.ToDouble(_options.NegBinomialTheta),
            _ => 1.0
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

        for (int outerIter = 0; outerIter < _options.MaxIterations; outerIter++)
        {
            // Compute linear predictor
            var eta = new Vector<T>(_nObservations);
            if (_fixedEffects != null)
            {
                for (int i = 0; i < _nObservations; i++)
                {
                    T pred = NumOps.Zero;
                    for (int j = 0; j < _nFixedParams; j++)
                    {
                        pred = NumOps.Add(pred, NumOps.Multiply(fixedX[i, j], _fixedEffects[j]));
                    }

                    // Add random effects
                    foreach (var re in _randomEffects)
                    {
                        double groupId = NumOps.ToDouble(fullX[i, re.GroupColumnIndex]);
                        var blup = re.GetGroupEffect(groupId);
                        if (blup.Length > 0)
                        {
                            pred = NumOps.Add(pred, blup[0]);
                        }
                    }

                    eta[i] = pred;
                }
            }

            // Compute fitted values
            var mu = ApplyInverseLink(eta);

            // Compute working response and weights
            var workingY = new Vector<T>(_nObservations);
            var weights = new Vector<T>(_nObservations);

            for (int i = 0; i < _nObservations; i++)
            {
                double muVal = NumOps.ToDouble(mu[i]);
                double yVal = NumOps.ToDouble(y[i]);
                double etaVal = NumOps.ToDouble(eta[i]);

                double gPrime = LinkDerivative(muVal);
                double v = VarianceFunction(muVal);

                // Working response: z = eta + (y - mu) * g'(mu)
                double z = etaVal + (yVal - muVal) * gPrime;

                // Working weight: w = 1 / (g'(mu)^2 * V(mu))
                double w = 1.0 / (gPrime * gPrime * v);

                workingY[i] = NumOps.FromDouble(z);
                weights[i] = NumOps.FromDouble(w);
            }

            // Fit weighted linear mixed model
            FitWeightedLME(fixedX, fullX, workingY, weights);

            // Compute log-likelihood
            double newLogLik = ComputeLogLikelihood(y, mu);

            // Check convergence
            if (outerIter > 0 && Math.Abs(newLogLik - _logLikelihood) < _options.Tolerance)
            {
                _logLikelihood = newLogLik;
                break;
            }

            _logLikelihood = newLogLik;
        }
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
        // Apply weights by transforming the problem
        var sqrtW = new Vector<T>(weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            sqrtW[i] = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(weights[i])));
        }

        // Weight-transform the design matrix and response
        var wX = new Matrix<T>(fixedX.Rows, fixedX.Columns);
        var wY = new Vector<T>(workingY.Length);

        for (int i = 0; i < fixedX.Rows; i++)
        {
            double w = NumOps.ToDouble(sqrtW[i]);
            for (int j = 0; j < fixedX.Columns; j++)
            {
                wX[i, j] = NumOps.FromDouble(NumOps.ToDouble(fixedX[i, j]) * w);
            }
            wY[i] = NumOps.FromDouble(NumOps.ToDouble(workingY[i]) * w);
        }

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
        var residuals = new Vector<T>(_nObservations);
        for (int i = 0; i < _nObservations; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < _nFixedParams; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(wX[i, j], _fixedEffects[j]));
            }
            residuals[i] = NumOps.Subtract(wY[i], pred);
        }

        foreach (var re in _randomEffects)
        {
            var groups = GroupObservations(fullX, re.GroupColumnIndex);

            double sigmaU = re.CovarianceMatrix != null ? NumOps.ToDouble(re.CovarianceMatrix[0, 0]) : 1.0;

            foreach (var kvp in groups)
            {
                double groupId = kvp.Key;
                var indices = kvp.Value;

                // Compute weighted sum of residuals
                double sumWResid = 0;
                double sumW = 0;
                foreach (int idx in indices)
                {
                    double w = NumOps.ToDouble(weights[idx]);
                    sumWResid += w * NumOps.ToDouble(residuals[idx]);
                    sumW += w;
                }

                // BLUP with shrinkage
                double shrinkage = sigmaU / (sigmaU + 1.0 / sumW);
                double blup = shrinkage * sumWResid / sumW;

                var blupVector = new Vector<T>(re.Dimension);
                if (re.IsRandomIntercept)
                {
                    blupVector[0] = NumOps.FromDouble(blup);
                }

                re.SetGroupEffect(groupId, blupVector);
            }
        }
    }

    /// <summary>
    /// Updates fixed effects coefficients.
    /// </summary>
    private void UpdateFixedEffects(Matrix<T> wX, Matrix<T> fullX, Vector<T> wY, Vector<T> weights)
    {
        // Adjust working response for random effects
        var adjustedY = new Vector<T>(wY.Length);
        for (int i = 0; i < wY.Length; i++)
        {
            adjustedY[i] = wY[i];

            double sqrtW = Math.Sqrt(NumOps.ToDouble(weights[i]));
            foreach (var re in _randomEffects)
            {
                double groupId = NumOps.ToDouble(fullX[i, re.GroupColumnIndex]);
                var blup = re.GetGroupEffect(groupId);
                if (blup.Length > 0)
                {
                    adjustedY[i] = NumOps.Subtract(adjustedY[i],
                        NumOps.FromDouble(NumOps.ToDouble(blup[0]) * sqrtW));
                }
            }
        }

        // Update fixed effects using weighted OLS
        _fixedEffects = SolveOLS(wX, adjustedY);
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private void UpdateVarianceComponents(Matrix<T> wX, Matrix<T> fullX, Vector<T> wY, Vector<T> weights)
    {
        foreach (var re in _randomEffects)
        {
            var groups = GroupObservations(fullX, re.GroupColumnIndex);
            double sumSqBlup = 0;

            foreach (var kvp in groups)
            {
                var blup = re.GetGroupEffect(kvp.Key);
                if (blup.Length > 0)
                {
                    double b = NumOps.ToDouble(blup[0]);
                    sumSqBlup += b * b;
                }
            }

            double newVar = sumSqBlup / groups.Count;

            // Bound variance to be non-negative
            if (_options.BoundVarianceComponents && newVar < 0)
            {
                newVar = 0;
            }

            if (re.CovarianceMatrix != null && re.CovarianceMatrix.Rows > 0)
            {
                re.CovarianceMatrix[0, 0] = NumOps.FromDouble(newVar);
            }
        }
    }

    /// <summary>
    /// Computes the log-likelihood.
    /// </summary>
    private double ComputeLogLikelihood(Vector<T> y, Vector<T> mu)
    {
        double logLik = 0;

        for (int i = 0; i < _nObservations; i++)
        {
            double yVal = NumOps.ToDouble(y[i]);
            double muVal = NumOps.ToDouble(mu[i]);

            // Prevent extreme values
            muVal = Math.Max(1e-10, Math.Min(1 - 1e-10, muVal));

            logLik += _options.Family switch
            {
                GLMMFamily.Gaussian => -0.5 * (yVal - muVal) * (yVal - muVal),
                GLMMFamily.Binomial => yVal * Math.Log(muVal) + (1 - yVal) * Math.Log(1 - muVal),
                GLMMFamily.Poisson => yVal * Math.Log(muVal) - muVal - LogGamma(yVal + 1),
                _ => -0.5 * (yVal - muVal) * (yVal - muVal)
            };
        }

        return logLik;
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
                Variance = re.CovarianceMatrix != null ? re.CovarianceMatrix[0, 0] : NumOps.Zero,
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

        int nCols = x.Columns - groupCols.Count;
        var fixedX = new Matrix<T>(x.Rows, nCols);

        int colIdx = 0;
        for (int j = 0; j < x.Columns; j++)
        {
            if (!groupCols.Contains(j))
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    fixedX[i, colIdx] = x[i, j];
                }
                colIdx++;
            }
        }

        return fixedX;
    }

    /// <summary>
    /// Initializes model parameters.
    /// </summary>
    private void InitializeParameters(Matrix<T> fixedX, Vector<T> y)
    {
        _nFixedParams = fixedX.Columns;

        // Initialize fixed effects using GLM (without random effects)
        // Transform y to linear predictor scale
        var etaY = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            etaY[i] = ApplyLink(y[i]);
        }

        // Initialize fixed effects using OLS on transformed response
        _fixedEffects = SolveOLS(fixedX, etaY);

        // Initialize variance components
        foreach (var re in _randomEffects)
        {
            int dim = re.Dimension;
            re.CovarianceMatrix = new Matrix<T>(dim, dim);
            for (int d = 0; d < dim; d++)
            {
                re.CovarianceMatrix[d, d] = NumOps.One;
            }
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
    private static Vector<T> SolveOLS(Matrix<T> x, Vector<T> y)
    {
        // X'X
        var xtx = x.Transpose().Multiply(x);

        // X'y
        var xty = x.Transpose().Multiply(y);

        // Add small ridge for numerical stability
        for (int i = 0; i < xtx.Rows; i++)
        {
            xtx[i, i] = NumOps.Add(xtx[i, i], NumOps.FromDouble(1e-10));
        }

        // Solve (X'X)^-1 X'y
        return xtx.Inverse().Multiply(xty);
    }

    /// <summary>
    /// Standard normal CDF approximation.
    /// </summary>
    private static double NormalCDF(double x)
    {
        // Approximation using error function
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2)));
    }

    /// <summary>
    /// Standard normal PDF.
    /// </summary>
    private static double NormalPDF(double x)
    {
        return Math.Exp(-0.5 * x * x) / Math.Sqrt(2 * Math.PI);
    }

    /// <summary>
    /// Standard normal quantile (inverse CDF) approximation.
    /// </summary>
    private static double NormalQuantile(double p)
    {
        // Abramowitz and Stegun approximation
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;

        double t = Math.Sqrt(-2.0 * Math.Log(p < 0.5 ? p : 1 - p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
        return p < 0.5 ? -result : result;
    }

    /// <summary>
    /// Error function approximation.
    /// </summary>
    private static double Erf(double x)
    {
        // Abramowitz and Stegun approximation
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double tau = t * Math.Exp(-x * x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));
        return x >= 0 ? 1 - tau : tau - 1;
    }

    /// <summary>
    /// Log-gamma function approximation.
    /// </summary>
    private static double LogGamma(double x)
    {
        if (x <= 0) return 0;

        // Stirling's approximation
        return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI) +
               1.0 / (12.0 * x) - 1.0 / (360.0 * x * x * x);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.GeneralizedLinearMixedModel;

    /// <summary>
    /// Creates a new instance of the model with the same configuration.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new GeneralizedLinearMixedModel<T>(_options, Regularization);

        // Copy random effect specifications
        foreach (var re in _randomEffects)
        {
            if (re.RandomSlopeColumns != null)
            {
                newModel.AddRandomSlope(re.Name, re.GroupColumnIndex, re.RandomSlopeColumns, re.IsRandomIntercept);
            }
            else if (re.IsRandomIntercept)
            {
                newModel.AddRandomIntercept(re.Name, re.GroupColumnIndex);
            }
        }

        return newModel;
    }
}
