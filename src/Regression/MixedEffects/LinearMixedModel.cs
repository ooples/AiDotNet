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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LinearMixedModel<T> : RegressionBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected new static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly MixedEffectsOptions<T> _options;

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
    /// Residual variance estimate.
    /// </summary>
    private T _residualVariance = default!;

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
    /// Marginal R-squared (fixed effects only).
    /// </summary>
    public T MarginalRSquared { get; private set; } = default!;

    /// <summary>
    /// Conditional R-squared (fixed + random effects).
    /// </summary>
    public T ConditionalRSquared { get; private set; } = default!;

    /// <summary>
    /// Initializes a new Linear Mixed-Effects Model.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public LinearMixedModel(
        MixedEffectsOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new MixedEffectsOptions<T>(), regularization)
    {
        _options = options ?? new MixedEffectsOptions<T>();
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

        // Set base class coefficients for prediction
        Coefficients = _fixedEffects ?? new Vector<T>(_nFixedParams);
        Intercept = NumOps.Zero;
    }

    /// <summary>
    /// Makes predictions for new data.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <returns>Predicted values.</returns>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_fixedEffects == null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        int n = input.Rows;
        var predictions = new Vector<T>(n);
        var fixedX = ExtractFixedEffectsMatrix(input);

        for (int i = 0; i < n; i++)
        {
            // Fixed effects contribution
            T pred = NumOps.Zero;
            for (int j = 0; j < _nFixedParams; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(fixedX[i, j], _fixedEffects[j]));
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
                        pred = NumOps.Add(pred, groupEffect[0]);
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
                                    pred = NumOps.Add(pred,
                                        NumOps.Multiply(fixedX[i, slopeCol], groupEffect[offset + s]));
                                }
                            }
                        }
                    }
                }
            }

            predictions[i] = pred;
        }

        return predictions;
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

        // Initialize fixed effects using OLS
        _fixedEffects = SolveOLS(fixedX, y);

        // Initialize variance components
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);
        double totalVar = ComputeVariance(residuals);

        // Start with equal variance split
        _residualVariance = NumOps.FromDouble(totalVar * 0.5);

        foreach (var re in _randomEffects)
        {
            int dim = re.Dimension;
            re.CovarianceMatrix = new Matrix<T>(dim, dim);
            for (int d = 0; d < dim; d++)
            {
                re.CovarianceMatrix[d, d] = NumOps.FromDouble(totalVar * 0.5 / _randomEffects.Count);
            }
        }
    }

    /// <summary>
    /// Fits the model using the EM algorithm.
    /// </summary>
    private void FitEM(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        double prevLogLik = double.NegativeInfinity;

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            // E-step: Compute expected random effects (BLUPs)
            ComputeBLUPs(fixedX, fullX, y);

            // M-step: Update fixed effects and variance components
            UpdateFixedEffects(fixedX, fullX, y);
            UpdateVarianceComponents(fixedX, fullX, y);

            // Compute log-likelihood
            _logLikelihood = ComputeLogLikelihood(fixedX, fullX, y);

            if (_options.Verbose)
            {
                Console.WriteLine($"Iteration {iter + 1}: Log-likelihood = {_logLikelihood:F4}");
            }

            // Check convergence
            if (Math.Abs(_logLikelihood - prevLogLik) < _options.Tolerance)
            {
                break;
            }

            prevLogLik = _logLikelihood;
        }
    }

    /// <summary>
    /// Computes BLUPs (Best Linear Unbiased Predictors) for random effects.
    /// </summary>
    private void ComputeBLUPs(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        // Compute residuals from fixed effects
        if (_fixedEffects == null) return;
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);

        foreach (var re in _randomEffects)
        {
            // Group observations
            var groups = GroupObservations(fullX, re.GroupColumnIndex);

            double sigmaE = NumOps.ToDouble(_residualVariance);
            double sigmaU = re.CovarianceMatrix != null ? NumOps.ToDouble(re.CovarianceMatrix[0, 0]) : 1.0;

            foreach (var kvp in groups)
            {
                double groupId = kvp.Key;
                var indices = kvp.Value;
                int ni = indices.Count;

                // Simple random intercept BLUP: u_i = (sigma_u^2 / (sigma_u^2 + sigma_e^2/n_i)) * mean(residuals_i)
                double sumResid = 0;
                foreach (int idx in indices)
                {
                    sumResid += NumOps.ToDouble(residuals[idx]);
                }
                double meanResid = sumResid / ni;

                double shrinkage = sigmaU / (sigmaU + sigmaE / ni);
                double blup = shrinkage * meanResid;

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
    private void UpdateFixedEffects(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        // Adjust y for random effects
        var adjustedY = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            adjustedY[i] = y[i];

            foreach (var re in _randomEffects)
            {
                double groupId = NumOps.ToDouble(fullX[i, re.GroupColumnIndex]);
                var blup = re.GetGroupEffect(groupId);
                if (blup.Length > 0)
                {
                    adjustedY[i] = NumOps.Subtract(adjustedY[i], blup[0]);
                }
            }
        }

        // Update fixed effects using OLS on adjusted response
        _fixedEffects = SolveOLS(fixedX, adjustedY);
    }

    /// <summary>
    /// Updates variance components.
    /// </summary>
    private void UpdateVarianceComponents(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return;

        // Compute residuals
        var residuals = ComputeResiduals(fixedX, y, _fixedEffects);

        // Update residual variance
        double ssResid = 0;
        int dfResid = _nObservations - _nFixedParams;
        for (int i = 0; i < residuals.Length; i++)
        {
            double r = NumOps.ToDouble(residuals[i]);

            // Subtract random effects
            foreach (var re in _randomEffects)
            {
                double groupId = NumOps.ToDouble(fullX[i, re.GroupColumnIndex]);
                var blup = re.GetGroupEffect(groupId);
                if (blup.Length > 0)
                {
                    r -= NumOps.ToDouble(blup[0]);
                }
            }

            ssResid += r * r;
        }
        _residualVariance = NumOps.FromDouble(ssResid / dfResid);

        // Update random effect variances
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
    private double ComputeLogLikelihood(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return double.NegativeInfinity;

        double sigmaE = NumOps.ToDouble(_residualVariance);
        int n = _nObservations;

        // Compute residual sum of squares
        var predictions = Predict(fullX);
        double ssResid = 0;
        for (int i = 0; i < n; i++)
        {
            double r = NumOps.ToDouble(y[i]) - NumOps.ToDouble(predictions[i]);
            ssResid += r * r;
        }

        // Log-likelihood for normal model
        double logLik = -0.5 * n * Math.Log(2 * Math.PI * sigmaE) - ssResid / (2 * sigmaE);

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
                Variance = _residualVariance
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
    /// Computes marginal and conditional R-squared.
    /// </summary>
    private void ComputeRSquaredValues(Matrix<T> fixedX, Matrix<T> fullX, Vector<T> y)
    {
        if (_fixedEffects == null) return;

        // Total variance
        double yMean = 0;
        for (int i = 0; i < y.Length; i++)
        {
            yMean += NumOps.ToDouble(y[i]);
        }
        yMean /= y.Length;

        double ssTot = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double d = NumOps.ToDouble(y[i]) - yMean;
            ssTot += d * d;
        }

        // Marginal: variance explained by fixed effects
        var fixedPred = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            T pred = NumOps.Zero;
            for (int j = 0; j < _nFixedParams; j++)
            {
                pred = NumOps.Add(pred, NumOps.Multiply(fixedX[i, j], _fixedEffects[j]));
            }
            fixedPred[i] = pred;
        }

        double ssFixed = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double d = NumOps.ToDouble(fixedPred[i]) - yMean;
            ssFixed += d * d;
        }

        MarginalRSquared = NumOps.FromDouble(ssFixed / ssTot);

        // Conditional: variance explained by fixed + random effects
        var fullPred = Predict(fullX);
        double ssFull = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double d = NumOps.ToDouble(fullPred[i]) - yMean;
            ssFull += d * d;
        }

        ConditionalRSquared = NumOps.FromDouble(ssFull / ssTot);
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
    private static Vector<T> ComputeResiduals(Matrix<T> x, Vector<T> y, Vector<T> beta)
    {
        var pred = x.Multiply(beta);
        var residuals = new Vector<T>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            residuals[i] = NumOps.Subtract(y[i], pred[i]);
        }
        return residuals;
    }

    /// <summary>
    /// Computes variance of a vector.
    /// </summary>
    private static double ComputeVariance(Vector<T> v)
    {
        double mean = 0;
        for (int i = 0; i < v.Length; i++)
        {
            mean += NumOps.ToDouble(v[i]);
        }
        mean /= v.Length;

        double variance = 0;
        for (int i = 0; i < v.Length; i++)
        {
            double d = NumOps.ToDouble(v[i]) - mean;
            variance += d * d;
        }
        return variance / (v.Length - 1);
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.MixedEffectsModel;

    /// <summary>
    /// Creates a new instance of the model with the same configuration.
    /// </summary>
    /// <returns>A new instance of LinearMixedModel.</returns>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new LinearMixedModel<T>(_options, Regularization);

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
