using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements the Cox Proportional Hazards model for survival analysis with covariates.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Cox model is a semi-parametric survival model that estimates the effect of covariates
/// on the hazard (risk) without assuming a specific form for the baseline hazard function.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cox Proportional Hazards is the most widely used survival model
/// because it can tell you HOW features affect survival risk without making assumptions
/// about the shape of survival over time.
///
/// The key equation is:
/// h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)
///
/// Where:
/// - h(t|X) = hazard (instantaneous risk) at time t for a subject with features X
/// - h₀(t) = baseline hazard (unknown, estimated non-parametrically)
/// - β = coefficients that tell you how each feature affects risk
/// - X = feature values
///
/// Interpreting coefficients:
/// - β &gt; 0: Higher feature value → Higher risk (shorter survival)
/// - β &lt; 0: Higher feature value → Lower risk (longer survival)
/// - exp(β) = Hazard Ratio: How much risk changes per unit increase in feature
///
/// Example: If β_age = 0.05, then exp(0.05) ≈ 1.05, meaning each year of age
/// increases the hazard (risk) by about 5%.
///
/// The "proportional hazards" assumption means that hazard ratios are constant over time.
/// If this assumption is violated, consider stratified Cox or time-varying coefficients.
///
/// References:
/// - Cox (1972). "Regression Models and Life-Tables"
/// </para>
/// </remarks>
public class CoxProportionalHazards<T> : SurvivalModelBase<T>
{
    /// <summary>
    /// The estimated coefficients (log hazard ratios).
    /// </summary>
    private Vector<T>? _coefficients;

    /// <summary>
    /// The learning rate for gradient descent.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Maximum number of iterations for optimization.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for optimization.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// L2 regularization strength (ridge penalty).
    /// </summary>
    private readonly double _l2Penalty;

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.CoxProportionalHazards;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cox Proportional Hazards supports JIT compilation because
    /// the hazard ratio prediction is a simple mathematical formula: exp(β·X).
    /// This can be compiled to optimized machine code for faster inference.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Initializes a new instance of the CoxProportionalHazards class.
    /// </summary>
    /// <param name="learningRate">Learning rate for gradient descent. Default is 0.01.</param>
    /// <param name="maxIterations">Maximum iterations. Default is 1000.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-6.</param>
    /// <param name="l2Penalty">L2 regularization strength. Default is 0.0.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cox model is fitted using Newton-Raphson or gradient descent
    /// to maximize the partial likelihood. Parameters:
    ///
    /// - learningRate: Step size for optimization (smaller = slower but more stable)
    /// - maxIterations: When to stop if not converged
    /// - tolerance: How small the improvement must be to declare convergence
    /// - l2Penalty: Regularization to prevent overfitting (higher = more regularization)
    ///
    /// Usage:
    /// <code>
    /// var cox = new CoxProportionalHazards&lt;double&gt;(l2Penalty: 0.1);
    /// cox.FitSurvival(features, times, events);
    /// var hazardRatios = cox.PredictHazardRatio(newPatients);
    /// </code>
    /// </para>
    /// </remarks>
    public CoxProportionalHazards(
        double learningRate = 0.01,
        int maxIterations = 1000,
        double tolerance = 1e-6,
        double l2Penalty = 0.0,
        int? seed = null)
    {
        _learningRate = learningRate;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _l2Penalty = l2Penalty;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Fits the Cox model using partial likelihood maximization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cox model is fitted by maximizing the "partial likelihood"
    /// which considers the order of events without needing to specify the baseline hazard.
    ///
    /// At each event time, we ask: "Given who was at risk, what's the probability that
    /// THIS subject had the event?" Subjects with higher risk scores (exp(β·X)) should
    /// have higher probability.
    ///
    /// The partial likelihood is:
    /// L(β) = ∏ [exp(β·Xᵢ) / Σⱼ∈Rᵢ exp(β·Xⱼ)]
    ///
    /// Where:
    /// - Product is over all event times
    /// - Rᵢ is the "risk set" (subjects still at risk at time i)
    /// - Xᵢ is the features of the subject who had the event
    /// </para>
    /// </remarks>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        int n = x.Rows;
        int p = x.Columns;

        // Initialize coefficients
        _coefficients = new Vector<T>(p);
        for (int i = 0; i < p; i++)
        {
            _coefficients[i] = NumOps.Zero;
        }

        // Get unique event times and sort data by time
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Create sorted indices by time (descending for efficient risk set computation)
        var sortedIndices = Enumerable.Range(0, n)
            .OrderByDescending(i => NumOps.ToDouble(times[i]))
            .ToArray();

        // Gradient descent optimization
        double prevLogLik = double.NegativeInfinity;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute gradient and update coefficients
            var (logLik, gradient) = ComputeGradient(x, times, events, sortedIndices);

            // Check convergence
            if (Math.Abs(logLik - prevLogLik) < _tolerance)
            {
                break;
            }
            prevLogLik = logLik;

            // Update coefficients
            for (int j = 0; j < p; j++)
            {
                T gradJ = gradient[j];
                T reg = NumOps.Multiply(NumOps.FromDouble(_l2Penalty), _coefficients[j]);
                T update = NumOps.Multiply(
                    NumOps.FromDouble(_learningRate),
                    NumOps.Subtract(gradJ, reg));
                _coefficients[j] = NumOps.Add(_coefficients[j], update);
            }
        }

        // Compute baseline survival using Breslow estimator
        ComputeBaselineSurvival(x, times, events);
    }

    /// <summary>
    /// Computes the gradient of the partial log-likelihood.
    /// </summary>
    private (double logLik, Vector<T> gradient) ComputeGradient(
        Matrix<T> x,
        Vector<T> times,
        Vector<int> events,
        int[] sortedIndices)
    {
        int n = x.Rows;
        int p = x.Columns;

        var gradient = new Vector<T>(p);
        double logLik = 0;

        // Compute risk scores exp(β·X) for all subjects
        var riskScores = new double[n];
        for (int i = 0; i < n; i++)
        {
            double linearPred = 0;
            for (int j = 0; j < p; j++)
            {
                linearPred += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[i, j]);
            }
            riskScores[i] = Math.Exp(Math.Min(linearPred, 700)); // Prevent overflow
        }

        // Process events in reverse time order (for efficient cumulative sums)
        double sumRisk = 0;
        var sumRiskX = new double[p];

        foreach (int i in sortedIndices)
        {
            sumRisk += riskScores[i];
            for (int j = 0; j < p; j++)
            {
                sumRiskX[j] += riskScores[i] * NumOps.ToDouble(x[i, j]);
            }

            if (events[i] == 1)
            {
                // Add contribution to log-likelihood
                double linearPred = 0;
                for (int j = 0; j < p; j++)
                {
                    linearPred += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[i, j]);
                }
                logLik += linearPred - Math.Log(sumRisk);

                // Add contribution to gradient
                for (int j = 0; j < p; j++)
                {
                    double xij = NumOps.ToDouble(x[i, j]);
                    double gradContrib = xij - sumRiskX[j] / sumRisk;
                    gradient[j] = NumOps.Add(gradient[j], NumOps.FromDouble(gradContrib));
                }
            }
        }

        return (logLik, gradient);
    }

    /// <summary>
    /// Computes the Breslow baseline survival function.
    /// </summary>
    private void ComputeBaselineSurvival(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
        {
            BaselineSurvivalFunction = new Vector<T>(1) { [0] = NumOps.One };
            return;
        }

        int numTimes = TrainedEventTimes.Length;
        BaselineSurvivalFunction = new Vector<T>(numTimes);

        // Compute risk scores
        var riskScores = new double[x.Rows];
        for (int i = 0; i < x.Rows; i++)
        {
            double linearPred = 0;
            for (int j = 0; j < x.Columns; j++)
            {
                linearPred += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[i, j]);
            }
            riskScores[i] = Math.Exp(Math.Min(linearPred, 700));
        }

        double cumulativeHazard = 0;

        for (int t = 0; t < numTimes; t++)
        {
            double eventTime = NumOps.ToDouble(TrainedEventTimes[t]);

            // Count events and sum of risk scores at this time
            int numEventsAtTime = 0;
            double sumRiskAtRisk = 0;

            for (int i = 0; i < x.Rows; i++)
            {
                double ti = NumOps.ToDouble(times[i]);

                // At risk if observation time >= event time
                if (ti >= eventTime)
                {
                    sumRiskAtRisk += riskScores[i];
                }

                // Event at this time
                if (events[i] == 1 && Math.Abs(ti - eventTime) < 1e-10)
                {
                    numEventsAtTime++;
                }
            }

            // Breslow estimator: H₀(t) = Σ dᵢ / Σⱼ∈Rᵢ exp(β·Xⱼ)
            if (sumRiskAtRisk > 0)
            {
                cumulativeHazard += numEventsAtTime / sumRiskAtRisk;
            }

            // S₀(t) = exp(-H₀(t))
            BaselineSurvivalFunction[t] = NumOps.FromDouble(Math.Exp(-cumulativeHazard));
        }
    }

    /// <summary>
    /// Predicts survival probabilities at specified time points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Individual survival is computed as:
    /// S(t|X) = S₀(t)^exp(β·X)
    ///
    /// Where:
    /// - S₀(t) is the baseline survival
    /// - exp(β·X) is the hazard ratio for this subject
    ///
    /// Higher risk (larger exp(β·X)) means lower survival probability.
    /// </para>
    /// </remarks>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        var result = new Matrix<T>(x.Rows, times.Length);
        var baselineSurv = GetBaselineSurvival(times);
        var hazardRatios = PredictHazardRatio(x);

        for (int i = 0; i < x.Rows; i++)
        {
            double hr = NumOps.ToDouble(hazardRatios[i]);
            for (int t = 0; t < times.Length; t++)
            {
                // S(t|X) = S₀(t)^HR
                double s0 = NumOps.ToDouble(baselineSurv[t]);
                double survProb = Math.Pow(s0, hr);
                result[i, t] = NumOps.FromDouble(survProb);
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts hazard ratios for each subject.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The hazard ratio is exp(β·X) - how much higher (or lower)
    /// a subject's risk is compared to baseline.
    ///
    /// - HR = 1.0: Same risk as baseline
    /// - HR = 2.0: Twice the risk
    /// - HR = 0.5: Half the risk
    /// </para>
    /// </remarks>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        var result = new Vector<T>(x.Rows);

        for (int i = 0; i < x.Rows; i++)
        {
            double linearPred = 0;
            for (int j = 0; j < x.Columns; j++)
            {
                linearPred += NumOps.ToDouble(_coefficients![j]) * NumOps.ToDouble(x[i, j]);
            }
            result[i] = NumOps.FromDouble(Math.Exp(Math.Min(linearPred, 700)));
        }

        return result;
    }

    /// <summary>
    /// Gets the baseline survival function at specified time points.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || BaselineSurvivalFunction is null)
        {
            var ones = new Vector<T>(times.Length);
            for (int i = 0; i < times.Length; i++)
            {
                ones[i] = NumOps.One;
            }
            return ones;
        }

        var result = new Vector<T>(times.Length);

        for (int i = 0; i < times.Length; i++)
        {
            double t = NumOps.ToDouble(times[i]);

            // Find survival at this time (step function)
            T survival = NumOps.One;
            for (int j = 0; j < TrainedEventTimes.Length; j++)
            {
                if (NumOps.ToDouble(TrainedEventTimes[j]) <= t)
                {
                    survival = BaselineSurvivalFunction[j];
                }
                else
                {
                    break;
                }
            }

            result[i] = survival;
        }

        return result;
    }

    /// <summary>
    /// Standard prediction - returns hazard ratios.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> x)
    {
        return PredictHazardRatio(x);
    }

    /// <summary>
    /// Gets the estimated coefficients (log hazard ratios).
    /// </summary>
    /// <returns>Vector of coefficients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each coefficient tells you how a one-unit increase in that
    /// feature affects the log hazard. To get the hazard ratio, compute exp(coefficient).
    /// </para>
    /// </remarks>
    public Vector<T>? GetCoefficients() => _coefficients;

    /// <summary>
    /// Gets the hazard ratios for each feature.
    /// </summary>
    /// <returns>Vector of hazard ratios (exp of coefficients).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are easier to interpret than raw coefficients:
    /// - HR = 1.5 means 50% higher risk per unit increase
    /// - HR = 0.8 means 20% lower risk per unit increase
    /// </para>
    /// </remarks>
    public Vector<T>? GetFeatureHazardRatios()
    {
        if (_coefficients is null) return null;

        var result = new Vector<T>(_coefficients.Length);
        for (int i = 0; i < _coefficients.Length; i++)
        {
            double coef = NumOps.ToDouble(_coefficients[i]);
            result[i] = NumOps.FromDouble(Math.Exp(coef));
        }

        return result;
    }

    #region JIT Compilation Support

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the hazard ratio prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cox model's prediction is exp(β·X), which can be
    /// JIT compiled for faster inference. The computation graph is:
    /// 1. Input X (features)
    /// 2. Coefficients β (constants)
    /// 3. Matrix multiplication: X @ β
    /// 4. Exponential: exp(result)
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!IsFitted || _coefficients is null)
        {
            throw new InvalidOperationException(
                "Model must be fitted before exporting computation graph.");
        }

        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        // Create input placeholder for features: [batchSize, numFeatures]
        var inputTensor = new Tensor<T>(new int[] { 1, NumFeatures });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "features");
        inputNodes.Add(inputNode);

        // Create constant node for coefficients: [numFeatures, 1]
        var coeffTensor = new Tensor<T>(new int[] { NumFeatures, 1 });
        for (int i = 0; i < NumFeatures; i++)
        {
            coeffTensor[i, 0] = _coefficients[i];
        }
        var coeffNode = TensorOperations<T>.Constant(coeffTensor, "coefficients");
        inputNodes.Add(coeffNode);

        // Matrix multiplication: linearPred = X @ β, shape [batchSize, 1]
        var linearPredNode = TensorOperations<T>.MatrixMultiply(inputNode, coeffNode);

        // Exponential: hazardRatio = exp(linearPred)
        var outputNode = TensorOperations<T>.Exp(linearPredNode);
        outputNode.Name = "hazard_ratio";

        return outputNode;
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Cox model, the parameters are the coefficients (β values)
    /// that determine how each feature affects the hazard rate. These are the log hazard ratios.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_coefficients is null)
        {
            return new Vector<T>(0);
        }

        return _coefficients;
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets the Cox model coefficients. Each coefficient
    /// represents the log hazard ratio for the corresponding feature.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length > 0)
        {
            _coefficients = parameters;
            NumFeatures = parameters.Length;
        }
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a copy of the Cox model with the given
    /// coefficients. Useful for model cloning and ensemble methods.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new CoxProportionalHazards<T>(_learningRate, _maxIterations, _tolerance, _l2Penalty);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new CoxProportionalHazards<T>(_learningRate, _maxIterations, _tolerance, _l2Penalty);
    }

    /// <summary>
    /// Gets the feature importance scores based on coefficient magnitudes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In Cox models, feature importance is often measured by the
    /// absolute value of coefficients - larger magnitudes mean stronger effects on survival.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        if (_coefficients is null) return result;

        for (int i = 0; i < _coefficients.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.FromDouble(Math.Abs(NumOps.ToDouble(_coefficients[i])));
        }

        return result;
    }

    #endregion
}
