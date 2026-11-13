using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

using AiDotNet.Models.Parameters;

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
/// <example>
/// <code>
/// // age and treatment arm per patient
/// var features = new Matrix&lt;double&gt;(new double[,]
/// {
///     { 45, 1 }, { 52, 0 }, { 38, 1 }, { 61, 0 }, { 47, 1 }, { 55, 0 }
/// });
/// // months each patient was observed
/// var times = new Vector&lt;double&gt;(new double[] { 5.0, 12.0, 3.0, 18.0, 9.0, 21.0 });
/// // 1 = the event happened; 0 = censored, still fine when the study ended or lost to follow-up
/// var events = new Vector&lt;int&gt;(new int[] { 1, 0, 1, 0, 1, 1 });
///
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new CoxProportionalHazards&lt;double&gt;(learningRate: 0.01, maxIterations: 1000))
///     .Build(features, times, events);
///
/// var risk = result.Predict(features);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Healthcare)]
[ModelCategory(ModelCategory.SurvivalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Regression Models and Life-Tables", "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x", Year = 1972, Authors = "David R. Cox")]
public partial class CoxProportionalHazards<T> : SurvivalModelBase<T>
{

    /// <inheritdoc />
    /// <remarks>The log-hazard-ratio coefficients, one per covariate. Cox has no intercept -- it is absorbed into the baseline hazard -- which is why NumFeatures is the full length here and length minus one in the propensity models.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorFieldParameterSource<T>(
            () => _coefficients,
            value =>
            {
                _coefficients = value;
                NumFeatures = value.Length;
            }));
    }
    /// <summary>
    /// The estimated coefficients (log hazard ratios).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _coefficients = new Vector<T>(0);

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
    // Keep the logarithm: exp(-H0) may round to one even when a subject's H0 * exp(eta) is material.
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private double[]? _logBaselineCumulativeHazard;


    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Gets the model type.
    /// </summary>

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
    /// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
    /// var times = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
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
        int p = x.Columns;
        var engine = AiDotNetEngine.Current;
        _coefficients = new Vector<T>(p);
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        // Sort once, descending in time. All tied subjects select the complete group's prefix.
        var timeValues = times.ToArray();
        var flags = events.ToArray();
        var groups = Enumerable.Range(0, times.Length).OrderByDescending(i => timeValues[i])
            .GroupBy(i => timeValues[i]).Select(group => group.ToArray()).ToArray();
        var order = groups.SelectMany(group => group).ToArray();
        int position = 0;
        var ends = new List<int>();
        var eventEnds = new List<int>();
        var eventCounts = new List<double>();
        foreach (var group in groups)
        {
            position += group.Length;
            ends.AddRange(Enumerable.Repeat(position - 1, group.Length));
            int count = group.Count(i => flags[i] == 1);
            if (count > 0)
            {
                eventEnds.Add(position - 1);
                eventCounts.Add(count);
            }
        }
        var indices = new Tensor<int>(new[] { order.Length }, new Vector<int>(order));
        var riskEnds = new Tensor<int>(new[] { ends.Count }, new Vector<int>(ends));
        var design = engine.TensorIndexSelect(Tensor<T>.FromMatrix(x).Cast<double>(), indices, 0);
        var eventMask = engine.TensorIndexSelect(events.AsTensor().Cast<double>(), indices, 0);
        var wideCoefficients = new Tensor<double>(new[] { p });
        double previous = double.NaN;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var eta = WideLinearPredictor(design, wideCoefficients);
            var (prefixLogRisk, prefixMean) = CoxRiskSetMath.PrefixMoments(eta, design);
            var logRisk = engine.TensorIndexSelect(prefixLogRisk, riskEnds, 0);
            var mean = engine.TensorIndexSelect(prefixMean, riskEnds, 0);
            var likelihoodTerms = engine.TensorSubtract(eta, logRisk);
            double likelihood = engine.TensorSum(engine.TensorMultiply(eventMask, likelihoodTerms));
            if (iter > 0 && Math.Abs(likelihood - previous) < _tolerance) break;
            previous = likelihood;
            var eventRows = engine.TensorBroadcastTo(eventMask.Reshape(new[] { x.Rows, 1 }), new[] { x.Rows, p });
            var gradient = engine.ReduceSum(engine.TensorMultiply(eventRows,
                engine.TensorSubtract(design, mean)), new[] { 0 });
            wideCoefficients = engine.TensorAdd(wideCoefficients, engine.TensorMultiplyScalar(
                engine.TensorSubtract(gradient, engine.TensorMultiplyScalar(wideCoefficients, _l2Penalty)), _learningRate));
        }
        _coefficients = wideCoefficients.Cast<T>().ToVector();
        if (eventEnds.Count == 0)
        {
            _logBaselineCumulativeHazard = Array.Empty<double>();
            BaselineSurvivalFunction = new Vector<T>(0);
            return;
        }

        // Use the stored coefficients for baseline estimation and all subsequent predictions.
        var (finalLogRisk, _) = CoxRiskSetMath.PrefixMoments(
            WideLinearPredictor(design, _coefficients.AsTensor().Cast<double>()), design);
        eventEnds.Reverse();
        eventCounts.Reverse();
        var selected = engine.TensorIndexSelect(finalLogRisk,
            new Tensor<int>(new[] { eventEnds.Count }, new Vector<int>(eventEnds)), 0);
        var counts = new Tensor<double>(new[] { eventCounts.Count }, new Vector<double>(eventCounts));
        var logHazard = CoxRiskSetMath.CumulativeLogSum(engine.TensorSubtract(engine.TensorLog(counts), selected));
        _logBaselineCumulativeHazard = logHazard.ToArray();
        BaselineSurvivalFunction = engine.TensorExp(engine.TensorNegate(engine.TensorExp(logHazard))).Cast<T>().ToVector();
    }

    private static Tensor<double> WideLinearPredictor(Tensor<double> design, Tensor<double> coefficients)
    {
        return AiDotNetEngine.Current.TensorMatMul(design,
            coefficients.Reshape(new[] { coefficients.Length, 1 })).Reshape(new[] { design.Shape[0] });
    }

    private Tensor<double> WideLinearPredictor(Matrix<T> x)
    {
        return WideLinearPredictor(Tensor<T>.FromMatrix(ExtractCovariates(x)).Cast<double>(),
            _coefficients.AsTensor().Cast<double>());
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

        var engine = AiDotNetEngine.Current;
        if (_logBaselineCumulativeHazard is null)
        {
            // Legacy checkpoints only contain baseline survival.
            var baseline = GetBaselineSurvival(times).AsTensor().Cast<double>();
            var risk = engine.TensorExp(engine.TensorClampMax(WideLinearPredictor(x), 700.0));
            var shape = new[] { x.Rows, times.Length };
            return engine.TensorPower(
                engine.TensorBroadcastTo(baseline.Reshape(new[] { 1, times.Length }), shape),
                engine.TensorBroadcastTo(risk.Reshape(new[] { x.Rows, 1 }), shape)).Cast<T>().ToMatrix();
        }
        var stepTimes = TrainedEventTimes ?? new Vector<T>(0);
        var positions = engine.TensorSearchSorted(stepTimes.AsTensor(), times.AsTensor(), right: true);
        var padded = engine.TensorConcatenate(new[]
        {
            Tensor<double>.CreateDefault(new[] { 1 }, double.NegativeInfinity),
            new Tensor<double>(new[] { _logBaselineCumulativeHazard.Length }, new Vector<double>(_logBaselineCumulativeHazard))
        }, 0);
        var logBaseline = engine.TensorGather(padded, positions, 0);
        var grid = new[] { x.Rows, times.Length };
        var logHazard = engine.TensorAdd(
            engine.TensorBroadcastTo(logBaseline.Reshape(new[] { 1, times.Length }), grid),
            engine.TensorBroadcastTo(WideLinearPredictor(x).Reshape(new[] { x.Rows, 1 }), grid));
        return engine.TensorExp(engine.TensorNegate(engine.TensorExp(logHazard))).Cast<T>().ToMatrix();
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

        // Accepts either the covariates alone or the [event | covariates] design matrix Train takes.
        var engine = AiDotNetEngine.Current;
        var risk = engine.TensorExp(engine.TensorClampMax(WideLinearPredictor(x), 700.0));
        return engine.TensorClampMax(risk, NumOps.ToDouble(NumOps.MaxValue)).Cast<T>().ToVector();
    }

    /// <summary>
    /// Gets the baseline survival function at specified time points.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || BaselineSurvivalFunction is null)
        {
            return Vector<T>.CreateDefault(times.Length, NumOps.One);
        }

        // Find survival at this time (step function)
        return EvaluateStepFunction(times, TrainedEventTimes, BaselineSurvivalFunction, NumOps.One).ToVector();
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

        return AiDotNetEngine.Current.Exp(_coefficients);
    }

    #region JIT Compilation Support

    #endregion

    #region IFullModel Implementation

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

        var importances = AiDotNetEngine.Current.Abs(_coefficients).ToArray();
        for (int i = 0; i < importances.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = importances[i];
        }

        return result;
    }

    #endregion
}
