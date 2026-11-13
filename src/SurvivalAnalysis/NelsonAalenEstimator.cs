using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements the Nelson-Aalen estimator for non-parametric cumulative hazard function estimation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Nelson-Aalen estimator calculates the cumulative hazard function H(t),
/// which represents the accumulated risk of an event up to time t. Unlike Kaplan-Meier which estimates
/// survival probability S(t), Nelson-Aalen estimates H(t) directly. They're related by S(t) = exp(-H(t)).</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>At each event time t, add d(t)/n(t) to the cumulative hazard</item>
/// <item>d(t) = number of events at time t</item>
/// <item>n(t) = number of subjects at risk just before time t</item>
/// </list>
/// </para>
///
/// <para><b>Variance estimation:</b> The Nelson-Aalen estimator uses the variance formula:
/// Var(H(t)) = sum over event times of d(t)/n(t)^2</para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you want to estimate cumulative hazard directly</item>
/// <item>As input to other models that work with cumulative hazard</item>
/// <item>When comparing hazard rates across groups</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Nelson (1972), Aalen (1978)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
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
///     .ConfigureModel(new NelsonAalenEstimator&lt;double&gt;())
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
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Theory of Counting Processes", "https://doi.org/10.1007/978-1-4612-4532-4")]
public partial class NelsonAalenEstimator<T> : SurvivalModelBase<T>
{
    /// <summary>
    /// The cumulative hazard values at each event time.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _cumulativeHazard;

    /// <summary>
    /// The variance estimates at each event time.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _variance;

    /// <summary>
    /// Gets the cumulative hazard function values at event times.
    /// </summary>
    public Vector<T>? CumulativeHazard => _cumulativeHazard;

    /// <summary>
    /// Gets the variance estimates at event times.
    /// </summary>
    public Vector<T>? Variance => _variance;

    /// <summary>
    /// Creates a new Nelson-Aalen estimator.
    /// </summary>
    public NelsonAalenEstimator() : base()
    {
    }

    /// <summary>
    /// Fits the Nelson-Aalen estimator to survival data.
    /// </summary>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        // Extract unique sorted event times
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        if (TrainedEventTimes.Length == 0)
        {
            _cumulativeHazard = new Vector<T>(0);
            _variance = new Vector<T>(0);
            BaselineSurvivalFunction = new Vector<T>(0);
            return;
        }

        var engine = AiDotNetEngine.Current;

        // Count events and at-risk at each event time
        var (atRisk, eventsAtTime) = CountAtRiskAndEvents(times, events, TrainedEventTimes);

        // A time with nobody at risk adds nothing; divide by 1 there to keep 0/0 out of the sums.
        var hasRisk = engine.TensorGreaterThan(atRisk, 0.0);
        var safeRisk = engine.TensorWhere(hasRisk, atRisk,
            Tensor<double>.CreateDefault(new[] { TrainedEventTimes.Length }, 1.0));

        // Nelson-Aalen increment: d(t)/n(t)
        var hazardIncrement = engine.TensorMultiply(hasRisk, engine.TensorDivide(eventsAtTime, safeRisk));

        // Variance increment: d(t)/n(t)^2, accumulated in double even for low-precision storage.
        var varianceIncrement = engine.TensorMultiply(hasRisk,
            engine.TensorDivide(eventsAtTime, engine.TensorMultiply(safeRisk, safeRisk)));

        var cumulativeHazard = engine.TensorCumSum(hazardIncrement, 0);
        _cumulativeHazard = cumulativeHazard.Cast<T>().ToVector();
        _variance = engine.TensorCumSum(varianceIncrement, 0).Cast<T>().ToVector();

        // Survival from cumulative hazard: S(t) = exp(-H(t))
        BaselineSurvivalFunction = engine.TensorExp(engine.TensorNegate(cumulativeHazard)).Cast<T>().ToVector();
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        var engine = AiDotNetEngine.Current;
        var survival = engine.TensorExp(engine.TensorNegate(CumulativeHazardAt(times)));

        return engine.TensorBroadcastTo(survival.Reshape(new[] { 1, times.Length }),
            new[] { x.Rows, times.Length }).ToMatrix();
    }

    /// <summary>
    /// Predicts cumulative hazard at specified times (override to use direct estimate).
    /// </summary>
    public new Matrix<T> PredictCumulativeHazard(Vector<T> times, Matrix<T>? features = null)
    {
        EnsureFitted();

        int numSubjects = features?.Rows ?? 1;

        return AiDotNetEngine.Current.TensorBroadcastTo(
            CumulativeHazardAt(times).Reshape(new[] { 1, times.Length }),
            new[] { numSubjects, times.Length }).ToMatrix();
    }

    /// <summary>
    /// Evaluates the cumulative hazard step function at each query time.
    /// </summary>
    private Tensor<T> CumulativeHazardAt(Vector<T> queryTimes)
    {
        if (TrainedEventTimes is null || _cumulativeHazard is null || TrainedEventTimes.Length == 0)
            return new Tensor<T>(new[] { queryTimes.Length });

        // Value at the last event time <= t; 0 before the first event time
        return EvaluateStepFunction(queryTimes, TrainedEventTimes, _cumulativeHazard, NumOps.Zero);
    }

    /// <summary>
    /// Returns hazard ratios (all 1s for non-parametric estimator).
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        return Vector<T>.CreateDefault(x.Rows, NumOps.One);
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        var engine = AiDotNetEngine.Current;

        return engine.TensorExp(engine.TensorNegate(CumulativeHazardAt(times))).ToVector();
    }

    /// <summary>
    /// Predicts survival probability at median event time.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
            return new Vector<T>(input.Rows);

        // Use median survival as prediction
        return PredictMedianSurvivalTime(input);
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new NelsonAalenEstimator<T>();
        copy.SetParameters(parameters);
        return copy;
    }

}
