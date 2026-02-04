using System.Text;
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
public class NelsonAalenEstimator<T> : SurvivalModelBase<T>
{
    /// <summary>
    /// The cumulative hazard values at each event time.
    /// </summary>
    private Vector<T>? _cumulativeHazard;

    /// <summary>
    /// The variance estimates at each event time.
    /// </summary>
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

        int numTimes = TrainedEventTimes.Length;
        _cumulativeHazard = new Vector<T>(numTimes);
        _variance = new Vector<T>(numTimes);
        BaselineSurvivalFunction = new Vector<T>(numTimes);

        double cumulativeH = 0;
        double cumulativeVar = 0;

        for (int t = 0; t < numTimes; t++)
        {
            T eventTime = TrainedEventTimes[t];
            double eventTimeDouble = NumOps.ToDouble(eventTime);

            // Count events and at-risk at this time
            int eventsAtTime = 0;
            int atRisk = 0;

            for (int i = 0; i < times.Length; i++)
            {
                double ti = NumOps.ToDouble(times[i]);
                if (ti >= eventTimeDouble)
                    atRisk++;
                if (Math.Abs(ti - eventTimeDouble) < 1e-10 && events[i] == 1)
                    eventsAtTime++;
            }

            if (atRisk > 0)
            {
                // Nelson-Aalen increment: d(t)/n(t)
                double hazardIncrement = (double)eventsAtTime / atRisk;
                cumulativeH += hazardIncrement;

                // Variance increment: d(t)/n(t)^2
                cumulativeVar += (double)eventsAtTime / (atRisk * atRisk);
            }

            _cumulativeHazard[t] = NumOps.FromDouble(cumulativeH);
            _variance[t] = NumOps.FromDouble(cumulativeVar);

            // Survival from cumulative hazard: S(t) = exp(-H(t))
            BaselineSurvivalFunction[t] = NumOps.FromDouble(Math.Exp(-cumulativeH));
        }
    }

    /// <summary>
    /// Predicts survival probabilities at specified times.
    /// </summary>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        int numSubjects = x.Rows;
        var result = new Matrix<T>(numSubjects, times.Length);

        for (int i = 0; i < numSubjects; i++)
        {
            for (int t = 0; t < times.Length; t++)
            {
                double queryTime = NumOps.ToDouble(times[t]);
                double cumHazard = InterpolateCumulativeHazard(queryTime);
                result[i, t] = NumOps.FromDouble(Math.Exp(-cumHazard));
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts cumulative hazard at specified times (override to use direct estimate).
    /// </summary>
    public new Matrix<T> PredictCumulativeHazard(Vector<T> times, Matrix<T>? features = null)
    {
        EnsureFitted();

        int numSubjects = features?.Rows ?? 1;
        var result = new Matrix<T>(numSubjects, times.Length);

        for (int i = 0; i < numSubjects; i++)
        {
            for (int t = 0; t < times.Length; t++)
            {
                double queryTime = NumOps.ToDouble(times[t]);
                double cumHazard = InterpolateCumulativeHazard(queryTime);
                result[i, t] = NumOps.FromDouble(cumHazard);
            }
        }

        return result;
    }

    /// <summary>
    /// Interpolates cumulative hazard at a specific time.
    /// </summary>
    private double InterpolateCumulativeHazard(double queryTime)
    {
        if (TrainedEventTimes is null || _cumulativeHazard is null || TrainedEventTimes.Length == 0)
            return 0;

        // Find the appropriate time point
        int idx = -1;
        for (int i = TrainedEventTimes.Length - 1; i >= 0; i--)
        {
            if (NumOps.ToDouble(TrainedEventTimes[i]) <= queryTime)
            {
                idx = i;
                break;
            }
        }

        if (idx < 0)
            return 0; // Before first event time

        return NumOps.ToDouble(_cumulativeHazard[idx]);
    }

    /// <summary>
    /// Returns hazard ratios (all 1s for non-parametric estimator).
    /// </summary>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
            result[i] = NumOps.One;
        return result;
    }

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        var result = new Vector<T>(times.Length);
        for (int t = 0; t < times.Length; t++)
        {
            double queryTime = NumOps.ToDouble(times[t]);
            double cumHazard = InterpolateCumulativeHazard(queryTime);
            result[t] = NumOps.FromDouble(Math.Exp(-cumHazard));
        }

        return result;
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
    public override Vector<T> GetParameters()
    {
        if (_cumulativeHazard is null)
            return new Vector<T>(0);

        return new Vector<T>(_cumulativeHazard.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _cumulativeHazard = new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new NelsonAalenEstimator<T>();
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new NelsonAalenEstimator<T>();
    }

    /// <inheritdoc />
    public override ModelType GetModelType() => ModelType.NelsonAalenEstimator;

    /// <inheritdoc />
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "NumFeatures", NumFeatures },
            { "IsFitted", IsFitted },
            { "EventTimes", TrainedEventTimes?.ToArray()?.Select(NumOps.ToDouble).ToArray() ?? Array.Empty<double>() },
            { "CumulativeHazard", _cumulativeHazard?.ToArray()?.Select(NumOps.ToDouble).ToArray() ?? Array.Empty<double>() },
            { "Variance", _variance?.ToArray()?.Select(NumOps.ToDouble).ToArray() ?? Array.Empty<double>() }
        };

        var metadata = GetModelMetadata();
        metadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(data));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(metadata));
    }

    /// <inheritdoc />
    public override void Deserialize(byte[] modelData)
    {
        var json = Encoding.UTF8.GetString(modelData);
        var metadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(json);

        if (metadata?.ModelData is null)
            throw new InvalidOperationException("Invalid model data.");

        var dataJson = Encoding.UTF8.GetString(metadata.ModelData);
        var data = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(dataJson);

        if (data is null)
            throw new InvalidOperationException("Invalid model data.");

        NumFeatures = data["NumFeatures"]?.ToObject<int>() ?? 0;
        IsFitted = data["IsFitted"]?.ToObject<bool>() ?? false;

        var eventTimes = data["EventTimes"]?.ToObject<double[]>() ?? Array.Empty<double>();
        var cumHazard = data["CumulativeHazard"]?.ToObject<double[]>() ?? Array.Empty<double>();
        var variance = data["Variance"]?.ToObject<double[]>() ?? Array.Empty<double>();

        TrainedEventTimes = new Vector<T>(eventTimes.Length);
        _cumulativeHazard = new Vector<T>(cumHazard.Length);
        _variance = new Vector<T>(variance.Length);

        for (int i = 0; i < eventTimes.Length; i++)
            TrainedEventTimes[i] = NumOps.FromDouble(eventTimes[i]);
        for (int i = 0; i < cumHazard.Length; i++)
            _cumulativeHazard[i] = NumOps.FromDouble(cumHazard[i]);
        for (int i = 0; i < variance.Length; i++)
            _variance[i] = NumOps.FromDouble(variance[i]);
    }
}
