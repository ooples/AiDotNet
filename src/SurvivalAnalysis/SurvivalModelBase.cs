using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Newtonsoft.Json;

namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Abstract base class for survival analysis models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This base class provides common functionality for survival models including
/// data validation, concordance index calculation, and baseline survival estimation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This class contains shared code that all survival models need,
/// so each specific model (like Cox or Kaplan-Meier) doesn't have to reimplement it.
///
/// Key shared functionality:
/// - Validating input data (times must be positive, events must be 0 or 1)
/// - Calculating the concordance index (how well the model predicts)
/// - Finding median survival times from survival curves
/// - Managing trained model state
/// </para>
/// </remarks>
public abstract class SurvivalModelBase<T> : ISurvivalModel<T>
{
    /// <summary>
    /// Numeric operations helper for generic math.
    /// </summary>
    protected readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The default loss function for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Stores the unique sorted event times from training data.
    /// </summary>
    protected Vector<T>? TrainedEventTimes;

    /// <summary>
    /// Stores the baseline survival function at each event time.
    /// </summary>
    protected Vector<T>? BaselineSurvivalFunction;

    /// <summary>
    /// Gets the unique event times from the training data.
    /// </summary>
    public Vector<T>? EventTimes => TrainedEventTimes;

    /// <summary>
    /// Gets the baseline survival function values at event times.
    /// </summary>
    public Vector<T>? BaselineSurvival => BaselineSurvivalFunction;

    /// <summary>
    /// Indicates whether the model has been fitted.
    /// </summary>
    protected bool IsFitted;

    /// <summary>
    /// Gets the number of features the model was trained on.
    /// </summary>
    protected int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets whether the model is trained.
    /// </summary>
    public bool IsTrained => IsFitted;

    /// <summary>
    /// Gets the default loss function.
    /// </summary>
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    public virtual int ParameterCount => NumFeatures;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> JIT (Just-In-Time) compilation can significantly accelerate
    /// model inference by compiling the computation graph to optimized machine code.
    /// Parametric models like Cox Proportional Hazards support JIT since their predictions
    /// follow a clear mathematical formula. Non-parametric models like Kaplan-Meier are
    /// harder to JIT compile since they rely on table lookups.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the SurvivalModelBase class.
    /// </summary>
    protected SurvivalModelBase()
    {
        _defaultLossFunction = new MeanSquaredErrorLoss<T>();
    }

    #region ISurvivalModel Interface Implementation

    /// <summary>
    /// Fits the survival model to time-to-event data (interface method).
    /// </summary>
    /// <param name="times">Observed times (event or censoring times).</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    /// <param name="features">Optional feature matrix for regression models.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method trains the survival model on your data.
    /// Times are how long each subject was observed. Events indicates whether the actual
    /// event occurred (1) or if we lost track of the subject (censored, 0).</para>
    /// </remarks>
    public virtual void Fit(Vector<T> times, Vector<T> events, Matrix<T>? features = null)
    {
        // Convert Vector<T> events to Vector<int>
        var eventInts = new Vector<int>(events.Length);
        for (int i = 0; i < events.Length; i++)
        {
            eventInts[i] = NumOps.ToDouble(events[i]) > 0.5 ? 1 : 0;
        }

        // Create dummy features if not provided
        if (features is null)
        {
            features = new Matrix<T>(times.Length, 1);
            for (int i = 0; i < times.Length; i++)
            {
                features[i, 0] = NumOps.One;
            }
        }

        FitSurvival(features, times, eventInts);
    }

    /// <summary>
    /// Predicts survival probability at specified times (interface method).
    /// </summary>
    /// <param name="times">Times at which to predict survival.</param>
    /// <param name="features">Features for new subjects (for regression models).</param>
    /// <returns>Survival probabilities S(t) for each time point.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> S(t) is the probability of surviving beyond time t.
    /// It starts at 1.0 (everyone starts alive) and decreases over time.</para>
    /// </remarks>
    public virtual Matrix<T> PredictSurvival(Vector<T> times, Matrix<T>? features = null)
    {
        EnsureFitted();

        // Create dummy features if not provided
        if (features is null)
        {
            features = new Matrix<T>(1, NumFeatures > 0 ? NumFeatures : 1);
            for (int j = 0; j < features.Columns; j++)
            {
                features[0, j] = NumOps.Zero;
            }
        }

        return PredictSurvivalProbability(features, times);
    }

    /// <summary>
    /// Predicts cumulative hazard at specified times (interface method).
    /// </summary>
    /// <param name="times">Times at which to predict cumulative hazard.</param>
    /// <param name="features">Features for new subjects (for regression models).</param>
    /// <returns>Cumulative hazard H(t) for each time point.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> H(t) represents the accumulated risk up to time t.
    /// It's related to survival by S(t) = exp(-H(t)).</para>
    /// </remarks>
    public virtual Matrix<T> PredictCumulativeHazard(Vector<T> times, Matrix<T>? features = null)
    {
        // H(t) = -ln(S(t))
        var survival = PredictSurvival(times, features);
        var cumHazard = new Matrix<T>(survival.Rows, survival.Columns);

        for (int i = 0; i < survival.Rows; i++)
        {
            for (int j = 0; j < survival.Columns; j++)
            {
                double s = Math.Max(1e-10, NumOps.ToDouble(survival[i, j]));
                cumHazard[i, j] = NumOps.FromDouble(-Math.Log(s));
            }
        }

        return cumHazard;
    }

    /// <summary>
    /// Predicts risk scores for subjects (interface method).
    /// </summary>
    /// <param name="features">Feature matrix for subjects.</param>
    /// <returns>Risk scores for each subject (higher = higher risk).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Risk scores indicate relative hazard compared to baseline.
    /// A score of 2.0 means twice the baseline hazard.</para>
    /// </remarks>
    public virtual Vector<T> PredictRisk(Matrix<T> features)
    {
        EnsureFitted();
        return PredictHazardRatio(features);
    }

    /// <summary>
    /// Gets the estimated median survival time (interface method).
    /// </summary>
    /// <param name="features">Features for subjects (for regression models).</param>
    /// <returns>Median survival times.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Median survival time is the time at which 50% of subjects
    /// are expected to have experienced the event.</para>
    /// </remarks>
    public virtual Vector<T> PredictMedianSurvivalTime(Matrix<T>? features = null)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
        {
            throw new InvalidOperationException("Model has no event times stored.");
        }

        // Create dummy features if not provided
        if (features is null)
        {
            features = new Matrix<T>(1, NumFeatures > 0 ? NumFeatures : 1);
            for (int j = 0; j < features.Columns; j++)
            {
                features[0, j] = NumOps.Zero;
            }
        }

        var survivalProbs = PredictSurvivalProbability(features, TrainedEventTimes);
        var medianTimes = new Vector<T>(features.Rows);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < features.Rows; i++)
        {
            T medianTime = NumOps.MaxValue;
            for (int t = 0; t < TrainedEventTimes.Length - 1; t++)
            {
                T prob = survivalProbs[i, t];
                T nextProb = survivalProbs[i, t + 1];

                if (NumOps.Compare(prob, half) >= 0 && NumOps.Compare(nextProb, half) < 0)
                {
                    T time1 = TrainedEventTimes[t];
                    T time2 = TrainedEventTimes[t + 1];
                    T probDiff = NumOps.Subtract(prob, nextProb);

                    if (NumOps.Compare(probDiff, NumOps.Zero) > 0)
                    {
                        T fraction = NumOps.Divide(
                            NumOps.Subtract(prob, half),
                            probDiff);
                        medianTime = NumOps.Add(time1,
                            NumOps.Multiply(fraction, NumOps.Subtract(time2, time1)));
                    }
                    else
                    {
                        medianTime = time1;
                    }
                    break;
                }
            }

            medianTimes[i] = medianTime;
        }

        return medianTimes;
    }

    #endregion

    #region Survival-Specific Methods

    /// <summary>
    /// Fits the survival model to time-to-event data.
    /// </summary>
    /// <param name="x">The feature matrix (covariates).</param>
    /// <param name="times">The observed times (event time or censoring time).</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    public virtual void FitSurvival(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        ValidateSurvivalData(x, times, events);
        NumFeatures = x.Columns;

        FitSurvivalCore(x, times, events);

        IsFitted = true;
    }

    /// <summary>
    /// Core fitting logic to be implemented by derived classes.
    /// </summary>
    protected abstract void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events);

    /// <summary>
    /// Predicts survival probabilities at specified time points.
    /// </summary>
    public abstract Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times);

    /// <summary>
    /// Predicts hazard ratios relative to a baseline.
    /// </summary>
    public abstract Vector<T> PredictHazardRatio(Matrix<T> x);

    /// <summary>
    /// Gets the baseline survival function.
    /// </summary>
    public abstract Vector<T> GetBaselineSurvival(Vector<T> times);

    /// <summary>
    /// Standard prediction - returns hazard ratios or survival at median time.
    /// </summary>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Calculates the concordance index (C-index) for model evaluation.
    /// </summary>
    public virtual T CalculateConcordanceIndex(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        EnsureFitted();

        var riskScores = PredictHazardRatio(x);
        int concordant = 0;
        int comparable = 0;

        for (int i = 0; i < x.Rows; i++)
        {
            if (events[i] == 0) continue;

            for (int j = 0; j < x.Rows; j++)
            {
                if (i == j) continue;

                double timeI = NumOps.ToDouble(times[i]);
                double timeJ = NumOps.ToDouble(times[j]);

                if (timeI < timeJ)
                {
                    comparable++;

                    double riskI = NumOps.ToDouble(riskScores[i]);
                    double riskJ = NumOps.ToDouble(riskScores[j]);

                    if (riskI > riskJ)
                    {
                        concordant++;
                    }
                    else if (Math.Abs(riskI - riskJ) < 1e-10)
                    {
                        concordant++;
                        comparable++;
                    }
                }
            }
        }

        if (comparable == 0)
        {
            return NumOps.FromDouble(0.5);
        }

        return NumOps.FromDouble((double)concordant / comparable);
    }

    /// <summary>
    /// Standard model training - redirects to survival-specific training.
    /// </summary>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        var events = new Vector<int>(y.Length);
        for (int i = 0; i < y.Length; i++)
        {
            events[i] = 1;
        }

        FitSurvival(x, y, events);
    }

    #endregion

    #region Validation

    /// <summary>
    /// Validates survival data inputs.
    /// </summary>
    protected void ValidateSurvivalData(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        if (x.Rows != times.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of times ({times.Length}).");
        }

        if (x.Rows != events.Length)
        {
            throw new ArgumentException(
                $"Number of samples in X ({x.Rows}) must match number of events ({events.Length}).");
        }

        for (int i = 0; i < times.Length; i++)
        {
            if (NumOps.Compare(times[i], NumOps.Zero) <= 0)
            {
                throw new ArgumentException($"All times must be positive. Found non-positive time at index {i}.");
            }
        }

        for (int i = 0; i < events.Length; i++)
        {
            if (events[i] != 0 && events[i] != 1)
            {
                throw new ArgumentException(
                    $"Event indicators must be 0 or 1. Found {events[i]} at index {i}.");
            }
        }
    }

    /// <summary>
    /// Ensures the model has been fitted before prediction.
    /// </summary>
    protected void EnsureFitted()
    {
        if (!IsFitted)
        {
            throw new InvalidOperationException(
                "Model must be fitted before making predictions. Call FitSurvival first.");
        }
    }

    /// <summary>
    /// Gets unique sorted event times from the data.
    /// </summary>
    protected Vector<T> GetUniqueEventTimes(Vector<T> times, Vector<int> events)
    {
        var eventTimes = new List<double>();
        for (int i = 0; i < times.Length; i++)
        {
            if (events[i] == 1)
            {
                double t = NumOps.ToDouble(times[i]);
                if (!eventTimes.Contains(t))
                {
                    eventTimes.Add(t);
                }
            }
        }

        eventTimes.Sort();
        var result = new Vector<T>(eventTimes.Count);
        for (int i = 0; i < eventTimes.Count; i++)
        {
            result[i] = NumOps.FromDouble(eventTimes[i]);
        }

        return result;
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Derived classes should override this to return their specific
    /// model type (e.g., KaplanMeierEstimator, CoxProportionalHazards).
    /// </para>
    /// </remarks>
    public virtual ModelType GetModelType() => ModelType.None;

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
            FeatureCount = NumFeatures,
            Complexity = NumFeatures,
            Description = $"{GetModelType()} survival model with {NumFeatures} features",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "IsFitted", IsFitted },
                { "NumEventTimes", TrainedEventTimes?.Length ?? 0 }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    public virtual byte[] Serialize()
    {
        var modelData = new Dictionary<string, object>
        {
            { "NumFeatures", NumFeatures },
            { "IsFitted", IsFitted }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    public virtual void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata?.ModelData is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var modelDataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (modelDataObj is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        NumFeatures = modelDataObj["NumFeatures"]?.ToObject<int>() ?? 0;
        IsFitted = modelDataObj["IsFitted"]?.ToObject<bool>() ?? false;
    }

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    public abstract void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the indices of features that are actively used in the model.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default: no-op
    }

    /// <summary>
    /// Determines whether a specific feature is used in the model.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        }
        return true;
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.One;
        }
        return result;
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        byte[] serialized = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(serialized);
        return copy;
    }

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Creates a clone of the model.
    /// </summary>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Computes gradients for the given input and target.
    /// </summary>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Survival models typically don't use standard gradient computation
        return new Vector<T>(ParameterCount);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Default: no-op for non-parametric models
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    public virtual void SaveModel(string filePath)
    {
        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    public virtual void LoadModel(string filePath)
    {
        byte[] serializedData = File.ReadAllBytes(filePath);
        Deserialize(serializedData);
    }

    /// <summary>
    /// Saves the model's state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        byte[] serializedData = Serialize();
        stream.Write(serializedData, 0, serializedData.Length);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        byte[] serializedData = memoryStream.ToArray();
        Deserialize(serializedData);
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for survival models.");
    }

    #endregion
}
