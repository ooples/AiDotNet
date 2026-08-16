using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Newtonsoft.Json;

using AiDotNet.Models.Parameters;
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
public abstract partial class SurvivalModelBase<T> : ISurvivalModel<T>, IModelShape, IParameterizable<T, Matrix<T>, Vector<T>>, IParameterManifestProvider
{
    // --- declared state (ModelStateRegistry) ---
    // Identical in every model base because these bases are siblings over the same interfaces rather
    // than one hierarchy; the logic itself lives once in ModelStateRegistry/ModelStateEnvelope.

    /// <summary>State that is not a parameter vector, declared once and persisted by this base.</summary>
    private readonly AiDotNet.Models.ModelStateRegistry<T> _declaredState = new();
    private bool _declaredStateRegistered;

    /// <summary>
    /// Declare state here that the parameter vector does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children. Both halves of the payload are driven
    /// by the declaration, so they cannot drift.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    protected virtual void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        // ModelStateGenerator emits RegisterGeneratedState for a model's OWN members only -- it does
        // not walk the base chain, and it cannot generate into this type at all, because FindHook
        // starts at BaseType and this class is where RegisterGeneratedState is declared. So the four
        // members below reach no generated file by either route and have to be declared by hand.
        //
        // They are not optional: NelsonAalenEstimator, WeibullAFT and LogNormalAFT each hand-wrote
        // these into their own Serialize/Deserialize, which is the duplication ADN0060 exists to
        // remove. Declaring them once here is what lets those three overrides go.
        state.Declare("SurvivalModelBase.TrainedEventTimes",
            () => TrainedEventTimes, v => TrainedEventTimes = v);
        state.Declare("SurvivalModelBase.BaselineSurvivalFunction",
            () => BaselineSurvivalFunction, v => BaselineSurvivalFunction = v);
        state.DeclareBoolean("SurvivalModelBase.IsFitted", () => IsFitted, v => IsFitted = v);
        state.DeclareInt32("SurvivalModelBase.NumFeatures", () => NumFeatures, v => NumFeatures = v);
    }
    /// <summary>Generated state declarations for fields declared across this model's hierarchy.</summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// Emitted by ModelStateGenerator into the partial model, so a model author declares nothing. The
    /// hand-written <c>RegisterState</c> beside it exists only for state the classifier genuinely
    /// cannot place; anything it CAN place belongs here, where it cannot be forgotten.
    /// </remarks>
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }

    /// <summary>The declared state, registered once and lazily so it runs after the constructor.</summary>
    protected AiDotNet.Models.ModelStateRegistry<T> DeclaredState
    {
        get
        {
            if (!_declaredStateRegistered)
            {
                _declaredStateRegistered = true;
                RegisterGeneratedState(_declaredState);
                RegisterState(_declaredState);
            }
            return _declaredState;
        }
    }
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
    /// The components the parameters of this model live in. Empty until the model registers
    /// some, in which case the surfaces below fall back to what they always did.
    /// </summary>
    private readonly ParameterComponentRegistry<T> _parameterRegistry = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to the surface of this model.
    /// Registration
    /// order is serialization order, so keep it stable.
    /// </summary>
    protected void RegisterParameterComponent(
        IParameterSource<T>? component,
        [System.Runtime.CompilerServices.CallerArgumentExpression(nameof(component))] string? componentExpression = null,
        [System.Runtime.CompilerServices.CallerMemberName] string? memberName = null)
        => _parameterRegistry.RegisterLegacy(GetType().FullName ?? GetType().Name,
            memberName, componentExpression, component);

    protected void RegisterParameterComponent(string stableId, IParameterSource<T>? component,
        ParameterSlotRole role = ParameterSlotRole.Trainable,
        ParameterAvailability availability = ParameterAvailability.Construction)
        => _parameterRegistry.Register(stableId, component, role, availability);

    /// <summary>
    /// Declare the trainable components of this model here with
    /// <see cref="RegisterParameterComponent"/>. Called once, lazily, so it runs after the
    /// constructor has built them.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    protected virtual void RegisterGeneratedParameterComponents(ParameterComponentRegistry<T> registry)
    {
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components.
    /// </summary>
    protected virtual void OnParametersRestored()
    {
    }

    private ParameterComponentRegistry<T> Registry
    {
        get
        {
            if (!_componentsRegistered)
            {
                RegisterGeneratedParameterComponents(_parameterRegistry);
                RegisterComponents();
                _componentsRegistered = true;
            }
            return _parameterRegistry;
        }
    }

    public ParameterLayoutSnapshot ParameterLayout => Registry.ParameterLayout;

    /// <inheritdoc/>
    /// <remarks>
    /// Virtual rather than abstract: a model that registers its components inherits all
    /// three surfaces and writes no parameter plumbing. It was abstract, which FORCED every
    /// descendant to hand-write the triple -- the same defect ModelBase and LayerBase had.
    /// </remarks>
    public virtual Vector<T> GetParameters()
        => Registry.HasComponents ? Registry.GetParameters() : new Vector<T>(0);

    /// <inheritdoc/>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (!Registry.HasComponents) return;
        Registry.SetParameters(parameters);
        OnParametersRestored();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Folds the same enumeration the vector does once components are registered. Models still
    /// awaiting manifest conversion fall back to their concrete vector length, so the count cannot
    /// invent values that their read/write surface does not own.
    /// </remarks>
    public virtual long ParameterCount
        => Registry.HasComponents ? Registry.ParameterCount : GetParameters().Length;
    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => Registry.CanInitializeOptimizerParameters;
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


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
            eventInts[i] = NumOps.GreaterThan(events[i], NumOps.FromDouble(0.5)) ? 1 : 0;
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
        double concordant = 0;
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
                        concordant += 1.0;
                    }
                    else if (Math.Abs(riskI - riskJ) < 1e-10)
                    {
                        // Harrell's C-index: tied risk scores get half credit
                        concordant += 0.5;
                    }
                }
            }
        }

        if (comparable == 0)
        {
            return NumOps.FromDouble(0.5);
        }

        return NumOps.FromDouble(concordant / comparable);
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

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = NumFeatures,
            Complexity = NumFeatures,
            Description = $"{GetType().Name} survival model with {NumFeatures} features",
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
        ModelPersistenceGuard.EnforceBeforeSerialize();
        return AiDotNet.Models.ModelStateEnvelope.Append(DeclaredState, SerializeInternalUnchecked());
    }

    /// <summary>
    /// Internal, non-virtual, no-guard serialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Serialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private byte[] SerializeInternalUnchecked()
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
        // Strips and applies any declared-state trailer, so the body below reads the payload
        // exactly as it did before this existed.
        modelData = AiDotNet.Models.ModelStateEnvelope.Extract(DeclaredState, modelData);
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        DeserializeInternalUnchecked(modelData);
    }

    /// <summary>
    /// Internal, non-virtual, no-guard deserialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Subclasses cannot override
    /// this method, so a subclass override of <see cref="Deserialize"/> cannot
    /// intercept the clone path.
    /// </summary>
    private void DeserializeInternalUnchecked(byte[] modelData)
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
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

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
        // In-memory clone, not a user save/load — wrap in InternalOperation
        // so the persistence guard does not treat this as a billable op, AND
        // route through the private non-virtual SerializeInternalUnchecked /
        // DeserializeInternalUnchecked helpers so a subclass override of the
        // public virtual Serialize / Deserialize methods cannot intercept the
        // clone path (closes the subclass-override bypass surface).
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] serialized = SerializeInternalUnchecked();
            var copy = CreateNewInstance();
            if (copy is SurvivalModelBase<T> copyBase)
            {
                copyBase.DeserializeInternalUnchecked(serialized);

                // SerializeInternalUnchecked captures only NumFeatures/IsFitted — NOT the model's
                // fitted parameters — so without this transfer every parametric survival model
                // (LogNormalAFT/WeibullAFT/CoxPH/etc.) would clone into an unfitted shell whose
                // Predict throws "Coefficients is null". Round-trip the fitted state through the
                // GetParameters/SetParameters contract each subclass already implements.
                // Non-parametric models (Kaplan-Meier, survival forests) return an empty/degenerate
                // parameter vector, so this is a no-op for them.
                if (IsFitted)
                {
                    copyBase.SetParameters(GetParameters());
                }
            }
            else
            {
                copy.Deserialize(serialized);
            }
            return copy;
        }
    }

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// No longer abstract. Every concrete model used to be forced to write this, and 1147 of them
    /// did -- each one a hand-copied list of constructor arguments that a new option could fall out
    /// of without anything failing. The clone plan records that constructor at compile time instead,
    /// so the base can rebuild the type and a model only overrides this when the generator says it
    /// cannot: a constructor parameter with nothing holding its value, which the build reports by
    /// name rather than leaving to be discovered by a clone that comes back subtly different.
    /// </para>
    /// </remarks>
    protected virtual IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
        => (IFullModel<T, Matrix<T>, Vector<T>>)AiDotNet.Models.CloneEngine.CopyConfiguration(this);

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
    /// <remarks>
    /// Many survival models (Kaplan-Meier, Cox PH on cached log-likelihood
    /// factorizations, RandomSurvivalForest) genuinely don't expose standard
    /// gradients. The base method previously returned a zero vector, which
    /// silently let gradient-based optimizers and gradient-norm metrics
    /// proceed as if the model were trainable while applying zero updates.
    /// Derived classes that DO produce gradients (e.g., Deep Cox networks)
    /// must override this; classes that legitimately don't should override
    /// it to throw a more specific NotSupportedException naming the
    /// constraint, rather than relying on the silent-zero default.
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            $"{GetType().Name}.ComputeGradients was not overridden. Survival models that " +
            "expose gradients must provide an implementation here. Models that don't expose " +
            "gradients (e.g., Kaplan-Meier, Cox-PH cached likelihood, survival forests) " +
            "should override this with a NotSupportedException describing why, rather than " +
            "leaving the default that previously returned a zero vector and silently let " +
            "gradient-based optimizers proceed with no real signal.");
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
    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { NumFeatures };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { 1 };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        byte[] serializedData = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            serializedData, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json,
            GetDynamicShapeInfo());
        File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        byte[] serializedData = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope if present; use raw bytes for legacy files
        if (ModelFileHeader.HasHeader(serializedData))
        {
            serializedData = ModelFileHeader.ExtractPayload(serializedData);
        }

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

    #endregion

    // --- IDisposable (issue #1136 plan part 3) ---

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        System.GC.SuppressFinalize(this);
    }

    /// <summary>Releases resources held by this survival model. Override + call base for layer/tensor cleanup.</summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
    }
}
