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
public abstract partial class SurvivalModelBase<T> : ISurvivalModel<T>, IModelShape, IParameterizable<T, Matrix<T>, Vector<T>>, IParameterManifestProvider, AiDotNet.Models.Parameters.IParameterChunkSource<T>
{
    /// <summary>Enumerates canonical persistent state without flattening tensor-backed components.</summary>
    public virtual IEnumerable<AiDotNet.Models.Parameters.ParameterChunk<T>> GetParameterStateChunks()
        => AiDotNet.Models.Parameters.ParameterChunkRestore<T>.Read(this, typeof(SurvivalModelBase<T>), Registry, GetParameters);

    /// <summary>Streams state in the same order as the parameter manifest.</summary>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks()
        => GetParameterStateChunks().Select(chunk => chunk.Tensor);

    /// <summary>Restores tensor-backed chunks on their device and retains custom-setter semantics.</summary>
    public virtual void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        AiDotNet.Models.Parameters.ParameterChunkRestore<T>.Restore(
            this, typeof(SurvivalModelBase<T>), chunks, Registry, SetParameters, OnParametersRestored);
    }

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
        // Common storage is emitted into RegisterGeneratedStateCore by ModelStateGenerator. This
        // hook remains only for state whose shape the registry cannot express declaratively.
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
        RegisterGeneratedStateCore(state);
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

    /// <inheritdoc />
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
        var eventInts = AiDotNetEngine.Current
            .TensorGreaterThan(events.AsTensor(), NumOps.FromDouble(0.5))
            .Cast<int>()
            .ToVector();

        // Create dummy features if not provided
        if (features is null)
        {
            features = Matrix<T>.CreateDefault(times.Length, 1, NumOps.One);
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
        var engine = AiDotNetEngine.Current;

        return engine.TensorNegate(engine.TensorLog(
            engine.TensorClampMin(Tensor<T>.FromMatrix(survival), NumOps.FromDouble(1e-10)))).ToMatrix();
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
        }

        var survivalProbs = PredictSurvivalProbability(features, TrainedEventTimes);
        var engine = AiDotNetEngine.Current;
        int n = survivalProbs.Rows;
        int m = TrainedEventTimes.Length;

        // Median as lifelines defines it (qth_survival_time with q = 0.5): the first event time at
        // which S(t) <= 0.5, with no interpolation, and +infinity (MaxValue) when the curve never
        // reaches 0.5. The survival curve is non-increasing, so the number of leading points with
        // S(t) > 0.5 indexes that time; summing the matching time steps reads it off per subject.
        var above = engine.TensorGreaterThan(Tensor<T>.FromMatrix(survivalProbs), NumOps.FromDouble(0.5));
        var times = TrainedEventTimes.AsTensor();
        var median = Tensor<T>.CreateDefault(new[] { n }, TrainedEventTimes[0]);
        if (m > 1)
        {
            var steps = engine.TensorSubtract(
                engine.TensorSlice(times, new[] { 1 }, new[] { m - 1 }),
                engine.TensorSlice(times, new[] { 0 }, new[] { m - 1 }));
            var advance = engine.TensorMatMul(
                engine.TensorSlice(above, new[] { 0, 0 }, new[] { n, m - 1 }),
                steps.Reshape(new[] { m - 1, 1 }));
            median = engine.TensorAdd(median, advance.Reshape(new[] { n }));
        }

        var neverReached = engine.TensorSlice(above, new[] { 0, m - 1 }, new[] { n, 1 }).Reshape(new[] { n });

        return engine.TensorWhere(neverReached, Tensor<T>.CreateDefault(new[] { n }, NumOps.MaxValue), median)
            .ToVector();
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
    /// Reconciles a predict-time feature matrix with the <c>[event | covariates]</c> layout that
    /// <see cref="Train(Matrix{T}, Vector{T})"/> consumes.
    /// </summary>
    /// <param name="input">Either the covariates alone, or the same design matrix that was trained on.</param>
    /// <returns>The covariate columns, with the event indicator dropped if it was present.</returns>
    /// <remarks>
    /// <para>
    /// A model fit through <c>Train</c> drops column 0 (the event indicator) before learning, so it
    /// holds <see cref="NumFeatures"/> coefficients rather than <c>input.Columns</c> of them. Handing
    /// the same matrix straight back to <c>Predict</c> would then run the coefficient loop one column
    /// past its end — an <see cref="ArgumentOutOfRangeException"/> from inside the numeric code, which
    /// says nothing about the actual mistake. This mirrors
    /// <c>CausalModelBase.ExtractCovariates</c>, which reconciles the identical
    /// <c>[treatment | covariates]</c> layout.
    /// </para>
    /// <para>
    /// A matrix that already has <see cref="NumFeatures"/> columns is returned unchanged, so the
    /// covariate-only path — <c>FitSurvival(x, times, events)</c>, or
    /// <c>AiModelBuilder.Build(features, times, events)</c>, where the event indicator was never folded
    /// into X — keeps working. The two widths differ by exactly one, so which one was passed is never
    /// ambiguous.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// <paramref name="input"/> has neither <see cref="NumFeatures"/> nor <c>NumFeatures + 1</c> columns.
    /// </exception>
    protected Matrix<T> ExtractCovariates(Matrix<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        if (input.Columns == NumFeatures)
        {
            return input;
        }

        if (input.Columns == NumFeatures + 1)
        {
            return AiDotNetEngine.Current
                .TensorSlice(Tensor<T>.FromMatrix(input), new[] { 0, 1 }, new[] { input.Rows, NumFeatures })
                .ToMatrix();
        }

        throw new ArgumentException(
            $"Predict input has {input.Columns} columns but the model was fit with {NumFeatures} " +
            $"covariates; expected {NumFeatures} (covariates only) or {NumFeatures + 1} " +
            "([event | covariates], matching Train).",
            nameof(input));
    }

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
        var engine = AiDotNetEngine.Current;
        int n = x.Rows;
        var square = new[] { n, n };

        // Pair (i, j) is comparable when subject i had the event and t_i < t_j (so i != j).
        var hadEvent = engine.TensorGreaterThan(
            engine.TensorAbs(engine.TensorSlice(events.AsTensor().Cast<T>(), new[] { 0 }, new[] { n })), NumOps.Zero).Reshape(new[] { n, 1 });
        var column = new[] { n, 1 };
        var row = new[] { 1, n };
        var timesTensor = engine.TensorSlice(times.AsTensor(), new[] { 0 }, new[] { n });
        var earlier = engine.TensorLessThan(
            engine.TensorBroadcastTo(timesTensor.Reshape(column), square),
            engine.TensorBroadcastTo(timesTensor.Reshape(row), square));
        var comparablePairs = engine.TensorMultiply(engine.TensorBroadcastTo(hadEvent, square), earlier);

        // Harrell's C-index: r_i > r_j scores 1, otherwise |r_i - r_j| < 1e-10 scores 1/2.
        var riskTensor = riskScores.AsTensor();
        var riskDiff = engine.TensorSubtract(
            engine.TensorBroadcastTo(riskTensor.Reshape(column), square),
            engine.TensorBroadcastTo(riskTensor.Reshape(row), square));
        var higher = engine.TensorGreaterThan(riskDiff, NumOps.Zero);
        var tied = engine.TensorLessThan(engine.TensorAbs(riskDiff), NumOps.FromDouble(1e-10));
        T halfCredit = NumOps.FromDouble(0.5);
        var credit = engine.TensorAdd(higher,
            engine.TensorMultiplyScalar(engine.TensorSubtract(tied, engine.TensorMultiply(tied, higher)), halfCredit));

        T comparable = engine.TensorSum(comparablePairs);
        if (NumOps.Equals(comparable, NumOps.Zero))
        {
            return halfCredit;
        }

        return NumOps.Divide(engine.TensorSum(engine.TensorMultiply(comparablePairs, credit)), comparable);
    }

    /// <summary>
    /// Event indicators handed over by the builder's three-argument Build, consumed by the next
    /// <see cref="Train(Matrix{T}, Vector{T})"/> and cleared, so a reused model cannot pick up a stale set.
    /// </summary>
    private Vector<int>? _suppliedEvents;

    /// <summary>
    /// Hands the next <see cref="Train(Matrix{T}, Vector{T})"/> its event indicators, so X can stay the
    /// covariate matrix. Internal because it is the builder's way of passing a third input through a
    /// two-argument contract, not a step a caller should have to know about.
    /// </summary>
    /// <param name="events">1 where the event was observed, 0 where the subject was censored.</param>
    internal void SupplyEvents(Vector<int>? events) => _suppliedEvents = events;

    /// <summary>
    /// Whether indicators are waiting to be consumed, which the build pipeline reads to decide that this
    /// model must see every row in the order it was given.
    /// </summary>
    internal bool HasSuppliedEvents => _suppliedEvents is not null;

    /// <summary>
    /// Standard model training — splits the design matrix and redirects to survival-specific training.
    /// </summary>
    /// <param name="x">
    /// The design matrix. Column 0 is the event indicator — 1 if the event was observed, 0 if the
    /// subject was censored — and columns 1.. are the covariates.
    /// </param>
    /// <param name="y">The observed time for each subject: time to event, or time to censoring.</param>
    /// <remarks>
    /// <para>
    /// Survival data is three things — a time, whether the event actually happened, and the covariates —
    /// but the <c>IFullModel</c> contract this base inherits exposes a single <c>Train(X, Y)</c>. The
    /// convention here matches the one <see cref="AiDotNet.CausalInference.CausalModelBase{T}"/> already
    /// uses for its treatment indicator: the extra per-subject signal is column 0 of X.
    /// </para>
    /// <para>
    /// This method used to set every event indicator to 1, which silently asserted that nobody was
    /// censored. Censoring is the entire reason survival analysis exists as a separate field: a subject
    /// still alive when the study ended has not had the event, and counting them as though they had
    /// biases every survival estimate downward. A model that quietly does that is worse than one that
    /// refuses the data, because the answer looks reasonable.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Say you follow 100 patients for a year. Some have the event you are
    /// measuring; others are still fine when the year ends, or move away and stop being followed. You do
    /// not know when — or whether — the event will happen for that second group; you only know it had not
    /// happened yet. That is "censored", and column 0 is where you say which is which.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// X has fewer than two columns, its row count does not match <paramref name="y"/>, or column 0
    /// holds something other than 0 and 1.
    /// </exception>
    public virtual void Train(Matrix<T> x, Vector<T> y)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));

        // Supplied by AiModelBuilder.Build(features, times, events), which has the three signals as
        // three arguments and no reason to fold one into the other two. Taking that path keeps X the
        // covariate matrix, which matters after training as much as during it: prediction takes
        // covariates, and whether the event occurred is the thing being predicted, so a design matrix
        // is not something a caller can produce at predict time.
        var supplied = System.Threading.Interlocked.Exchange(ref _suppliedEvents, null);
        if (supplied is not null && supplied.Length == y.Length && x.Rows == y.Length)
        {
            FitSurvival(x, y, supplied);
            return;
        }

        if (x.Columns < 2)
        {
            throw new ArgumentException(
                "Survival models require at least 2 columns in X: column 0 is the event indicator " +
                "(1 = the event was observed, 0 = the subject was censored) and columns 1.. are the " +
                "covariates. Y is the observed time for each subject.",
                nameof(x));
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException(
                $"Sample count mismatch: X has {x.Rows} rows but Y has {y.Length} times.",
                nameof(y));
        }

        int n = x.Rows;
        int p = x.Columns - 1;
        var engine = AiDotNetEngine.Current;
        var design = Tensor<T>.FromMatrix(x);
        var indicators = engine.TensorSlice(design, new[] { 0, 0 }, new[] { n, 1 }).Reshape(new[] { n });

        // Reading the indicator strictly rather than rounding: a covariate left in column 0 by
        // mistake is the one failure this cannot recover from, and it would otherwise be read as
        // censoring and silently change the answer.
        var valid = engine.TensorAdd(
            engine.TensorEquals(indicators, NumOps.Zero),
            engine.TensorEquals(indicators, NumOps.One));
        int firstInvalid = valid.ToVector().IndexOf(NumOps.Zero);
        if (firstInvalid >= 0)
        {
            T indicator = indicators.GetFlatIndexValue(firstInvalid);
            throw new ArgumentException(
                $"Column 0 of X is the event indicator and must be 0 or 1; found {indicator} at row " +
                $"{firstInvalid}. If that column is a covariate, move it: survival training needs to know which " +
                "subjects were censored, and cannot infer it.",
                nameof(x));
        }

        var events = indicators.Cast<int>().ToVector();
        var features = engine.TensorSlice(design, new[] { 0, 1 }, new[] { n, p }).ToMatrix();

        FitSurvival(features, y, events);
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

        int firstNonPositive = AiDotNetEngine.Current
            .TensorGreaterThan(times.AsTensor(), NumOps.Zero)
            .ToVector()
            .IndexOf(NumOps.Zero);
        if (firstNonPositive >= 0)
        {
            throw new ArgumentException(
                $"All times must be positive. Found non-positive time at index {firstNonPositive}.");
        }

        var eventValues = events.ToArray();
        int firstInvalid = Array.FindIndex(eventValues, e => e != 0 && e != 1);
        if (firstInvalid >= 0)
        {
            throw new ArgumentException(
                $"Event indicators must be 0 or 1. Found {eventValues[firstInvalid]} at index {firstInvalid}.");
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
        var eventFlags = events.ToArray();
        var eventTimes = times.GetElements(Enumerable.Range(0, eventFlags.Length).Where(i => eventFlags[i] == 1));
        if (eventTimes.Length == 0)
        {
            return new Vector<T>(0);
        }

        return AiDotNetEngine.Current.TensorUnique(eventTimes.AsTensor(), sorted: true).ToVector();
    }

    /// <summary>
    /// Counts, for each sorted event time, the subjects at risk (observed time &gt;= event time) and
    /// the events observed at that time (within 1e-10).
    /// </summary>
    /// <param name="times">The observed times.</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    /// <param name="eventTimes">The sorted unique event times.</param>
    /// <returns>The at-risk counts and event counts, one per event time.</returns>
    protected (Tensor<T> AtRisk, Tensor<T> Events) CountAtRiskAndEvents(
        Vector<T> times, Vector<int> events, Vector<T> eventTimes)
    {
        var engine = AiDotNetEngine.Current;
        int numTimes = eventTimes.Length;
        int n = times.Length;
        var grid = new[] { numTimes, n };
        var observed = engine.TensorBroadcastTo(times.AsTensor().Reshape(new[] { 1, n }), grid);
        var eventGrid = engine.TensorBroadcastTo(eventTimes.AsTensor().Reshape(new[] { numTimes, 1 }), grid);

        // At risk if observation time >= event time
        var atRiskMask = engine.TensorAdd(
            engine.TensorGreaterThan(observed, eventGrid),
            engine.TensorEquals(observed, eventGrid));

        // Event at this time if event occurred and time matches
        var hadEvent = engine.TensorBroadcastTo(
            engine.TensorEquals(events.AsTensor().Cast<T>(), NumOps.One).Reshape(new[] { 1, n }), grid);
        var atEventTime = engine.TensorLessThan(
            engine.TensorAbs(engine.TensorSubtract(observed, eventGrid)), NumOps.FromDouble(1e-10));

        return (engine.ReduceSum(atRiskMask, new[] { 1 }),
            engine.ReduceSum(engine.TensorMultiply(hadEvent, atEventTime), new[] { 1 }));
    }

    /// <summary>
    /// Evaluates a right-continuous step function at each query time: the value at the last step time
    /// &lt;= t, or <paramref name="before"/> when t precedes the first step time.
    /// </summary>
    /// <param name="queryTimes">The times to evaluate at.</param>
    /// <param name="stepTimes">The sorted step times.</param>
    /// <param name="values">The function value from each step time on.</param>
    /// <param name="before">The value before the first step time.</param>
    /// <returns>The step function value at each query time.</returns>
    protected Tensor<T> EvaluateStepFunction(Vector<T> queryTimes, Vector<T> stepTimes, Vector<T> values, T before)
    {
        var engine = AiDotNetEngine.Current;
        int q = queryTimes.Length;
        int m = stepTimes.Length;
        var grid = new[] { q, m };
        var query = engine.TensorBroadcastTo(queryTimes.AsTensor().Reshape(new[] { q, 1 }), grid);
        var steps = engine.TensorBroadcastTo(stepTimes.AsTensor().Reshape(new[] { 1, m }), grid);

        // The count of sorted step times <= t indexes [before, v_0, v_1, ...].
        var reached = engine.TensorAdd(
            engine.TensorLessThan(steps, query),
            engine.TensorEquals(steps, query));
        var stepIndex = engine.ReduceSum(reached, new[] { 1 }).Cast<int>();
        var padded = engine.TensorConcatenate(new[]
        {
            Tensor<T>.CreateDefault(new[] { 1 }, before),
            values.AsTensor()
        });

        return engine.TensorGather(padded, stepIndex, 0);
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
            byte[] serialized = AiDotNet.Models.ModelStateEnvelope.Append(
                DeclaredState, SerializeInternalUnchecked());
            var copy = CreateNewInstance();
            if (copy is SurvivalModelBase<T> copyBase)
            {
                byte[] inner = AiDotNet.Models.ModelStateEnvelope.Extract(
                    copyBase.DeclaredState, serialized);
                copyBase.DeserializeInternalUnchecked(inner);

                // Declared state is restored first because it materializes fitted vector lengths;
                // the flat parameter vector then remains authoritative for every registered value.
                if (IsFitted)
                {
                    AiDotNet.Models.Parameters.ParameterTransfer.Copy(this, copyBase);
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


    /// <inheritdoc />
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
