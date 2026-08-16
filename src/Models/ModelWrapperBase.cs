using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Validation;

namespace AiDotNet.Models;

/// <summary>
/// Abstract base class for model wrappers that delegate to an underlying <see cref="IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Provides default implementations for most <see cref="IFullModel{T, TInput, TOutput}"/>
/// interface members by delegating to the wrapped base model. Subclasses only need to override
/// prediction logic and parameter management specific to their wrapping strategy.
/// </para>
/// <para><b>For Beginners:</b> Some models work by wrapping another model and adding extra behavior.
/// For example, a transfer-learning model wraps a pre-trained model with a feature mapper,
/// or an adversarial defense wraps a model with input preprocessing. This base class handles
/// all the common delegation so wrapper classes only implement what's different.
/// </para>
/// </remarks>
public abstract class ModelWrapperBase<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>,
    IParameterizable<T, TInput, TOutput>, IFeatureAware, IGradientComputable<T, TInput, TOutput>,
    AiDotNet.Models.Parameters.IParameterManifestProvider
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
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Hardware-accelerated computation engine (CPU SIMD / GPU).
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The underlying full model being wrapped.
    /// </summary>
    protected IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelWrapperBase{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="baseModel">The underlying model to wrap.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="baseModel"/> is null.</exception>
    protected ModelWrapperBase(IFullModel<T, TInput, TOutput> baseModel)
    {
        Guard.NotNull(baseModel);
        BaseModel = baseModel;
    }

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => BaseModel.DefaultLossFunction;

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public virtual void Train(TInput input, TOutput expectedOutput)
        => BaseModel.Train(input, expectedOutput);

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata() => BaseModel.GetModelMetadata();

    // --- IParameterizable ---

    /// <summary>
    /// The components this wrapper's parameters live in, in registration order, which is also the
    /// serialization order. Empty for a plain wrapper, which forwards to the model it wraps.
    /// </summary>
    private readonly AiDotNet.Models.Parameters.ParameterComponentRegistry<T> _parameterRegistry = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to this wrapper's own surface, for a wrapper
    /// that holds parameters INSTEAD of the model it wraps.
    /// </summary>
    /// <remarks>
    /// A meta-learning adapted model is the case this exists for: it wraps a base model but carries
    /// its own adapted vector, and forwarding to the wrapped model would read the wrong weights.
    /// Registration order is serialization order, so keep it stable. Null is tolerated and
    /// registration is idempotent by reference.
    /// </remarks>
    protected void RegisterParameterComponent(
        IParameterSource<T>? component,
        [System.Runtime.CompilerServices.CallerArgumentExpression(nameof(component))] string? componentExpression = null,
        [System.Runtime.CompilerServices.CallerMemberName] string? memberName = null)
        => _parameterRegistry.RegisterLegacy(GetType().FullName ?? GetType().Name,
            memberName, componentExpression, component);

    protected void RegisterParameterComponent(string stableId, IParameterSource<T>? component,
        AiDotNet.Models.Parameters.ParameterSlotRole role = AiDotNet.Models.Parameters.ParameterSlotRole.Trainable)
        => _parameterRegistry.Register(stableId, component, role);

    /// <summary>
    /// Declare this wrapper's own trainable components here with
    /// <see cref="RegisterParameterComponent"/>. Leave it alone to forward to the wrapped model.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    protected virtual void RegisterGeneratedParameterComponents(
        AiDotNet.Models.Parameters.ParameterComponentRegistry<T> registry)
    {
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components.
    /// </summary>
    protected virtual void OnParametersRestored()
    {
    }

    private IReadOnlyList<IParameterSource<T>> Components
    {
        get
        {
            if (!_componentsRegistered)
            {
                RegisterGeneratedParameterComponents(_parameterRegistry);
                RegisterComponents();
                _componentsRegistered = true;
            }
            return _parameterRegistry.Components;
        }
    }

    public AiDotNet.Models.Parameters.ParameterLayoutSnapshot ParameterLayout
    {
        get
        {
            var components = Components;
            if (components.Count > 0) return _parameterRegistry.ParameterLayout;
            if (BaseModel is AiDotNet.Models.Parameters.IParameterManifestProvider manifest)
                return manifest.ParameterLayout;
            var parameterizable = InterfaceGuard.TryParameterizable(BaseModel);
            long count = parameterizable?.ParameterCount ?? 0;
            return new AiDotNet.Models.Parameters.ParameterLayoutSnapshot(new[]
            {
                new AiDotNet.Models.Parameters.ParameterSlotDescriptor(
                    $"{BaseModel.GetType().FullName}::wrapped-model",
                    AiDotNet.Models.Parameters.ParameterSlotRole.Trainable,
                    count == 0 ? AiDotNet.Models.Parameters.ParameterReadiness.ParameterFree
                               : AiDotNet.Models.Parameters.ParameterReadiness.Materialized,
                    count)
            });
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Registered components first; a wrapper that registers none forwards to the model it wraps,
    /// which is what a wrapper should do and what this always did.
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        var components = Components;
        if (components.Count == 0)
            return InterfaceGuard.TryParameterizable(BaseModel)?.GetParameters() ?? new Vector<T>(0);

        return _parameterRegistry.GetParameters();
    }

    /// <inheritdoc/>
    /// <remarks>The inverse of <see cref="GetParameters"/>, down whichever of the two paths that
    /// took.</remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var components = Components;
        if (components.Count == 0)
        {
            InterfaceGuard.TryParameterizable(BaseModel)?.SetParameters(parameters);
            return;
        }

        _parameterRegistry.SetParameters(parameters);
        OnParametersRestored();
    }

    /// <inheritdoc/>
    /// <remarks>Folds the same enumeration the vector does, so the two cannot disagree.</remarks>
    public virtual long ParameterCount
    {
        get
        {
            var components = Components;
            if (components.Count == 0)
                return InterfaceGuard.TryParameterizable(BaseModel)?.ParameterCount ?? 0;

            return _parameterRegistry.ParameterCount;
        }
    }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization =>
        InterfaceGuard.TryParameterizable(BaseModel) is { SupportsParameterInitialization: true };
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    // --- ICloneable ---

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// No longer abstract. Configuration is rebuilt from the compile-time clone plan, which records
    /// the constructor the type was built with; learned state is carried through the model's own
    /// public Serialize and Deserialize, so a model that persists something extra keeps it. The
    /// persistence guard is told this is an internal operation because a clone is not a save.
    /// </para>
    /// <para>
    /// A model overrides this only when the generator reports that it cannot rebuild the type --
    /// a constructor parameter with no member holding its value -- and the build names which one.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, TInput, TOutput> DeepCopy()
    {
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] state = Serialize();
            var copy = (ModelWrapperBase<T, TInput, TOutput>)AiDotNet.Models.CloneEngine.CopyConfiguration(this);
            copy.Deserialize(state);
            return copy;
        }
    }

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> Clone() => DeepCopy();

    // --- IGradientComputable ---

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
        => InterfaceGuard.GradientComputable(BaseModel).ComputeGradients(input, target, lossFunction ?? DefaultLossFunction);

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
        => InterfaceGuard.GradientComputable(BaseModel).ApplyGradients(gradients, learningRate);

    // --- IModelSerializer ---

    /// <inheritdoc/>
    /// <remarks>
    /// Appends the declared-state trailer that <see cref="Deserialize"/> already strips. Without
    /// this the two halves disagreed: Extract was called on the way in, Append was never called on
    /// the way out, so anything a wrapper declared was read back but never written - which is the
    /// "two places to forget the same field" defect this base exists to remove, in the base itself.
    /// </remarks>
    public virtual byte[] Serialize()
        => AiDotNet.Models.ModelStateEnvelope.Append(DeclaredState, BaseModel.Serialize());

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        // Strips and applies any declared-state trailer, so the body below reads the payload
        // exactly as it did before this existed.
        data = AiDotNet.Models.ModelStateEnvelope.Extract(DeclaredState, data);
        Guard.NotNull(data);
        BaseModel.Deserialize(data);
    }

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath) => BaseModel.SaveModel(filePath);

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath) => BaseModel.LoadModel(filePath);

    // --- ICheckpointableModel ---

    /// <inheritdoc/>
    public virtual void SaveState(Stream stream) => BaseModel.SaveState(stream);

    /// <inheritdoc/>
    public virtual void LoadState(Stream stream) => BaseModel.LoadState(stream);

    // --- IFeatureAware ---

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
        => InterfaceGuard.TryFeatureAware(BaseModel)?.GetActiveFeatureIndices() ?? Enumerable.Empty<int>();

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        => InterfaceGuard.TryFeatureAware(BaseModel)?.SetActiveFeatureIndices(featureIndices);

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex)
        => InterfaceGuard.TryFeatureAware(BaseModel)?.IsFeatureUsed(featureIndex) ?? false;

    // --- IFeatureImportance ---

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance() => BaseModel.GetFeatureImportance();

    // --- IDisposable (issue #1136 plan part 3) ---

    private bool _disposed;

    /// <inheritdoc/>
    /// <remarks>
    /// Forwards Dispose to the wrapped <see cref="BaseModel"/> when
    /// it implements IDisposable. The wrapper itself owns no
    /// additional disposable state beyond the base model reference.
    /// </remarks>
    public void Dispose()
    {
        Dispose(disposing: true);
        System.GC.SuppressFinalize(this);
    }

    /// <summary>Disposes the wrapped base model if it is disposable. Override + call base for additional cleanup.</summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            (BaseModel as System.IDisposable)?.Dispose();
        }
        _disposed = true;
    }
}
