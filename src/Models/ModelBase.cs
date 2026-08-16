using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models;

/// <summary>
/// Abstract base class for standalone models that directly implement <see cref="IFullModel{T, TInput, TOutput}"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Provides common infrastructure and sensible defaults for standalone model implementations
/// that are not wrappers around other models. Subclasses must implement core model behavior:
/// prediction, training, parameter management, loss function, and cloning.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for building standalone machine learning models.
/// Models like linear regression, expression trees, gradient boosting, and ensembles all inherit
/// from this class. It handles boilerplate like serialization and feature tracking so each model
/// only needs to implement its core prediction and training logic.
/// </para>
/// </remarks>
public abstract class ModelBase<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>,
    IParameterizable<T, TInput, TOutput>, IFeatureAware, IGradientComputable<T, TInput, TOutput>,
    IParameterManifestProvider, IParameterChunkSource<T>
{
    /// <summary>
    /// Gets the hardware-accelerated computation engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public abstract ILossFunction<T> DefaultLossFunction { get; }

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public abstract void Train(TInput input, TOutput expectedOutput);

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata() => new();

    // --- IParameterizable ---

    /// <summary>
    /// The components this model's parameters live in, in registration order, which is also the
    /// serialization order.
    /// </summary>
    private readonly ParameterComponentRegistry<T> _parameterRegistry = new();
    private bool _componentsRegistered;

    /// <summary>
    /// Declares a component whose parameters belong to this model's surface. Registration order is
    /// serialization order, so keep it stable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Register only what is TRAINED. A frozen teacher, a target copy or a cached projection is not
    /// an independent parameter: registering one inflates the count and hands an optimizer weights
    /// that are only ever meant to be written to.
    /// </para>
    /// <para>Null is tolerated, so a model may register a component a configuration did not build,
    /// and registration is idempotent by reference.</para>
    /// </remarks>
    protected void RegisterParameterComponent(
        IParameterSource<T>? component,
        [System.Runtime.CompilerServices.CallerArgumentExpression(nameof(component))]
        string? componentExpression = null,
        [System.Runtime.CompilerServices.CallerMemberName] string? memberName = null)
        => _parameterRegistry.RegisterLegacy(
            GetType().FullName ?? GetType().Name, memberName, componentExpression, component);

    /// <summary>Registers an exceptional component by stable identity.</summary>
    protected void RegisterParameterComponent(
        string stableId,
        IParameterSource<T>? component,
        ParameterSlotRole role = ParameterSlotRole.Trainable,
        ParameterAvailability availability = ParameterAvailability.Construction)
        => _parameterRegistry.Register(stableId, component, role, availability);

    /// <summary>
    /// Declare this model's trainable components here with <see cref="RegisterParameterComponent"/>.
    /// Called once, lazily, so it runs after the constructor has built them.
    /// </summary>
    protected virtual void RegisterComponents()
    {
    }

    /// <summary>Generated override chain for fields declared across the model hierarchy.</summary>
    protected virtual void RegisterGeneratedParameterComponents(ParameterComponentRegistry<T> registry)
    {
    }

    /// <summary>State that is not a flat parameter vector, declared once and persisted by the base.</summary>
    private readonly ModelStateRegistry<T> _stateRegistry = new();
    private bool _stateRegistered;

    /// <summary>
    /// Declare state here that <see cref="GetParameters"/> does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// <para>
    /// Every model whose learned state IS its parameter vector needs nothing here. The rest used to
    /// hand-write a Serialize/Deserialize pair, because there was nowhere to say "this is state too"
    /// -- and a hand-written pair is two places to forget the same field.
    /// </para>
    /// <para>
    /// A declaration is a name and an accessor pair. Both halves of the payload are driven by it, so
    /// they cannot drift; nothing here touches a writer or a reader.
    /// </para>
    /// </remarks>
    protected virtual void RegisterState(ModelStateRegistry<T> state)
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
    private ModelStateRegistry<T> State
    {
        get
        {
            if (!_stateRegistered)
            {
                _stateRegistered = true;
                RegisterGeneratedState(_stateRegistry);
                RegisterState(_stateRegistry);
            }
            return _stateRegistry;
        }
    }

    /// <summary>
    /// Runs after <see cref="SetParameters"/> has distributed values into the components. Override
    /// to refresh anything DERIVED from them.
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

                // Generated sources hold accessors rather than snapshots, so registration can be
                // latched even when a fitted field is currently null. Its manifest entry remains
                // ShapeDeferred and observes the field when Fit materializes it later.
                _componentsRegistered = true;

            }
            return _parameterRegistry.Components;
        }
    }

    /// <inheritdoc />
    public ParameterLayoutSnapshot ParameterLayout
    {
        get
        {
            _ = Components;
            return _parameterRegistry.ParameterLayout;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Concatenates the registered components in registration order, so the length is
    /// <see cref="ParameterCount"/> by construction rather than by agreement.
    /// <para>
    /// Virtual rather than abstract: a model that registers its components inherits all three
    /// surfaces and writes no parameter plumbing at all. It was abstract, which FORCED every one of
    /// the 299 types in this hierarchy to hand-write the triple -- the identical defect LayerBase
    /// and DiffusionModelBase had, and the reason count and vector could read different sources and
    /// silently disagree.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        var components = Components;
        if (components.Count == 0) return new Vector<T>(0);
        return _parameterRegistry.GetParameters();
    }

    /// <summary>
    /// Whether this model's weights can be written. Override and return <c>false</c> for a model
    /// running a loaded graph it does not own; the default is <c>true</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A model wrapping an ONNX session cannot honour a restore: the weights belong to the loaded
    /// graph, and writing the native-side tensors would leave the model reporting new parameters
    /// while still computing with the old graph. Hundreds of models expressed that by overriding
    /// the parameter surfaces with <c>if (!_useNativeMode) throw new NotSupportedException(...)</c>
    /// and hand-rolling the rest of the method around it -- the same refusal written out hundreds
    /// of times, each free to word it differently or to guard one surface and forget another.
    /// </para>
    /// <para>
    /// Stating it once here means a model declares the fact and inherits the behaviour:
    /// <c>protected override bool SupportsParameterMutation =&gt; _useNativeMode;</c>. Read-only
    /// access -- <see cref="ParameterCount"/> and <see cref="GetParameters"/> -- is deliberately
    /// NOT gated: an ONNX-mode model can still be counted and inspected, it just cannot be written.
    /// </para>
    /// </remarks>
    protected virtual bool SupportsParameterMutation => true;

    /// <summary>
    /// Throws the standard refusal when <see cref="SupportsParameterMutation"/> is <c>false</c>.
    /// </summary>
    protected void GuardParameterMutation()
    {
        if (!SupportsParameterMutation)
        {
            throw new NotSupportedException(
                $"{GetType().Name} is not in a mode where its parameters can be written: its weights "
                + "belong to a loaded graph rather than to this model. Construct it in native mode "
                + "to train, restore or otherwise mutate parameters. Reading them -- ParameterCount "
                + "and GetParameters -- is still supported.");
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Distributes a flat vector over the same components <see cref="GetParameters"/> folds, which
    /// is exactly what <see cref="SetParameters"/> does -- the two names are one operation. Models
    /// used to override this because the interface demanded an answer and the base gave none; every
    /// such override either re-sliced the vector across the same components by hand, delegated
    /// straight back, or refused. All three are now the base's job.
    /// </remarks>
    public virtual void UpdateParameters(Vector<T> parameters) => SetParameters(parameters);

    /// <inheritdoc/>
    /// <remarks>The inverse of <see cref="GetParameters"/>: each component takes back the slice it
    /// contributed, then <see cref="OnParametersRestored"/> refreshes whatever derives from them.
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        GuardParameterMutation();

        var components = Components;
        if (components.Count == 0 && parameters.Length != 0)
            throw new ArgumentException("This model has no registered parameter layout.", nameof(parameters));
        if (components.Count > 0) _parameterRegistry.SetParameters(parameters);

        OnParametersRestored();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Folds the same enumeration the vector does, so the two cannot disagree. A model that has not
    /// registered components falls back to measuring its own <see cref="GetParameters"/>, which is
    /// what every unconverted model in this hierarchy still relies on.
    /// </remarks>
    public virtual long ParameterCount
    {
        get
        {
            var components = Components;
            if (components.Count == 0) return GetParameters().Length;
            return _parameterRegistry.ParameterCount;
        }
    }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization
    {
        get
        {
            _ = Components;
            return _parameterRegistry.HasComponents
                ? _parameterRegistry.CanInitializeOptimizerParameters
                : ParameterCount > 0;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The registry is the single owner of ordering for count, flat values, restore, and chunks.
    /// Classical sources that do not natively store tensors are represented by one exact payload
    /// tensor per registered component, so every generated model receives a correct chunk surface
    /// without another per-model override.
    /// </remarks>
    public virtual IEnumerable<ParameterChunk<T>> GetParameterStateChunks()
        => _parameterRegistry.GetParameterStateChunks();

    /// <inheritdoc/>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks()
    {
        foreach (var chunk in GetParameterStateChunks())
            yield return chunk.Tensor;
    }

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

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
            var copy = (ModelBase<T, TInput, TOutput>)AiDotNet.Models.CloneEngine.CopyConfiguration(this);
            copy.Deserialize(state);
            return copy;
        }
    }

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> Clone() => DeepCopy();

    // --- IGradientComputable ---

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        throw new NotSupportedException(
            $"Gradient computation is not supported for {GetType().Name}. " +
            "Override ComputeGradients to provide an implementation.");
    }

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();
        if (gradients.Length != parameters.Length)
        {
            throw new ArgumentException(
                $"Gradient length mismatch: expected {parameters.Length}, got {gradients.Length}.",
                nameof(gradients));
        }

        // Vectorized SGD: params = params - lr * gradients
        var scaledGradients = Engine.Multiply(gradients, learningRate);
        parameters = Engine.Subtract(parameters, scaledGradients);

        SetParameters(parameters);
    }

    // --- IModelSerializer ---

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// THIS USED TO THROW, with the message "Override Serialize to provide an implementation", and
    /// that instruction is the whole reason 368 hand-written Serialize/Deserialize halves exist. A
    /// base that refuses the job conscripts every author into doing it by hand, and each hand-written
    /// pair is two places to forget the same field.
    /// </para>
    /// <para>
    /// It does the job now, from what the model has already DECLARED: components registered through
    /// <see cref="RegisterParameterComponent"/> and
    /// <see cref="RegisterGeneratedParameterComponents"/> are folded by
    /// <see cref="GetParameters"/>, so the base can persist all of them without knowing anything
    /// about a particular model. Configuration is not written here -- a clone gets it from the
    /// recorded constructor, and a load applies it to a model the caller already constructed.
    /// </para>
    /// <para>
    /// The type token is not decoration. Without it, loading one model's bytes into another whose
    /// parameter vector happens to be the same length succeeds silently and yields a model that is
    /// confidently wrong, which is precisely the class of defect this work exists to remove.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        // ModelSave is a licensed capability. It used to be enforced only by each model's
        // hand-written Serialize, so deleting one of those in favour of this base -- which is
        // exactly what ADN0060 asks for -- silently removed the gate for that model. Enforcing
        // here means the replacement carries it for every model, and a model that still has its
        // own override keeps enforcing there. Re-entry is harmless: InternalOperation scopes
        // suppress the nested call.
        ModelPersistenceGuard.EnforceBeforeSerialize();

        var parameters = GetParameters();

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        writer.Write(ModelSerializationMagic);
        writer.Write(GetType().FullName ?? GetType().Name);
        writer.Write(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            writer.Write(Convert.ToDouble(parameters[i]));
        }

        // Whatever the model declared that the parameter vector does not carry: a retained training
        // set, fitted knots, kernel centres, an ensemble's children.
        State.WriteAll(writer);

        writer.Flush();
        return stream.ToArray();
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        // Load is not a paid gate, but it is still gated on an Active licence, and for the same
        // reason as Serialize above: this base is now the replacement for the hand-written halves.
        ModelPersistenceGuard.EnforceBeforeDeserialize();

        if (data is null) throw new ArgumentNullException(nameof(data));

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        int magic = reader.ReadInt32();
        if (magic != ModelSerializationMagic)
        {
            throw new InvalidDataException(
                $"{GetType().Name}: payload is not an AiDotNet model state block. A checkpoint written "
                + "by an earlier hand-written Serialize must be regenerated.");
        }

        string savedType = reader.ReadString();
        string liveType = GetType().FullName ?? GetType().Name;
        if (!string.Equals(savedType, liveType, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"State was saved from '{savedType}' and is being loaded into '{liveType}'. Loading it "
                + "would produce a model that is confidently wrong rather than one that fails.");
        }

        int count = reader.ReadInt32();
        var parameters = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        SetParameters(parameters);

        // AFTER SetParameters: declared state may be derived from, or consistent with, the parameter
        // vector, and restoring it first would let the parameter restore overwrite it.
        if (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            State.ReadAll(reader);
        }
    }

    /// <summary>Identifies a model state payload written by <see cref="Serialize"/>.</summary>
    private const int ModelSerializationMagic = unchecked((int)0xA1D00DE1);

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    // --- ICheckpointableModel ---

    /// <inheritdoc/>
    public virtual void SaveState(Stream stream)
    {
        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    /// <inheritdoc/>
    public virtual void LoadState(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        Deserialize(ms.ToArray());
    }

    // --- IFeatureAware ---

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices() => Array.Empty<int>();

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex) => false;

    // --- IFeatureImportance ---

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance() => new(StringComparer.Ordinal);

    // --- IDisposable ---

    private bool _disposed;

    /// <inheritdoc/>
    /// <remarks>
    /// Implements <see cref="System.IDisposable.Dispose"/>. Calls
    /// <see cref="Dispose(bool)"/> with disposing=true and
    /// suppresses finalization. Derived classes that own disposable
    /// resources (neural-network layers, GPU handles, rented tensor
    /// buffers from <c>TensorAllocator</c>) should override the
    /// protected <see cref="Dispose(bool)"/> overload — issue #1136
    /// plan part 3.
    /// </remarks>
    public void Dispose()
    {
        Dispose(disposing: true);
        System.GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources held by this model. Derived classes with
    /// disposable state (layers, GPU handles, rented tensors)
    /// override and call <c>base.Dispose(disposing)</c> at the end.
    /// Default is a no-op for value-only models (linear regressors,
    /// naive Bayes, etc.) that have nothing to release.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
    }
}
