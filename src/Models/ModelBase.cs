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
    IParameterManifestProvider
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
    protected void RegisterParameterComponent(IParameterSource<T>? component)
        => _parameterRegistry.Register(component);

    /// <summary>Registers an exceptional component by stable identity.</summary>
    protected void RegisterParameterComponent(
        string stableId,
        IParameterSource<T>? component,
        ParameterSlotRole role = ParameterSlotRole.Trainable)
        => _parameterRegistry.Register(stableId, component, role);

    /// <summary>
    /// Declare this model's trainable components here with <see cref="RegisterParameterComponent"/>.
    /// Called once, lazily, so it runs after the constructor has built them.
    /// </summary>
    protected virtual void RegisterComponents()
    {
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
                if (this is IGeneratedParameterRegistrar<T> generated)
                    generated.RegisterGeneratedParameters(_parameterRegistry);
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
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc/>
    /// <remarks>
    /// Default implementation yields nothing. Concrete model bases that
    /// represent a real layer stack (NeuralNetworkBase) override to walk
    /// trainable parameters per-tensor; classical / sklearn-style models
    /// (linear regressors, trees, clustering) keep the empty default
    /// because their flat <see cref="GetParameters"/> path is sufficient
    /// (parameter counts are well below int.MaxValue). Foundation-scale
    /// diffusion models override at <c>DiffusionModelBase</c> / per-model
    /// level — tracked by issue #1237.
    /// </remarks>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks() => System.Linq.Enumerable.Empty<Tensor<T>>();

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    // --- ICloneable ---

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();

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
    public virtual byte[] Serialize()
    {
        throw new NotSupportedException(
            $"Serialization is not supported for {GetType().Name}. Override Serialize to provide an implementation.");
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        throw new NotSupportedException(
            $"Deserialization is not supported for {GetType().Name}. Override Deserialize to provide an implementation.");
    }

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
