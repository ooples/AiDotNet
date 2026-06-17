#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.MixedPrecision;
// 0.68.0 of AiDotNet.Tensors introduced its own MixedPrecisionConfig under
// Engines.Autodiff (the engine-side fp16/bf16 mixed-precision plumbing the
// repo asked for in ooples/AiDotNet.Tensors#276). Alias the local one to a
// distinct name so the two coexist without ambiguity at every reference.
using LocalMixedPrecisionConfig = AiDotNet.MixedPrecision.MixedPrecisionConfig;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;
using System.Reflection;
using System.Threading;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Base class for all neural network implementations in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A neural network is a computing system inspired by the human brain. It consists of 
/// interconnected "layers" of artificial neurons that process information and learn patterns from data.
/// This class provides the foundation for building different types of neural networks.
/// </para>
/// </remarks>
public abstract class NeuralNetworkBase<T> : INeuralNetworkModel<T>, IInterpretableModel<T>, IInputGradientComputable<T>, IConfigurableModel<T>, IModelShape, IDisposable,
    IParameterizable<T, Tensor<T>, Tensor<T>>, IFeatureAware, IGradientComputable<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// The internal collection of layers that make up this neural network.
    /// </summary>
    /// <remarks>
    /// This field is private to ensure parameter count cache invalidation.
    /// Use the Layers property for read access or AddLayerToCollection/RemoveLayerFromCollection methods for modifications.
    /// </remarks>
    private readonly List<ILayer<T>> _layers;


    /// <summary>
    /// Gets the collection of layers that make up this neural network (read-only access).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Layers are the building blocks of neural networks. Each layer contains
    /// neurons that process information and pass it to the next layer. A typical network has
    /// an input layer (receives data), hidden layers (process data), and an output layer (produces results).
    /// <para>
    /// <b>Important:</b> Do not directly modify this collection (e.g., Layers.Add()).
    /// Use AddLayerToCollection() or RemoveLayerFromCollection() instead to ensure proper cache invalidation.
    /// </para>
    /// </remarks>
    public List<ILayer<T>> Layers => _layers;

    /// <summary>
    /// Gets the collection of layers that make up this neural network (internal read-only access).
    /// </summary>
    /// <remarks>
    /// This accessor enables internal integrations (e.g., builder-time augmentation) without exposing the mutable layer list
    /// as part of the public API surface area.
    /// </remarks>
    internal IReadOnlyList<ILayer<T>> LayersReadOnly => _layers;

    /// <summary>
    /// Inserts a layer into the internal layer collection and invalidates the parameter count cache.
    /// </summary>
    /// <remarks>
    /// This is intended for internal composition features that need to augment a network safely while maintaining cache correctness.
    /// </remarks>
    internal void InsertLayerIntoCollection(int index, ILayer<T> layer)
    {
        _layers.Insert(index, layer);
        InvalidateParameterCountCache();
    }

    /// <summary>
    /// Gets the number of layers in this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many processing stages (layers) your network has.
    /// More layers generally means the network can learn more complex patterns.
    /// </remarks>
    public int LayerCount => _layers.Count;

    /// <summary>
    /// The architecture definition for this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The architecture defines the structure of your neural network - how many layers it has,
    /// how many neurons are in each layer, and how they're connected. Think of it as the blueprint for your network.
    /// </remarks>
    /// <summary>
    /// The architecture descriptor. Always non-null — layer-only models
    /// (backbones composed of lazy layers, sub-modules whose shape is
    /// derived from a parent network) pass a layer-only stub via
    /// <see cref="NeuralNetworkArchitecture{T}.CreateLayerOnly"/> so
    /// the existing 100+ <c>Architecture.X</c> reads keep working.
    /// Use <see cref="IsLayerOnlyModel"/> to detect the stub case when
    /// you need to fall back to layer-derived shape resolution.
    /// </summary>
    public readonly NeuralNetworkArchitecture<T> Architecture;

    /// <summary>
    /// True when the model was constructed via the layer-only ctor
    /// (architecture is a stub with no semantic input contract). Use
    /// this to gate code that would otherwise read meaningful values
    /// like <c>Architecture.InputHeight</c> — for layer-only models
    /// those are sentinel <c>-1</c> dims, and the real input shape
    /// comes from <see cref="Layers"/>[0].GetInputShape().
    /// </summary>
    // Lazy-shape plumbing, not a user-facing model capability — exposed
    // protected internal so derived networks (e.g. layer-only test
    // scaffolds, internal lazy-shape probes) can read the flag, but
    // external consumers can't take a dependency on this internal
    // implementation detail.
    protected internal bool IsLayerOnlyModel => Architecture.IsLayerOnly;

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This set contains feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on feature importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tracks which parts of your input data have been manually
    /// selected as important for the neural network, regardless of what the network would
    /// automatically determine based on weights.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// Configuration options for this neural network model.
    /// </summary>
    /// <remarks>
    /// Derived classes should set this to their specific options type in their constructor.
    /// </remarks>
    protected ModelOptions Options { get; set; } = new NeuralNetworkOptions();

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Mathematical operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Stores the input values for each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When data flows through the network, we need to remember what values went into each layer.
    /// This is necessary for the learning process (backpropagation).
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerInputs = new Dictionary<int, Tensor<T>>();

    /// <summary>
    /// Stores the output values from each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Similar to layer inputs, we also need to remember what values came out of each layer
    /// during the learning process.
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerOutputs = new Dictionary<int, Tensor<T>>();

    /// <summary>
    /// Gets the thread-safe random number generator for initialization.
    /// </summary>
    /// <remarks>
    /// Uses the centralized RandomHelper which is thread-safe and avoids creating multiple instances per thread.
    /// </remarks>
    protected static Random Random => RandomHelper.ThreadSafeRandom;

    /// <summary>
    /// The loss function used to calculate error during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss function measures how wrong the network's predictions are.
    /// Different types of problems need different loss functions:
    /// - Classification problems often use Cross Entropy Loss
    /// - Regression problems often use Mean Squared Error
    /// - Ranking problems might use Hinge Loss
    /// 
    /// This is like having different ways to score different games - you wouldn't use the same
    /// scoring system for basketball and golf.
    /// </para>
    /// </remarks>
    protected ILossFunction<T> LossFunction;

    /// <summary>
    /// The last calculated loss value during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss value tells you how well your neural network is performing.
    /// A lower loss means better performance. This field stores the most recent loss value
    /// calculated during training, which you can use to track progress.
    /// </para>
    /// </remarks>
    protected T? LastLoss;

    /// <summary>
    /// Indicates whether the network is currently in training mode.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when they're making predictions.
    /// In training mode, the network keeps track of additional information needed for learning.
    /// </remarks>
    public bool IsTrainingMode { get; internal set; } = true;

    /// <summary>
    /// Indicates whether this network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not all neural networks can learn. Some are designed only for making predictions
    /// with pre-set parameters. This property tells you if the network can learn from data.
    /// </remarks>
    public virtual bool SupportsTraining => Layers.Count > 0;

    /// <summary>
    /// Gets whether all layers in the network support GPU-resident training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GPU-resident training keeps all data on GPU during the entire training loop:
    /// - Forward pass runs on GPU
    /// - Loss computation on GPU
    /// - Backward pass on GPU
    /// - Parameter updates on GPU
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When this returns true, training can be much faster because
    /// data doesn't need to be copied back and forth between CPU and GPU each step.
    /// </para>
    /// </remarks>
    public virtual bool SupportsGpuTraining
    {
        get
        {
            if (Layers.Count == 0) return false;

            // All layers must support GPU training
            foreach (var layer in Layers)
            {
                // Non-LayerBase<T> layers don't support GPU training
                if (layer is not LayerBase<T>)
                    return false;
                if (layer is LayerBase<T> layerBase && !layerBase.SupportsGpuTraining)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Gets whether GPU-resident training can be used right now.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This combines layer support with GPU engine availability.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Check this before calling TrainBatchGpu(). If false,
    /// use the standard TrainBatch() method instead.
    /// </para>
    /// </remarks>
    public virtual bool CanTrainOnGpu => SupportsGpuTraining && AiDotNetEngine.Current is DirectGpuTensorEngine;

    /// <summary>
    /// Gets the GPU tensor engine when available, or null if not using GPU.
    /// </summary>
    protected DirectGpuTensorEngine? GpuEngine => AiDotNetEngine.Current as DirectGpuTensorEngine;

    /// <summary>
    /// Backing storage for the gradient-clip max norm. Subclasses may assign
    /// to this in their constructor / init path; the public double-typed
    /// accessor <see cref="MaxGradNormValue"/> reads through to it.
    /// </summary>
    protected T MaxGradNorm;

    /// <summary>
    /// Maximum allowed global L2 norm for gradients per training step,
    /// exposed as a <c>double</c>. When the total gradient norm across all
    /// trainable parameters exceeds this value, every gradient is scaled
    /// down by <c>maxNorm / totalNorm</c> before the optimizer step.
    /// Mirrors PyTorch's <c>torch.nn.utils.clip_grad_norm_(params, max_norm)</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Default: 1.0 from the ctor. Subclasses that need per-model control
    /// (e.g. GraFPrint sourcing the value from <c>GraFPrintOptions</c>)
    /// should override this property and return the option-driven value;
    /// returning 0 disables clipping for that subclass.
    /// </para>
    /// <para>
    /// Active in BOTH the eager training path (TrainWithTape calls
    /// ApplyGradientClipping just before optimizer.Step) and the compiled
    /// fused-optimizer path (CompiledTapeTrainingStep wires this onto the
    /// plan so it's applied inside <c>plan.Step()</c> between the backward
    /// pass and the fused Adam update). Without clipping, deep networks
    /// with batch-norm at small batch sizes can produce single-step Adam
    /// explosions that drive loss up by orders of magnitude on the very
    /// first iteration (GraFPrint's 53-layer pyramid is the canonical case
    /// — single-sample BN drives variance toward zero, the eps-floored
    /// reciprocal sqrt blows up activations, the chain amplifies).
    /// </para>
    /// </remarks>
    public virtual double MaxGradNormValue => NumOps.ToDouble(MaxGradNorm);

    /// <summary>
    /// Backwards-compatible <c>T</c>-typed accessor for code that historically
    /// read the protected field directly. Routes through the public
    /// <see cref="MaxGradNormValue"/> virtual so subclasses that override
    /// the public accessor (e.g. GraFPrint reading from its options) flow
    /// the option-driven value into every clipping helper. Reading the
    /// backing field directly would bypass those overrides — including the
    /// "return 0 to disable clipping" contract.
    /// </summary>
    protected T MaxGradNormT => NumOps.FromDouble(MaxGradNormValue);

    /// <summary>
    /// Cached parameter count to avoid repeated Sum() calculations.
    /// Null when invalid (layers modified).
    /// </summary>
    private long? _cachedParameterCount;

    /// <summary>
    /// Mixed-precision training context (null if mixed-precision is disabled).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training uses both 16-bit (FP16) and 32-bit (FP32) floating-point
    /// numbers to speed up training while maintaining accuracy. When enabled, this context manages the conversion
    /// between different precisions and handles loss scaling to prevent numerical issues.
    /// </para>
    /// </remarks>
    protected MixedPrecisionContext? _mixedPrecisionContext;

    /// <summary>
    /// Memory manager for gradient checkpointing and activation pooling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The memory manager helps train larger models by reducing memory usage.
    /// It implements gradient checkpointing (trading compute for memory) and activation pooling
    /// (reusing tensor allocations to reduce garbage collection).
    /// </para>
    /// </remarks>
    protected Training.Memory.TrainingMemoryManager<T>? _memoryManager;

    /// <summary>
    /// Gets whether memory management (gradient checkpointing/pooling) is enabled.
    /// </summary>
    public bool IsMemoryManagementEnabled => _memoryManager is not null;

    /// <summary>
    /// Gets whether gradient checkpointing is enabled.
    /// </summary>
    public bool IsGradientCheckpointingEnabled => _memoryManager?.IsCheckpointingEnabled ?? false;

    /// <summary>
    /// Gets whether mixed-precision training is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property tells you if the network is using mixed-precision training.
    /// Mixed-precision can provide 2-3x faster training on modern GPUs with Tensor Cores.
    /// </para>
    /// </remarks>
    public bool IsMixedPrecisionEnabled => _mixedPrecisionContext != null;

    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the network.</param>
    /// <param name="lossFunction">The loss function used for training.</param>
    /// <param name="maxGradNorm">Optional gradient-clipping max norm.</param>
    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
    {
        Architecture = architecture;
        // Begin a deterministic per-layer initialization-seed sequence for this
        // model's construction BEFORE the derived constructor builds its layers.
        // Each LayerBase<T> constructor pulls a seed from this scope so weight
        // init derives from architecture.RandomSeed instead of the process-shared
        // RandomHelper.ThreadSafeRandom (whose cumulative state made initial
        // weights depend on prior, unrelated work — the cross-test / cross-process
        // training-determinism bug). Inert when no seed was requested (production
        // default), preserving the existing non-reproducible init behaviour.
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(architecture.RandomSeed);
        _layers = new List<ILayer<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        MaxGradNorm = NumOps.FromDouble(maxGradNorm);
        LossFunction = lossFunction;
        _cachedParameterCount = null;
        _sensitiveFeatures = new Vector<int>(0);
        // Concrete subclass's type name threads through so disk-cached plans in
        // PlanCache.Current don't collide between different model classes.
        _compileHost = new CompiledModelHost<T>(
            shapeMode: SymbolicShapeMode.BatchDynamic,
            modelIdentity: GetType().FullName ?? GetType().Name);
    }

    /// <summary>
    /// Creates a layer-only neural network with no semantic architecture.
    /// The base receives a stub architecture (all-sentinel dims, IsLayerOnly=true)
    /// so existing <c>Architecture.X</c> consumers keep compiling, and methods
    /// that need a real input shape consult <see cref="Layers"/>[0] instead.
    /// Intended for sub-modules (residual blocks used inside ResNet, etc.) and
    /// detection backbones whose input contract is owned by a parent network.
    /// </summary>
    /// <param name="lossFunction">The loss function used for training.</param>
    /// <param name="maxGradNorm">Optional gradient-clipping max norm.</param>
    protected NeuralNetworkBase(ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
        : this(architecture: NeuralNetworkArchitecture<T>.CreateLayerOnly(),
               lossFunction: lossFunction,
               maxGradNorm: maxGradNorm)
    {
    }

    /// <summary>
    /// Enables deterministic CPU inference by setting MKL to single-threaded.
    /// Call from model constructors that need bitwise-identical forward passes.
    /// Note: This sets a process-global BLAS flag. Consider calling it once at
    /// application startup rather than from individual model constructors if
    /// multiple models coexist.
    /// </summary>
    public static void EnableDeterministicMode()
    {
        BlasProvider.SetDeterministicMode(true);
    }

    /// <summary>
    /// Applies gradient clipping to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">A list of tensors containing the gradients to be clipped.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the total norm of all gradients and scales them down if the norm exceeds
    /// the maximum allowed gradient norm (_maxGradNorm).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as a safety mechanism. Sometimes, the network might try to
    /// make very large adjustments, which can make learning unstable. This method checks if the
    /// adjustments are too big, and if they are, it scales them down to a safe level. It's like
    /// having a speed limiter on a car to prevent it from going too fast and losing control.
    /// </para>
    /// </remarks>
    protected void ClipGradients(List<Tensor<T>> gradients)
    {
        for (int i = 0; i < gradients.Count; i++)
        {
            gradients[i] = ClipTensorGradient(gradients[i], MaxGradNormT);
        }
    }

    /// <summary>
    /// Clips a single gradient tensor if its norm exceeds the specified maximum norm.
    /// </summary>
    /// <param name="gradient">The gradient tensor to be clipped.</param>
    /// <param name="maxNorm">The maximum allowed norm. If null, uses MaxGradNorm.</param>
    /// <returns>The clipped gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the total norm of the gradient and scales it down if it exceeds
    /// the specified maximum norm. This is the core gradient clipping logic used by all other
    /// gradient clipping methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a safety mechanism to prevent the "exploding gradient" problem.
    /// If the gradient (which represents how much to change the network's parameters) becomes too large,
    /// it can cause the training to become unstable. This method checks if the gradient is too big,
    /// and if so, it scales it down to a safe level.
    /// </para>
    /// <para>
    /// Think of it like having a speed limiter on a car. If the car (gradient) tries to go too fast,
    /// this method slows it down to a safe speed to prevent losing control during training.
    /// </para>
    /// </remarks>
    private Tensor<T> ClipTensorGradient(Tensor<T> gradient, T maxNorm)
    {
        // Compute L2 norm using vectorized operations: sqrt(sum(gradient^2))
        var gradSquared = Engine.TensorMultiply(gradient, gradient);
        T totalNormSquared = Engine.TensorSum(gradSquared);
        T totalNorm = NumOps.Sqrt(totalNormSquared);

        if (NumOps.GreaterThan(totalNorm, maxNorm))
        {
            // Scale gradient using vectorized multiplication
            T scalingFactor = NumOps.Divide(maxNorm, totalNorm);
            gradient = Engine.TensorMultiplyScalar(gradient, scalingFactor);
        }

        return gradient;
    }

    /// <summary>
    /// Clips the gradient tensor if its norm exceeds the maximum allowed gradient norm.
    /// </summary>
    /// <param name="gradient">The gradient tensor to be clipped.</param>
    /// <returns>The clipped gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method is a convenience wrapper that clips a gradient tensor using the default MaxGradNorm.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a safety mechanism to prevent the "exploding gradient" problem.
    /// It ensures gradients don't become too large during training, which helps keep the learning process stable.
    /// </para>
    /// </remarks>
    protected Tensor<T> ClipGradient(Tensor<T> gradient)
    {
        return ClipTensorGradient(gradient, MaxGradNormT);
    }

    /// <summary>
    /// Clips the gradient vector if its norm exceeds the maximum allowed gradient norm.
    /// </summary>
    /// <param name="gradient">The gradient vector to be clipped.</param>
    /// <returns>The clipped gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the vector to a tensor, applies gradient clipping, and converts back to a vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is another safety mechanism to prevent the "exploding gradient" problem,
    /// but specifically for vector inputs. It works just like the tensor version but handles vector data.
    /// </para>
    /// </remarks>
    protected Vector<T> ClipGradient(Vector<T> gradient)
    {
        return ClipTensorGradient(Tensor<T>.FromVector(gradient), MaxGradNormT).ToVector();
    }

    /// <summary>
    /// Gets all trainable parameters of the network as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks learn by adjusting their "parameters" (also called weights and biases).
    /// This method collects all those adjustable values into a single list so they can be updated during training.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        // Pre-resolve any lazy layer shapes from the architecture before
        // walking parameters. Issue #1209's lazy-shape-inference layers
        // return empty vectors for their GetParameters when their input
        // shape is still the -1 sentinel — pre-Forward queries on the
        // parent network would otherwise return a 0-length vector even
        // for a fully-architecturally-defined model. Idempotent: chain
        // resolution only happens once per network instance, guarded by
        // each layer's IsShapeResolved short-circuit.
        ResolveLazyLayerShapes();

        // Sum per-layer parameter counts via layer.ParameterCount (cheap
        // metadata read), NOT via layer.GetParameters().Length (which
        // forces every layer to materialize and return its full
        // parameter Vector<T> just to read the Length). The previous
        // implementation walked Layers TWICE — once to count via
        // GetParameters().Length, once to actually copy — doubling the
        // allocation pressure and the per-layer parameter materialization
        // work for deep models. ResolveLazyLayerShapes above has
        // already materialized every lazy layer's shape, so
        // ParameterCount is now safe to read here. Closes review-comment
        // #1271.uxip. Long accumulator + int.MaxValue gate below
        // forwards the same overflow protection as before.
        long totalParameterCountLong = 0;
        foreach (var layer in Layers)
        {
            totalParameterCountLong += layer.ParameterCount;
        }
        int totalParameterCount = ParameterCountHelper.ToFlatVectorSize(totalParameterCountLong);

        var parameters = new Vector<T>(totalParameterCount);
        var destSpan = parameters.AsWritableSpan();

        int currentIndex = 0;
        foreach (var layer in Layers)
        {
            var layerParameters = layer.GetParameters();
            int copyLength = layerParameters.Length;
            if (copyLength == 0) continue;
            layerParameters.AsSpan().Slice(0, copyLength)
                .CopyTo(destSpan.Slice(currentIndex, copyLength));
            currentIndex += copyLength;
        }

        return parameters;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Yields each layer's trainable parameter tensors in order. For
    /// <see cref="ITrainableLayer{T}"/> layers this returns the per-tensor
    /// weight references registered via <c>RegisterTrainableParameter</c>
    /// (zero-copy). For non-trainable / parameterless layers this yields
    /// nothing. Mirrors PyTorch's <c>nn.Module.parameters()</c> generator.
    /// Use this for foundation-scale models where the flat
    /// <see cref="GetParameters"/> path overflows <see cref="int"/> in
    /// either <see cref="Vector{T}"/>.Length or
    /// <see cref="ParameterCount"/>.
    /// </remarks>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks()
    {
        ResolveLazyLayerShapes();

        // SCOPE CONTRACT: chunks must match exactly the parameter set
        // that ParameterCount / GetParameters / SetParameters operate on.
        // Those flat APIs walk only `Layers`; widening this enumeration
        // to include GetExtraTrainableLayers / GetExtraTrainableTensors
        // would make `sum(chunk.Length) > ParameterCount` for models
        // with network-level extras (ViT cls/pos, Conformer subsamplers),
        // causing callers that mix the flat and chunked APIs to mis-size
        // buffers or skip parameters on round-trip.
        //
        // Extras still flow through TrainWithTape via the separate extra-
        // trainable handling path — they're just not surfaced in the
        // chunked enumeration here. If a future PR widens the flat APIs
        // to include extras, this enumeration can match in lockstep.
        //
        // The recursive CollectTrainableLayers walk DOES descend into
        // composite-layer sublayers (DenseBlock BN/Conv, MoE experts) —
        // those are still part of `Layers` from the flat APIs' point of
        // view because GetParameters walks each top-level layer's
        // ParameterCount which already aggregates sublayer params.
        var trainableLayers = Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers, _layerStructureVersion);
        foreach (var trainable in trainableLayers)
        {
            foreach (var t in trainable.GetTrainableParameters())
            {
                if (t is null || t.Length == 0) continue;
                yield return t;
            }
        }
    }

    #region GPU Training Methods

    /// <summary>
    /// Performs a forward pass through the network entirely on GPU.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>The GPU-resident output tensor.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support GPU execution.</exception>
    /// <remarks>
    /// <para>
    /// This method passes data through all layers on GPU without CPU round-trips.
    /// The output remains on GPU and can be used directly for loss computation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Like ForwardWithMemory() but everything stays on the GPU.
    /// This is much faster for training because there's no copying between CPU and GPU.
    /// </para>
    /// </remarks>
    /// <summary>
    /// GPU forward pass with named auxiliary inputs routed to multi-port layers.
    /// Mirrors <see cref="ForwardWithMemory(Tensor{T}, IReadOnlyDictionary{string, Tensor{T}})"/>
    /// but keeps all data on GPU.
    /// </summary>
    /// <param name="input">Primary GPU input tensor.</param>
    /// <param name="auxiliaryInputs">Named GPU auxiliary tensors (e.g., "time_embed").</param>
    /// <returns>Final GPU output tensor.</returns>
    public virtual Tensor<T> ForwardGpu(
        Tensor<T> input,
        IReadOnlyDictionary<string, Tensor<T>> auxiliaryInputs)
    {
        if (!CanTrainOnGpu)
            throw new InvalidOperationException("GPU forward pass is not supported.");

        var current = input;
        foreach (var layer in Layers)
        {
            if (layer is not LayerBase<T> layerBase)
                throw new InvalidOperationException(
                    $"Layer {layer.GetType().Name} does not inherit from LayerBase<T>.");

            if (layerBase.InputPorts.Count > 1)
            {
                var namedInputs = new Dictionary<string, Tensor<T>> { ["input"] = current };
                foreach (var port in layerBase.InputPorts)
                {
                    if (port.Name != "input" && auxiliaryInputs.TryGetValue(port.Name, out var aux))
                        namedInputs[port.Name] = aux;
                }
                current = layerBase.ForwardGpu(namedInputs);
            }
            else
            {
                current = layerBase.ForwardGpu(current);
            }
        }

        return current;
    }


    /// <summary>
    /// Performs backpropagation through all layers with deferred GPU execution.
    /// </summary>
    /// <param name="outputGradients">The GPU-resident gradient of loss with respect to network output.</param>
    /// <param name="options">Optional GPU execution options.</param>
    /// <returns>The GPU-resident gradient with respect to network input.</returns>
    /// <remarks>
    /// <para>
    /// Uses deferred execution to batch all backward pass operations into a single GPU command buffer.
    /// This reduces CPU-GPU synchronization overhead and improves performance.
    /// </para>
    /// </remarks>
    // BackpropagateGpuDeferred removed — tape-based autodiff handles GPU gradient computation.

    /// <summary>
    /// Updates all trainable parameters in the network using GPU-computed gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <param name="momentum">Optional momentum factor (default 0).</param>
    /// <param name="weightDecay">Optional weight decay / L2 regularization factor (default 0).</param>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support GPU training.</exception>
    /// <remarks>
    /// <para>
    /// This method updates weights and biases directly on GPU using gradients computed by BackpropagateGpu.
    /// The update uses: w = w - lr * (grad + weightDecay * w) + momentum * velocity
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After computing gradients with BackpropagateGpu(), call this
    /// to actually update the weights. Everything happens on GPU for maximum speed.
    /// </para>
    /// </remarks>
    [Obsolete("Use UpdateParametersGpu(IGpuOptimizerConfig) instead for full optimizer support.")]
    public virtual void UpdateParametersGpu(T learningRate, T? momentum = default, T? weightDecay = default)
    {
        // Convert to new config-based API
        var config = new SgdGpuConfig(
            NumOps.ToFloat(learningRate),
            momentum: momentum != null ? NumOps.ToFloat(momentum) : 0.0f,
            weightDecay: weightDecay != null ? NumOps.ToFloat(weightDecay) : 0f);
        UpdateParametersGpu(config);
    }

    /// <summary>
    /// Updates all trainable parameters in the network using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration specifying the update algorithm and hyperparameters.</param>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support GPU training.</exception>
    /// <remarks>
    /// <para>
    /// This method updates weights and biases directly on GPU using gradients computed by BackpropagateGpu.
    /// Supports all GPU optimizer types: SGD, Adam, AdamW, RMSprop, Adagrad, NAG, LARS, LAMB.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After computing gradients with BackpropagateGpu(), call this
    /// to actually update the weights. The config determines which optimizer algorithm to use:
    /// - SGD: Simple gradient descent with optional momentum
    /// - Adam: Adaptive learning rates (most popular)
    /// - AdamW: Adam with proper weight decay (best for transformers)
    /// </para>
    /// </remarks>
    public virtual void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (!CanTrainOnGpu)
        {
            throw new InvalidOperationException(
                "GPU parameter updates are not supported. Check CanTrainOnGpu before calling this method.");
        }

        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> layerBase && layerBase.SupportsGpuTraining)
            {
                layerBase.UpdateParametersGpu(config);
            }
        }
    }

    /// <summary>
    /// Updates all trainable parameters with deferred GPU execution.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    /// <param name="options">Optional GPU execution options.</param>
    /// <remarks>
    /// <para>
    /// Uses deferred execution to batch all parameter update operations into a single GPU command buffer.
    /// This reduces CPU-GPU synchronization overhead and improves training performance.
    /// </para>
    /// </remarks>
    public virtual void UpdateParametersGpuDeferred(
        IGpuOptimizerConfig config,
        GpuExecutionOptions? options = null)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        if (engine?.GetBackend() == null)
        {
            // Fallback to non-deferred if no GPU backend
            UpdateParametersGpu(config);
            return;
        }

        var backend = engine.GetBackend() as IAsyncGpuBackend;
        if (backend == null)
        {
            UpdateParametersGpu(config);
            return;
        }

        backend.ExecuteDeferred(
            scope => UpdateParametersGpu(config),
            options);
    }

    /// <summary>
    /// Performs a complete training step (forward + backward + update) on GPU with deferred execution.
    /// </summary>
    /// <param name="input">The GPU-resident input batch.</param>
    /// <param name="target">The GPU-resident target batch.</param>
    /// <param name="config">The GPU optimizer configuration.</param>
    /// <param name="options">Optional GPU execution options for deferred execution.</param>
    /// <returns>The loss value for this batch.</returns>
    /// <remarks>
    /// <para>
    /// This method wraps forward, backward, and update in a deferred execution scope, allowing the GPU
    /// to optimize the entire training step as a single execution graph. This provides significant
    /// performance improvements through:
    /// - Kernel fusion
    /// - Memory optimization
    /// - Stream parallelization
    /// - Reduced synchronization overhead
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the fastest way to train on GPU. Instead of executing each
    /// operation immediately, it records all operations and executes them as one optimized graph.
    /// Think of it like batch processing - more efficient than doing things one at a time.
    /// </para>
    /// </remarks>
    public virtual T TrainBatchGpuDeferred(
        Tensor<T> input,
        Tensor<T> target,
        IGpuOptimizerConfig config,
        GpuExecutionOptions? options = null)
    {
        if (!CanTrainOnGpu)
        {
            throw new InvalidOperationException(
                "GPU training is not supported. Check CanTrainOnGpu before calling this method.");
        }

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException("GPU training requires a GPU engine.");
        }

        options ??= new GpuExecutionOptions
        {
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            EnableGpuResidency = true
        };

        T lossValue = NumOps.Zero;

        var backend = gpuEngine.GetBackend() as IAsyncGpuBackend;
        if (backend == null)
        {
            throw new InvalidOperationException("GPU training requires an async GPU backend.");
        }

        // Execute the entire training step as a deferred graph
        backend.ExecuteDeferred(
            scope =>
            {
                // Forward pass
                var output = ForwardGpu(input);

                // Compute loss
                var lossResult = LossFunction.CalculateLossAndGradientGpu(output, target);
                lossValue = lossResult.Loss;

                // Backward pass

                // Update parameters
                UpdateParametersGpu(config);
            },
            options);

        return lossValue;
    }

    /// <summary>
    /// Performs a complete training step (forward + backward + update) on GPU with deferred execution asynchronously.
    /// </summary>
    /// <param name="input">The GPU-resident input batch.</param>
    /// <param name="target">The GPU-resident target batch.</param>
    /// <param name="config">The GPU optimizer configuration.</param>
    /// <param name="options">Optional GPU execution options for deferred execution.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The loss value for this batch.</returns>
    public virtual async Task<T> TrainBatchGpuDeferredAsync(
        Tensor<T> input,
        Tensor<T> target,
        IGpuOptimizerConfig config,
        GpuExecutionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        if (!CanTrainOnGpu)
        {
            throw new InvalidOperationException(
                "GPU training is not supported. Check CanTrainOnGpu before calling this method.");
        }

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException("GPU training requires a GPU engine.");
        }

        options ??= new GpuExecutionOptions
        {
            EnableGraphCompilation = true,
            EnableAutoFusion = true,
            EnableGpuResidency = true
        };

        T lossValue = NumOps.Zero;

        var backend = gpuEngine.GetBackend() as IAsyncGpuBackend;
        if (backend == null)
        {
            throw new InvalidOperationException("GPU training requires an async GPU backend.");
        }

        // Execute the entire training step as a deferred graph asynchronously
        await backend.ExecuteDeferredAsync(
            scope =>
            {
                // Forward pass
                var output = ForwardGpu(input);

                // Compute loss
                var lossResult = LossFunction.CalculateLossAndGradientGpu(output, target);
                lossValue = lossResult.Loss;

                // Backward pass

                // Update parameters
                UpdateParametersGpu(config);

                return lossValue;
            },
            options,
            cancellationToken);

        return lossValue;
    }

    /// <summary>
    /// Uploads all layer weights to GPU for GPU-resident training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this once before starting GPU training to:
    /// - Create GPU buffers for all weights and biases
    /// - Copy current CPU values to GPU
    /// - Create GPU buffers for gradients and optimizer states
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This prepares the network for GPU training by copying
    /// all learned values to the GPU. After this, training can happen entirely on GPU.
    /// </para>
    /// </remarks>
    public virtual void UploadWeightsToGpu()
    {
        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                layerBase.UploadWeightsToGpu();
            }
        }
    }

    /// <summary>
    /// Downloads all layer weights from GPU back to CPU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this after GPU training to sync updated weights back to CPU for:
    /// - Model saving/checkpointing
    /// - CPU inference
    /// - Weight inspection
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During GPU training, weights are updated on GPU.
    /// The CPU copy becomes stale. Call this to get the latest values back to CPU.
    /// </para>
    /// </remarks>
    public virtual void DownloadWeightsFromGpu()
    {
        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                layerBase.DownloadWeightsFromGpu();
            }
        }
    }

    /// <summary>
    /// Zeros all GPU gradient accumulators in preparation for a new batch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this at the start of each training batch to clear gradients from the previous batch.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Before processing a new batch, you need to clear the old gradients.
    /// Otherwise they accumulate and training goes wrong.
    /// </para>
    /// </remarks>
    public virtual void ZeroGradientsGpu()
    {
        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                layerBase.ZeroGradientsGpu();
            }
        }
    }

    #endregion

    /// <summary>
    /// Extracts a single example from a batch tensor and formats it as a tensor with shape [1, features].
    /// </summary>
    /// <param name="batchTensor">The batch tensor to extract from.</param>
    /// <param name="index">The index of the example to extract.</param>
    /// <returns>A tensor containing a single example with shape [1, features].</returns>
    protected Tensor<T> ExtractSingleExample(Tensor<T> batchTensor, int index)
    {
        // Get the vector for this example
        Vector<T> row = batchTensor.GetRow(index);

        // Create a tensor with shape [1, features]
        return new Tensor<T>([1, row.Length], row);
    }

    /// <summary>
    /// Performs a forward pass through the network while storing intermediate values for backpropagation.
    /// </summary>
    /// <param name="input">The input data to the network.</param>
    /// <returns>The output of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method passes data through the network from input to output, but also
    /// remembers all the intermediate values. This is necessary for the learning process, as the network
    /// needs to know these values when figuring out how to improve.
    /// </para>
    /// <para>
    /// <b>API Change Note:</b> The signature changed from Vector&lt;T&gt; to Tensor&lt;T&gt; to support multi-dimensional
    /// inputs. This is a breaking change. For backward compatibility, consider adding an overload that accepts
    /// Vector&lt;T&gt; and converts it internally to Tensor&lt;T&gt;.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support training.</exception>
    public virtual Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support training mode");
        }

        // Use memory-managed forward if gradient checkpointing is enabled
        if (_memoryManager is not null && _memoryManager.IsCheckpointingEnabled)
        {
            return ForwardWithCheckpointing(input);
        }

        // Standard forward pass - store all activations
        Tensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            // Store input to each layer for backpropagation
            _layerInputs[i] = current;

            // Forward pass through layer with mixed-precision awareness
            // ForwardWithPrecisionCheck automatically handles precision based on
            // the current MixedPrecisionScope and LayerPrecisionPolicy
            current = Layers[i].ForwardWithPrecisionCheck(current);

            // Store output from each layer for backpropagation
            _layerOutputs[i] = current;
        }

        return current;
    }

    /// <summary>
    /// Forward pass with named auxiliary inputs that are routed to layers declaring matching ports.
    /// Layers with a single "input" port receive the sequential output as usual.
    /// Layers with additional ports (e.g., "time_embed") receive the matching tensor from <paramref name="auxiliaryInputs"/>.
    /// </summary>
    /// <param name="input">Primary input tensor (routed as "input" port to each layer).</param>
    /// <param name="auxiliaryInputs">Named auxiliary tensors (e.g., "time_embed" → embedding tensor).</param>
    /// <returns>Final output tensor.</returns>
    public virtual Tensor<T> ForwardWithMemory(Tensor<T> input, IReadOnlyDictionary<string, Tensor<T>> auxiliaryInputs)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("This network does not support training mode");

        Tensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            _layerInputs[i] = current;

            var layer = Layers[i];
            if (layer is LayerBase<T> typedLayer && typedLayer.InputPorts.Count > 1)
            {
                var namedInputs = new Dictionary<string, Tensor<T>> { ["input"] = current };
                foreach (var port in typedLayer.InputPorts)
                {
                    if (port.Name != "input" && auxiliaryInputs.TryGetValue(port.Name, out var aux))
                        namedInputs[port.Name] = aux;
                }
                current = typedLayer.Forward(namedInputs);
            }
            else
            {
                current = layer.ForwardWithPrecisionCheck(current);
            }

            _layerOutputs[i] = current;
        }

        return current;
    }

    /// <summary>
    /// Performs forward pass with gradient checkpointing to reduce memory usage.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient checkpointing trades compute for memory:
    /// - Instead of storing ALL layer activations (high memory), only store SOME checkpoints
    /// - During backprop, recompute the missing activations from the nearest checkpoint
    /// - Typical memory savings: 40-50% with only ~20% extra compute time
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> ForwardWithCheckpointing(Tensor<T> input)
    {
        if (_memoryManager is null)
            throw new InvalidOperationException("Memory manager is not configured.");

        Tensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            // Only store inputs for checkpointed layers
            if (_memoryManager.ShouldCheckpoint(i))
            {
                _layerInputs[i] = current;
            }

            // Forward pass with checkpointing
            current = _memoryManager.ForwardWithCheckpoint(Layers[i], current, i);

            // Only store outputs for checkpointed layers
            if (_memoryManager.ShouldCheckpoint(i))
            {
                _layerOutputs[i] = current;
            }
        }

        return current;
    }

    /// <summary>
    /// Checks if all layers in the network support GPU execution.
    /// Used to determine if the GPU-resident optimization path can be used.
    /// </summary>
    /// <returns>True if all layers can execute on GPU; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks if every layer in your network can run on the GPU.
    /// If even one layer needs the CPU, we can't use the fast GPU-only path.
    /// </para>
    /// </remarks>
    protected virtual bool CanUseGpuResidentPath()
    {
        return Layers.All(layer => layer.CanExecuteOnGpu);
    }

    /// <summary>
    /// Attempts to perform a GPU-resident forward pass with automatic fallback to CPU.
    /// Use this in derived class Forward() methods to get GPU optimization with minimal code.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="result">The output tensor if GPU path succeeded.</param>
    /// <returns>True if GPU path was used successfully; false if CPU path should be used.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Derived Classes:</b> Call this at the start of your Forward() method:
    /// </para>
    /// <code>
    /// public Tensor&lt;T&gt; Forward(Tensor&lt;T&gt; input)
    /// {
    ///     if (TryForwardGpuOptimized(input, out var result))
    ///         return result;
    ///     
    ///     // CPU fallback path
    ///     ...
    /// }
    /// </code>
    /// </remarks>
    protected bool TryForwardGpuOptimized(Tensor<T> input, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out Tensor<T>? result)
    {
        result = null;

        if (Engine is not DirectGpuTensorEngine)
            return false;

        if (!CanUseGpuResidentPath())
            return false;

        try
        {
            // Fix for #1625: the previous `using var gpuResult = ForwardGpu(input)` disposed
            // the GPU tensor before the caller could read its values, so every Predict call
            // returned all-zero output. The returned tensor's lifetime now belongs to the
            // caller (which is how every other Predict path works — the result is the model's
            // output, not an intermediate that can be safely freed).
            result = ForwardGpu(input);
            return true;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not System.Threading.ThreadAbortException)
        {
            // Log GPU failure for diagnostics before falling back to CPU path
            System.Diagnostics.Debug.WriteLine($"[NeuralNetworkBase] GPU forward failed ({ex.GetType().Name}): {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Performs a GPU-resident forward pass, keeping intermediate results on GPU.
    /// Only downloads the final result to CPU when the returned tensor is accessed.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>GPU-resident output tensor. Only downloads when <see cref="IGpuTensor{T}.ToTensor"/> is called.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no GPU backend is available or the engine is not a DirectGpuTensorEngine.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is like the regular forward pass, but keeps all
    /// intermediate calculations on the GPU instead of moving data back and forth between
    /// CPU and GPU. This can be 10-50x faster for multi-layer networks!
    /// </para>
    /// <para>
    /// <b>Performance Tip:</b> Use this method for inference when you have multiple layers
    /// that all support GPU execution. The data stays on the GPU until you call ToTensor()
    /// on the result.
    /// </para>
    /// <code>
    /// // Example: GPU-resident inference
    /// using var gpuResult = network.ForwardGpu(input);
    /// var output = gpuResult; // Only downloads here
    /// </code>
    /// </remarks>
    public virtual Tensor<T> ForwardGpu(Tensor<T> input)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires DirectGpuTensorEngine. Current engine: " +
                Engine.GetType().Name);
        }

        // Upload input to GPU once
        Tensor<T>? current = null;
        bool ownsCurrentTensor = false;

        try
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];

                if (layer.CanExecuteOnGpu)
                {
                    // Layer supports GPU-resident execution
                    if (current is null)
                    {
                        // First GPU layer - upload input
                        current = gpuEngine.UploadToGpu(input, GpuTensorRole.Activation);
                        ownsCurrentTensor = true;
                    }

                    var next = layer.ForwardGpu(current);

                    // Dispose intermediate if we own it (but not the input)
                    if (ownsCurrentTensor && current is not null)
                    {
                        current.Dispose();
                    }

                    current = next;
                    ownsCurrentTensor = true;
                }
                else
                {
                    // Layer doesn't support GPU - fall back to CPU
                    Tensor<T> cpuInput;
                    if (current is not null)
                    {
                        // Download current GPU tensor to CPU
                        cpuInput = current;
                        if (ownsCurrentTensor)
                        {
                            current.Dispose();
                        }
                        current = null;
                        ownsCurrentTensor = false;
                    }
                    else
                    {
                        // Haven't uploaded yet, use original input
                        cpuInput = input;
                    }

                    // Execute on CPU
                    var cpuOutput = layer.Forward(cpuInput);

                    // Check if next layer supports GPU
                    bool nextLayerSupportsGpu = i + 1 < Layers.Count &&
                        Layers[i + 1].CanExecuteOnGpu;

                    if (nextLayerSupportsGpu || i == Layers.Count - 1)
                    {
                        // Upload result to GPU for next layer or final output
                        current = gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
                        ownsCurrentTensor = true;
                    }
                    else
                    {
                        // Keep on CPU for next CPU layer
                        input = cpuOutput; // Reuse input variable for next iteration
                    }
                }
            }

            // Ensure we return a GPU tensor
            if (current is null)
            {
                // All layers were CPU - upload final result
                current = gpuEngine.UploadToGpu(input, GpuTensorRole.Activation);
            }

            return current;
        }
        catch (Exception)
        {
            // Clean up on error
            if (ownsCurrentTensor && current is not null)
            {
                current.Dispose();
            }
            throw;
        }
    }

    /// <summary>
    /// Performs a forward pass using deferred execution for optimized GPU performance.
    /// Operations are recorded and batched into an execution graph that runs with a single sync point.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor from the network.</returns>
    /// <remarks>
    /// <para>
    /// This method uses deferred execution to batch all GPU operations and execute them
    /// as an optimized graph. This provides significant performance improvements over
    /// eager execution by:
    /// - Avoiding synchronization between layers
    /// - Enabling kernel fusion optimizations
    /// - Minimizing CPU-GPU data transfers
    /// </para>
    /// <para><b>Execution Flow:</b></para>
    /// <code>
    /// BeginDeferredScope()
    ///   Layer1.ForwardGpu() → Record GPU op (no sync)
    ///   Layer2.ForwardGpu() → Record GPU op (no sync)
    ///   Layer3.ForwardGpu() → Record GPU op (no sync)
    /// EndDeferredScope() → Execute all → Single sync → Download final result
    /// </code>
    /// <para><b>For Beginners:</b> Think of this like batch cooking vs cooking one dish at a time.
    /// Instead of starting and finishing each layer separately, we plan out all the operations
    /// and then execute them together more efficiently.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the engine doesn't support deferred execution.
    /// </exception>
    public virtual Tensor<T> ForwardDeferred(Tensor<T> input)
    {
        // Check if we have a DirectGpuTensorEngine
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            // Fall back to regular predict if no GPU engine
            return Predict(input);
        }

        // Try to use deferred scope for batched execution with optimization
        var deferredScope = gpuEngine.BeginDeferredScope();
        if (deferredScope != null)
        {
            try
            {
                // Wrap the GPU forward pass in a deferred scope
                // Operations are recorded to the graph builder and executed as a batch
                using (deferredScope)
                {
                    // Upload input and record it to the graph
                    var gpuInput = gpuEngine.UploadToGpu(input, GpuTensorRole.Input);

                    // Forward through layers - operations chain on GPU
                    // With full integration, these would record to scope.GraphBuilder
                    Tensor<T> current = gpuInput;
                    bool ownsCurrentTensor = true;

                    try
                    {
                        for (int i = 0; i < Layers.Count; i++)
                        {
                            var layer = Layers[i];

                            if (layer.CanExecuteOnGpu)
                            {
                                var next = layer.ForwardGpu(current);

                                if (ownsCurrentTensor && current is not null)
                                {
                                    current.Dispose();
                                }

                                current = next;
                                ownsCurrentTensor = true;
                            }
                            else
                            {
                                // CPU fallback for layers without GPU support
                                var cpuInput = current;
                                if (ownsCurrentTensor)
                                {
                                    current.Dispose();
                                }

                                var cpuOutput = layer.Forward(cpuInput);
                                current = gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
                                ownsCurrentTensor = true;
                            }
                        }

                        // Execute the deferred scope to run all batched operations
                        deferredScope.Execute();

                        // Download final result
                        var result = current;
                        if (ownsCurrentTensor)
                        {
                            current.Dispose();
                        }

                        return result;
                    }
                    catch (Exception)
                    {
                        if (ownsCurrentTensor && current is not null)
                        {
                            current.Dispose();
                        }
                        throw;
                    }
                }
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not System.Threading.ThreadAbortException)
            {
                // Fall back to non-deferred GPU execution if deferred fails
                System.Diagnostics.Debug.WriteLine($"Deferred execution failed, falling back: {ex.Message}");
            }
        }

        // Fall back to GPU-resident forward without deferred execution
        try
        {
            using var result = ForwardGpu(input);
            return result;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException and not System.Threading.ThreadAbortException)
        {
            // Log GPU failure for diagnostics before final fallback to CPU
            System.Diagnostics.Debug.WriteLine($"[NeuralNetworkBase] GPU forward failed in ForwardDeferred ({ex.GetType().Name}): {ex.Message}");
            return Predict(input);
        }
    }

    /// <summary>
    /// Performs an asynchronous forward pass using deferred execution for optimized GPU performance.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="cancellationToken">Cancellation token to cancel the operation.</param>
    /// <returns>A task representing the async operation with the output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This is the async version of <see cref="ForwardDeferred"/>. The GPU execution
    /// runs asynchronously, allowing the CPU to do other work while waiting.
    /// </para>
    /// </remarks>
    public virtual async Task<Tensor<T>> ForwardDeferredAsync(Tensor<T> input, CancellationToken cancellationToken = default)
    {
        // Check if we have a DirectGpuTensorEngine
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            // Fall back to regular predict if no GPU engine
            return Predict(input);
        }

        // Try to use deferred scope with async execution
        var deferredScope = gpuEngine.BeginDeferredScope();
        if (deferredScope != null)
        {
            try
            {
                using (deferredScope)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Upload input
                    var gpuInput = gpuEngine.UploadToGpu(input, GpuTensorRole.Input);
                    Tensor<T> current = gpuInput;
                    bool ownsCurrentTensor = true;

                    try
                    {
                        for (int i = 0; i < Layers.Count; i++)
                        {
                            cancellationToken.ThrowIfCancellationRequested();
                            var layer = Layers[i];

                            if (layer.CanExecuteOnGpu)
                            {
                                var next = layer.ForwardGpu(current);

                                if (ownsCurrentTensor && current is not null)
                                {
                                    current.Dispose();
                                }

                                current = next;
                                ownsCurrentTensor = true;
                            }
                            else
                            {
                                var cpuInput = current;
                                if (ownsCurrentTensor)
                                {
                                    current.Dispose();
                                }

                                var cpuOutput = layer.Forward(cpuInput);
                                current = gpuEngine.UploadToGpu(cpuOutput, GpuTensorRole.Activation);
                                ownsCurrentTensor = true;
                            }
                        }

                        // Execute the deferred scope asynchronously
                        await deferredScope.ExecuteAsync(cancellationToken);

                        // Download final result
                        var result = current;
                        if (ownsCurrentTensor)
                        {
                            current.Dispose();
                        }

                        return result;
                    }
                    catch (Exception)
                    {
                        if (ownsCurrentTensor && current is not null)
                        {
                            current.Dispose();
                        }
                        throw;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not System.Threading.ThreadAbortException)
            {
                // Fall back to non-deferred execution
                System.Diagnostics.Debug.WriteLine($"Async deferred execution failed, falling back: {ex.Message}");
            }
        }

        // Fall back to GPU-resident forward without async deferred execution
        return await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                using var result = ForwardGpu(input);
                return result;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException and not System.Threading.ThreadAbortException)
            {
                // Log GPU failure for diagnostics before falling back to CPU
                System.Diagnostics.Debug.WriteLine($"[NeuralNetworkBase] Async GPU forward failed ({ex.GetType().Name}): {ex.Message}");
                return Predict(input);
            }
        }, cancellationToken);
    }

    #region GPU Execution Context Integration

    /// <summary>
    /// Begins a GPU execution context for managing GPU-resident tensor lifecycle.
    /// </summary>
    /// <param name="options">Optional GPU execution options.</param>
    /// <returns>A GPU execution context that should be disposed when done.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU backend is available.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a scope for GPU operations where tensors stay
    /// on the GPU and are only downloaded when explicitly needed. This avoids redundant
    /// CPU-GPU transfers during batch inference or training.</para>
    /// <code>
    /// // Example: Batch inference with GPU context
    /// using (var ctx = network.BeginGpuExecution())
    /// {
    ///     foreach (var batch in batches)
    ///     {
    ///         var result = network.ForwardWithGpuContext(batch);
    ///         // Results are GPU-resident until ToTensor() is called
    ///         predictions.Add(result);
    ///     }
    /// } // All GPU tensors are cleaned up here
    /// </code>
    /// </remarks>
    public virtual GpuExecutionContext BeginGpuExecution(GpuExecutionOptions? options = null)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "BeginGpuExecution requires DirectGpuTensorEngine. Current engine: " +
                Engine.GetType().Name);
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null || !backend.IsAvailable)
        {
            throw new InvalidOperationException("No GPU backend available.");
        }

        return GpuExecutionContext.Begin(backend, options);
    }

    /// <summary>
    /// Performs a GPU-resident forward pass within a GPU execution context.
    /// Uses the current thread's GpuExecutionContext for tensor management.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>GPU-resident output tensor managed by the current context.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU context is active.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is like ForwardGpu but uses the GPU execution context
    /// to track all tensor allocations. The context handles memory management automatically,
    /// preventing memory leaks and enabling memory pressure monitoring.</para>
    /// </remarks>
    public virtual Tensor<T> ForwardWithGpuContext(Tensor<T> input)
    {
        var ctx = GpuExecutionContext.Current;
        if (ctx is null)
        {
            throw new InvalidOperationException(
                "ForwardWithGpuContext requires an active GpuExecutionContext. " +
                "Call BeginGpuExecution() first.");
        }

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardWithGpuContext requires DirectGpuTensorEngine. Current engine: " +
                Engine.GetType().Name);
        }

        // Use context's size threshold for GPU decision
        int inputElements = input.Data.Length;

        if (!ctx.ShouldUseGpu(inputElements))
        {
            // Context says stay on CPU - wrap result in GPU tensor for API consistency
            var cpuResult = Predict(input);
            return ctx.Upload(cpuResult, GpuTensorRole.Activation);
        }

        // Upload input to GPU using context (tracked in registry)
        Tensor<T> current = ctx.Upload(input, GpuTensorRole.Activation);

        try
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];

                if (layer.CanExecuteOnGpu)
                {
                    var next = layer.ForwardGpu(current);

                    // Register output with context (if not already registered by layer)
                    if (next is Tensor<float> gpuNextFloat)
                    {
                        ctx.Registry.TryRegister(gpuNextFloat);
                    }

                    current = next;
                }
                else
                {
                    // Layer doesn't support GPU - fall back to CPU
                    var cpuInput = current;
                    var cpuOutput = layer.Forward(cpuInput);

                    // Upload result back using context
                    current = ctx.Upload(cpuOutput, GpuTensorRole.Activation);
                }
            }

            return current;
        }
        catch (Exception)
        {
            // On error, tensors are cleaned up when context is disposed
            throw;
        }
    }

    /// <summary>
    /// Gets GPU memory statistics if running within a GPU execution context.
    /// </summary>
    /// <returns>Memory statistics, or null if no context is active.</returns>
    public virtual GpuMemoryStats? GetGpuMemoryStats()
    {
        return GpuExecutionContext.Current?.GetMemoryStats();
    }

    #endregion

    /// <summary>
    /// Performs a forward pass and returns intermediate layer activations for feature extraction.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <param name="layerIndices">Optional array of layer indices to extract features from. If null, returns all layer outputs.</param>
    /// <returns>A tuple containing the final output tensor and a dictionary of intermediate features indexed by layer number.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network while capturing intermediate layer
    /// activations. This is useful for feature extraction, transfer learning, style transfer,
    /// and advanced training techniques like feature matching in GANs.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you see what's happening inside the network at each layer.
    ///
    /// Think of it like watching a factory assembly line:
    /// - Normally, you only see the final product (output)
    /// - This method lets you inspect the product at each station (layer)
    /// - You can choose specific stations to inspect (layerIndices)
    ///
    /// This is useful for:
    /// - Understanding what features the network has learned
    /// - Using intermediate representations for other tasks (transfer learning)
    /// - Debugging network behavior
    /// - Advanced training techniques like feature matching in GANs
    ///
    /// Example:
    /// - layerIndices = new[] { -2, -1 } means "last two layers" (negative indices count from end)
    /// - layerIndices = null means "all layers"
    /// </para>
    /// <para><b>Industry Standard:</b> This pattern is common in modern ML frameworks:
    /// - PyTorch: model.forward_features() or register_forward_hook()
    /// - TensorFlow/Keras: Model(inputs=..., outputs=[layer1.output, layer2.output])
    /// - This implementation follows the TensorFlow-style approach
    /// </para>
    /// </remarks>
    public virtual (Tensor<T> output, Dictionary<int, Tensor<T>> features) ForwardWithFeatures(
        Tensor<T> input,
        int[]? layerIndices = null)
    {
        // Clear previous outputs
        _layerOutputs.Clear();

        // Perform forward pass and store outputs
        Tensor<T> current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            // Clone to prevent modification of cached values
            // Note: This is memory-intensive for large networks but necessary for correctness.
            // If layers modify their output tensors in-place during subsequent operations,
            // we need independent copies to avoid corrupting feature extraction results.
            // For memory-constrained scenarios, consider using features only when needed.
            _layerOutputs[i] = current.Clone();
        }

        // Build features dictionary
        Dictionary<int, Tensor<T>> features;

        if (layerIndices == null)
        {
            // Return all layer outputs
            features = new Dictionary<int, Tensor<T>>(_layerOutputs);
        }
        else
        {
            // Return only requested layers
            features = new Dictionary<int, Tensor<T>>();
            foreach (int idx in layerIndices)
            {
                // Support negative indices (count from end)
                int actualIdx = idx < 0 ? Layers.Count + idx : idx;

                if (actualIdx >= 0 && actualIdx < Layers.Count && _layerOutputs.TryGetValue(actualIdx, out var value))
                {
                    features[actualIdx] = value;
                }
            }
        }

        return (current, features);
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many adjustable values (weights and biases) your neural network has.
    /// More complex networks typically have more parameters and can learn more complex patterns, but also
    /// require more data to train effectively. This is part of the IFullModel interface for consistency with other model types.
    /// <para>
    /// <b>Performance:</b> This property uses caching to avoid recomputing the sum on every access.
    /// The cache is invalidated when layers are modified.
    /// </para>
    /// </remarks>
    public virtual long ParameterCount
    {
        get
        {
            if (_cachedParameterCount == null)
            {
                // Pre-resolve any lazy layers' shapes from the architecture
                // BEFORE summing per-layer ParameterCount. Lazy DenseLayer /
                // ConvolutionalLayer / FullyConnectedLayer / FeedForwardLayer
                // return 0 from ParameterCount when InputShape[0] is still
                // the -1 sentinel (issue #1209's lazy-shape-inference
                // migration), which makes a freshly-constructed model
                // report ParameterCount == 0 even though the architecture
                // fully defines the layer chain. Resolving shapes via
                // chain-walked input/output shapes turns the -1 sentinel
                // into the architecture's concrete dim and lets the test
                // (issue #1136 plan part 5) `network.ParameterCount > 0`
                // pre-Forward query work as designed.
                ResolveLazyLayerShapes();

                // Sum the per-layer counts in long throughout — both the
                // accumulator and the cache are long, so this getter returns
                // the genuine count even for >2.1B-parameter models. The
                // public signature has been long since PR #1244 (#1237);
                // this is the matching internal storage migration that
                // unblocks weight-streaming auto-detect on PaLM-E-class
                // models (#1271 / #1222) where ParameterCount is the input
                // to the threshold check. Consumers of the flat-Vector<T>
                // parameter API (GetParameters / SetParameters /
                // GetAllLayerInfo) STILL cannot represent more than
                // int.MaxValue elements in a single Vector<T> — that
                // invariant is enforced at THOSE call sites, where the
                // throw is actionable, rather than blocking every read of
                // the count itself (e.g. weight-streaming auto-detect,
                // model-metadata reporting, telemetry).
                long total = 0L;
                for (int i = 0; i < Layers.Count; i++)
                {
                    total += Layers[i].ParameterCount;
                }
                _cachedParameterCount = total;
            }
            return _cachedParameterCount.Value;
        }
    }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <returns>The total number of parameters in the neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the total count of all trainable parameters across all layers
    /// in the neural network. It uses the cached ParameterCount property for efficiency.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many adjustable values (weights and biases)
    /// your neural network has. More parameters mean the network can learn more complex patterns,
    /// but also requires more training data and computational resources.
    /// </para>
    /// </remarks>
    public long GetParameterCount()
    {
        return ParameterCount;
    }

    /// <summary>
    /// Invalidates the parameter count cache.
    /// Call this method whenever layers are added, removed, or modified.
    /// </summary>
    protected void InvalidateParameterCountCache()
    {
        _cachedParameterCount = null;
        _layerStructureVersion++;
        _parameterBuffer = null;
        // Layer structure changed — re-test the skip-buffer threshold next
        // training step. Without this, a model that grew from 100M -> 200M
        // params (e.g., LoRA rank bump, layer addition) would keep trying
        // to build a buffer until the new threshold check forced skip on
        // the next param-set sample.
        _skipParameterBuffer = false;
        _skipParameterBufferVersion = -1;
        Training.TapeTrainingStep<T>.InvalidateCache();
        InvalidateLayerInfoCache();
        // Layer structure changed — drop stale compiled inference plans.
        _compileHost.Invalidate();
        // Also drop compiled fused training plans and reset sticky-disable
        // so the next training run gets a fresh chance at the fused path.
        Training.CompiledTapeTrainingStep<T>.Invalidate();
        _fusedTrainingDisabled = false;
        _fusedTrainingCommitted = false;
    }

    /// <summary>
    /// Invalidates the cached layer info so that <see cref="GetAllLayerInfo"/> recomputes
    /// layer metadata on the next call. Called automatically from all layer mutation methods
    /// and deserialization.
    /// </summary>
    private void InvalidateLayerInfoCache()
    {
        _cachedLayerInfo = null;
        _cachedLayerCount = -1;
    }

    /// <summary>
    /// Adds a layer to the internal layers collection and invalidates the parameter count cache.
    /// </summary>
    /// <param name="layer">The layer to add</param>
    /// <remarks>
    /// This method ensures that the parameter count cache is properly invalidated when layers are added.
    /// Derived classes should use this method instead of directly accessing Layers.Add().
    /// </remarks>
    /// <summary>
    /// Monotonically increasing version counter, incremented when layers are added/removed.
    /// Used by TapeTrainingStep caching to detect structural changes.
    /// </summary>
    private int _layerStructureVersion;

    /// <summary>
    /// Gets the current layer structure version for cache invalidation.
    /// </summary>
    internal int LayerStructureVersion => _layerStructureVersion;


    protected void AddLayerToCollection(ILayer<T> layer)
    {
        _layers.Add(layer);
        InvalidateParameterCountCache();
    }

    /// <summary>
    /// Removes a layer from the internal layers collection and invalidates the parameter count cache.
    /// </summary>
    /// <param name="layer">The layer to remove</param>
    /// <returns>True if the layer was successfully removed, false otherwise</returns>
    /// <remarks>
    /// This method ensures that the parameter count cache is properly invalidated when layers are removed.
    /// Derived classes should use this method instead of directly accessing Layers.Remove().
    /// </remarks>
    protected bool RemoveLayerFromCollection(ILayer<T> layer)
    {
        bool removed = _layers.Remove(layer);
        if (removed)
        {
            InvalidateParameterCountCache();
        }
        return removed;
    }

    /// <summary>
    /// Freeze-time BatchNorm folding (compiled-inference, Phase 6). Folds every
    /// <see cref="BatchNormalizationLayer{T}"/> that directly follows a linear op
    /// with identity activation (a <see cref="ConvolutionalLayer{T}"/> here) into
    /// that op's weights and bias, then removes the BatchNorm layer. At inference a
    /// BatchNorm is the fixed per-channel affine
    /// <c>y = γ·(x − μ)/√(σ² + ε) + β</c>; for the preceding linear
    /// <c>z = W·x + b</c> the composition <c>BN(z)</c> equals
    /// <c>W'·x + b'</c> with per-output-channel <c>s = γ/√(σ² + ε)</c>,
    /// <c>W' = W·s</c>, <c>b' = (b − μ)·s + β</c> — so folding is lossless (to
    /// floating-point rounding) and eliminates a full per-element pass plus an
    /// intermediate tensor per block.
    ///
    /// <para><b>Safety:</b> only folds when the preceding layer's activation is the
    /// identity (no activation, or <see cref="IdentityActivation{T}"/>) — BatchNorm
    /// must sit directly on the linear output, before any nonlinearity, or the fold
    /// is invalid. Channel counts must match. Anything else is left untouched.</para>
    /// </summary>
    /// <returns>The number of BatchNorm layers folded (0 if none applicable).</returns>
    internal int FoldBatchNormForInference()
    {
        int folded = 0;
        // Walk back-to-front so removing a BN doesn't shift the indices of layers
        // we still have to examine.
        for (int i = _layers.Count - 1; i >= 1; i--)
        {
            if (_layers[i] is not BatchNormalizationLayer<T> bn)
                continue;
            var prev = _layers[i - 1];
            if (!IsIdentityActivation(prev))
                continue;
            bool didFold = prev switch
            {
                ConvolutionalLayer<T> conv => TryFoldBatchNormIntoConv(conv, bn),
                DenseLayer<T> dense => TryFoldBatchNormIntoDense(dense, bn),
                _ => false,
            };
            if (didFold)
            {
                RemoveLayerFromCollection(bn);
                folded++;
            }
        }

        if (folded > 0)
        {
            // The folds above mutate the weight/bias arrays IN PLACE, which leaves
            // the engine's identity-keyed inference weight caches stale — chiefly
            // SimdGemm.SgemmWithCachedB's pre-packed B panels, which are keyed by
            // the weight ARRAY OBJECT and never re-read its contents
            // (InferenceWeightCache's documented contract: "callers that mutate
            // weights in place must call InvalidateAll afterwards"). The per-tensor
            // Engine.InvalidatePersistentTensor calls inside the fold helpers cover
            // the GPU-resident persistent buffers but are a NO-OP on CpuEngine, so
            // the CPU packed-GEMM path kept serving the PRE-fold weights while the
            // BatchNorm layer was already removed — observed as a deterministic
            // ~0.17 Dense→BN folded-output divergence on AVX2-only hosts (CI),
            // invisible on AVX-512 hosts where SgemmWithCachedB bypasses the pack
            // cache entirely (the `Avx512Sgemm.CanUse` gate).
            InvalidateWeightCachesAfterSuccessfulWeightUpdate();
        }

        return folded;
    }

    /// <summary>A layer's activation is the identity iff it applies no activation
    /// or an <see cref="IdentityActivation{T}"/> (folding across a real nonlinearity
    /// would be incorrect).</summary>
    private static bool IsIdentityActivation(ILayer<T> layer)
    {
        if (layer is not LayerBase<T> lb)
            return false;
        bool scalarOk = lb.ScalarActivation is null || lb.ScalarActivation is IdentityActivation<T>;
        bool vectorOk = lb.VectorActivation is null;
        return scalarOk && vectorOk;
    }

    /// <summary>
    /// Folds <paramref name="bn"/>'s inference affine into <paramref name="conv"/>'s
    /// kernels/biases in place. Kernel layout is <c>[outChannels, inChannels, kH, kW]</c>;
    /// the BatchNorm operates per output channel. Returns false (folding nothing) when
    /// the channel counts don't line up.
    /// </summary>
    private bool TryFoldBatchNormIntoConv(ConvolutionalLayer<T> conv, BatchNormalizationLayer<T> bn)
    {
        var kernels = conv.GetFilters();   // live [outC, inC, kH, kW]
        var biases = conv.GetBiases();     // live [outC]
        int outC = kernels.Shape[0];
        if (biases.Length != outC) return false;

        var gamma = bn.GetGamma();
        var beta = bn.GetBeta();
        var mean = bn.GetRunningMean();
        var variance = bn.GetRunningVariance();
        if (gamma.Length != outC || beta.Length != outC || mean.Length != outC || variance.Length != outC)
            return false;

        T eps = bn.GetEpsilon();
        int perChannel = kernels.Length / outC;   // inC * kH * kW
        var kSpan = kernels.Data.Span;
        var bSpan = biases.Data.Span;
        var gSpan = gamma.Data.Span;
        var beSpan = beta.Data.Span;
        var mSpan = mean.Data.Span;
        var vSpan = variance.Data.Span;

        for (int oc = 0; oc < outC; oc++)
        {
            // s = γ / sqrt(var + ε);  W'[oc] = W[oc]·s;  b'[oc] = (b[oc] − μ)·s + β
            T denom = NumOps.Sqrt(NumOps.Add(vSpan[oc], eps));
            T scale = NumOps.Divide(gSpan[oc], denom);
            int baseIdx = oc * perChannel;
            for (int j = 0; j < perChannel; j++)
                kSpan[baseIdx + j] = NumOps.Multiply(kSpan[baseIdx + j], scale);
            bSpan[oc] = NumOps.Add(NumOps.Multiply(NumOps.Subtract(bSpan[oc], mSpan[oc]), scale), beSpan[oc]);
        }

        // Per-tensor invalidation of the GPU-resident persistent buffers keyed by
        // these arrays. NO-OP on CpuEngine — the CPU-side identity-keyed caches
        // (pre-packed GEMM panels, transposed conv kernels) are flushed by the
        // InvalidateWeightCachesAfterSuccessfulWeightUpdate() call in
        // FoldBatchNormForInference after all folds complete (see the Dense fold
        // below for the full staleness story).
        Engine.InvalidatePersistentTensor(kernels);
        Engine.InvalidatePersistentTensor(biases);
        return true;
    }

    /// <summary>
    /// Folds <paramref name="bn"/>'s inference affine into <paramref name="dense"/>'s
    /// weights/biases in place. Weight layout is <c>[inputSize, outputSize]</c> and the
    /// BatchNorm operates per output feature, so the scale for output <c>oc</c> multiplies
    /// column <c>oc</c> of the weight matrix (stride <c>outputSize</c>). Returns false when
    /// the feature counts don't line up.
    /// </summary>
    private bool TryFoldBatchNormIntoDense(DenseLayer<T> dense, BatchNormalizationLayer<T> bn)
    {
        var weights = dense.GetWeights();   // live [inputSize, outputSize]
        var biases = dense.GetBiases();     // live [outputSize]
        if (weights.Rank != 2) return false;
        int inSize = weights.Shape[0];
        int outSize = weights.Shape[1];
        if (biases.Length != outSize) return false;

        var gamma = bn.GetGamma();
        var beta = bn.GetBeta();
        var mean = bn.GetRunningMean();
        var variance = bn.GetRunningVariance();
        if (gamma.Length != outSize || beta.Length != outSize || mean.Length != outSize || variance.Length != outSize)
            return false;

        T eps = bn.GetEpsilon();
        var wSpan = weights.Data.Span;
        var bSpan = biases.Data.Span;
        var gSpan = gamma.Data.Span;
        var beSpan = beta.Data.Span;
        var mSpan = mean.Data.Span;
        var vSpan = variance.Data.Span;

        for (int oc = 0; oc < outSize; oc++)
        {
            T scale = NumOps.Divide(gSpan[oc], NumOps.Sqrt(NumOps.Add(vSpan[oc], eps)));
            for (int i = 0; i < inSize; i++)
            {
                int idx = i * outSize + oc;   // row-major [inputSize, outputSize]
                wSpan[idx] = NumOps.Multiply(wSpan[idx], scale);
            }
            bSpan[oc] = NumOps.Add(NumOps.Multiply(NumOps.Subtract(bSpan[oc], mSpan[oc]), scale), beSpan[oc]);
        }

        // Per-tensor invalidation of the GPU-resident persistent buffers keyed by
        // these arrays. NOTE: this is a NO-OP on CpuEngine — the CPU-side
        // identity-keyed caches (SgemmWithCachedB pre-packed panels, int8 packs,
        // transposed conv kernels) are flushed by the single
        // InvalidateWeightCachesAfterSuccessfulWeightUpdate() call in
        // FoldBatchNormForInference after all folds complete. Relying on this call
        // alone left the CPU pack cache stale (the ~0.17 Dense→BN divergence on
        // AVX2-only CI hosts).
        Engine.InvalidatePersistentTensor(weights);
        Engine.InvalidatePersistentTensor(biases);
        return true;
    }

    /// <summary>
    /// Clears all layers from the internal layers collection and invalidates the parameter count cache.
    /// </summary>
    /// <remarks>
    /// This method ensures that the parameter count cache is properly invalidated when layers are cleared.
    /// Derived classes should use this method instead of directly accessing Layers.Clear().
    /// </remarks>
    protected void ClearLayers()
    {
        _layers.Clear();
        InvalidateParameterCountCache();
    }

    /// <summary>
    /// Validates that the provided layers form a valid neural network architecture.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Not all combinations of layers make a valid neural network. This method checks that
    /// the layers can properly connect to each other (like making sure puzzle pieces fit together).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the layer configuration is invalid.</exception>
    protected virtual void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        ValidateCustomLayersInternal(layers);
    }

    protected void ValidateCustomLayersInternal(List<ILayer<T>> layers)
    {
        if (layers == null || layers.Count == 0)
        {
            throw new ArgumentException("Neural network must have at least 1 layer.");
        }

        var errors = new List<string>();

        // Check input shape. Custom layers should be accepted by contract,
        // not by concrete layer type: if the first layer can consume the
        // architecture input shape, it is a valid boundary layer.
        if (!IsFirstLayerShapeCompatible(layers[0]))
        {
            var firstLayerShape = TryGetLayerShape(layers[0], static l => l.GetInputShape());
            var archShape = TryGetArchitectureDeclaredInputShape();
            errors.Add(
                $"The first layer's input shape must be compatible with the architecture input shape. " +
                $"(first layer = {layers[0].GetType().Name}, input shape = {FormatShape(firstLayerShape)}; " +
                $"architecture input shape = {FormatShape(archShape)}).");
        }

        // Check layer connections
        if (layers.Count > 1)
        {
            for (int i = 1; i < layers.Count; i++)
            {
                var prevLayer = layers[i - 1];
                var currentLayer = layers[i];

                if (!AreLayersCompatible(prevLayer, currentLayer))
                {
                    var prevOut = TryGetLayerShape(prevLayer, static l => l.GetOutputShape());
                    var currIn = TryGetLayerShape(currentLayer, static l => l.GetInputShape());
                    errors.Add(
                        $"Layer {i - 1} ({prevLayer.GetType().Name}, output {FormatShape(prevOut)}) " +
                        $"is not compatible with Layer {i} ({currentLayer.GetType().Name}, " +
                        $"input {FormatShape(currIn)}).");
                }
            }
        }

        // Check output shape. The final layer may be a custom readout that
        // produces the expected target representation without being a Dense
        // or activation layer.
        if (!IsLastLayerShapeCompatible(layers[layers.Count - 1], out var outputShapeError))
        {
            errors.Add(outputShapeError);
        }

        // Throw exception if any errors were found
        if (errors.Count > 0)
        {
            throw new ArgumentException($"Invalid layer configuration:\n{string.Join("\n", errors)}");
        }
    }

    private bool IsFirstLayerShapeCompatible(ILayer<T> layer)
    {
        // Embedding-category first layers (EmbeddingLayer, positional encodings, custom subclasses)
        // consume token INDICES through a per-token broadcast contract and declare a [1]-style input
        // shape that intentionally does not match the architecture input shape (the sequence length).
        // Accept them outright (#1321/#1323); runtime shape resolution validates the real dimensions.
        if (NeuralNetworkArchitecture<T>.IsEmbeddingCategoryLayer(layer))
            return true;

        int[]? layerInputShape = TryGetLayerShape(layer, shapeSelector: static l => l.GetInputShape());
        if (IsDeferredOrAgnosticShape(layerInputShape))
            return true;

        int[]? architectureInputShape = TryGetArchitectureDeclaredInputShape();
        if (IsDeferredOrAgnosticShape(architectureInputShape))
            return true;

        if (AreShapesCompatible(architectureInputShape!, layerInputShape!))
            return true;

        // Flat-reshape boundary: architecture may declare a structured
        // shape (e.g. [H, W] image, [batch, ctxLen]) while a flatten-style
        // first layer like InputLayer<T>(N) declares a single flat dim.
        // Accept when the total element count matches and all dims are
        // known positive values — this is exactly the "flatten" boundary
        // (TransformerArchitecture with inputType=TwoDimensional + inputSize=64
        // yields shape [8, 8]; a custom chain starting with InputLayer(64)
        // expects the same 64 elements arriving as rank-1). Without this
        // path the validator rejects a legitimate flatten contract that
        // the layer would have happily accepted at first forward.
        //
        // This acceptance is directional: it only covers a rank-1 (flat) first
        // layer CONSUMING a (possibly structured) architecture input — the
        // natural flatten boundary. A higher-rank first layer demanding a
        // specific structured RESHAPE of a flat architecture input (e.g. a
        // layer declaring [2, 8] fed by a flat [16]) must use an explicit
        // ReshapeLayer, mirroring the layer-to-layer "no implicit flatten"
        // contract — so it is NOT accepted here.
        if (layerInputShape!.Length == 1 && TotalElementsMatch(architectureInputShape!, layerInputShape!))
            return true;

        return false;
    }

    /// <summary>
    /// Two shape arrays have the same total element count, with all dims
    /// known and positive. Used at the architecture / first-layer boundary
    /// to recognise legitimate flatten / reshape contracts that
    /// <see cref="AreShapesCompatible"/>'s rank-strict check would reject.
    /// </summary>
    private static bool TotalElementsMatch(int[] a, int[] b)
    {
        if (a is null || b is null || a.Length == 0 || b.Length == 0) return false;
        long pa = 1, pb = 1;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] <= 0) return false;
            pa *= a[i];
        }
        for (int i = 0; i < b.Length; i++)
        {
            if (b[i] <= 0) return false;
            pb *= b[i];
        }
        return pa == pb;
    }

    private bool IsLastLayerShapeCompatible(ILayer<T> layer, out string error)
    {
        int[]? layerOutputShape = TryGetLayerShape(layer, shapeSelector: static l => l.GetOutputShape());
        if (IsDeferredOrAgnosticShape(layerOutputShape))
        {
            error = string.Empty;
            return true;
        }

        int[] outputShape = layerOutputShape!;

        // Partially-deferred shapes — e.g. a transposed-conv generator that
        // emits [3, -1, -1] until first Forward resolves H/W from runtime
        // input — can't be compared dimensionally against the architecture's
        // flat OutputSize. Defer the check; the layer will fail at runtime
        // with a precise shape error if the resolved output doesn't honour
        // the contract. This guard was added by PR #1329 and inadvertently
        // removed by the grafprint PR (c8cac237b) — restoring it closes the
        // DCGAN cluster-1 false-rejection where the generator's last
        // ConvTranspose2D layer reports [channels, -1, -1] before first
        // Forward and is compared against OutputSize = channels * H * W
        // (a single flat scalar that can't be element-wise-matched).
        //
        // Only -1 sentinels trigger the deferral. Zero-sized dimensions
        // would ALSO escape the AreShapesCompatible check below — the
        // ShapesMatchKnownDimensions wildcard rule treats any d <= 0
        // as matching anything, so a stray d == 0 in either operand
        // silently passes. That's the existing wildcard convention
        // across this validator (matches lazy-layer rank-2 [-1, F] vs
        // resolved rank-3 [B, S, F] etc); a layer that emits d == 0 in
        // a resolved output shape is a bug a different validator should
        // catch — not this one's job. Restricting deferral to d < 0
        // keeps the surface narrow: any future "d > 0 but doesn't
        // match" path still gets the clear "output shape ... must match
        // architecture output size" error.
        if (outputShape.Any(d => d < 0))
        {
            error = string.Empty;
            return true;
        }

        if (Architecture.OutputSize > 0 && !AreShapesCompatible([Architecture.OutputSize], outputShape))
        {
            error = $"The last layer's output shape [{string.Join(", ", outputShape)}] must match the architecture output size ({Architecture.OutputSize}).";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private int[]? TryGetArchitectureDeclaredInputShape()
    {
        if (Architecture.IsLayerOnly)
            return null;

        try
        {
            return Architecture.GetInputShape();
        }
        catch (Exception ex)
        {
            throw new ArgumentException(
                "Failed to resolve the architecture input shape for custom-layer validation.",
                ex);
        }
    }

    private static int[]? TryGetLayerShape(ILayer<T> layer, Func<ILayer<T>, int[]> shapeSelector)
    {
        try
        {
            return shapeSelector(layer);
        }
        catch (Exception ex)
        {
            throw new ArgumentException(
                $"Failed to resolve shape metadata from layer '{layer.GetType().Name}' during custom-layer validation.",
                ex);
        }
    }

    private static bool IsDeferredOrAgnosticShape(int[]? shape)
    {
        return shape is null || shape.Length == 0 || shape.All(d => d <= 0);
    }

    /// <summary>Renders a shape array as <c>[d0, d1, ...]</c> for validator error messages.</summary>
    private static string FormatShape(int[]? shape)
    {
        if (shape is null) return "<null>";
        if (shape.Length == 0) return "[]";
        return "[" + string.Join(", ", shape) + "]";
    }

    private static bool AreShapesCompatible(int[] expectedShape, int[] actualShape)
    {
        if (ShapesMatchKnownDimensions(expectedShape, actualShape))
            return true;

        var expectedWithoutLeadingBatch = TrimLeadingBatchLikeDimensions(expectedShape);
        if (ShapesMatchKnownDimensions(expectedWithoutLeadingBatch, actualShape))
            return true;

        var actualWithoutLeadingBatch = TrimLeadingBatchLikeDimensions(actualShape);
        if (ShapesMatchKnownDimensions(expectedShape, actualWithoutLeadingBatch))
            return true;

        if (ShapesMatchKnownDimensions(expectedWithoutLeadingBatch, actualWithoutLeadingBatch))
            return true;

        // Rank-N declared vs rank-(N+1) post-forward-resolved batch path.
        // TrimLeadingBatchLikeDimensions only strips dims <= 1, so a layer
        // whose OnFirstForward resolved its input shape to a concrete
        // [batch>=2, ...rest] keeps the batch dim in place — and the rank
        // ends up off by one against an upstream layer that declares
        // [-1, ...rest]. This is the path
        //
        //   MultiHeadAttentionLayer.GetOutputShape() == [-1, embDim]   (rank 2 declared)
        //   SequenceTokenSliceLayer.GetInputShape()  == [batch, seq, embDim]
        //                                              (rank 3, resolved by OnFirstForward)
        //
        // exercised by NeuralNetworkBase.DeepCopy → Transformer..ctor →
        // ValidateCustomLayers re-running on an already-forwarded chain
        // (HarmonicEngine PR #149 / #1333 repro).
        //
        // Crucially, the shorter side MUST explicitly declare a wildcard (-1)
        // leading dim. Otherwise we'd bless unrelated shapes — e.g.
        // [32, 32] (a concrete rank-2) vs [3, 32, 32] (a concrete rank-3)
        // would silently pass by stripping the leading 3. The wildcard guard
        // restricts the strip to the genuine "I accept any batch" contract
        // that MHA/Slice/etc actually declare.
        if (expectedShape.Length + 1 == actualShape.Length
            && actualShape[0] >= 1
            && expectedShape.Length > 0
            && expectedShape[0] <= 0)
        {
            var actualNoBatch = new int[actualShape.Length - 1];
            Array.Copy(actualShape, 1, actualNoBatch, 0, actualNoBatch.Length);
            if (ShapesMatchKnownDimensions(expectedShape, actualNoBatch))
                return true;
        }
        if (actualShape.Length + 1 == expectedShape.Length
            && expectedShape[0] >= 1
            && actualShape.Length > 0
            && actualShape[0] <= 0)
        {
            var expectedNoBatch = new int[expectedShape.Length - 1];
            Array.Copy(expectedShape, 1, expectedNoBatch, 0, expectedNoBatch.Length);
            if (ShapesMatchKnownDimensions(expectedNoBatch, actualShape))
                return true;
        }

        return false;
    }

    private static bool ShapesMatchKnownDimensions(int[] expectedShape, int[] actualShape)
    {
        if (expectedShape.Length != actualShape.Length)
            return false;

        for (int i = 0; i < expectedShape.Length; i++)
        {
            if (expectedShape[i] > 0 && actualShape[i] > 0 && expectedShape[i] != actualShape[i])
                return false;
        }

        return true;
    }

    private static int[] TrimLeadingBatchLikeDimensions(int[] shape)
    {
        int start = 0;
        while (start < shape.Length - 1 && shape[start] <= 1)
            start++;

        if (start == 0)
            return shape;

        var trimmed = new int[shape.Length - start];
        Array.Copy(shape, start, trimmed, 0, trimmed.Length);
        return trimmed;
    }

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an input layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The input layer is the first layer of your neural network. It receives the raw data 
    /// you want to process (like image pixels or text features). This method checks if a layer is suitable 
    /// to be the first layer in your network.
    /// </para>
    /// </remarks>
    protected virtual bool IsValidInputLayer(ILayer<T> layer)
    {
        // Check if the layer is specifically designed as an input layer
        if (layer is InputLayer<T>)
            return true;

        // For convolutional networks, the first layer is often a ConvolutionalLayer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // For simple feedforward networks, the first layer might be Dense
        if (layer is DenseLayer<T> denseLayer)
        {
            // Lazy DenseLayer carries InputShape = [-1] until first forward; treat as valid.
            var shape = denseLayer.GetInputShape();
            return shape.Length == 1 && (shape[0] > 0 || shape[0] == -1);
        }

        // For recurrent networks, the first layer might be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // If none of the above, it's not a valid input layer
        return false;
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an output layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The output layer is the last layer of your neural network. It produces the final result 
    /// (like a prediction or classification). This method checks if a layer is suitable to be the final layer 
    /// in your network. Different tasks need different types of output layers - for example, image classification 
    /// might use a Softmax activation, while regression might use a linear activation.
    /// </para>
    /// </remarks>
    protected virtual bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Most commonly, the output layer is a Dense layer
        if (layer is DenseLayer<T> denseLayer)
        {
            // Ensure the dense layer has an output (it's not empty)
            return denseLayer.GetOutputShape().Length == 1 && denseLayer.GetOutputShape()[0] > 0;
        }

        // NoisyDenseLayer is functionally a dense linear layer with parametric
        // noise on its weights (Fortunato et al. 2017 "Noisy Networks for
        // Exploration") — used as the output head in Noisy-DQN / Rainbow-DQN
        // (Hessel et al. 2018 §3.4). Same output-shape contract as DenseLayer.
        if (layer is Layers.NoisyDenseLayer<T> noisyDense)
        {
            return noisyDense.GetOutputShape().Length == 1 && noisyDense.GetOutputShape()[0] > 0;
        }

        // DuelingCombinationLayer is the output head of a Wang et al. 2016
        // dueling-DQN architecture (also Hessel et al. 2018 §3.4 in Rainbow):
        // it internally projects the trunk features into V(s) and A(s, a) and
        // emits Q(s, a) = V(s) + (A − mean_a A). The emitted Q vector is the
        // network's final output — same shape contract as a regular dense
        // output layer.
        if (layer is Layers.DuelingCombinationLayer<T> dueling)
        {
            return dueling.GetOutputShape().Length == 1 && dueling.GetOutputShape()[0] > 0;
        }

        // For some specific tasks, the output might be from other layer types
        // For example, in sequence-to-sequence models, it could be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // For image segmentation tasks, it might be a Convolutional layer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // For generative-image models (DCGAN, VAE decoders, segmentation
        // upsamplers), the final layer is a transposed convolution that
        // produces an image-shaped tensor. The activation (typically Tanh
        // for [-1, 1] image range, or Sigmoid for [0, 1]) is the deconv's
        // built-in activation, so the output is valid even though the layer
        // itself is a deconvolution.
        if (layer is DeconvolutionalLayer<T>)
            return true;

        // Check if the layer has an activation function typically used in output layers
        if (layer is ActivationLayer<T> activationLayer)
        {
            // Check if the layer has an activation function typically used in output layers
            var activationTypes = layer.GetActivationTypes();
            return activationTypes.Any(type => type == ActivationFunction.Softmax || type == ActivationFunction.Sigmoid);
        }

        // If none of the above, it's not a valid output layer
        return false;
    }

    /// <summary>
    /// Checks if two consecutive layers can be connected in a neural network.
    /// </summary>
    /// <param name="prevLayer">The preceding layer.</param>
    /// <param name="currentLayer">The current layer to check compatibility with.</param>
    /// <returns>True if the layers can be connected; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks work by connecting layers in sequence. For two layers to connect properly, 
    /// the output of one layer must match what the next layer expects as input. This is like making sure puzzle 
    /// pieces fit together. This method checks if two layers can be properly connected.
    /// </para>
    /// <para>
    /// For example, if a layer outputs 100 values, the next layer should expect 100 values as input. Some layer 
    /// combinations also have special rules - like needing a "Flatten" layer between image processing layers and 
    /// regular dense layers.
    /// </para>
    /// </remarks>
    protected virtual bool AreLayersCompatible(ILayer<T> prevLayer, ILayer<T> currentLayer)
    {
        // Embedding-category layers (EmbeddingLayer, positional encodings, custom subclasses)
        // participate in a per-token broadcast contract whose static shapes intentionally omit the
        // sequence dimension. On the INPUT side they declare a [1]-style input that does not match a
        // preceding layer's output (e.g. an InputLayer or DenseLayer feeding token ids). On the
        // OUTPUT side they declare a per-token [dModel] shape that omits the sequence length (which
        // is data-dependent), so a downstream sequence layer's [seqLen, dModel] input does not match
        // statically. Treat any transition INTO or OUT OF such a layer as compatible (#1323);
        // runtime shape resolution validates the real dimensions on first forward.
        if (NeuralNetworkArchitecture<T>.IsEmbeddingCategoryLayer(currentLayer)
            || NeuralNetworkArchitecture<T>.IsEmbeddingCategoryLayer(prevLayer))
            return true;

        // Lazy layers report InputShape = [-1] (or empty) until first Forward —
        // skip the strict shape-equality check; resolution happens at first
        // forward. Empty-shape layers are shape-agnostic by design (e.g.
        // DropoutLayer, ActivationLayer constructed without an explicit
        // InputShape) and would have been incorrectly rejected by the
        // SequenceEqual check otherwise.
        var currentInputShape = TryGetLayerShape(currentLayer, shapeSelector: static l => l.GetInputShape());
        var prevOutputShape = TryGetLayerShape(prevLayer, shapeSelector: static l => l.GetOutputShape());
        bool currentIsLazy = IsDeferredOrAgnosticShape(currentInputShape);
        if (!currentIsLazy
            && !IsDeferredOrAgnosticShape(prevOutputShape)
            && !AreShapesCompatible(prevOutputShape!, currentInputShape!))
            return false;

        // Special checks for specific layer combinations
        if (prevLayer is ConvolutionalLayer<T> && currentLayer is DenseLayer<T>)
        {
            // Ensure there's a Flatten layer between Conv and Dense
            return false;
        }

        if (prevLayer is PoolingLayer<T> && currentLayer is LSTMLayer<T>)
        {
            // Pooling directly to LSTM is usually not valid
            return false;
        }

        // Check for dimension compatibility in case of Reshape or Flatten layers.
        // Lazy current layers carry -1 placeholders; defer the product check to first forward.
        if (prevLayer is ReshapeLayer<T> reshapeLayer && !currentIsLazy)
        {
            // First, try the same-rank shape match. This path handles the
            // common case where Reshape's output shape lines up dim-for-dim
            // with the next layer's input shape, possibly with -1 wildcards
            // on either side (e.g. ReshapeLayer([4, 32]) → MHA whose input
            // is declared as [-1, 32] because the seq-len dim is lazy until
            // first forward). The naive product comparison below trims
            // leading dims <= 1, which silently strips a leading -1 too —
            // turning [-1, 32] into [32] and breaking the product check.
            // Closes the #1330 InferenceOptimizerIntegrationTests cluster.
            if (AreShapesCompatible(reshapeLayer.GetOutputShape(), currentInputShape!))
                return true;

            // Fall back to product-of-known-dims for the implicit-flatten
            // case where ranks differ (e.g. Reshape([4, 32]) feeding a
            // DenseLayer that takes a flat [128]). TrimLeadingBatchLikeDimensions
            // strips leading dims <= 1 to ignore an implicit batch dim, so
            // a -1 in any non-leading position still survives and short-
            // circuits the comparison.
            var reshapeOutputShape = TrimLeadingBatchLikeDimensions(reshapeLayer.GetOutputShape());
            var normalizedCurrentInputShape = TrimLeadingBatchLikeDimensions(currentInputShape!);

            if (reshapeOutputShape.Any(d => d <= 0) || normalizedCurrentInputShape.Any(d => d <= 0))
                return true;

            return reshapeOutputShape.Aggregate((a, b) => a * b) ==
                   normalizedCurrentInputShape.Aggregate((a, b) => a * b);
        }

        // If no incompatibilities found, layers are considered compatible
        return true;
    }

    /// <summary>
    /// Retrieves the gradients for all trainable parameters in the network.
    /// </summary>
    /// <returns>A vector containing all parameter gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When a neural network learns, it needs to know how to adjust each of its internal values 
    /// (parameters). These adjustments are called "gradients" - they tell the network which direction and how much 
    /// to change each parameter. This method collects all those adjustment values into a single list.
    /// </para>
    /// <para>
    /// Think of gradients as a recipe for improvement: "increase this weight by 0.01, decrease that one by 0.03," etc.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameterGradients()
    {
        // Collect gradients from all layers
        List<Vector<T>> allGradients = new List<Vector<T>>();

        foreach (var layer in Layers.Where(l => l.SupportsTraining && l.ParameterCount > 0))
        {
            allGradients.Add(layer.GetParameterGradients());
        }

        // Concatenate all gradients into a single vector
        if (allGradients.Count == 0)
        {
            return new Vector<T>(0);
        }

        return Vector<T>.Concatenate(allGradients.ToArray());
    }

    /// <summary>
    /// Ensures the architecture is initialized before training begins.
    /// </summary>
    protected void EnsureArchitectureInitialized()
    {
        if (Architecture.IsLayerOnly)
        {
            // Layer-only model — no semantic architecture to initialize.
            // Layers were registered directly by the subclass; skip
            // cached-data hydration but still drive InitializeLayers +
            // lazy-shape resolution so ParameterCount works pre-Forward.
            // Idempotency guard (#1259 review): callers may invoke
            // EnsureArchitectureInitialized multiple times (e.g.,
            // ParameterCount probe before Train, then again at first
            // Predict). The runtime _layerOnlyInitialized flag handles
            // that within a session; the Layers.Count check ALSO catches
            // the post-deserialize case where Layers has been hydrated
            // from disk but the runtime flag is still false on the fresh
            // instance — without this, Deserialize → ParameterCount would
            // double up the entire network.
            if (!_layerOnlyInitialized && Layers.Count == 0)
            {
                InitializeLayers();
            }
            _layerOnlyInitialized = true;
            ResolveLazyLayerShapes();
            // Eager auto-detect for streaming (#1222 / #183). For models with
            // non-lazy layer construction (e.g. ResNet, VGG, classical CNNs)
            // ParameterCount is reliable here and the threshold check fires
            // immediately. For lazy models (Transformer / MultiHeadAttention),
            // ParameterCount may report 0 at this point — the first-Predict
            // call will retry once weights have materialized.
            TryAutoEnableWeightStreaming();
            return;
        }

        if (!Architecture.IsInitialized)
        {
            // Initialize from cached data
            Architecture.InitializeFromCachedData();

            // Initialize network-specific layers
            InitializeLayers();

            // Pre-resolve lazy layers' shapes from the architecture so
            // ParameterCount / GetParameters / Clone / ONNX export work
            // before the first Forward — issue #1209 lazy migration left
            // every DenseLayer / FullyConnectedLayer / FeedForwardLayer /
            // Conv variant with InputShape[0] = -1 sentinel until first
            // Forward, which made `network.ParameterCount` return 0 for
            // a freshly-constructed model and broke the
            // `Parameters_ShouldBeNonEmpty` invariant test that runs
            // BEFORE any Forward by design (the docstring says
            // "ParameterCount reads the declared count without forcing
            // lazy layers to materialize", which only works once the
            // chain has been resolved through architecture-known shapes).
            //
            // Walks layer-by-layer: the architecture's input shape feeds
            // the first layer; each layer's resolved output shape feeds
            // the next. ResolveFromShape is idempotent and only allocates
            // weight tensors for actually-lazy layers (eager layers
            // short-circuit on their already-resolved IsShapeResolved).
            ResolveLazyLayerShapes();
            // Eager auto-detect for streaming (#1222 / #183). See note on
            // the layer-only branch above — this is the parallel call for
            // architecture-driven models. Both paths converge through
            // TryAutoEnableWeightStreaming, which is idempotent.
            TryAutoEnableWeightStreaming();
        }
    }

    /// <summary>
    /// Tracks whether <see cref="ResolveLazyLayerShapes"/> has already
    /// run once on this network instance. Once every lazy layer's shape
    /// is resolved (or once we've decided we can't resolve them from
    /// the architecture alone), there's no point re-walking on every
    /// ParameterCount / GetParameters call — a no-op pass through 100+
    /// layers in a deep DiT/UNet on every parameter query is the
    /// difference between sub-second tests and 120-second timeouts.
    /// </summary>
    private bool _layerShapesResolved;

    /// <summary>
    /// One-shot flag for the layer-only branch of <see cref="EnsureArchitectureInitialized"/>.
    /// Without this, repeated calls (ParameterCount probe → Train →
    /// Predict) would re-invoke <see cref="InitializeLayers"/> and any
    /// subclass that appends to <see cref="Layers"/> would duplicate
    /// the network. Mirrors the role of <see cref="NeuralNetworkArchitecture{T}.IsInitialized"/>
    /// for the architecture-driven branch.
    /// </summary>
    private bool _layerOnlyInitialized;

    /// <summary>
    /// Walks <see cref="Layers"/> in order, propagating concrete input
    /// shapes through the chain so every lazy layer has its
    /// <c>InputShape</c> / <c>OutputShape</c> resolved before any
    /// Forward / GetParameters / ParameterCount call. Bridges the gap
    /// between #1209's lazy-shape-inference layers and pre-Forward
    /// queries on the parent network. Issue #1136 plan part 3 cleanup.
    /// Idempotent — runs at most once per network instance.
    /// </summary>
    protected void ResolveLazyLayerShapes()
    {
        if (_layerShapesResolved) return;
        if (Layers is null || Layers.Count == 0) return;

        int[]? currentShape = TryGetArchitectureInputShape();
        if (currentShape is null)
        {
            // Architecture can't yield a concrete input shape — still
            // mark "resolved" so subsequent queries don't re-attempt
            // the walk. Layers will resolve lazily on first Forward as
            // before.
            _layerShapesResolved = true;
            return;
        }

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            if (layer is null) continue;

            try
            {
                if (layer is LayerBase<T> lb && !lb.IsShapeResolved)
                {
                    // Shape-only resolution — does NOT allocate or initialize
                    // weights, so we don't consume RNG state and perturb
                    // training. Weight allocation still happens lazily on the
                    // first real Forward via EnsureInitializedFromInput.
                    lb.ResolveShapesOnly(currentShape);
                }

                // Advance the chain via the layer's now-resolved output
                // shape. We re-read GetOutputShape after ResolveFromShape
                // so a lazy layer that just resolved contributes its
                // real shape to the next iteration.
                int[] outShape = layer.GetOutputShape();
                if (outShape != null && outShape.Length > 0 && System.Array.TrueForAll(outShape, d => d > 0))
                {
                    currentShape = outShape;
                }
                else
                {
                    // Layer can't yield a concrete output shape (e.g., a
                    // dynamic-size layer with -1 dims). Stop pre-resolution;
                    // remaining layers will lazy-resolve on first Forward.
                    break;
                }
            }
            catch
            {
                // ResolveFromShape can fail for layers that need richer
                // shape info than we can derive from a flat array (e.g.,
                // some attention layers expect contextual metadata).
                // Swallow so InitializeLayers always succeeds — those
                // layers retain their lazy state and resolve on first
                // Forward as before.
                break;
            }
        }

        _layerShapesResolved = true;
    }

    /// <summary>
    /// Returns the input shape that <see cref="Layers"/>[0] actually
    /// observes — the starting point for <see cref="ResolveLazyLayerShapes"/>'s
    /// chain walk. Default implementation returns the architecture's
    /// declared input shape (with a unit batch prepended); models that
    /// pre-process inputs before the layer stack — patch-embedding image
    /// tokenizers in vision-language encoders, audio mel-spectrogram front
    /// ends, etc. — should override to return the post-pre-processing
    /// shape so lazy <c>LayerNormalization</c>/<c>Dense</c>/etc. layers
    /// resolve their gamma/weights to the right channel count up-front
    /// instead of binding to the raw architecture shape and mismatching
    /// at first real Forward. (Was the cause of the BiomedCLIP /
    /// DFNCLIP <c>"Gamma shape (128) does not match the last 1 dimensions
    /// of input shape (1, 256, 768)"</c> failure mode in #1224 — the
    /// pre-norm <c>LayerNormalizationLayer</c> resolved its gamma to the
    /// raw NCHW spatial dim 128 from <c>TryGetArchitectureInputShape</c>
    /// instead of the post-patch-embed channel dim 768.)
    /// </summary>
    /// <remarks>
    /// Intentionally silent on missing data so dynamic-spatial models
    /// that depend on runtime shape feeding still work — return null to
    /// defer all shape resolution to the first real Forward.
    /// </remarks>
    protected virtual int[]? TryGetArchitectureInputShape()
    {
        // Layer-only models carry a stub architecture; fall back to the
        // first layer's input shape so the lazy-resolution chain still
        // has a starting point. If even that's lazy (-1 dims), we
        // return null and the caller leaves layers to resolve on first
        // Forward.
        if (Architecture.IsLayerOnly)
        {
            if (Layers is null || Layers.Count == 0) return null;
            int[] firstShape;
            try { firstShape = Layers[0].GetInputShape(); }
            catch { return null; }
            if (firstShape is null || firstShape.Length == 0) return null;
            for (int i = 0; i < firstShape.Length; i++)
            {
                if (firstShape[i] <= 0) return null;
            }
            int[] withBatchFromLayer = new int[firstShape.Length + 1];
            withBatchFromLayer[0] = 1;
            for (int i = 0; i < firstShape.Length; i++) withBatchFromLayer[i + 1] = firstShape[i];
            return withBatchFromLayer;
        }

        try
        {
            int[] shape = Architecture.GetInputShape();
            if (shape == null || shape.Length == 0) return null;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] <= 0) return null;
            }
            // Prepend a unit batch dim so layers expecting [B, ...] see
            // a coherent rank — matches what the first real Forward
            // would feed. Layers that don't care about rank are
            // unaffected.
            int[] withBatch = new int[shape.Length + 1];
            withBatch[0] = 1;
            for (int i = 0; i < shape.Length; i++) withBatch[i + 1] = shape[i];
            return withBatch;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method sets up all the layers in your neural network according to the architecture 
    /// you've defined. It's like assembling the parts of your network before you can use it.
    /// </remarks>
    protected abstract void InitializeLayers();

    /// <summary>
    /// Makes a prediction using the neural network.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>The network's prediction.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main method you'll use to get results from your trained neural network.
    /// You provide some input data (like an image or text), and the network processes it through all its
    /// layers to produce an output (like a classification or prediction).
    /// <para>
    /// The default implementation routes through the eager forward path
    /// (<see cref="PredictEager"/>) and is wrapped in a <see cref="NoGradScope{T}"/> so inference never
    /// records onto the gradient tape (matches PyTorch <c>torch.no_grad()</c> semantics). This was the
    /// compiled-replay path historically, but the compiled-plan cache in <see cref="PredictCompiled"/>
    /// binds to the trace-time input tensor reference and replay returns the first call's output for any
    /// subsequent call with the same shape but different values — the canonical "DifferentInputs /
    /// ScaledInput produces identical output" failure. Eager forward is correct for any input values at
    /// the cost of skipping plan-replay reuse.
    /// </para>
    /// <para>
    /// <see cref="PredictCompiled"/> remains available for callers that explicitly opt in via
    /// <see cref="CompileForward"/> + identical-tensor replay (or by overriding <see cref="Predict"/> in a
    /// subclass to call <see cref="PredictCompiled"/> directly when their tracing is value-stable).
    /// </para>
    /// <para>
    /// Subclasses that need custom inference behavior (e.g., diffusion models that run a multi-step
    /// denoising loop, GANs that sample from a generator, networks that produce structured outputs) should
    /// override this method.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        using var _ = new NoGradScope<T>();

        // Predict means inference. Temporarily flip the network into eval
        // mode so stateful layers (Dropout, BatchNorm batch-stats vs running-
        // stats, GaussianNoise, etc.) behave deterministically — and restore
        // the prior mode in finally so a Predict-mid-training-loop call
        // doesn't permanently flip the network out of training mode.
        // Without this, callers who never explicitly called
        // SetTrainingMode(false) get non-deterministic Predict outputs (the
        // default IsTrainingMode is true on construction), which is a real
        // model-behavior bug — see Generated Layers
        // PriorGradTests.Predict_ShouldBeDeterministic and similar.
        //
        // Concurrency contract: this temporary flip is intentionally NOT
        // thread-safe. Callers running parallel Predict on a single model
        // instance must serialize externally OR call SetTrainingMode(false)
        // once before the parallel batch (the second call is a no-op once
        // already in eval mode, so the inner toggle becomes side-effect-
        // free). This matches PyTorch nn.Module's non-thread-safe
        // `model.eval()` / `model.train()` convention — the framework
        // doesn't synchronize a global model state for concurrent inference.
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);

        // Pre-forward auto-detect attempt (#1222 / #183). Catches eager
        // networks (ResNet/VGG/CNN) whose ParameterCount is reliable as
        // soon as InitializeLayers ran. For lazy networks (Transformer,
        // MultiHeadAttention with 0×0 placeholders) ParameterCount is 0
        // here, so this is a no-op and the post-forward retry below
        // catches them once their weights have materialized through the
        // first PredictEager call. Idempotent once finalized.
        TryAutoEnableWeightStreaming();

        // L3b (#1622): latch the auto compiled-inference path for foundation-scale
        // models. We are guaranteed to be in inference mode here (the flip above ran),
        // so keying on _streamingEngagedByAutoDetect (set iff the model is over the
        // foundation-scale threshold and the user didn't opt out of auto-streaming) is
        // correct regardless of whether the streaming auto-detect first finalized from a
        // training- or inference-mode call. Idempotent. Cleared again on SetTrainingMode(true).
        if (_streamingEngagedByAutoDetect && !s_autoCompileDisabled)
            _autoCompiledInferenceEnabled = true;

        try
        {
            // Universal batch-dim auto-promotion (mirrors the Train path).
            // When the caller passes an unbatched single sample whose rank
            // exactly matches the architecture's effective unbatched rank,
            // prepend a unit batch dim before flowing through Layers.
            // Without this, FlattenLayer (and any other layer that treats
            // axis 0 as batch) would interpret the channels axis of a rank-3
            // [C, H, W] image as 32 separate batch samples and emit
            // [32, H*W] instead of [1, C*H*W] — collapsing the forward path
            // to one filter's pre-softmax distribution.
            var promoted = NormalizeInputBatchDim(input);

            // Route through the eager forward instead of PredictCompiled.
            //
            // HISTORY (corrected 2026-06, #1622 L3): an earlier revision of this
            // comment claimed compiled replay binds the trace-time input as a
            // constant and returns stale output for a *different* same-shape input.
            // That is NO LONGER TRUE — CompiledModelHost.Predict rebinds the fresh
            // input via plan.SetInputs(new[] { input }) before Execute() on both the
            // hot path and the slow path, so replay reflects the current input's
            // values. CompiledInferenceParityTests.{Mlp,Cnn}_CompiledReplay_
            // DifferentSameShapeInput_IsNotStale pin this (trace on A, replay on B,
            // assert eager(B)).
            //
            // The reason Predict still defaults to EAGER is a different, narrower
            // risk: some model families compile a plan that succeeds but is
            // NUMERICALLY WRONG (the AIsEval benchmark's "compiled 2/16 vs eager
            // 4/16"; CompiledInferenceParityTests Defect 1 was one such CNN plan).
            // A plan that compiles "successfully" but computes the wrong answer would
            // be SILENTLY wrong if we replayed it blind. So compiled-by-default needs
            // a verify-then-trust gate (run eager once, compare, adopt compiled only
            // on parity; else stay eager for that model/shape) — tracked as #1622 L3b.
            // NOTE for the L3b implementer: the gate canNOT just live here — almost
            // every concrete model (all but ClipNeuralNetwork) OVERRIDES Predict and
            // never calls this base method, so a base-only gate would cover ~1 model.
            // L3b needs either a sealed Predict + virtual PredictCore refactor (single
            // funnel for the gate) or per-model wiring of a shared verify helper.
            // Until that lands, eager is the safe default; callers can opt into
            // compiled replay explicitly via CompileForward + PredictCompiled.
            //
            // #1622 L3b: for foundation-scale models in inference,
            // _autoCompiledInferenceEnabled routes through PredictAccelerated — the
            // verify-then-trust gate the note above asks for. It runs eager once per
            // shape to validate the compiled plan (adopting it only on numerical
            // parity, else staying eager for that shape), and adds a collision-safe
            // value-hash memo. Output is numerically identical to PredictEager for
            // every input, so it is safe as the base-Predict default for the models
            // that route here; small models / training stay eager.
            var output = _autoCompiledInferenceEnabled
                ? PredictAccelerated(promoted)
                : PredictEager(promoted);

            // If we promoted the input by prepending a unit batch dim,
            // squeeze the same dim back off the output so callers
            // passing unbatched inputs get unbatched outputs. Without
            // this, ResNet/VGG/MobileNet etc. (which used to squeeze
            // inside Forward when they added their own batch dim) now
            // return a phantom batch axis on what should be a single-
            // sample inference.
            bool wasPromoted = !ReferenceEquals(promoted, input);
            if (wasPromoted && output.Rank > 1 && output.Shape[0] == 1)
            {
                int[] squeezed = new int[output.Rank - 1];
                for (int i = 0; i < squeezed.Length; i++)
                    squeezed[i] = output.Shape[i + 1];
                output = output.Reshape(squeezed);
            }

            // Mark first-forward-completed and run the auto-detect retry
            // ONLY on a successful forward. The previous version did this
            // in `finally`, which fired even when PredictEager threw; on
            // a lazy model that flipped _firstForwardCompleted=true with
            // ParameterCount still at 0 (the placeholder weights weren't
            // materialized because forward failed), so the next forward
            // call's auto-detect saw "already past first forward" and
            // skipped the retry path that's the WHOLE POINT of this
            // logic. Closes review-comment #1271.s-Ng.
            //
            // For lazy networks (Transformer / MultiHeadAttention) the
            // pre-forward attempt at the top of Predict saw
            // ParameterCount=0 and bailed without finalizing; now that
            // PredictEager has materialized the placeholder weights,
            // ParameterCount returns the real value and the threshold
            // comparison can engage streaming for the NEXT call. The
            // first call of an above-threshold lazy model still runs in
            // eager mode — engaging streaming mid-forward would require
            // a more invasive RegisterTrainableParameter-time hook
            // (filed as a follow-up). Eager networks finalized in the
            // pre-forward call so this is a no-op for them.
            if (!_firstForwardCompleted)
            {
                _firstForwardCompleted = true;
                // Invalidate the cached ParameterCount before the retry.
                // The pre-forward auto-detect attempt at the top of
                // Predict (above) called ParameterCount when lazy
                // layers still reported 0, populating the cache with
                // 0. Without this invalidation the post-forward retry
                // would re-read the SAME cached 0 (because the cache
                // sticks until layer mutation) and never engage
                // streaming for the NEXT call — defeating the entire
                // point of the lazy-layer retry path. Closes review-
                // comments #1271.uxiB / .vbtb.
                _cachedParameterCount = null;
                TryAutoEnableWeightStreaming();
            }
            return output;
        }
        finally
        {
            // Restore the training-mode flag regardless of whether
            // PredictEager threw — that's a state restore, not a
            // success-only side effect, so it stays in finally.
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>
    /// Chunked inference path: splits <paramref name="input"/> along axis 0
    /// into slices of size <paramref name="batchSize"/>, runs <see cref="Predict"/>
    /// on each slice, and concatenates the per-slice outputs back along axis 0.
    /// Output shape matches what <c>Predict(input)</c> would have produced —
    /// chunking is a memory-bounding refactor of the same forward computation,
    /// not a semantic change. Single-chunk inputs short-circuit to a direct
    /// <see cref="Predict"/> call.
    /// </summary>
    /// <param name="input">Batched input tensor with at least one axis-0 sample.</param>
    /// <param name="batchSize">Maximum samples per forward pass. Clamped to <c>≥ 1</c>.</param>
    /// <remarks>
    /// Use this in any caller that may receive a full-dataset tensor and
    /// would otherwise allocate O(N · seq²) attention scores in one shot —
    /// optimizer evaluators, cross-validation predict, AutoML, etc. The
    /// chunked path eliminates the OOM surface described in #1296 (and the
    /// sibling-bug audit at the Predict-eval site) for Transformer-class
    /// models while leaving small-batch calls (the common case) unchanged.
    ///
    /// Layers that maintain per-sample state across batch elements (none of
    /// the canonical Dense/Conv/Attention/Norm layers do — they treat
    /// axis-0 as i.i.d. samples) would observe different behaviour vs a
    /// single big forward. Callers that build such layers must opt out by
    /// invoking <see cref="Predict"/> directly. This is the same contract
    /// PyTorch eval-mode forward implies when iterating a DataLoader.
    /// </remarks>
    public virtual Tensor<T> PredictInBatches(Tensor<T> input, int batchSize)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (batchSize < 1) batchSize = 1;

        // A rank-0 / unbatched-leading tensor has no meaningful axis-0
        // chunking semantics — fall through to plain Predict and let the
        // existing batch-dim promotion path inside Predict handle it.
        //
        // The leading-axis-larger-than-batchSize check alone is not enough:
        // a genuine single sample with shape [seq, F] (or even [features]
        // where features > batchSize) would otherwise be sliced on its
        // sequence / feature axis instead of a batch axis, corrupting
        // semantics. When the architecture's expected unbatched input
        // rank equals the input's rank, this is a single sample — short-
        // circuit to Predict so the auto-promote path treats it correctly.
        int expectedUnbatchedRank = GetExpectedUnbatchedInputRank();
        bool appearsUnbatched = expectedUnbatchedRank > 0
            && input.Rank == expectedUnbatchedRank;

        if (input.Rank == 0 || appearsUnbatched || input.Shape[0] <= batchSize)
        {
            return Predict(input);
        }

        int n = input.Shape[0];
        int nChunks = (n + batchSize - 1) / batchSize;
        var perChunkOutputs = new Tensor<T>[nChunks];

        // Per-chunk forwards route through plain Predict (which uses
        // PredictEager by default). Two attempts to engage the compile
        // cache here surfaced an inherent memory tradeoff: even with
        // tail-padding so all chunks share a single shape (and therefore
        // hit one cached plan instead of two), the plan itself retains
        // all-layer pinned intermediate buffers (~96 MB at d=128 / L=4
        // /heads=4 / ctx=64) that coexist with the training tape's own
        // intermediates across the optimizer's epoch loop. The host's
        // unmanaged-commit limit gets exhausted before the test finishes.
        // The value-stable rebind landed in CompiledModelHost in the same
        // PR still benefits any caller that explicitly invokes
        // PredictCompiled; chunked PredictInBatches just stays on the
        // eager path the default Predict already uses. A future Tensors-
        // side pass that bounds the compiled plan's resident memory
        // (e.g. by eagerly releasing intermediate buffers between
        // Executes) is the prerequisite for safely re-enabling compile-
        // by-default on chunked dispatch.
        for (int chunkIdx = 0; chunkIdx < nChunks; chunkIdx++)
        {
            int start = chunkIdx * batchSize;
            int end = Math.Min(start + batchSize, n);
            var chunk = SliceAlongAxis0(input, start, end);
            perChunkOutputs[chunkIdx] = Predict(chunk);
        }

        return Tensor<T>.Concatenate(perChunkOutputs, axis: 0);
    }

    /// <summary>
    /// Chunked training with TRUE gradient accumulation. Walks
    /// <paramref name="input"/> / <paramref name="target"/> in axis-0
    /// chunks of size <paramref name="batchSize"/>, runs forward + backward
    /// per chunk WITHOUT firing the optimizer step, sums per-chunk
    /// gradients into a single accumulator keyed by parameter-tensor
    /// reference, then applies one optimizer step at the end with the
    /// averaged gradient. Matches PyTorch's
    /// <c>optimizer.zero_grad / loss.backward / optimizer.step</c> idiom
    /// that users normally hand-roll for memory-bounded full-batch
    /// gradients.
    /// </summary>
    /// <param name="input">Full input tensor; leading axis is chunked.</param>
    /// <param name="target">Full target tensor; leading axis must match input.</param>
    /// <param name="batchSize">Per-chunk leading-axis size. Sub-threshold inputs short-circuit to a single <see cref="Train"/> call.</param>
    /// <remarks>
    /// Preserves full-batch-equivalent gradient DIRECTION and keeps the
    /// optimizer's per-parameter state (Adam's <c>_m</c> / <c>_v</c>) at
    /// one update per logical batch — the semantic that distinguishes
    /// gradient accumulation from naive mini-batched SGD. Falls through
    /// to <see cref="Train"/> when the input is already small enough to
    /// fit in one forward.
    /// </remarks>
    public virtual void TrainWithGradientAccumulation(
        Tensor<T> input, Tensor<T> target, int batchSize)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (batchSize < 1) batchSize = 1;

        // Match Train's pre-conditions and training-mode bracketing so
        // unbatched [seq,F] inputs don't get chunked on their sequence
        // axis, dropout/batchnorm run with the right semantics, and
        // input / target leading-axis mismatches surface as a clear
        // ArgumentException rather than silent truncation.
        (input, target) = NormalizeBatchDim(input, target);

        if (input.Rank == 0 || target.Rank == 0 || input.Shape[0] <= batchSize)
        {
            Train(input, target);
            return;
        }

        if (input.Shape[0] != target.Shape[0])
        {
            throw new ArgumentException(
                $"TrainWithGradientAccumulation: input leading axis ({input.Shape[0]}) " +
                $"must equal target leading axis ({target.Shape[0]}). " +
                $"input.Shape=[{string.Join(",", input._shape)}] " +
                $"target.Shape=[{string.Join(",", target._shape)}].",
                nameof(target));
        }

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "TrainWithGradientAccumulation requires a LossFunctionBase<T> for ComputeTapeLoss.");

        // Collect any network-level trainable tensors so models with
        // raw trainable parameters on the network (not on a layer) keep
        // receiving updates under gradient accumulation, matching
        // TrainWithTape's parameter set.
        var extraTrainableTensors = new List<Tensor<T>>();
        foreach (var t in GetExtraTrainableTensors())
        {
            if (t is not null && t.Length > 0) extraTrainableTensors.Add(t);
        }

        bool wasTraining = IsTrainingMode;
        if (!wasTraining) SetTrainingMode(true);
        try
        {
            int n = input.Shape[0];
            int nChunks = (n + batchSize - 1) / batchSize;
            Dictionary<Tensor<T>, Tensor<T>>? accumGrads = null;
            T accumLoss = NumOps.Zero;
            int totalSamples = 0;

            for (int chunkIdx = 0; chunkIdx < nChunks; chunkIdx++)
            {
                int start = chunkIdx * batchSize;
                int end = Math.Min(start + batchSize, n);
                int chunkSamples = end - start;
                totalSamples += chunkSamples;
                var xChunk = SliceAlongAxis0(input, start, end);
                var yChunk = SliceAlongAxis0(target, start, end);

                using var arena = TensorArena.Create();
                using var tape = new GradientTape<T>();
                var prediction = ForwardForTraining(xChunk);

                // Align target to prediction shape — same policy as TrainWithTape.
                var alignedTarget = AlignTargetToOutputShape(prediction, yChunk);

                var lossTensor = loss.ComputeTapeLoss(prediction, alignedTarget);
                T chunkLoss = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

                // Weight each chunk's loss by its sample count so a final
                // short chunk doesn't overweight the accumulator. Dividing
                // by nChunks alone is only correct when every chunk is the
                // same size, which is rarely true at the tail.
                T chunkSamplesT = NumOps.FromDouble(chunkSamples);
                accumLoss = NumOps.Add(accumLoss, NumOps.Multiply(chunkLoss, chunkSamplesT));

                var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers, _layerStructureVersion);
                var allGrads = tape.ComputeGradients(lossTensor, sources: null);
                // Walk both the layer-collected params AND any network-
                // level trainable tensors so the latter actually receive
                // accumulated gradient updates.
                foreach (var param in trainableParams.Concat(extraTrainableTensors))
                {
                    if (!allGrads.TryGetValue(param, out var grad)) continue;
                    if (accumGrads is null)
                    {
                        accumGrads = new Dictionary<Tensor<T>, Tensor<T>>(
                            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                    }
                    if (accumGrads.TryGetValue(param, out var existing))
                    {
                        var scaledGrad = Engine.TensorMultiplyScalar(grad, chunkSamplesT);
                        Engine.TensorAddInPlace(existing, scaledGrad);
                    }
                    else
                    {
                        // Clone-and-scale so the gradient survives tape
                        // disposal at the arena-using scope end AND is
                        // pre-weighted by its chunk's sample count.
                        var cloned = Engine.TensorMultiplyScalar(grad, chunkSamplesT);
                        accumGrads[param] = cloned;
                    }
                }
            }

            if (accumGrads is null || accumGrads.Count == 0) return;

            // Average across SAMPLES to match a single full-batch SGD
            // step under mean-reduced losses. Each chunk's grad and loss
            // were pre-weighted by its sample count above; dividing the
            // sum by total samples gives the true mean over the logical
            // full batch even when the final chunk is short.
            if (totalSamples <= 0) return;
            T inverseSamples = NumOps.Divide(NumOps.One, NumOps.FromDouble(totalSamples));
            var avgGrads = new Dictionary<Tensor<T>, Tensor<T>>(
                Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var kvp in accumGrads)
            {
                avgGrads[kvp.Key] = Engine.TensorMultiplyScalar(kvp.Value, inverseSamples);
            }
            T avgLoss = NumOps.Multiply(accumLoss, inverseSamples);

            // Build one TapeStepContext and fire the optimizer once. The
            // optimizer's per-parameter state (Adam m/v) advances exactly
            // once for the logical full batch, matching PyTorch's grad-
            // accum pattern.
            var optimizer = GetOrCreateBaseOptimizer();
            var paramsList = Training.TapeTrainingStep<T>.CollectParameters(Layers, _layerStructureVersion);

            // Mirror TrainWithTape's pre-step global gradient-norm clipping
            // on the ACCUMULATED gradient (PyTorch grad-accum idiom: clip
            // once on the summed/averaged gradient right before
            // optimizer.step()). Without this, every other training entry
            // point clips at MaxGradNorm (default 1.0) while this path
            // didn't — so a single full-batch Train step and an equivalent
            // TrainWithGradientAccumulation step produced visibly different
            // updates whenever the gradient norm exceeded the clip threshold
            // (the Issue1296 GradientAccumulation_MatchesFullBatchGradient
            // probe caught exactly this). Clips over the layer-collected
            // params only — the same set TrainWithTape clips — using the
            // deterministic list order for the norm reduction.
            double maxGradNorm = MaxGradNormValue;
            if (maxGradNorm > 0.0 && avgGrads.Count > 0)
            {
                ApplyGradientClipping(avgGrads, maxGradNorm, paramsList);
            }
            // Include extra trainable tensors in the optimizer's
            // parameter list too, so its update step (and any per-
            // parameter state it maintains) covers them.
            var fullParams = paramsList.Concat(extraTrainableTensors).ToList();
            var context = new TapeStepContext<T>(
                fullParams, avgGrads, avgLoss,
                input, target,
                (inp, tgt) => ForwardForTraining(inp),
                (pred, tgt) => loss.ComputeTapeLoss(pred, tgt),
                parameterBuffer: null);
            optimizer.Step(context);
            // GPU weight-cache coherence after the grad-accum aggregated step.
            // See InvalidateWeightCachesAfterSuccessfulWeightUpdate.
            InvalidateWeightCachesAfterSuccessfulWeightUpdate();
            StepSchedulerIfSupported(optimizer);
            LastLoss = avgLoss;
        }
        finally
        {
            if (!wasTraining) SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Axis-0 slice helper for <see cref="PredictInBatches"/>: returns
    /// <c>input[start..end, …]</c> as a contiguous tensor. Uses the
    /// O(1) <see cref="Tensor{T}.Slice(int, int, int?)"/> view and
    /// materialises it with <see cref="Tensor{T}.Contiguous"/> so downstream
    /// engine ops (BLAS GEMM, attention SDP, etc.) that assume contiguous
    /// storage observe a strict copy of the slice, not a strided view into
    /// the parent.
    /// </summary>
    private static Tensor<T> SliceAlongAxis0(Tensor<T> input, int start, int end)
    {
        return input.Slice(axis: 0, start: start, end: end).Contiguous();
    }

    /// <summary>
    /// Read-only counterpart to <see cref="NormalizeBatchDim"/> for the
    /// inference path: only the input is shape-normalized; targets aren't
    /// involved in Predict. Returns the original tensor if the architecture
    /// has no usable input shape or if input is already batched.
    /// </summary>
    private Tensor<T> NormalizeInputBatchDim(Tensor<T> input)
    {
        int expectedUnbatchedRank = GetExpectedUnbatchedInputRank();
        if (expectedUnbatchedRank <= 0) return input;
        if (input.Rank != expectedUnbatchedRank) return input;
        return PromoteToBatchedTensor(input);
    }

    /// <summary>
    /// Computes the effective unbatched input rank from the architecture's
    /// input dimensions. <see cref="NeuralNetworkArchitecture{T}.GetInputShape"/>
    /// is inconsistent across <see cref="InputType"/> variants — TwoDimensional
    /// returns [H, W] (rank 2) but ConvNN-style consumers internally treat it
    /// as [1, H, W] (rank 3) when InputDepth ≥ 1. This helper picks the rank
    /// the model's first layer actually expects so auto-promote can fire on
    /// the right unbatched signal.
    /// </summary>
    /// <remarks>
    /// Test-only / helper-only accessor. <see cref="NeuralBatchHelper"/>
    /// uses this via <see cref="GetExpectedUnbatchedInputRankInternal"/>
    /// to decide whether a tensor with a leading axis exceeding its chunk
    /// size is truly a batched input or an unbatched single sample whose
    /// leading axis is sequence/features.
    /// </remarks>
    internal int GetExpectedUnbatchedInputRankInternal() => GetExpectedUnbatchedInputRank();

    private int GetExpectedUnbatchedInputRank()
    {
        if (Architecture is null) return 0;
        try
        {
            // Video / spatiotemporal: InputFrames > 0 means
            // [Frames, C, H, W] is the unbatched layout (rank 4).
            // Use this branch BEFORE the spatial check so video
            // architectures (which also have InputHeight > 0) don't
            // resolve to rank 3.
            if (Architecture.InputFrames > 0
                && Architecture.InputHeight > 0
                && Architecture.InputWidth > 0)
                return 4;
            // Vision / spatial: InputHeight > 0 means [C, H, W] is the
            // unbatched layout (rank 3). InputDepth defaults to 1 when not
            // explicitly set on a TwoDimensional arch — paper-faithful CNN
            // models (LeNet on MNIST, etc.) all assume a channel axis.
            if (Architecture.InputHeight > 0 && Architecture.InputWidth > 0)
                return 3;
            // Sequence / time-series: when the architecture's
            // GetInputShape() reports a rank-2 unbatched layout
            // (e.g. `[seq, F]` for transformer/RNN models), honour it.
            // Architectures built via NeuralNetworkArchitecture's
            // sequence-aware ctors expose a 2-element input shape with
            // both axes positive.
            var inShape = Architecture.GetInputShape();
            if (inShape is { Length: 2 } && inShape[0] > 0 && inShape[1] > 0)
                return 2;
            // Sequence / feature: InputSize is the per-sample feature count;
            // unbatched is rank 1 [F].
            if (Architecture.InputSize > 0)
                return 1;
        }
        catch { /* fall through */ }
        return 0;
    }

    /// <summary>
    /// Runs the forward pass through all layers WITHOUT suppressing tape recording.
    /// Used for tape-based training where engine ops must be recorded.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The network output.</returns>
    /// <inheritdoc />
    private bool _layerRandomSeedsWired;

    /// <summary>
    /// Propagates <see cref="NeuralNetworkArchitecture{T}.RandomSeed"/> to every layer (and nested
    /// sub-layer) so seed-respecting stochastic layers — chiefly <see cref="Layers.DropoutLayer{T}"/>,
    /// whose mask derives from <see cref="Layers.LayerBase{T}.RandomSeed"/> plus a per-forward
    /// counter — produce a reproducible stream. No-op when no seed is set (production default), so
    /// it never changes entropy-seeded behavior; only the test harness's
    /// <see cref="NeuralNetworkArchitecture{T}.DefaultRandomSeedOverride"/> (or an explicit seed)
    /// engages it. Each layer gets a distinct derived seed so their RNG streams don't collide.
    /// </summary>
    private void WireLayerRandomSeeds()
    {
        if (Architecture?.RandomSeed is not int seed) return;
        var seedRng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        foreach (var layer in Layers)
        {
            WireLayerRandomSeedRecursive(layer, seedRng);
        }
    }

    /// <summary>
    /// Runs <see cref="WireLayerRandomSeeds"/> exactly once, before the first training forward.
    /// Subclasses that override <see cref="ForwardForTraining"/> without calling the base
    /// implementation (e.g. finance models that route through their own native forward) MUST call
    /// this first, or their stochastic layers (DropoutLayer) train with order-dependent masks and
    /// the training-trajectory invariants flake / fail only when other tests advance the shared RNG.
    /// </summary>
    protected void EnsureLayerRandomSeedsWired()
    {
        if (_layerRandomSeedsWired) return;
        _layerRandomSeedsWired = true;
        WireLayerRandomSeeds();
    }

    private static void WireLayerRandomSeedRecursive(ILayer<T> layer, Random seedRng)
    {
        if (layer is Layers.LayerBase<T> baseLayer)
        {
            baseLayer.RandomSeed = seedRng.Next();
            foreach (var sub in baseLayer.GetSubLayers())
            {
                WireLayerRandomSeedRecursive(sub, seedRng);
            }
        }
    }

    public virtual Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Propagate the architecture's RandomSeed to every layer once, before the first training
        // forward, so stochastic layers (notably DropoutLayer, whose mask derives from
        // RandomSeed + a per-forward counter) produce a REPRODUCIBLE sequence. Without this a
        // model whose CreateDefault layer chain wasn't seed-wired trains with fresh random dropout
        // masks each run, making training-trajectory invariants (e.g. MoreData_ShouldNotDegrade)
        // flake run-to-run. Only runs when a seed is actually set (explicit per-architecture seed,
        // or the test harness's process-wide DefaultRandomSeedOverride); production with no seed is
        // unaffected. Eager-init layers already chose their weights, but the loss-comparison tests
        // clone a single network so init is shared — the dropout stream is the run-to-run variable.
        EnsureLayerRandomSeedsWired();

        // Gradient checkpointing opt-in path. Unify on a single source of
        // truth by consulting BOTH:
        //   (a) the builder-set GradientCheckpointingSegmentSize (populated by
        //       ConfigureMemoryManagement with UseGradientCheckpointing=true), and
        //   (b) the existing _memoryManager (populated by the public
        //       EnableMemoryManagement method for direct-model usage)
        // If either path says checkpointing is active, use it — resolving the
        // "two sources disagree" reviewer concern. Precedence: builder flag
        // wins when both specify a segment size; the memory-manager fallback
        // preserves behavior for users who configured via EnableMemoryManagement
        // directly without touching the builder.
        int segmentSize = GradientCheckpointingSegmentSize;
        if (segmentSize <= 0 && _memoryManager is not null && _memoryManager.IsCheckpointingEnabled)
        {
            // Memory-manager-enabled but builder-set size is 0 → use the
            // existing default heuristic based on sqrt(N), where N is the
            // current layer count. sqrt(N) gives the optimal memory/compute
            // tradeoff: ~sqrt(N) checkpoints with ~33% extra compute.
            segmentSize = Math.Max(1, (int)Math.Sqrt(Math.Max(1, Layers.Count)));
        }
        if (segmentSize > 0 && Layers.Count > segmentSize && CanUseGradientCheckpointingForCurrentLayerGraph())
        {
            // Cache the layer-forward delegate array so checkpointed training
            // doesn't allocate N closures + a delegate array on every call.
            // Rebuild only when the layer graph changes (structure version
            // bump). Saves ~Layers.Count delegate allocations per training step.
            if (_checkpointLayerFunctions is null
                || _checkpointFunctionsVersion != _layerStructureVersion)
            {
                var fns = new Func<Tensor<T>, Tensor<T>>[Layers.Count];
                for (int i = 0; i < Layers.Count; i++)
                {
                    // Use the method group directly so the delegate binds to the
                    // layer instance's Forward method with no captured local —
                    // no per-layer closure allocation. The delegate holds a
                    // reference to the layer (its implicit Target) but no
                    // extra heap closure is created.
                    //
                    // Training-mode correctness: SetTrainingMode(true) is called
                    // by the caller (TrainWithTape) for the entire training step,
                    // and layer.Forward reads IsTrainingMode internally to select
                    // training-time behavior (e.g. Dropout applies masks, LayerNorm
                    // uses batch stats). Since the mode flag is set ONCE per step
                    // and stays true across both the original forward and the
                    // checkpoint recomputation, both code paths see identical
                    // training-time behavior for mode-aware layers.
                    //
                    // Note on stochastic/stateful layers: if a segment contains
                    // Dropout or BatchNormalization, the recomputation during
                    // backward generates a NEW dropout mask / updates the running
                    // stats AGAIN. This is a known property of activation
                    // checkpointing (identical to PyTorch's checkpoint without
                    // preserve_rng_state) — callers who need bit-exact gradients
                    // across recomputation should either disable checkpointing or
                    // restrict it to segments without stochastic/stateful layers.
                    // The checkpoint segment-size contract is unchanged; this is
                    // documented behavior, not a bug in the delegate binding.
                    fns[i] = Layers[i].Forward;
                }
                _checkpointLayerFunctions = fns;
                _checkpointFunctionsVersion = _layerStructureVersion;
            }
            return AiDotNet.Tensors.Engines.Autodiff.GradientCheckpointing<T>.Checkpoint(
                _checkpointLayerFunctions, input, segmentSize);
        }

        // Streaming-aware training forward (#1222 / #185). When weight
        // streaming is configured, drive the same prefetch + materialize-
        // scope orchestration the inference path uses (PredictEager →
        // PredictEagerStreaming, see #184) so a Train() call's forward
        // pulls weights through the streaming pool with the LRU keeping
        // them warm into the backward replay. The tape-based backward
        // accesses the SAME tensor instances the forward read; LRU
        // recency-of-access is what keeps them resident through the
        // backward pass, so we don't need a parallel reverse-order
        // materialize loop here. Verified by inspection: the autodiff
        // tape stores tensor references, not copies, so a tensor
        // already pinned by forward's MaterializeScope is the SAME
        // object the backward-replay reads.
        Tensor<T> result;
        if (_weightLifetimeConfigured)
        {
            result = PredictEagerStreaming(input);
        }
        else
        {
            var current = input;
            foreach (var layer in Layers)
            {
                current = layer.Forward(current);
            }
            result = current;
        }

        // Post-first-forward auto-detect retry (mirrors Predict's path).
        // Lazy training-only networks that never call Predict (e.g. a
        // fine-tuning loop where forward is always paired with
        // backward) need this hook to ever engage streaming. Invalidate
        // the cached ParameterCount first — the pre-forward attempt may
        // have populated it with 0 from lazy placeholder layers, and a
        // retry that reuses the stale cache would never engage. Closes
        // review-comment #1271.vbtm.
        if (!_firstForwardCompleted)
        {
            _firstForwardCompleted = true;
            _cachedParameterCount = null;
            TryAutoEnableWeightStreaming();
        }

        return result;
    }

    protected bool CanUseGradientCheckpointingForCurrentLayerGraph()
    {
        foreach (var layer in Layers)
        {
            if (ContainsCheckpointUnsafeLayer(layer))
                return false;
        }

        return true;
    }

    private static bool ContainsCheckpointUnsafeLayer(ILayer<T> layer)
    {
        var typeName = layer.GetType().Name;
        if (typeName.Contains("Dropout", StringComparison.OrdinalIgnoreCase)
            || typeName.Contains("BatchNormalization", StringComparison.OrdinalIgnoreCase)
            || typeName.Contains("BatchNorm", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (layer is Layers.LayerBase<T> baseLayer)
        {
            foreach (var child in baseLayer.GetSubLayers())
            {
                if (ContainsCheckpointUnsafeLayer(child))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Per-instance gradient-checkpointing segment size. 0 = disabled.
    /// </summary>
    internal int GradientCheckpointingSegmentSize { get; private set; }

    private Func<Tensor<T>, Tensor<T>>[]? _checkpointLayerFunctions;
    private int _checkpointFunctionsVersion = -1;

    internal void SetGradientCheckpointingSegmentSize(int segmentSize)
    {
        if (segmentSize < 0)
            throw new ArgumentOutOfRangeException(nameof(segmentSize),
                "Segment size must be non-negative. Use 0 to disable checkpointing.");
        GradientCheckpointingSegmentSize = segmentSize;
    }

    /// <summary>
    /// Composable inference-compilation helper — traces the forward pass on first
    /// call at each input shape and replays the compiled plan on subsequent calls.
    /// Falls back to eager execution on failure. Invalidated automatically when
    /// <see cref="_layerStructureVersion"/> changes.
    /// </summary>
    /// <remarks>
    /// Single attachment point for future compilation features: AOT plan
    /// serialization, CUDA Graph capture, symbolic shape plans, persistent
    /// autotune. All of those attach to <see cref="CompiledModelHost{T}"/>
    /// rather than re-implementing the compile+cache+fallback dance per
    /// model family.
    /// </remarks>
    private readonly CompiledModelHost<T> _compileHost;

    /// <summary>
    /// Tracks input shapes whose compilation has previously failed on this
    /// model instance. Once a shape is in this set, <see cref="PredictCompiled"/>
    /// short-circuits to <see cref="PredictEager"/> without re-attempting
    /// compilation — re-trying every Predict would re-burn the same trace cost
    /// for no benefit and re-emit the same Trace warning on every call.
    /// Uses <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey, TValue}"/>
    /// (value type <see cref="byte"/> is unused) so concurrent Predict calls
    /// on the same model instance (request-pool sharing) can safely add and
    /// read without external synchronization.
    /// </summary>
    private readonly System.Collections.Concurrent.ConcurrentDictionary<long, byte> _knownBadCompileShapes = new();

    /// <summary>
    /// Computes a deterministic 64-bit shape key (FNV-1a) for the bad-compile
    /// cache. Using a numeric key keeps the set entries cheap to hash without
    /// allocating a wrapper object per Predict call.
    /// </summary>
    private static long ComputeShapeKey(int[] shape)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        for (int i = 0; i < shape.Length; i++)
        {
            hash ^= shape[i];
            hash *= unchecked((long)0x100000001b3L);
        }
        return hash;
    }

    /// <summary>
    /// Executes the forward pass using a compiled plan for maximum performance.
    /// First call traces and compiles; subsequent calls replay the compiled plan.
    /// Falls back to eager execution if compilation fails. Plans auto-invalidate
    /// when <see cref="_layerStructureVersion"/> changes.
    /// </summary>
    protected internal Tensor<T> PredictCompiled(Tensor<T> input)
    {
        // Mirror Predict()'s inference-mode contract: temporary training-
        // mode flip so stateful layers (Dropout, BatchNorm running-stats,
        // GaussianNoise) behave deterministically + cheaply. Without this,
        // PredictCompiled's eager fallback or replay both ran with Dropout
        // sampling and BatchNorm batch-stat updates every call — 3-5× per-
        // call slowdown vs Predict() at the same model shape, and outputs
        // were non-deterministic across repeated calls with the same input.
        // NoGradScope mirrors Predict() too so the autodiff tape doesn't
        // record any of the inference work as a training step.
        using var _ = new NoGradScope<T>();
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try
        {
            return _compileHost.Predict(input, _layerStructureVersion, () => PredictEager(input));
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>
    /// Eagerly traces and compiles the forward pass for the given input shape,
    /// storing the compiled plan in the per-instance cache. Subsequent calls to
    /// <see cref="Predict"/> / <see cref="PredictCompiled"/> with the same
    /// input shape replay the plan with zero re-compile overhead.
    /// </summary>
    /// <param name="sampleInput">
    /// A sample input tensor whose shape keys the compiled plan. The tensor's
    /// values are used during tracing but are not persisted; any tensor of
    /// the target inference shape works.
    /// </param>
    /// <returns>
    /// <c>true</c> when compilation succeeds and the plan is cached. <c>false</c>
    /// when compilation is disabled via
    /// <see cref="AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.EnableCompilation"/>
    /// or when tracing throws (in which case <see cref="Predict"/> transparently
    /// falls back to eager execution).
    /// </returns>
    /// <remarks>
    /// <para>
    /// Addresses the "NeuralNetworkBase.CompileForward()" checklist item on
    /// github.com/ooples/AiDotNet#1015. Gives applications an explicit pre-warm
    /// hook so the first production inference doesn't pay the one-time trace +
    /// compile cost. Call once at startup with a representative input shape
    /// for lowest-latency first-inference.
    /// </para>
    /// <para>
    /// Multiple calls with different shapes pre-warm multiple plans in the
    /// same cache — useful for applications that accept variable batch sizes
    /// or sequence lengths.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this like a "pre-heat the oven"
    /// step. The first time you ask the model to predict something, it spends
    /// extra time analyzing the network and building an optimized plan.
    /// Calling this method does that work up-front, so the real predictions
    /// are fast from the very first call.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var network = new MyNetwork(...);
    /// var warmupInput = new Tensor&lt;float&gt;(new[] { 1, 3, 224, 224 }); // batch=1, RGB 224x224
    /// if (network.CompileForward(warmupInput))
    /// {
    ///     // Pre-warmed — first real Predict is zero-overhead replay.
    /// }
    /// // Otherwise compilation is off / failed; Predict transparently falls back.
    /// </code>
    /// </example>
    public bool CompileForward(Tensor<T> sampleInput)
    {
        if (sampleInput is null)
            throw new ArgumentNullException(nameof(sampleInput));
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return false;

        try
        {
            // _compileHost.Predict handles trace-and-compile-on-miss / replay-on-hit
            // internally and invalidates stale plans when _layerStructureVersion changes.
            // Executing once up-front warms the plan so the first production inference
            // pays no trace + compile cost.
            var warmupOutput = _compileHost.Predict(
                sampleInput,
                _layerStructureVersion,
                () => PredictEager(sampleInput));
            if (warmupOutput is IDisposable disposableOutput)
                disposableOutput.Dispose();
            return true;
        }
        catch (Exception ex) when (
            ex is not OutOfMemoryException &&
            ex is not StackOverflowException &&
            ex is not AccessViolationException)
        {
            // Trace/compile failed — fall back to lazy-compile-on-first-Predict.
            // PredictCompiled still handles the eager fallback transparently.
            // Fatal CLR exceptions (OOM, SO, AV) propagate rather than being
            // silently swallowed.
            System.Diagnostics.Trace.TraceWarning(
                $"CompileForward failed for shape [{string.Join(",", sampleInput._shape)}]: " +
                $"{ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // #1622 L3 — auto-enabled, correctness-gated compiled inference + value memo.
    // The verify-then-trust gate + collision-safe value memo live in the reusable
    // VerifiedInferenceGate<T> component (shared with NoisePredictorBase so the
    // diffusion family gets the same behavior); this class just composes one and
    // feeds it the eager + compiled forwards.
    // ─────────────────────────────────────────────────────────────────────────

    private readonly VerifiedInferenceGate<T> _inferenceGate = new();

    /// <summary>
    /// True when this model is foundation-scale and in inference mode, so
    /// <see cref="Predict"/> routes through the verify-then-trust compiled path +
    /// value memo instead of plain eager. Latched in <see cref="Predict"/> /
    /// <see cref="SetTrainingMode"/> off <see cref="WeightStreamingAutoDetected"/>;
    /// cleared when training is (re)enabled. Small models and training stay eager.
    /// </summary>
    private volatile bool _autoCompiledInferenceEnabled;

    /// <summary>
    /// Process-wide opt-out for the auto compiled-inference path. Set
    /// <c>AIDOTNET_DISABLE_AUTO_COMPILE=1</c> to force every model onto the plain
    /// eager path (debugging a suspected compiled-plan defect, or pinning bit-exact
    /// eager numerics). Read once.
    /// </summary>
    private static readonly bool s_autoCompileDisabled =
        string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_AUTO_COMPILE"), "1", StringComparison.Ordinal);

    /// <summary>Whether the auto compiled-inference path is currently engaged for this model. Diagnostic/test hook.</summary>
    internal bool AutoCompiledInferenceEngaged => _autoCompiledInferenceEnabled;

    /// <summary>Count of value-memo hits (identical-input short-circuits) on the accelerated path. Diagnostic/test hook.</summary>
    internal long AcceleratedMemoHits => _inferenceGate.MemoHits;

    /// <summary>Count of verify-then-trust verifications performed (one per new shape+version). Diagnostic/test hook.</summary>
    internal long AcceleratedVerifyCount => _inferenceGate.VerifyCount;

    /// <summary>
    /// Forces the verify-then-trust compiled inference path on/off for this model, bypassing the
    /// foundation-scale auto-detect gate. Test hook only — production enablement flows through
    /// <see cref="TryAutoEnableWeightStreaming"/> + <see cref="Predict"/>.
    /// </summary>
    internal void ForceAutoCompiledInferenceForTesting(bool enabled) => _autoCompiledInferenceEnabled = enabled;

    /// <summary>
    /// Returns the current verify-then-trust verdict for an input shape at the live structure
    /// version: 0 = unknown/not-yet-verified, 1 = trusted (compiled replayed), 2 = rejected (eager).
    /// Test/diagnostic hook.
    /// </summary>
    internal int CompiledVerdictForTesting(int[] shape) => _inferenceGate.VerdictFor(shape, _layerStructureVersion);

    /// <summary>
    /// Verify-then-trust compiled inference for foundation-scale models, with a
    /// collision-safe value-hash memo. Returns output numerically identical to
    /// <see cref="PredictEager"/> for every input: compiled is adopted per shape
    /// only after it matches eager, and the memo only returns a cached output for a
    /// confirmed-identical input. Invoked by <see cref="Predict"/> when
    /// <see cref="_autoCompiledInferenceEnabled"/> is set.
    /// </summary>
    /// <param name="input">The (already batch-normalized) inference input.</param>
    protected internal Tensor<T> PredictAccelerated(Tensor<T> input)
        => Accelerate(input, () => PredictEager(input));

    /// <summary>
    /// Opt a model's forward into the #1622 verify-then-trust compiled-inference acceleration with a
    /// single wrap. The canonical use is the LAST line of a <see cref="Predict"/> override:
    /// <code>
    /// public override Tensor&lt;T&gt; Predict(Tensor&lt;T&gt; input)
    /// {
    ///     // ... model-specific setup (eval flip, reshape, etc.) ...
    ///     return Accelerate(input, () =&gt; RunMyForward(input));
    /// }
    /// </code>
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the foundation-scale auto-compiled path is NOT engaged (small models, training), this just
    /// invokes <paramref name="eagerForward"/> — zero behavior change and zero overhead, so it is safe
    /// to wrap EVERY model's forward unconditionally. When engaged, it routes through the shared
    /// <see cref="VerifiedInferenceGate{T}"/>: the SAME <paramref name="eagerForward"/> lambda is both
    /// the verify reference AND what the compile host traces, so the gate verifies the model's REAL
    /// forward (whatever custom logic it contains) and adopts the compiled plan only on numerical
    /// parity — otherwise it stays eager for that shape. Output is numerically identical to
    /// <paramref name="eagerForward"/> for every input.
    /// </para>
    /// <para>
    /// Models with a genuinely non-traceable forward (multi-step loops, control flow the GraphMode
    /// tracer can't capture) are handled transparently: the compile attempt fails, the gate records the
    /// shape as eager-only, and every call runs <paramref name="eagerForward"/> — so wrapping them is
    /// harmless, never wrong.
    /// </para>
    /// </remarks>
    /// <param name="input">The input tensor whose shape keys the compile cache.</param>
    /// <param name="eagerForward">The model's eager forward (verify reference + trace source + fallback).</param>
    protected Tensor<T> Accelerate(Tensor<T> input, Func<Tensor<T>> eagerForward)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (eagerForward is null) throw new ArgumentNullException(nameof(eagerForward));

        // Self-latch for foundation-scale eval so a wrapped model engages without depending on the
        // base Predict / SetTrainingMode latch (overriding models may manage mode themselves). Only in
        // inference mode (training mutates weights → plans/memo would be stale) and only when weight-
        // streaming auto-detect already flagged this model as foundation-scale.
        if (!_autoCompiledInferenceEnabled
            && _streamingEngagedByAutoDetect && !s_autoCompileDisabled && !IsTrainingMode)
            _autoCompiledInferenceEnabled = true;

        // Not engaged → behave exactly like the plain eager forward (no gate, no trace, no overhead).
        // (Training safety is already enforced upstream: the self-latch above only engages in eval, and
        // SetTrainingMode(true) clears the flag + the gate — so the flag is never set during training.)
        if (!_autoCompiledInferenceEnabled)
            return eagerForward();

        return _inferenceGate.Run(
            input,
            _layerStructureVersion,
            eager: eagerForward,
            compiled: () => _compileHost.Predict(input, _layerStructureVersion, eagerForward),
            onDecision: (enabled, reason) => AiDotNet.Helpers.InferenceDiagnostics.RecordDecision(
                area: "NeuralNetworkBase", feature: "AutoCompiledInference", enabled: enabled, reason: reason));
    }

    /// <summary>
    /// Releases every compiled inference plan this network holds, returning their
    /// pre-allocated intermediate buffers to the GC. The next compiled call (or
    /// <see cref="CompileForward"/>) re-traces and re-compiles from scratch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A compiled plan pre-allocates one output buffer per step so replay is
    /// allocation-free — for activation-heavy models (conv nets at large batch)
    /// that is tens of MB per plan, held for the network's lifetime. A process
    /// that warms plans across many models/shapes accumulates enough resident
    /// buffer memory to degrade everything else running beside it (GC pressure,
    /// no-GC-region reservation failures — the AIsEval compiled-mode benchmark
    /// measured even non-compiled models slowing ~2x once ~16 plans were warm).
    /// Call this when a model goes cold (request lull, model rotation in a
    /// serving pool, after a batch-size sweep) to bound that residency.
    /// </para>
    /// <para><b>For Beginners:</b> Compiled prediction keeps scratch memory
    /// around so repeat predictions are fast. If you're done predicting with
    /// this model for a while, call this to give that memory back; the next
    /// prediction just rebuilds it automatically.</para>
    /// <para>
    /// Composite models that own compiled state OUTSIDE <see cref="Layers"/>
    /// (child networks, sub-modules with their own compile hosts) should
    /// override <see cref="OnReleaseCompiledPlans"/> to release it — the base
    /// release runs first, then the hook.
    /// </para>
    /// </remarks>
    public void ReleaseCompiledPlans()
    {
        _compileHost.Invalidate();

        // Disposing the plans is not enough: during the trace each layer stashed
        // its inputs/outputs in per-layer state fields (_lastInput / cached
        // activations, kept for a potential Backward), and those references are
        // the PLAN's pre-allocated buffers. As long as the layers hold them, the
        // released plan's tens-of-MB of step buffers stay GC-reachable through
        // the network itself. ResetState() clears that per-layer cache; it is
        // inference-safe (the next Forward repopulates it) and only forfeits a
        // Backward against the now-released forward, which is meaningless anyway.
        foreach (var layer in Layers)
            layer.ResetState();

        OnReleaseCompiledPlans();
    }

    /// <summary>
    /// Extension hook for <see cref="ReleaseCompiledPlans"/>. The base release
    /// only covers this network's own compile host and <see cref="Layers"/>;
    /// composite models that hold compiled state elsewhere (child
    /// <see cref="NeuralNetworkBase{T}"/> instances, wrapped sub-modules)
    /// override this to forward the release — typically by calling each
    /// child's <see cref="ReleaseCompiledPlans"/>. Called AFTER the base
    /// release completes. Default: no-op.
    /// </summary>
    protected virtual void OnReleaseCompiledPlans()
    {
    }

    /// <summary>
    /// Inference forward pass that threads a per-call <see cref="AiDotNet.Inference.InferenceForwardContext"/>
    /// to context-aware layers (paged/cached attention), enabling concurrent multi-sequence decode
    /// over a shared KV cache (#99). Non-context-aware layers run their normal forward.
    /// </summary>
    /// <remarks>
    /// Concurrency contract (mirrors <see cref="Predict"/>): this does NOT toggle training mode, so
    /// callers driving concurrent sequences on one instance must call <c>SetTrainingMode(false)</c>
    /// once (and warm up lazy caches with one forward) before fanning out. KV routing is isolated by
    /// <see cref="AiDotNet.Inference.InferenceForwardContext.SequenceId"/>; the remaining per-forward
    /// scratch fields (<c>_lastInput</c>/<c>_lastOutput</c>) are written but only read by Backward,
    /// which inference never calls, so concurrent forwards produce correct independent outputs.
    /// </remarks>
    internal Tensor<T> PredictWithContext(Tensor<T> input, AiDotNet.Inference.InferenceForwardContext context)
    {
        if (context is null) throw new ArgumentNullException(nameof(context));

        using var _ = new NoGradScope<T>();

        var promoted = NormalizeInputBatchDim(input);
        var current = promoted;
        foreach (var layer in Layers)
        {
            current = layer is AiDotNet.Inference.IContextAwareInferenceLayer<T> contextAware
                ? contextAware.ForwardWithContext(current, context)
                : layer.Forward(current);
        }

        // Squeeze the unit batch dim back off if we promoted (mirrors Predict).
        bool wasPromoted = !ReferenceEquals(promoted, input);
        if (wasPromoted && current.Rank > 1 && current.Shape[0] == 1)
        {
            int[] squeezed = new int[current.Rank - 1];
            for (int i = 0; i < squeezed.Length; i++)
                squeezed[i] = current.Shape[i + 1];
            current = current.Reshape(squeezed);
        }

        return current;
    }

    /// <summary>
    /// Eager forward pass through all layers. Used as fallback when compilation fails.
    /// </summary>
    protected virtual Tensor<T> PredictEager(Tensor<T> input)
    {
        // Hot path: no streaming configured — every layer's weights are
        // already resident in RAM, so the simple foreach-and-forward loop
        // matches the pre-#1222 behavior bit-for-bit. Adding the streaming
        // orchestration overhead unconditionally would penalize every
        // small / mid model that fits in memory without help.
        if (!_weightLifetimeConfigured)
        {
            var c = input;
            foreach (var layer in Layers)
                c = layer.Forward(c);
            return c;
        }

        // Streaming path (#1222 / #184): prefetch ahead by W=2 layers,
        // materialize the active layer's weights for the duration of its
        // forward, then release so the LRU pool can evict if memory is
        // tight. This keeps the working set bounded by ~3 layers' worth
        // of weights regardless of total model size — the difference
        // between OOM-ing on a 562B PaLM-E and running through.
        return PredictEagerStreaming(input);
    }

    /// <summary>
    /// Streaming-aware forward path used when
    /// <see cref="ConfigureWeightLifetime"/> has been called (whether
    /// explicitly or via the auto-detect threshold). Walks layers
    /// sequentially while:
    ///   1. Prefetching weights for the next <c>W = StreamingPrefetchWindow</c>
    ///      layers asynchronously while the current layer's forward runs;
    ///   2. Materializing the current layer's weights inside an
    ///      <c>IDisposable</c> scope so they stay resident during forward;
    ///   3. Releasing the scope after forward so the LRU pool can evict
    ///      the layer's weights when memory pressure demands.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The prefetch window is fixed at W=2 per the locked v1 design — one
    /// layer being computed, two layers' worth of weights paging in
    /// behind it. Larger windows would amortize disk-read latency better
    /// but require correspondingly larger pool capacity to avoid
    /// thrashing.
    /// </para>
    /// <para>
    /// Layers with zero trainable tensors (Activation, Dropout, Reshape,
    /// Add, Concat, …) skip the materialize/release dance — there's no
    /// weight tensor to page. They also DON'T consume prefetch budget:
    /// the prefetch advance walks past weightless layers via
    /// <see cref="FindNextWeightedLayerAfter"/> so a sequence of weightless
    /// ops between two conv blocks doesn't shrink the effective
    /// lookahead.
    /// </para>
    /// </remarks>
    private const int StreamingPrefetchWindow = 2;

    /// <summary>
    /// Returns the index of the <paramref name="stepsAhead"/>th weighted
    /// layer (one with at least one non-empty trainable tensor) at or
    /// after <paramref name="fromIndex"/>. Returns <c>Layers.Count</c>
    /// (i.e. past the end) when fewer than <paramref name="stepsAhead"/>
    /// weighted layers remain. Used by the streaming forward loop's
    /// prefetch advance to avoid spending lookahead budget on weightless
    /// layers (review-comment #1271.rRxc).
    /// </summary>
    private int FindNextWeightedLayerAfter(int fromIndex, int stepsAhead)
    {
        if (stepsAhead <= 0) return fromIndex;
        int weightedFound = 0;
        for (int idx = fromIndex + 1; idx < Layers.Count; idx++)
        {
            if (LayerHasWeights(idx))
            {
                weightedFound++;
                if (weightedFound == stepsAhead) return idx;
            }
        }
        return Layers.Count;
    }

    /// <summary>
    /// True iff the layer at <paramref name="layerIndex"/> exposes at
    /// least one non-empty trainable tensor. Distinguishes weighted
    /// layers (Dense, Convolutional, Embedding, MultiHeadAttention, ...)
    /// from weightless ones (Activation, Dropout, Reshape, Add, Concat).
    /// </summary>
    private bool LayerHasWeights(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count) return false;
        if (Layers[layerIndex] is not LayerBase<T> layer) return false;
        foreach (var tensor in layer.GetTrainableParameters())
        {
            if (tensor is not null && tensor.Length > 0) return true;
        }
        return false;
    }

    private Tensor<T> PredictEagerStreaming(Tensor<T> input)
    {
        // Pre-flight: prefetch the first W *weighted* layers so the
        // first weighted layer's weights are warm by the time we hit
        // Forward. Walk weighted-only so a model that opens with a
        // weightless head (e.g., Reshape → Conv → ...) doesn't burn
        // prefetch budget on the no-op.
        int primeStart = LayerHasWeights(0) ? 0 : FindNextWeightedLayerAfter(-1, 1);
        int primed = 0;
        for (int j = primeStart; j < Layers.Count && primed < StreamingPrefetchWindow; j++)
        {
            if (!LayerHasWeights(j)) continue;
            PrefetchLayerWeights(j);
            primed++;
        }

        var current = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            // Slide the prefetch window forward: by the time we're
            // computing layer i, the next W *weighted* layers ahead of
            // us should be in the pool. Walk forward skipping weightless
            // layers (Dropout, Activation, Reshape, Add, Concat) which
            // have nothing to fetch — those don't consume the prefetch
            // budget. Without this, a model with sparse weighted layers
            // (e.g., a ConvBlock followed by 3 weightless ops, then
            // another ConvBlock) would eat its prefetch lookahead on
            // no-ops, leaving the next real conv unprefetched and
            // forcing a cold disk read on the critical path. Closes
            // review-comment #1271.rRxc.
            int prefetchTarget = FindNextWeightedLayerAfter(i, StreamingPrefetchWindow);
            if (prefetchTarget < Layers.Count)
            {
                PrefetchLayerWeights(prefetchTarget);
            }

            // Materialize this layer's weights for the duration of its
            // forward. The scope's IDisposable release is what allows
            // the LRU pool to evict if memory pressure builds.
            using (BeginLayerMaterializeScope(i))
            {
                current = Layers[i].Forward(current);
            }

            // Post-Forward: lazy layers (Transformer's MultiHeadAttention,
            // lazy ConvolutionalLayer, etc.) materialize their weight
            // tensors on first Forward — before the call those tensors
            // had Length == 0 and were silently skipped by
            // RegisterTrainableTensorsWithWeightRegistry's
            // `tensor.Length == 0` guard, after the call they're real
            // parameter buffers that the streaming pool MUST be tracking
            // or eviction / prefetch would never see them. Re-register
            // this layer's tensors after its forward completes. The
            // per-layer helper skips zero-length tensors (still-lazy
            // weights stay skipped) and is idempotent on already-
            // registered tensors (the registry's RegisterWeight upserts
            // by tensor reference). Closes review-comment #1271.rT-V.
            RegisterLayerTrainableTensorsWithWeightRegistry(i);
        }
        return current;
    }

    /// <summary>
    /// Registers a single layer's trainable tensors with the weight
    /// registry. Per-layer counterpart to
    /// <see cref="RegisterTrainableTensorsWithWeightRegistry"/> — used
    /// during streaming forward to pick up tensors that materialized on
    /// the layer's first Forward call (review-comment #1271.rT-V).
    /// </summary>
    private void RegisterLayerTrainableTensorsWithWeightRegistry(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count) return;
        if (Layers[layerIndex] is not LayerBase<T> layer) return;
        foreach (var tensor in layer.GetTrainableParameters())
        {
            if (tensor is null || tensor.Length == 0) continue;
            // Already-registered tensors (StreamingPoolHandle >= 0)
            // must be skipped: re-entering the streaming branch of
            // RegisterWeight would call SerializeToBytes on a tensor
            // whose storage was already dropped, throwing
            // ArgumentOutOfRangeException from AsSpan(). Idempotency
            // gate.
            if (tensor.StreamingPoolHandle >= 0) continue;
            tensor.Lifetime = _registrationLifetime;
            WeightRegistry.RegisterWeight(tensor);
        }
    }

    private bool _streamingScheduleSet;

    /// <summary>
    /// Feeds the streaming pool this network's per-step weight-access SCHEDULE —
    /// forward layer order then backward (training) — so it pages with Belady-optimal
    /// eviction (evict the weight whose next use is furthest) instead of LRU. For the
    /// cyclic forward→backward pattern LRU is near-pessimal (it evicts the weight
    /// about to be reused); Belady is provably minimal-fault. No-op unless weight
    /// streaming is engaged; runs once after the first forward, when every weight's
    /// StreamingPoolHandle is assigned.
    /// </summary>
    private void RefreshStreamingSchedule()
    {
        if (_streamingScheduleSet) return;
        var pool = WeightRegistry.StreamingPool;
        if (pool is null) return;

        var order = new System.Collections.Generic.List<long>();
        for (int i = 0; i < Layers.Count; i++) CollectStreamingHandles(Layers[i], order);          // forward 0..N
        // Backward reuses the same weights in reverse — include it so Belady knows the
        // late layers are needed again right after the forward turn (training path).
        if (IsTrainingMode)
            for (int i = Layers.Count - 1; i >= 0; i--) CollectStreamingHandles(Layers[i], order);
        if (order.Count == 0) return; // nothing streamed (all weights resident) — keep LRU

        TrySetStreamingAccessSchedule(pool, order.ToArray());
        _streamingScheduleSet = true;
    }

    private static MethodInfo? _streamingPoolSetAccessScheduleMethod;
    private static bool _streamingPoolSetAccessScheduleLookupDone;

    private static void TrySetStreamingAccessSchedule(object pool, long[] order)
    {
        // Optional hook: present in some AiDotNet.Tensors versions, absent in others.
        // Keep the Belady schedule when available without binding this assembly to it.
        if (!_streamingPoolSetAccessScheduleLookupDone)
        {
            _streamingPoolSetAccessScheduleMethod = pool.GetType().GetMethod(
                "SetAccessSchedule",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                binder: null,
                types: new[] { typeof(long[]) },
                modifiers: null);
            _streamingPoolSetAccessScheduleLookupDone = true;
        }

        _streamingPoolSetAccessScheduleMethod?.Invoke(pool, new object[] { order });
    }

    // WeightRegistry.SetStreamingExecutionTraining takes a Nullable<bool>
    // (true=training, false=inference, null=unknown), so the reflected lookup
    // must request typeof(bool?) — a typeof(bool) probe finds no overload and
    // silently returns null, which left the execution-mode hint permanently
    // "unknown" and disabled both bf16-Auto store selection AND the zero-copy
    // mmap inference fast path for every model. Fall back to a name-only lookup
    // so an older package that still declared the bool overload keeps working.
    private static readonly MethodInfo? StreamingExecutionTrainingMethod =
        typeof(WeightRegistry).GetMethod(
            "SetStreamingExecutionTraining",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(bool?) },
            modifiers: null)
        ?? typeof(WeightRegistry).GetMethod(
            "SetStreamingExecutionTraining",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(bool) },
            modifiers: null);

    private static void SetStreamingExecutionTrainingIfSupported(bool isTraining)
    {
        StreamingExecutionTrainingMethod?.Invoke(null, new object?[] { isTraining });
    }

    private static void CollectStreamingHandles(ILayer<T> layer, System.Collections.Generic.List<long> order)
    {
        if (layer is not LayerBase<T> lb) return;
        foreach (var t in lb.GetTrainableParameters())
            if (t is not null && t.StreamingPoolHandle >= 0) order.Add(t.StreamingPoolHandle);
    }

    /// <summary>
    /// Hook into <c>WeightRegistry.PrefetchAsyncMany</c> for the given
    /// layer's trainable tensors. No-op for layers without weights
    /// (Activation, Dropout, …) and for tensors that are already
    /// resident in the pool — the registry's own short-circuit handles
    /// the common case where streaming was enabled but the working set
    /// is small enough to keep everything resident.
    /// </summary>
    /// <remarks>
    /// Issued fire-and-forget — the returned Task isn't awaited because
    /// the next layer's forward begins as soon as this one's weights
    /// are materialized, and prefetching N+1's weights while computing
    /// N is exactly the parallelism we want. If the prefetch hasn't
    /// finished by the time layer N+1 needs its weights,
    /// MaterializeScope will block briefly until it completes —
    /// equivalent to a synchronous read but without serializing every
    /// disk-read with every forward.
    /// </remarks>
    private void PrefetchLayerWeights(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count) return;
        if (Layers[layerIndex] is not LayerBase<T> layer) return;
        var tensors = layer.GetTrainableParameters();
        if (tensors is null) return;
        foreach (var tensor in tensors)
        {
            if (tensor is null || tensor.Length == 0) continue;
            // PrefetchAsync is the per-tensor entry point and returns
            // void (the registry's async-internally fire-and-forget
            // contract — no Task to await from the caller). We loop
            // over tensors rather than calling PrefetchAsyncMany so a
            // future lazy-yield enumerator doesn't materialize the
            // whole list just to prefetch it.
            WeightRegistry.PrefetchAsync(tensor);
        }
    }

    /// <summary>
    /// Returns an <see cref="IDisposable"/> that pins the given layer's
    /// trainable tensors as resident in the streaming pool for the scope's
    /// lifetime. Disposing the scope releases the pin, allowing the LRU
    /// pool to evict if other layers' weights need the slot.
    /// </summary>
    /// <remarks>
    /// For weight-less layers this returns a no-op disposable so the
    /// caller's <c>using</c> block doesn't need to special-case rank-0
    /// trainable parameter sets.
    /// </remarks>
    private IDisposable BeginLayerMaterializeScope(int layerIndex)
    {
        if (Layers[layerIndex] is not LayerBase<T> layer) return NoOpDisposable.Instance;
        var tensors = layer.GetTrainableParameters();
        if (tensors is null || tensors.Count == 0) return NoOpDisposable.Instance;

        // Fast path — common case after first forward: all trainable
        // tensors are non-empty, so pass the layer's own
        // IReadOnlyList<Tensor<T>> directly to MaterializeScope without
        // copying into a fresh List<Tensor<T>>. This is the path a
        // 100-layer model takes on every steady-state forward, and was
        // previously allocating one List<Tensor<T>> per layer per call
        // (review-comment #1271.rRxy). Probe once before deciding to
        // copy — for the typical 2-4 trainable tensors per layer the
        // probe is a few-element loop with branch-prediction-friendly
        // early exit.
        bool needsFilter = false;
        int materializableCount = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            if (t is null || t.Length == 0)
            {
                needsFilter = true;
            }
            else
            {
                materializableCount++;
            }
        }
        if (materializableCount == 0) return NoOpDisposable.Instance;
        if (!needsFilter)
        {
            return new WeightRegistry.MaterializeScope<T>(tensors);
        }

        // Slow path — at least one tensor is empty (lazy placeholder
        // pre-first-forward). Build a filtered list. This branch only
        // triggers during the first-forward materialization phase; once
        // weights are concrete the fast path takes over.
        var live = new List<Tensor<T>>(materializableCount);
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            if (t is null || t.Length == 0) continue;
            live.Add(t);
        }
        return new WeightRegistry.MaterializeScope<T>(live);
    }

    /// <summary>
    /// Returned in place of a real materialize scope when there's nothing
    /// to materialize (weight-less layer, or layer with only empty
    /// placeholder tensors). Lets the caller's <c>using</c> block stay
    /// uniform across both branches.
    /// </summary>
    private sealed class NoOpDisposable : IDisposable
    {
        public static readonly NoOpDisposable Instance = new();
        public void Dispose() { }
    }

    /// <summary>
    /// Updates the network's parameters with new values.
    /// </summary>
    /// <param name="parameters">The new parameter values to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, a neural network's internal values (parameters) get adjusted to improve 
    /// its performance. This method allows you to update all those values at once by providing a complete set 
    /// of new parameters.
    /// </para>
    /// <para>
    /// This is typically used by optimization algorithms that calculate better parameter values based on 
    /// training data.
    /// </para>
    /// </remarks>
    public abstract void UpdateParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the neural network to either training or inference mode.
    /// </summary>
    /// <param name="isTraining">True to enable training mode; false to enable inference mode.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when making predictions.
    /// </para>
    /// <para>
    /// When in training mode (isTraining = true):
    /// - The network keeps track of intermediate calculations needed for learning
    /// - Certain layers like Dropout and BatchNormalization behave differently
    /// - The network uses more memory but can learn from its mistakes
    /// </para>
    /// <para>
    /// When in inference/prediction mode (isTraining = false):
    /// - The network only performs forward calculations
    /// - It uses less memory and runs faster
    /// - It cannot learn or update its parameters
    /// </para>
    /// <para>
    /// Think of it like the difference between taking a practice test (training mode) where you 
    /// can check your answers and learn from mistakes, versus taking the actual exam (inference mode)
    /// where you just give your best answers based on what you've already learned.
    /// </para>
    /// </remarks>
    public virtual void SetTrainingMode(bool isTraining)
    {
        // Auto-detect weight streaming on first model touch. Every Predict
        // override (~933 of them across the codebase) calls SetTrainingMode
        // before iterating layers; the base-class Predict's auto-detect
        // hooks (lines 3080, 3170) never fire because the subclass
        // overrides bypass them. Wiring detection here covers all forward-
        // path callers in one place. Both calls are idempotent (gated by
        // _layerShapesResolved / _streamingAutoDetectFinalized) so the
        // hot path is one branch + early-return after the first call.
        ResolveLazyLayerShapes();
        // Pass the INCOMING mode explicitly: this runs before IsTrainingMode is updated below,
        // so relying on the field would finalize the streaming store dtype from the prior mode.
        TryAutoEnableWeightStreaming(isTrainingOverride: isTraining);

        // Tell the weight-streaming pool which execution mode we're in so its
        // StreamingStoreDtype.Auto policy is safe: store paged-out weights as bf16
        // (2x I/O) during INFERENCE — where a read-only weight is quantized once and
        // never accumulates error — but at full fp32/fp64 precision during TRAINING,
        // so the pool's canonical copy (the master weight) isn't silently truncated
        // to bf16 and small optimizer updates aren't lost. No-op unless streaming is
        // engaged and the dtype is Auto.
        SetStreamingExecutionTrainingIfSupported(isTraining);

        if (isTraining)
        {
            // Entering training invalidates the inference acceleration state: weights
            // will change, so the verify-then-trust verdicts and the value memo (keyed
            // to the current weights) are stale. Turn the auto-compiled path off and
            // clear the gate; a later return to inference re-verifies from scratch.
            _autoCompiledInferenceEnabled = false;
            _inferenceGate.Clear();
        }
        else if (_streamingEngagedByAutoDetect && !s_autoCompileDisabled)
        {
            // L3b (#1622): switching a foundation-scale model into inference latches the
            // auto-compiled eligibility. SetTrainingMode is the universal funnel every
            // Predict override calls, so this marks eligibility even for models that
            // override Predict (the base-Predict routing only reaches non-overriding
            // models; overriding families opt their forward into PredictAccelerated).
            _autoCompiledInferenceEnabled = true;
        }

        if (SupportsTraining)
        {
            IsTrainingMode = isTraining;
            // Propagate to stateful layers (Dropout, BatchNormalization,
            // GaussianNoise, etc.). LayerBase.IsTrainingMode defaults to
            // true at construction, so without this propagation a network
            // in "eval mode" still has Dropout dropping random units and
            // BatchNorm using batch stats — matching PyTorch's
            // model.eval() contract, which walks the module tree.
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].SetTrainingMode(isTraining);
            }
        }

        // If weight streaming is engaged, hand the pool the fwd/bwd access schedule
        // so it pages Belady-optimally. No-op until the first forward has assigned
        // every weight's StreamingPoolHandle (then it sets the schedule once), and a
        // no-op when streaming isn't engaged.
        RefreshStreamingSchedule();
    }

    /// <summary>
    /// Enables mixed-precision training for the neural network.
    /// </summary>
    /// <param name="config">Configuration for mixed-precision training (optional, uses defaults if null).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixed-precision training is a technique that uses a mix of 16-bit (FP16) and
    /// 32-bit (FP32) floating-point numbers to train neural networks faster while maintaining accuracy.
    ///
    /// Benefits:
    /// - **2-3x faster training** on modern GPUs with Tensor Cores (e.g., NVIDIA V100, A100, RTX 3000+)
    /// - **~50% memory reduction** allows training larger models or using bigger batch sizes
    /// - **Maintained accuracy** through careful use of FP32 for critical operations
    ///
    /// How it works:
    /// 1. Forward pass: Computations done in FP16 (faster, less memory)
    /// 2. Loss calculation: Done in FP32 (maintains numerical stability)
    /// 3. Backward pass: Gradients computed in FP16
    /// 4. Loss scaling: Prevents small gradients from becoming zero
    /// 5. Parameter updates: Done in FP32 master weights (maintains precision)
    ///
    /// When to use:
    /// - ✅ Training large models (>100M parameters)
    /// - ✅ Using modern GPUs with Tensor Core support
    /// - ✅ Memory-constrained scenarios
    /// - ❌ CPU-only training (minimal benefit)
    /// - ❌ Very small models (<1M parameters)
    ///
    /// Note: This feature requires that the network's numeric type T is float.
    /// Mixed-precision training with Half or double is not supported.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Enable with default settings (recommended for most cases)
    /// network.EnableMixedPrecision();
    ///
    /// // Enable with custom configuration
    /// var config = new MixedPrecisionConfig
    /// {
    ///     InitialLossScale = 4096.0,
    ///     ScaleGrowthInterval = 2000
    /// };
    /// network.EnableMixedPrecision(config);
    ///
    /// // Enable with conservative settings for sensitive models
    /// network.EnableMixedPrecision(MixedPrecisionConfig.Conservative());
    /// </code>
    /// </example>
    /// <exception cref="NotSupportedException">Thrown when T is not float.</exception>
    /// <exception cref="InvalidOperationException">Thrown when mixed-precision is already enabled.</exception>
    internal virtual void EnableMixedPrecision(LocalMixedPrecisionConfig? config = null)
    {
        // Check that T is float
        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException(
                $"Mixed-precision training is only supported for neural networks with type parameter float. " +
                $"Current type: {typeof(T).Name}. " +
                $"Create your network as NeuralNetwork<float> to use mixed-precision training.");
        }

        if (_mixedPrecisionContext != null)
        {
            throw new InvalidOperationException(
                "Mixed-precision training is already enabled. Call DisableMixedPrecision() first if you want to change the configuration.");
        }

        _mixedPrecisionContext = new MixedPrecisionContext(config);
    }

    /// <summary>
    /// Disables mixed-precision training and releases associated resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This turns off mixed-precision training and returns the network to
    /// standard FP32 training. This is useful for:
    /// - Comparing performance with/without mixed-precision
    /// - Debugging numerical issues
    /// - Switching to FP32 training for the final epochs
    /// </para>
    /// </remarks>
    internal virtual void DisableMixedPrecision()
    {
        if (_mixedPrecisionContext != null)
        {
            _mixedPrecisionContext.Dispose();
            _mixedPrecisionContext = null;
        }
    }

    /// <summary>
    /// Enables memory management with the specified configuration.
    /// </summary>
    /// <param name="config">Memory management configuration. If null, uses default settings.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory management helps train larger models by:
    ///
    /// 1. <b>Gradient Checkpointing</b>: Instead of storing all layer activations (which uses lots of memory),
    ///    only store some "checkpoints". During backpropagation, recompute the missing activations from
    ///    the checkpoints. This trades compute time for memory (typically 40-50% memory savings).
    ///
    /// 2. <b>Activation Pooling</b>: Reuse tensor memory instead of allocating new tensors each time.
    ///    This reduces garbage collection overhead and memory fragmentation.
    ///
    /// Example:
    /// <code>
    /// // Enable memory-efficient training
    /// network.EnableMemoryManagement(TrainingMemoryConfig.MemoryEfficient());
    ///
    /// // Or with custom settings
    /// network.EnableMemoryManagement(new TrainingMemoryConfig
    /// {
    ///     UseGradientCheckpointing = true,
    ///     CheckpointEveryNLayers = 2,
    ///     UseActivationPooling = true
    /// });
    /// </code>
    /// </para>
    /// </remarks>
    public virtual void EnableMemoryManagement(Training.Memory.TrainingMemoryConfig? config = null)
    {
        if (_memoryManager is not null)
        {
            _memoryManager.Dispose();
        }

        _memoryManager = new Training.Memory.TrainingMemoryManager<T>(config, Layers);

        // Precompute which layers should be checkpointed
        var layerTypes = Layers.Select(l => l.GetType().Name).ToList();
        _memoryManager.ComputeCheckpointIndices(Layers.Count, layerTypes);
    }

    /// <summary>
    /// Opts the model into the AiDotNet.Tensors weight-lifetime machinery —
    /// streaming pool, pinned-host, and GPU offload — so models larger than
    /// RAM can run without OOMing. Requires the version of AiDotNet.Tensors
    /// pinned in <c>Directory.Packages.props</c>; the underlying
    /// <see cref="WeightRegistry"/> APIs were introduced in the 0.68 line and
    /// have remained stable since.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Process-global side effect:</b> this method calls
    /// <see cref="WeightRegistry.Configure"/>, which mutates a process-wide
    /// singleton. Calling it on one model instance changes the offload
    /// configuration seen by every other model in the process. The method is
    /// instance-scoped only because it also walks <see cref="Layers"/> and
    /// registers their parameter tensors with the registry.
    /// </para>
    /// <para>
    /// Composite networks that own sub-networks outside <see cref="Layers"/>
    /// (e.g. <c>InfoGAN</c>, <c>CycleGAN</c>, encoder-decoder VLMs) override
    /// this method to also forward the call to each sub-network so their
    /// trainable tensors are registered.
    /// </para>
    /// </remarks>
    internal virtual void ConfigureWeightLifetime(
        GpuOffloadOptions options,
        IGpuOffloadAllocator? allocator = null)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));

        // WeightRegistry.Configure's `offloadAllocator` parameter is itself
        // optional and defaults to null when omitted, so a null forward
        // here is correct (no need for the null-forgiving `!` operator,
        // which the project bans). Branching keeps the signature explicit
        // for callers that did pass an allocator.
        if (allocator is null)
            WeightRegistry.Configure(options);
        else
            WeightRegistry.Configure(options, allocator);
        _weightLifetimeConfigured = true;
        // Pick the per-tensor lifetime to apply: GpuOffload when the user
        // wired in a real GPU offload allocator, Streaming otherwise (the
        // disk-backed pool path PaLM-E and other foundation-scale models
        // need). Without this, RegisterTrainableTensorsWithWeightRegistry
        // would call WeightRegistry.RegisterWeight on every layer's tensors
        // — but those tensors keep their default Lifetime=Default, and
        // RegisterWeight's switch early-returns for Default. Result before
        // this fix: ConfigureWeightLifetime was completely inert for
        // streaming, the pool tracked nothing, and PaLM-E OOMed exactly
        // as if streaming were never configured. Setting the lifetime
        // BEFORE the register loop is what makes the pool actually
        // start tracking weights.
        _registrationLifetime = allocator is null
            ? WeightLifetime.Streaming
            : WeightLifetime.GpuOffload;
        RegisterTrainableTensorsWithWeightRegistry();
    }

    /// <summary>
    /// Per-instance lifetime applied by
    /// <see cref="RegisterTrainableTensorsWithWeightRegistry"/> to every
    /// trainable tensor it walks. Set in <see cref="ConfigureWeightLifetime"/>
    /// based on whether the user wired in a real GPU offload allocator
    /// (→ GpuOffload) or just the disk-backed streaming path (→ Streaming).
    /// Stays <see cref="WeightLifetime.Default"/> until explicitly set, so
    /// pre-streaming-configuration register calls (rare; mostly defensive)
    /// remain inert.
    /// </summary>
    private WeightLifetime _registrationLifetime = WeightLifetime.Default;

    /// <summary>
    /// True once <see cref="ConfigureWeightLifetime"/> has been called on this
    /// network. Used by lazy-aware re-registration paths so newly-allocated
    /// weights from a first forward pass can join the registry retroactively.
    /// </summary>
    private bool _weightLifetimeConfigured;

    /// <summary>
    /// Re-walks <see cref="Layers"/> and registers any trainable tensor whose
    /// length is now positive (i.e., the layer's lazy weights resolved during
    /// a forward pass after <see cref="ConfigureWeightLifetime"/> was called).
    /// Idempotent: <see cref="WeightRegistry.RegisterWeight"/> is safe to invoke
    /// on a tensor that's already registered.
    /// </summary>
    /// <remarks>
    /// Call this after a warm-up forward pass on lazy models so layers like
    /// <c>MultiHeadAttentionLayer</c> (whose Q/K/V/O start as 0×0 placeholders)
    /// get their real weights into the streaming pool. <see cref="ConfigureWeightLifetime"/>
    /// alone runs before any forward and can therefore only see the
    /// already-allocated subset.
    /// </remarks>
    internal void RefreshWeightRegistry()
    {
        if (!_weightLifetimeConfigured) return;
        RegisterTrainableTensorsWithWeightRegistry();
    }

    /// <summary>
    /// Walks <see cref="Layers"/> and registers each layer's trainable tensors
    /// with the process-wide <see cref="WeightRegistry"/>. Used by
    /// <see cref="ConfigureWeightLifetime"/> on this network and recursively
    /// invoked by composite-network overrides on their sub-networks.
    /// </summary>
    /// <remarks>
    /// Tensors with <c>Length == 0</c> are skipped — they are placeholder
    /// allocations from lazy layers (e.g. <c>MultiHeadAttentionLayer</c>'s
    /// 0×0 Q/K/V/O before first forward) and have nothing to offload yet.
    /// Once a real forward pass resolves their shape, callers should invoke
    /// <see cref="RefreshWeightRegistry"/> to pick up the now-allocated tensors.
    /// </remarks>
    protected void RegisterTrainableTensorsWithWeightRegistry()
    {
        // Propagate the streaming-allocator hint to every layer BEFORE
        // we walk their tensors. When a lazy layer's OnFirstForward later
        // calls AllocateLazyWeight, it'll route through
        // WeightRegistry.AllocateStreaming so the pool pre-evicts
        // competing weights to disk before the new GC byte[] lands.
        // Without this propagation, layers default to plain
        // `new Tensor<T>(shape)` and the peak-GC-heap reduction the
        // streaming pool was built for never fires for lazy models like
        // PaLM-E. _registrationLifetime is set to Streaming or GpuOffload;
        // the GpuOffload path doesn't need pool pre-eviction (allocation
        // goes to the pinned-host allocator, not the GC heap), so we
        // only enable the streaming-allocator hint for Streaming.
        bool useStreamingAlloc = _registrationLifetime == WeightLifetime.Streaming;

        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is not LayerBase<T> layer) continue;
            layer.UseStreamingAllocator = useStreamingAlloc;
            foreach (var tensor in layer.GetTrainableParameters())
            {
                if (tensor is null || tensor.Length == 0) continue;
                // Already-registered tensors (StreamingPoolHandle >= 0
                // from a prior RegisterWeight) MUST be skipped on the
                // refresh pass: a re-register would re-enter the
                // streaming branch, hit SerializeToBytes on a tensor
                // whose storage was already dropped by the first
                // register's DropStorageForStreaming, and throw
                // ArgumentOutOfRangeException from AsSpan(). Idempotency
                // gate.
                if (tensor.StreamingPoolHandle >= 0) continue;
                // CRITICAL: set Lifetime BEFORE RegisterWeight. The
                // registry's switch early-returns for Lifetime=Default,
                // so without this assignment every register call is a
                // silent no-op and the streaming pool never tracks the
                // tensor. _registrationLifetime is set in
                // ConfigureWeightLifetime to Streaming (disk-backed) or
                // GpuOffload (pinned-host) based on whether the user
                // wired in a real GPU offload allocator.
                tensor.Lifetime = _registrationLifetime;
                WeightRegistry.RegisterWeight(tensor);
            }
        }

        // Composite models often own trainable layers outside of the
        // <see cref="Layers"/> list — e.g. VLMs that lazy-construct a
        // patch-embedding ConvolutionalLayer in a separate field once an
        // image input arrives. Subclasses override
        // <see cref="GetExtraTrainableLayers"/> to surface those so their
        // weights also land in the offload registry. Default returns an
        // empty enumerable and is a no-op.
        foreach (var extra in GetExtraTrainableLayers())
        {
            if (extra is null) continue;
            extra.UseStreamingAllocator = useStreamingAlloc;
            foreach (var tensor in extra.GetTrainableParameters())
            {
                if (tensor is null || tensor.Length == 0) continue;
                if (tensor.StreamingPoolHandle >= 0) continue;
                tensor.Lifetime = _registrationLifetime;
                WeightRegistry.RegisterWeight(tensor);
            }
        }

        // Some networks own RAW trainable tensors directly on the
        // network (not inside any layer) — e.g. a Vision Transformer's
        // cls_token / positional_embeddings. Surface them through the
        // weight registry too so they're paged in/out by the streaming
        // pool when offload is configured.
        foreach (var tensor in GetExtraTrainableTensors())
        {
            if (tensor is null || tensor.Length == 0) continue;
            // Skip tensors that already have a streaming-pool handle.
            // RefreshWeightRegistry can fire after the model is already
            // streaming (e.g. when Predict re-syncs its extra-tensor
            // surface), and re-registering an already-streamed raw tensor
            // tries to AsSpan() its dropped storage and throws — same
            // failure mode the layer-backed branches already guard against.
            if (tensor.StreamingPoolHandle >= 0) continue;
            tensor.Lifetime = _registrationLifetime;
            WeightRegistry.RegisterWeight(tensor);
        }
    }

    // ---------------------------------------------------------------------
    // Auto-detect default weight streaming (#1222 / #183)
    // ---------------------------------------------------------------------

    /// <summary>
    /// Default parameter-count threshold above which weight streaming is
    /// auto-enabled. 500 million parameters covers the memory-pressure
    /// inflection point across precisions and host environments:
    /// <list type="bullet">
    /// <item>fp64 (default for ModelFamily tests): 4 GB of weights — CI
    /// runners with 7 GB total RAM need this headroom for activations
    /// + optimizer state + runtime, otherwise paper-scale VLMs (EVA-ViT-G
    /// in InstructBLIP/MiniGPT4/BLIP3 ≈ 1 B params, Gemma3 vision ≈ 1.2 B)
    /// OOM at first DenseLayer allocation.</item>
    /// <item>fp32: 2 GB of weights — consumer GPUs (8–24 GB) keep ample
    /// headroom for activations.</item>
    /// <item>fp16: 1 GB of weights — workstation systems unaffected.</item>
    /// </list>
    /// Was 10 B (fp32-only sizing); raised the floor for paper-scale fp64
    /// model tests that OOM'd before the threshold check ever fired.
    /// Models below this train eagerly with no streaming overhead. Override
    /// per-process via the <c>AIDOTNET_STREAMING_THRESHOLD_PARAMS</c>
    /// environment variable, or per-instance via
    /// <see cref="DisableAutoStreaming"/> / the explicit
    /// <see cref="ConfigureWeightLifetime"/> call.
    /// </summary>
    private const long DefaultStreamingThresholdParams = 500_000_000L;

    /// <summary>
    /// Public-readable view of the auto-detect threshold for telemetry
    /// callers (e.g. <c>AiModelBuilder.BuildWeightStreamingReport</c>).
    /// Reflects the effective value (env-var override applied if present
    /// at process start, else the compiled default 10B). Per-instance
    /// overrides via <c>WeightStreamingConfig.ThresholdParameters</c> are
    /// not visible here — callers that need the per-instance value
    /// should consult the config they passed in.
    /// </summary>
    internal static long DefaultStreamingThresholdParamsForReport => s_streamingThresholdParams;

    /// <summary>
    /// Resolved threshold: env-var override if set + parseable, else the
    /// compiled-in default. Read once at first use rather than per-call so
    /// the env var is captured at process start (matching the conventions
    /// of <c>DOTNET_*</c> / <c>ASPNETCORE_*</c> tunables).
    /// </summary>
    private static readonly long s_streamingThresholdParams = ResolveStreamingThreshold();

    private static long ResolveStreamingThreshold()
    {
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_STREAMING_THRESHOLD_PARAMS");
        if (!string.IsNullOrWhiteSpace(raw)
            && long.TryParse(raw.Trim(), System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out long parsed)
            && parsed > 0)
        {
            return parsed;
        }
        return DefaultStreamingThresholdParams;
    }

    /// <summary>
    /// Quant-resident inference store selection (Tier 1 / AiDotNet#1622). Picks the streaming-store
    /// precision for a foundation-scale model in INFERENCE so its weight set stays RESIDENT — no
    /// per-forward paging I/O, which is the dominant cost of the multi-forward Predict the
    /// ModelFamily tests run — at the best accuracy that fits <paramref name="residentBudgetBytes"/>:
    /// <list type="bullet">
    /// <item>bf16 (2 B/param) when it fits — least lossy, and the current Auto-inference behaviour,
    /// so models that already fit bf16-resident are unchanged.</item>
    /// <item>int8-resident (1 B/param, 4x vs fp32, fed by the no-upcast int8 GEMM) when bf16 will
    /// not fit but int8 will — this is the lever that keeps the mid-size OOM models resident.</item>
    /// <item>int4-resident (0.5 B/param, 8x) slots in here once the Tensors package exposes
    /// <c>StreamingStoreDtype.Int4</c>; until then the &gt;int8 case falls through.</item>
    /// <item>If even the tightest available precision will not fit, return bf16 and let the pool
    /// page to stay under budget (the existing streaming fallback).</item>
    /// </list>
    /// Inference-only: training keeps the Auto policy so fp/bf16 masters are preserved.
    /// </summary>
    internal static StreamingStoreDtype ResolveInferenceStoreDtype(long paramCount, long residentBudgetBytes)
    {
        if (paramCount <= 0 || residentBudgetBytes <= 0) return StreamingStoreDtype.Auto;
        long bf16Bytes = paramCount * 2L;
        if (bf16Bytes <= residentBudgetBytes) return StreamingStoreDtype.Bf16; // bf16-resident
        if (paramCount <= residentBudgetBytes) return StreamingStoreDtype.Int8; // int8-resident (1 B/param)
        // int4-resident (paramCount/2 bytes) is the next rung — enabled once the int4-capable
        // Tensors release is consumed; until then, fall back to bf16 + paging.
        return StreamingStoreDtype.Bf16;
    }

    /// <summary>
    /// True once auto-detect has FINALIZED on this instance. Distinct from
    /// "attempted once" because lazy models legitimately need a second
    /// look after the first forward materializes their placeholder weights
    /// — the ctor's pre-forward attempt sees ParameterCount=0 (placeholders)
    /// and would otherwise latch a "below threshold, never retry" state.
    ///
    /// Finalized when:
    ///   * Auto-detect engaged (ConfigureWeightLifetime called → stream
    ///     mode active for the model's lifetime).
    ///   * Explicit ConfigureWeightLifetime by user code outside auto-detect.
    ///   * User opted out via DisableAutoStreaming.
    ///   * Post-first-forward retry ran AND model is still under threshold
    ///     (so the parameter count is now reliable).
    /// </summary>
    private bool _streamingAutoDetectFinalized;

    /// <summary>
    /// Set to true ONLY when the auto-detect path itself called
    /// <see cref="ConfigureWeightLifetime"/> (vs the user calling it
    /// explicitly via <see cref="ConfigureWeightLifetime"/> or
    /// <c>ConfigureWeightStreaming(Enabled: true)</c>). Drives the
    /// <c>WeightStreamingReport.AutoDetected</c> telemetry field —
    /// dashboards distinguishing "framework caught a too-big model" from
    /// "user explicitly opted in" rely on this flag.
    /// </summary>
    private bool _streamingEngagedByAutoDetect;

    /// <summary>
    /// True after this network's first <see cref="Predict"/> or
    /// <see cref="ForwardForTraining"/> call has completed. Used by the
    /// auto-detect retry to know whether ParameterCount is reliable
    /// (placeholders materialize during the first forward, so post-first-
    /// forward the count reflects actual allocated weights).
    /// </summary>
    private bool _firstForwardCompleted;

    /// <summary>
    /// True when the user explicitly opted out of auto-streaming for this
    /// instance via <see cref="DisableAutoStreaming"/> (e.g.
    /// <c>PredictionModelBuilder.ConfigureWeightStreaming(disabled: true)</c>).
    /// Honoured by <see cref="TryAutoEnableWeightStreaming"/>.
    /// </summary>
    private bool _streamingAutoDetectDisabled;

    /// <summary>
    /// Per-instance threshold override applied by
    /// <see cref="ApplyAutoDetectThresholdOverride"/> when the user passes
    /// <c>WeightStreamingConfig.ThresholdParameters</c>. Null falls back
    /// to <see cref="s_streamingThresholdParams"/>.
    /// </summary>
    private long? _streamingThresholdOverride;

    /// <summary>
    /// Opts this model OUT of auto-streaming detection. Useful for tests
    /// that need predictable in-memory behavior, or for callers who know
    /// their working set fits in RAM regardless of total parameter count
    /// (e.g. they only run inference on a fraction of the parameters).
    /// </summary>
    /// <remarks>
    /// <para>
    /// No effect once <see cref="ConfigureWeightLifetime"/> has been called
    /// — that's a deliberate explicit opt-IN that takes precedence over
    /// auto-detect either direction.
    /// </para>
    /// </remarks>
    internal void DisableAutoStreaming() => _streamingAutoDetectDisabled = true;

    /// <summary>
    /// Sets a per-instance threshold for auto-detect. Used by
    /// <c>AiModelBuilder.ConfigureWeightStreaming(config)</c> so a caller's
    /// <c>WeightStreamingConfig.ThresholdParameters</c> actually drives the
    /// per-instance auto-detect comparison.
    /// </summary>
    internal void ApplyAutoDetectThresholdOverride(long thresholdParams)
    {
        if (thresholdParams > 0) _streamingThresholdOverride = thresholdParams;
    }

    /// <summary>
    /// Returns true iff streaming was engaged by auto-detect on THIS
    /// instance (vs. the user explicitly forcing it on via
    /// <see cref="ConfigureWeightLifetime"/> /
    /// <c>ConfigureWeightStreaming(Enabled: true)</c>). Used by
    /// <see cref="WeightStreamingReport.AutoDetected"/> so operator
    /// dashboards can distinguish framework-engaged from user-requested
    /// streaming.
    /// </summary>
    internal bool WeightStreamingAutoDetected => _streamingEngagedByAutoDetect;

    /// <summary>
    /// True when the model is currently in streaming mode for ANY reason
    /// (auto-detect or explicit). Drives the public-facing "is streaming?"
    /// query the report DTO surfaces as <c>StreamingEnabled</c>.
    /// </summary>
    internal bool IsWeightStreamingActive => _weightLifetimeConfigured;

    /// <summary>
    /// Live read of the streaming pool's resident-bytes counter. Used by
    /// tests that need to verify the pool is actually tracking weights
    /// (ResidentBytes > 0 after ConfigureWeightLifetime → register flow
    /// engaged correctly; ResidentBytes == 0 → registration was a no-op,
    /// likely because Lifetime wasn't set to Streaming pre-RegisterWeight).
    /// Returns 0 when streaming isn't configured at all.
    /// </summary>
    internal long WeightStreamingResidentBytes
    {
        get
        {
            // Narrow the catch to the documented failure modes for the
            // streaming-pool report: schema mismatches surface as
            // InvalidOperationException, transient-state inconsistencies as
            // InvalidOperationException, and a deliberately-disposed pool
            // would surface as ObjectDisposedException. Any OTHER exception
            // (NRE, OOM, real bugs) propagates so we don't hide it behind
            // a 0-byte return that looks like "streaming inactive". The
            // caller (a test or telemetry emitter) is best positioned to
            // decide whether to skip / fail.
            try
            {
                return WeightRegistry.GetStreamingReport().ResidentBytes;
            }
            catch (ObjectDisposedException) { return 0; }
            catch (InvalidOperationException) { return 0; }
        }
    }

    /// <summary>
    /// Forces the process-wide WeightRegistry into a clean state — drops
    /// all registered entries, disposes the streaming pool, clears any
    /// offload allocator. ONLY for tests that need to exercise multiple
    /// streaming-configured networks in the same test process: the
    /// registry is a singleton and Configure() throws when there are
    /// live entries from a prior test, so tests must reset between
    /// configurations to avoid leaking state.
    /// </summary>
    /// <remarks>
    /// Production code should never call this — resetting mid-flight
    /// breaks every tensor whose StreamingPoolHandle was assigned from
    /// the disposed pool. Even within tests, only call between
    /// fully-finished test runs (after Predict / Train completes), not
    /// mid-forward.
    /// </remarks>
    internal static void ResetWeightStreamingForTests()
    {
        // Don't swallow — WeightRegistry.Reset() mutates a process-wide
        // singleton, and a silent failure leaves the next test running
        // against contaminated state (live pool entries, leaked handles).
        // The caller (test code) is best positioned to decide whether to
        // skip / fail the test; we surface the original exception with a
        // wrapping note so the failure points at the right place.
        try
        {
            WeightRegistry.Reset();
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "ResetWeightStreamingForTests: WeightRegistry.Reset() failed. " +
                "The next test will run against a contaminated singleton — fix " +
                "the underlying pool error rather than ignoring it.", ex);
        }
    }

    /// <summary>
    /// Auto-enables weight streaming if this model's total parameter count
    /// crosses the threshold AND the user hasn't already opted in or out
    /// explicitly. Idempotent: subsequent calls are no-ops once the flag
    /// is set. Both ctor (eager) and first-forward (lazy) call this so
    /// models with non-lazy layers get streaming early, while models that
    /// only know their parameter count after first forward (lazy
    /// MultiHeadAttention / lazy Conv) catch up at that point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Ctor invocation:</b> ParameterCount may report 0 for fully-lazy
    /// models in the ctor (placeholders haven't materialized). That's not
    /// a missed opportunity — it's the right behavior, because the threshold
    /// is in absolute parameter count and a lazy model legitimately doesn't
    /// know its size yet. The first-forward call rechecks once weights
    /// have allocated.
    /// </para>
    /// <para>
    /// <b>Process-wide side effect:</b> like
    /// <see cref="ConfigureWeightLifetime"/>, this method ultimately calls
    /// <see cref="WeightRegistry.Configure"/>, which mutates a process-wide
    /// singleton. The first network in a process to cross the threshold
    /// installs the offload configuration that every other network will
    /// see. Multi-model processes that need different policies per model
    /// must explicitly call <see cref="ConfigureWeightLifetime"/> with
    /// matched options on each network.
    /// </para>
    /// <para>
    /// Visible to <c>AiDotNet.Tests</c> via <c>InternalsVisibleTo</c> so
    /// regression tests can assert on the auto-detect branch firing.
    /// </para>
    /// </remarks>
    /// <param name="isTrainingOverride">
    /// The execution mode the caller is about to enter, when it is known to differ from the
    /// current <see cref="IsTrainingMode"/> field. <see cref="SetTrainingMode"/> calls this
    /// BEFORE it updates the field, so it must pass the incoming mode here — otherwise the
    /// permanent inference-vs-training store-dtype decision below is made from the stale prior
    /// mode (e.g. finalizing an inference dtype while transitioning INTO training, which would
    /// silently truncate the master weights and lose the Auto training-safety policy).
    /// When null, the current <see cref="IsTrainingMode"/> is used (correct for every call site
    /// that runs after the field already reflects the intended mode).
    /// </param>
    internal void TryAutoEnableWeightStreaming(bool? isTrainingOverride = null)
    {
        if (_streamingAutoDetectFinalized) return;
        if (_weightLifetimeConfigured)
        {
            // User opted in explicitly via ConfigureWeightLifetime. Catch
            // up the registry with any lazy weights that materialized
            // since then — without this, layers whose OnFirstForward
            // allocated via AllocateLazyWeight + AllocateStreaming would
            // hold their reservations forever (RegisterWeight never
            // runs on them, so the bytes stay on the GC heap and the
            // pool's _reservedBytes drifts up). Idempotent: tensors
            // already registered are skipped by the Length==0 / handle>=0
            // gates inside RegisterTrainableTensorsWithWeightRegistry.
            //
            // Only finalize auto-detect AFTER first forward — otherwise
            // the pre-forward call (from EnsureLayersInitialized) would
            // latch the flag before lazy layers materialize, and the
            // post-forward retry hook at line 2747 would early-return
            // on `if (_streamingAutoDetectFinalized) return`, skipping
            // the registry refresh that picks up the just-materialized
            // weights. Tests
            // Streaming_LazyLayer_RoutesAllocationThroughPool_OnFirstForward
            // pin this flow; without the gate, ResidentBytes stays 0
            // for explicitly-configured-streaming + lazy-layer flows.
            if (_firstForwardCompleted)
            {
                RefreshWeightRegistry();
                _streamingAutoDetectFinalized = true;
            }
            return;
        }
        if (_streamingAutoDetectDisabled) { _streamingAutoDetectFinalized = true; return; }

        long paramCount;
        try
        {
            paramCount = ParameterCount;
        }
        catch (Exception ex) when (
            ex is InvalidOperationException ||
            ex is OverflowException ||
            ex is NullReferenceException)
        {
            // ParameterCount can throw on partially-constructed models in
            // these specific ways:
            //   - InvalidOperationException: subclass ctor failing before
            //     InitializeLayers completes (most common).
            //   - OverflowException: int sum wraps mid-property; the
            //     caller can fix it but we shouldn't crash auto-detect.
            //   - NullReferenceException: a sublayer field is still null
            //     during partial construction.
            // Other exceptions (real bugs, GPU faults, etc.) propagate so
            // they aren't hidden behind a silent skip. Don't finalize
            // either: we want to retry once the model is fully built.
            // Surface to System.Diagnostics so telemetry pipelines can see
            // that auto-detect bailed.
            System.Diagnostics.Debug.WriteLine(
                $"[NeuralNetworkBase] WeightStreaming auto-detect deferred — " +
                $"ParameterCount threw {ex.GetType().Name}: {ex.Message}");
            return;
        }

        long threshold = _streamingThresholdOverride ?? s_streamingThresholdParams;

        // Lazy models (Transformer/VLM decoders built from FullyConnected /
        // MultiHeadAttention layers that resolve their input dim on first
        // forward) report ParameterCount == 0 here because no weight tensor
        // has materialized yet. Relying on the post-first-forward retry is
        // too late: that first forward is exactly what allocates the entire
        // (multi-GB) weight set on the GC heap and OOMs on a memory-bounded
        // runner BEFORE the retry can engage streaming. So when the
        // materialized count is below threshold, fall back to the model's
        // structural estimate (computed from its architecture/options, no
        // materialization required) so streaming turns on in the ctor and
        // the lazy weights stream as they materialize. See
        // EstimateStructuralParameterCount.
        long effectiveCount = paramCount;
        if (effectiveCount < threshold)
        {
            long structural = SafeEstimateStructuralParameterCount();
            if (structural > effectiveCount) effectiveCount = structural;
        }

        if (effectiveCount < threshold)
        {
            // Below threshold. For lazy models (Transformer, etc.),
            // ParameterCount may legitimately report 0 here because
            // weights haven't materialized yet. We DON'T finalize on
            // the pre-forward call — the post-first-forward retry will
            // re-check with materialized weights. Once we've seen the
            // first forward, the count is stable and we can safely
            // latch off.
            if (_firstForwardCompleted)
            {
                _streamingAutoDetectFinalized = true;
            }
            return;
        }

        // Above threshold: enable streaming with conservative defaults.
        // The locked design (#1222 weight-streaming v1) calls for LZ4
        // compression on the disk-backing store and a prefetch window of
        // W=2 layers. Both live on GpuOffloadOptions; we use the
        // parameterless ctor so any future field additions inherit the
        // Tensors-side default rather than getting frozen at the value we
        // pinned today.
        var options = new GpuOffloadOptions
        {
            // Auto-detect only fires for foundation-scale models (paramCount or
            // structural estimate over the threshold), and the model-family /
            // inference paths drive a single forward at a time. Transparent
            // auto-eviction lets a layer's first-forward Materialize drop cold
            // weights once the resident budget is crossed, so the full weight
            // set never has to be simultaneously resident — without it, a lazy
            // foundation model materializes every layer on the GC heap during
            // the first forward and OOMs a memory-bounded runner even though
            // streaming was "enabled". Safe here because auto-detect targets
            // single-threaded foundation-scale inference (see
            // GpuOffloadOptions.TransparentAutoEviction remarks); the streaming
            // training path manages its own residency explicitly.
            TransparentAutoEviction = true,
        };
        // Cap the resident budget conservatively. The Tensors default
        // (0.7 × available, ceiling 16 GB) is tuned for "model is a fraction of
        // RAM"; for a foundation model whose weights approach the machine's
        // total RAM (e.g. a ~15 GB float model on a 16 GB runner) a 0.7 budget
        // leaves no room for activations, the rehydrated active weight, GEMM
        // scratch, and the framework, so the forward still OOMs even with
        // streaming. Holding the resident weight set to ~0.4 × available leaves
        // generous headroom for everything else while keeping enough weights
        // hot to avoid thrashing. Only lowers the budget — never raises it
        // above the Tensors default — so large-RAM hosts are unaffected.
        long availableForBudget = ResolveAvailableMemoryForStreaming();
        if (availableForBudget > 0)
        {
            long conservative = Math.Max(1L * 1024 * 1024 * 1024, (long)(availableForBudget * 0.25));
            if (conservative < options.StreamingPoolMaxResidentBytes)
                options.StreamingPoolMaxResidentBytes = conservative;
        }
        // Quant-resident inference (Tier 1, #1622): for a foundation-scale model in inference, keep
        // the weights resident at the loosest precision that fits the (now-capped) budget so
        // multi-forward Predict pays no per-forward paging I/O. Use the entry point's intended mode,
        // not the (possibly stale) field — see the isTrainingOverride remarks. Training keeps the
        // Auto policy (fp/bf16 masters), so it leaves StreamingStoreDtype at its default.
        bool isTrainingForThisAttempt = isTrainingOverride ?? IsTrainingMode;
        if (!isTrainingForThisAttempt)
            options.StreamingStoreDtype = ResolveInferenceStoreDtype(paramCount, options.StreamingPoolMaxResidentBytes);
        // L3b (#1622): a foundation-scale model auto-enables the verify-then-trust
        // compiled forward path (PredictAccelerated) too, composing the compiled plan
        // with the quant-resident streamed weights configured just above (L1 + L3
        // unified). The enable itself is latched in Predict (guaranteed inference mode
        // there, so it is correct regardless of whether this finalize ran from a
        // training- or inference-mode call) keyed on _streamingEngagedByAutoDetect.
        ConfigureWeightLifetime(options);
        // Declare the default (inference) execution mode so the zero-copy mmap
        // fast path is eligible immediately — a freshly-built model is in
        // inference mode until Train() flips it via SetTrainingMode(true), which
        // forwards the training state here and correctly disables zero-copy
        // (training writes weights; the mmap alias is read-only).
        SetStreamingExecutionTrainingIfSupported(IsTrainingMode);
        _streamingEngagedByAutoDetect = true;
        _streamingAutoDetectFinalized = true;
    }

    /// <summary>
    /// Override-point for subclasses that own trainable layers outside
    /// <see cref="Layers"/> (e.g. a lazy patch-embedding conv kept in its
    /// own field) so those layers' tensors also flow through the weight
    /// registry. Default implementation returns an empty enumerable.
    /// </summary>
    protected virtual IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => System.Linq.Enumerable.Empty<LayerBase<T>?>();

    /// <summary>
    /// Structural estimate of this model's trainable parameter count, computed
    /// from its architecture/options WITHOUT materializing any lazy weight
    /// tensors. Returns 0 by default (meaning "no estimate available — rely on
    /// the materialized <see cref="ParameterCount"/>").
    /// </summary>
    /// <remarks>
    /// Foundation-scale models built from lazy layers (large Transformer/VLM
    /// decoders) report <see cref="ParameterCount"/> == 0 until the first
    /// forward materializes their weights. That is too late for the weight-
    /// streaming auto-detector: the first forward is what allocates the full
    /// multi-GB weight set on the GC heap and OOMs a memory-bounded runner.
    /// Overriding this lets <see cref="TryAutoEnableWeightStreaming"/> engage
    /// streaming in the constructor so those weights stream as they
    /// materialize. The estimate only needs to be accurate enough to land on
    /// the correct side of the streaming threshold — a coarse lower bound from
    /// the dominant layers (decoder depth × hidden² , etc.) is sufficient.
    /// </remarks>
    protected virtual long EstimateStructuralParameterCount() => 0L;

    /// <summary>
    /// Best-effort query of the memory ceiling the process can actually use —
    /// the GC heap hard limit when one is configured (containers / constrained
    /// CI runners), else physical RAM. Used to size the weight-streaming
    /// resident budget conservatively. Returns 0 when unavailable so the caller
    /// keeps the streaming pool's own default.
    /// </summary>
    private static long ResolveAvailableMemoryForStreaming()
    {
#if NET5_0_OR_GREATER
        try
        {
            long avail = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            return avail > 0 ? avail : 0L;
        }
        catch
        {
            return 0L;
        }
#else
        // net471: no portable available-memory query — let the streaming pool
        // keep its own default budget.
        return 0L;
#endif
    }

    /// <summary>
    /// Calls <see cref="EstimateStructuralParameterCount"/> defensively: the
    /// override runs during construction (before subclass fields may be fully
    /// set) and must never break auto-detect, so any failure degrades to 0
    /// ("no estimate").
    /// </summary>
    private long SafeEstimateStructuralParameterCount()
    {
        try
        {
            long est = EstimateStructuralParameterCount();
            return est > 0 ? est : 0L;
        }
        catch (Exception ex) when (
            ex is InvalidOperationException ||
            ex is NullReferenceException ||
            ex is OverflowException ||
            ex is ArithmeticException)
        {
            System.Diagnostics.Debug.WriteLine(
                $"[NeuralNetworkBase] EstimateStructuralParameterCount threw " +
                $"{ex.GetType().Name}: {ex.Message}; treating as no estimate.");
            return 0L;
        }
    }

    /// <summary>
    /// Override-point for subclasses that own raw trainable
    /// <see cref="Tensor{T}"/> parameters directly on the network
    /// (NOT inside any layer) — for example a Vision Transformer's
    /// <c>cls_token</c> and <c>positional_embeddings</c>. The tape
    /// training path collects parameters from <see cref="Layers"/>
    /// only, so any raw tensor not surfaced through this hook will
    /// receive gradients via the tape but never be updated by the
    /// optimizer (issue surfaced on AiDotNet#1231 ViT review).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this when the parameter is a raw tensor with no associated
    /// Forward semantics (cls tokens, positional embeddings,
    /// learnable scale factors). For trainable LAYERS outside
    /// <see cref="Layers"/>, prefer
    /// <see cref="GetExtraTrainableLayers"/> instead — that path
    /// also threads the layer through ZeroGrad / sub-layer recursion.
    /// </para>
    /// <para>
    /// Default implementation returns an empty enumerable.
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<Tensor<T>> GetExtraTrainableTensors()
        => System.Linq.Enumerable.Empty<Tensor<T>>();

    /// <summary>
    /// Disables memory management and releases associated resources.
    /// </summary>
    public virtual void DisableMemoryManagement()
    {
        if (_memoryManager is not null)
        {
            _memoryManager.Dispose();
            _memoryManager = null;
        }
    }

    /// <summary>
    /// Gets memory usage statistics if memory management is enabled.
    /// </summary>
    /// <returns>Memory savings estimate, or null if memory management is disabled.</returns>
    public Training.Memory.MemorySavingsEstimate? GetMemoryEstimate(int batchSize = 32, int sequenceLength = 512)
    {
        if (_memoryManager is null)
            return null;

        return _memoryManager.EstimateMemorySavings(ParameterCount, batchSize, sequenceLength);
    }

    /// <summary>
    /// Gets the mixed-precision training context (if enabled).
    /// </summary>
    /// <returns>The mixed-precision context, or null if mixed-precision is disabled.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This provides access to the mixed-precision training internals,
    /// such as the current loss scale and overflow statistics. Useful for monitoring and debugging.
    /// </para>
    /// </remarks>
    internal virtual MixedPrecisionContext? GetMixedPrecisionContext()
    {
        return _mixedPrecisionContext;
    }

    /// <summary>
    /// Gets the loss value from the most recent training iteration.
    /// </summary>
    /// <returns>The loss value from the last training iteration, or zero if no training has occurred.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the error/loss value calculated during the most recent call to the Train method.
    /// It's useful for monitoring the training progress and implementing early stopping.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how well your network is learning.
    /// 
    /// The loss value is a measure of how far off your network's predictions are from the correct answers.
    /// - A high loss means the network is making big mistakes
    /// - A low loss means the network is getting closer to the right answers
    /// 
    /// By tracking this value over time, you can:
    /// - See if your network is improving
    /// - Decide when to stop training (when the loss stops decreasing)
    /// - Compare different network designs to see which learns better
    /// 
    /// Think of it like a score in a game - the lower the score, the better your network is performing.
    /// </para>
    /// </remarks>
    public virtual T GetLastLoss()
    {
        // If we haven't calculated a loss yet, return a default value
        if (LastLoss == null || NumOps.IsNaN(LastLoss))
        {
            return NumOps.Zero;
        }

        return LastLoss;
    }

    /// <summary>
    /// Trains the neural network on a single input-output pair, OR on a batch when the caller
    /// passes a tensor whose leading dimension is the batch axis (shape <c>[B, …]</c>).
    /// </summary>
    /// <param name="input">
    /// The input data. May be either a single sample (e.g. <c>[ctxLen, F]</c> / <c>[C, H, W]</c> /
    /// <c>[F]</c>) or an explicitly batched tensor (e.g. <c>[B, ctxLen, F]</c> / <c>[B, C, H, W]</c>
    /// / <c>[B, F]</c>). Single-sample inputs are auto-promoted to <c>[1, …]</c> by
    /// <see cref="NormalizeBatchDim"/>; batched inputs are passed through unchanged.
    /// </param>
    /// <param name="expectedOutput">
    /// The expected output. Must match the batch arity of <paramref name="input"/>: pass a single
    /// target with a single input, or a batched target with a batched input.
    /// </param>
    /// <remarks>
    /// <para>
    /// This method performs <i>one</i> gradient step on the network. With <c>B = 1</c> (per-sample
    /// training) the gradient is exactly the loss gradient on that one example; with <c>B &gt; 1</c>
    /// the gradient is the per-sample loss summed across the batch (which the optimizer then
    /// averages via its scaled update step).
    /// </para>
    /// <para>
    /// <b>Per-sample vs. batched training — when to use which:</b> per-sample updates
    /// (<c>B = 1</c>) are mathematically slow at output cardinalities <c>V</c> ≥ ~32 because
    /// each step's gradient signal must compete with <c>V − 1</c> negative classes. For tasks
    /// like character / byte language modelling (V = 256), token classification, or any
    /// downstream task with a vocabulary head, prefer batched training: pass a tensor with
    /// <c>B</c> = 16…64. The gradient averaging across a batch reduces noise by a factor of
    /// <c>√B</c> and is what allows the model to escape unigram-prior accuracy in practical
    /// step budgets. See <see cref="TrainBatched"/> for a convenience overload that stacks an
    /// array of single-sample tensors into a batch for you.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how your neural network learns. You provide:
    /// - An input (what the network should process)
    /// - The expected output (what the correct answer should be)
    ///
    /// The network then:
    /// 1. Makes a prediction based on the input
    /// 2. Compares its prediction to the expected output
    /// 3. Calculates how wrong it was (the loss)
    /// 4. Adjusts its internal values to do better next time
    ///
    /// After training, you can get the loss value using the GetLastLoss() method to see how well the network is learning.
    /// For training language models (Transformers especially), pass a BATCH of inputs at once
    /// rather than calling Train one example at a time — see TrainBatched().
    /// </para>
    /// </remarks>
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        try
        {
            TrainCore(input, expectedOutput);
        }
        catch (Exception ex) when (IsGpuTransientFailure(ex))
        {
            // A sticky GPU fault tripped the circuit breaker (engine is now CPU); retry on CPU.
            TrainCore(input, expectedOutput);
        }
    }

    private void TrainCore(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Universal batch-dim auto-promotion. When the caller passes an
        // unbatched single sample (matching the architecture's declared rank
        // exactly), prepend a unit batch dim so downstream Conv/BN/Dense
        // layers see the canonical [B, …] shape. Same logic for the target
        // when its rank also matches an unbatched output. This removes the
        // per-CNN-model EnsureBatchForCnnTraining boilerplate (CNN, VGG,
        // ResNet, MobileNetV2, EfficientNet) and gives all NN models the
        // same input-shape contract: pass either single-sample
        // <c>[C,H,W]</c> / <c>[seq,F]</c> / <c>[F]</c> or batched
        // <c>[B,C,H,W]</c> / <c>[B,seq,F]</c> / <c>[B,F]</c> — both work.
        (input, expectedOutput) = NormalizeBatchDim(input, expectedOutput);

        SetTrainingMode(true);
        try
        {
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
            // Subclasses may own raw trainable tensors outside Layers
            // (e.g. ViT's cls_token / positional_embeddings). Treat
            // their presence as also satisfying the trainable-params
            // check so the tape-training branch still kicks in for
            // networks whose only trainable params live on the network
            // itself rather than in a layer.
            bool hasExtraTensors = false;
            using (var enumerator = GetExtraTrainableTensors().GetEnumerator())
            {
                hasExtraTensors = enumerator.MoveNext();
            }

            if (trainableParams.Count > 0 || hasExtraTensors)
            {
                // Tape-based training: delegates forward/backward/update to TrainWithTape
                // which uses the configured optimizer via Step(TapeStepContext)
                TrainWithTape(input, expectedOutput, optimizer: null);
            }
            else
            {
                // Fallback for networks without ITrainableLayer layers:
                // use the legacy per-layer UpdateParameters path. Step
                // the scheduler at the batch boundary via the shared
                // helper so all training entry points keep the same
                // OnBatchEnd contract. Closes #1270.yYuK.
                var opt = GetOrCreateBaseOptimizer();
                opt.UpdateParameters(Layers);
                StepSchedulerIfSupported(opt);
            }
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Trains the network on a batch of input/target pairs in a single optimizer step.
    /// Stacks the per-sample tensors along a new leading batch dimension and dispatches
    /// to <see cref="Train(Tensor{T}, Tensor{T})"/>, so the gradient is computed against
    /// the SUM of per-sample losses — the optimizer's update is therefore the
    /// average gradient signal across <paramref name="inputs"/>.
    /// </summary>
    /// <param name="inputs">
    /// One or more single-sample input tensors with identical shapes (e.g. each <c>[ctxLen, F]</c>
    /// or each <c>[C, H, W]</c>). The method stacks them into <c>[B, …]</c> where <c>B = inputs.Length</c>.
    /// Must contain at least one element; all elements must share the same shape.
    /// </param>
    /// <param name="targets">
    /// Per-sample target tensors, one per input, with identical shapes. Stacked into <c>[B, …]</c>
    /// the same way as <paramref name="inputs"/>.
    /// </param>
    /// <exception cref="ArgumentNullException">Either array is null.</exception>
    /// <exception cref="ArgumentException">
    /// The arrays have different lengths, are empty, or contain tensors with mismatched shapes.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>When to use this:</b> any classification / sequence task where the per-sample
    /// gradient signal is small relative to the noise floor — most notably language modelling
    /// (V ≥ 32 vocabularies). Per-sample <see cref="Train(Tensor{T}, Tensor{T})"/> calls each
    /// produce a gradient that has to compete against <c>V − 1</c> negative classes; over many
    /// steps these cancel out and the model stalls at the unigram-prior accuracy. Batched
    /// updates with <c>B</c> = 16…64 reduce gradient noise by <c>√B</c> and let the model
    /// actually escape that floor in a practical step budget.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Calling <c>Train(input, target)</c> in a loop one sample at a time
    /// works for tiny tasks but converges very slowly — and on language-model-style tasks
    /// (predicting the next character / token from a 256-class output) it can fail to
    /// converge at all in any reasonable step budget. <c>TrainBatched</c> takes a list of
    /// samples and updates the network from all of them at once, which is dramatically more
    /// stable. The standard practice is batches of 16–64 samples; experiment to find what
    /// fits your task. This is what every modern training script (PyTorch / TensorFlow /
    /// JAX) does under the hood — they batch by default.
    /// </para>
    /// </remarks>
    public virtual void TrainBatched(Tensor<T>[] inputs, Tensor<T>[] targets)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (inputs.Length == 0) throw new ArgumentException("At least one input is required.", nameof(inputs));
        if (inputs.Length != targets.Length)
            throw new ArgumentException($"inputs.Length ({inputs.Length}) must equal targets.Length ({targets.Length}).");

        // No single-sample fast path: a one-element TrainBatched still adds
        // the batch dimension via the same stack-and-dispatch flow as a
        // multi-sample call. Previously the fast path delegated to
        // Train(inputs[0], targets[0]) on the assumption the override
        // would call NormalizeBatchDim itself, but Transformer.Train and
        // similar overrides do not — they pass the (unbatched) sample
        // straight into TrainWithTape, which reaches Layers with a rank
        // one-lower than what a 2+ sample call produces. Reaching layers
        // with a different rank for B=1 vs B≥2 is a class of bug that
        // bites only when callers happen to feed a single-element batch
        // (e.g. last batch of an uneven epoch). Stacking the unit batch
        // here keeps the layer pipeline rank-stable.

        // Validate consistent shapes across the batch — concrete element-shape
        // mismatch is a programming error and the resulting concat would fail
        // anyway, but with a less actionable error message. Surface it here.
        var inputShape = inputs[0]._shape;
        var targetShape = targets[0]._shape;
        for (int i = 1; i < inputs.Length; i++)
        {
            if (!inputs[i]._shape.SequenceEqual(inputShape))
                throw new ArgumentException(
                    $"All inputs must have identical shapes. inputs[0]={string.Join("x",inputShape)} but inputs[{i}]={string.Join("x",inputs[i]._shape)}.",
                    nameof(inputs));
            if (!targets[i]._shape.SequenceEqual(targetShape))
                throw new ArgumentException(
                    $"All targets must have identical shapes. targets[0]={string.Join("x",targetShape)} but targets[{i}]={string.Join("x",targets[i]._shape)}.",
                    nameof(targets));
        }

        // Detect whether per-sample inputs are already in batched form
        // (rank == expectedUnbatchedRank + 1 with a leading dim that the user
        // intends as the batch axis — typically 1 for the per-sample Predict
        // shape, but can be any positive value when the caller is passing
        // already-batched chunks). In that case, CONCATENATE along the
        // existing leading batch dim rather than stacking a new dim:
        //
        //   STACK  : N × [1, ctxLen]   →  [N, 1, ctxLen]   ← double-batch, wrong
        //   CONCAT : N × [1, ctxLen]   →  [N, ctxLen]      ← correct
        //   STACK  : N × [B_i, ctxLen] →  [N, B_i, ctxLen] ← double-batch
        //   CONCAT : N × [B_i, ctxLen] →  [sum(B_i), ctxLen] ← correct
        //
        // For unbatched per-sample inputs (rank == expectedUnbatchedRank):
        //   STACK  : N × [ctxLen]      →  [N, ctxLen]      ← correct
        //
        // Detection is shape-driven rather than architecture-driven because
        // models with custom Train overrides (Transformer) bypass
        // NormalizeBatchDim and treat input shapes flexibly. Architecture-
        // reported expectedUnbatchedRank can also disagree with the runtime
        // shape contract: e.g., a Transformer with InputType.TwoDimensional
        // + inputSize=8 gets InputHeight=1, InputWidth=8 auto-assigned and
        // its GetInputShape reports rank-2, but its Train override accepts
        // [batch, ctxLen] inputs whose effective unbatched form is rank-1.
        //
        // Heuristic: if the per-sample shape has a leading dim that looks
        // like a batch axis (any positive size — typically 1 for the
        // standard Predict shape, but could be a partial-batch chunk),
        // concatenate along that existing axis. Otherwise the per-sample
        // shape is genuinely unbatched and we stack a new dim.
        //
        // Conservative: only fires when the per-sample input has rank > 1.
        // Rank-1 per-sample inputs are unambiguously unbatched (a single
        // feature/token vector); always stack.
        int batchSize = inputs.Length;
        bool concatAlongLeadingDim =
            inputShape.Length > 1
            && inputShape[0] > 0;

        Tensor<T> batchedInput;
        Tensor<T> batchedTarget;

        if (concatAlongLeadingDim)
        {
            // Concat: leading dim sums per-sample batch sizes.
            int totalBatch = batchSize * inputShape[0];
            var batchedInputShape = new int[inputShape.Length];
            batchedInputShape[0] = totalBatch;
            for (int d = 1; d < inputShape.Length; d++) batchedInputShape[d] = inputShape[d];

            // Mirror the same concat for targets when they have the same
            // structure (rank matches expected output rank + 1). Otherwise
            // stack targets along a new leading dim — labels can be flat
            // [vocab] or matrix [B, vocab] independent of input layout.
            int[] batchedTargetShape;
            bool concatTargets =
                targetShape.Length >= 2
                && targetShape[0] == inputShape[0];
            if (concatTargets)
            {
                int totalTargetBatch = batchSize * targetShape[0];
                batchedTargetShape = new int[targetShape.Length];
                batchedTargetShape[0] = totalTargetBatch;
                for (int d = 1; d < targetShape.Length; d++) batchedTargetShape[d] = targetShape[d];
            }
            else
            {
                batchedTargetShape = new int[targetShape.Length + 1];
                batchedTargetShape[0] = batchSize;
                for (int d = 0; d < targetShape.Length; d++) batchedTargetShape[d + 1] = targetShape[d];
            }

            batchedInput = new Tensor<T>(batchedInputShape);
            batchedTarget = new Tensor<T>(batchedTargetShape);

            int inputStride = inputs[0].Length;
            int targetStride = targets[0].Length;
            for (int b = 0; b < batchSize; b++)
            {
                int inputOffset = b * inputStride;
                for (int j = 0; j < inputStride; j++) batchedInput[inputOffset + j] = inputs[b][j];
                int targetOffset = b * targetStride;
                for (int j = 0; j < targetStride; j++) batchedTarget[targetOffset + j] = targets[b][j];
            }
        }
        else
        {
            // Stack along a new leading batch dim. Allocate once for the batch
            // and copy each sample's flat-index range into its slice; this avoids
            // any per-sample tape allocation and keeps the resulting tensor
            // contiguous so downstream Conv / matmul kernels stay on their
            // fast paths. The output shape is [B, *inputShape].
            var batchedInputShape = new int[inputShape.Length + 1];
            batchedInputShape[0] = batchSize;
            for (int d = 0; d < inputShape.Length; d++) batchedInputShape[d + 1] = inputShape[d];
            var batchedTargetShape = new int[targetShape.Length + 1];
            batchedTargetShape[0] = batchSize;
            for (int d = 0; d < targetShape.Length; d++) batchedTargetShape[d + 1] = targetShape[d];

            batchedInput = new Tensor<T>(batchedInputShape);
            batchedTarget = new Tensor<T>(batchedTargetShape);

            int inputStride = inputs[0].Length;
            int targetStride = targets[0].Length;
            for (int b = 0; b < batchSize; b++)
            {
                int inputOffset = b * inputStride;
                for (int j = 0; j < inputStride; j++) batchedInput[inputOffset + j] = inputs[b][j];
                int targetOffset = b * targetStride;
                for (int j = 0; j < targetStride; j++) batchedTarget[targetOffset + j] = targets[b][j];
            }
        }

        Train(batchedInput, batchedTarget);
    }

    /// <summary>
    /// Universal rank-N → rank-(N+1) batch-dim promotion. If the caller
    /// supplied an unbatched single sample (input rank exactly equal to the
    /// architecture's declared input rank), prepends a unit batch dim so
    /// downstream layers see <c>[1, …]</c>. The target is promoted on the
    /// same condition (its rank matches the unbatched output rank). When the
    /// architecture lacks a usable shape, both tensors pass through
    /// unchanged.
    /// </summary>
    /// <remarks>
    /// Subsumes the per-CNN <c>EnsureBatchForCnnTraining</c> helper. CNN
    /// models that already override <c>Train</c> can drop their explicit
    /// promotion call once they delegate to <c>base.Train</c>; until then the
    /// double-promote is suppressed because their override is what's invoked
    /// (this base path runs only for models that don't override <c>Train</c>).
    /// </remarks>
    protected (Tensor<T> Input, Tensor<T> Target) NormalizeBatchDim(Tensor<T> input, Tensor<T> target)
    {
        int expectedUnbatchedRank = GetExpectedUnbatchedInputRank();
        if (expectedUnbatchedRank <= 0) return (input, target);

        // Only promote when input rank matches the unbatched rank exactly —
        // when input rank is one more, the caller already supplied a batched
        // tensor and we must NOT promote.
        bool inputNeedsPromote = input.Rank == expectedUnbatchedRank;
        if (!inputNeedsPromote) return (input, target);

        int origInputRank = input.Rank;
        var processedInput = PromoteToBatchedTensor(input);

        // Promote the target when (and only when) it looks truly
        // unbatched. Two patterns cover every supported architecture
        // family without double-promoting pre-batched targets:
        //
        //   (1) target.Rank == 1
        //       Per-sample label / scalar target. Universal across
        //       classification, regression, multi-class, etc. Examples:
        //         • CNN classifier: input [C,H,W] + target [numClasses]
        //         • MLP regression: input [F] + target [O]
        //         • Sequence-to-vector: input [seq,F] + target [O]
        //
        //   (2) target.Rank == origInputRank
        //       Per-sample target whose dimensionality mirrors the input.
        //       Examples:
        //         • Autoencoder: input [F] + target [F]
        //         • Segmentation: input [C,H,W] + target [C,H,W]
        //         • Sequence-to-sequence: input [seq,F] + target [seq,O]
        //
        // Pre-batched targets fall through unchanged:
        //   • target.Rank == processedInput.Rank (== origInputRank+1) —
        //     same number of axes as the now-batched input ⇒ already
        //     batched, must NOT promote.
        //   • CNN classifier with pre-batched [1, numClasses]: rank 2,
        //     origInputRank=3 ⇒ neither rule fires, passes through.
        //
        // The earlier `target.Rank < processedInput.Rank` rule promoted
        // pre-batched targets too, the `< processedInput.Rank - 2` rule
        // missed every non-CNN case. This unified pattern handles MLP /
        // sequence / CNN / segmentation / autoencoder shapes uniformly.
        Tensor<T> processedTarget = target;
        if (target.Rank == 1 || target.Rank == origInputRank)
        {
            processedTarget = PromoteToBatchedTensor(target);
        }

        return (processedInput, processedTarget);
    }

    /// <summary>
    /// Promotes a rank-3 <c>[C,H,W]</c> tensor to rank-4 <c>[1,C,H,W]</c>. Named
    /// <c>PromoteToBatchedTensor</c> to avoid collision with per-subclass
    /// <c>AddBatchDimension</c> helpers that predate this shared utility.
    /// </summary>
    protected static Tensor<T> PromoteToBatchedTensor(Tensor<T> tensor)
    {
        int[] inputShape = tensor._shape;
        int[] resultShape = new int[inputShape.Length + 1];
        resultShape[0] = 1;
        for (int i = 0; i < inputShape.Length; i++)
        {
            resultShape[i + 1] = inputShape[i];
        }
        return tensor.Reshape(resultShape);
    }

    /// <summary>
    /// Normalizes a (input, target) pair for CNN training loops: when input is
    /// a single rank-3 <c>[C,H,W]</c> sample, adds a batch dim so downstream
    /// layers see <c>[1,C,H,W]</c>. When the caller supplied a rank-1
    /// classification label, promotes it to match the promoted input so
    /// tape-based training sees consistent shapes.
    /// </summary>
    /// <returns>(processedInput, processedTarget) ready for <c>TrainWithTape</c>.</returns>
    protected static (Tensor<T> Input, Tensor<T> Target) EnsureBatchForCnnTraining(
        Tensor<T> input, Tensor<T> target)
    {
        if (input.Rank != 3)
        {
            return (input, target);
        }
        var processedInput = PromoteToBatchedTensor(input);
        var processedTarget = target.Rank < processedInput.Rank - 2
            ? PromoteToBatchedTensor(target)
            : target;
        return (processedInput, processedTarget);
    }

    /// <summary>
    /// Reentrancy sentinel for <see cref="TrainWithTape(Tensor{T}, Tensor{T}, IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}}?)"/>:
    /// 0 = idle, 1 = a training step is currently in flight on this instance.
    /// </summary>
    /// <remarks>
    /// <para>Mutated only via <see cref="Interlocked.CompareExchange(ref int, int, int)"/>
    /// in <see cref="AcquireTrainSentinel"/> and released via
    /// <see cref="Volatile.Write(ref int, int)"/> in
    /// <see cref="TrainSentinel.Dispose"/>. Detects two threads calling
    /// <c>TrainWithTape</c> on the SAME network instance concurrently — that
    /// path interleaves Adam moment updates, GradientTape state, and the
    /// fused-optimizer commit pipeline in ways the per-tensor
    /// <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey, TValue}"/>
    /// storage (swept across all 15 optimizers + the MP FP32 snapshot pool
    /// in this PR) is not sufficient to make race-free at the step level.</para>
    /// <para>The right path to multi-thread training is one network per
    /// worker plus a separate distributed-trainer aggregating gradients
    /// (HOGWILD!-style lock-free, or DDP-style all-reduce). That design is
    /// tracked separately (see the multi-thread training design issue).
    /// Until that lands, concurrent calls on a single instance throw with
    /// a clear redirect.</para>
    /// </remarks>
    private int _trainInFlight;

    /// <summary>
    /// Acquires the per-instance training sentinel. Throws if another
    /// thread is currently inside <see cref="TrainWithTape"/> on this
    /// instance.
    /// </summary>
    private TrainSentinel AcquireTrainSentinel()
    {
        if (Interlocked.CompareExchange(ref _trainInFlight, 1, 0) != 0)
        {
            throw new InvalidOperationException(
                "Concurrent TrainWithTape invocations on the same NeuralNetwork instance " +
                "are not supported. Per-tensor optimizer state (Adam m/v, AdaMax u, " +
                "AdaDelta E[g^2]/E[Δx^2], etc.) is thread-safe via ConcurrentDictionary, " +
                "but the training step interleaves forward/backward/optimizer-step in a " +
                "way that requires step-level mutual exclusion. To train in parallel, " +
                "create one NeuralNetwork instance per worker and aggregate gradients via " +
                "a distributed trainer (HOGWILD!-style lock-free or DDP-style all-reduce). " +
                "See issue ooples/AiDotNet#1369 for the multi-thread training design.");
        }
        return new TrainSentinel(this);
    }

    /// <summary>
    /// IDisposable struct released at the end of <see cref="TrainWithTape"/>'s
    /// scope via <c>using var</c>; clears <see cref="_trainInFlight"/> on every
    /// return path including exceptions. Allocation-free (value type) and
    /// readonly so the compiler doesn't defensively-copy.
    /// </summary>
    private readonly struct TrainSentinel : IDisposable
    {
        private readonly NeuralNetworkBase<T> _owner;
        internal TrainSentinel(NeuralNetworkBase<T> owner) { _owner = owner; }
        public void Dispose() { Volatile.Write(ref _owner._trainInFlight, 0); }
    }

    /// <summary>
    /// Performs tape-based forward/backward pass and delegates the parameter update to the
    /// provided optimizer via <see cref="IGradientBasedOptimizer{T,TInput,TOutput}.Step"/>.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expected">The target tensor.</param>
    /// <param name="optimizer">The optimizer to apply. If null, uses a default Adam optimizer.</param>
    /// <summary>
    /// Controls the memory-bounded streaming training path. Default
    /// <see cref="StreamingTrainingMode.Auto"/> engages streaming only when the
    /// model is too large to train in memory the classic way; small/medium
    /// models are unaffected. See <see cref="StreamingTrainingMode"/>.
    /// </summary>
    public StreamingTrainingMode StreamingTraining { get; set; } = StreamingTrainingMode.Auto;

    /// <summary>Learning rate used by the default streaming 8-bit Adam epilogue. Defaults
    /// to 1e-4 — the conservative rate large-transformer/foundation-model
    /// training uses, which stays stable under 8-bit moment quantization. When
    /// callers configure a gradient optimizer explicitly, the streaming resolver
    /// uses that optimizer's current learning rate instead.</summary>
    public double StreamingTrainingLearningRate { get; set; } = 1e-4;

    /// <summary>Decoupled weight decay used by the default streaming Adam epilogue. 0 = plain Adam.</summary>
    public double StreamingTrainingWeightDecay { get; set; } = 0.0;

    // Per-parameter 8-bit optimizer state for the streaming path; persists
    // across Train calls so the moments accumulate over a multi-step run.
    private Training.IStreamingOptimizer<T>? _streamingOptimizerState;
    private string? _streamingOptimizerKey;

    /// <summary>
    /// Autotuner: decides whether this Train step should take the memory-bounded
    /// streaming path. In <see cref="StreamingTrainingMode.Auto"/> it engages
    /// only when the estimated full-precision training footprint (weights + grad
    /// + Adam m/v ≈ 4× the weights) would not comfortably fit in available
    /// memory, so models that already fit are never penalized.
    /// </summary>
    private bool ShouldUseStreamingTraining()
    {
        switch (StreamingTraining)
        {
            case StreamingTrainingMode.ForceOff:
                return false;
            case StreamingTrainingMode.ForceOn:
                return true;
            default:
                // Auto: engage streaming when the model's estimated full-precision
                // training footprint (weights + grads + Adam first/second moments)
                // would not comfortably fit in available memory. TrainWithTapeStreaming
                // now drives the real topological-min streaming backward
                // (tape.ComputeGradientsStreaming, AiDotNet.Tensors#564 — shipped in
                // 0.92.0+), so the gradient set is never fully resident and registered
                // weights are paged in/out around each backward step. Models that
                // already fit stay on the classic path below (zero overhead).
                long paramCount = ParameterCount;
                if (paramCount <= 0) return false;
                long elemSize = typeof(T) == typeof(float) ? 4L : 8L;
                // weights + gradients + Adam first/second moments at full precision.
                double footprintBytes = (double)paramCount * elemSize * 4.0;
                double available;
#if NET5_0_OR_GREATER
                // GC.GetGCMemoryInfo().TotalAvailableMemoryBytes is .NET 5+. The try/catch
                // guards a runtime throw; the #if guards the COMPILE on net471, where the
                // API doesn't exist at all (a try/catch can't rescue a missing method).
                try
                {
                    available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                }
                catch
                {
                    available = 0;
                }
#else
                // net471: no GC memory-info API — fall through to the conservative default
                // below so the autotuner stays well-behaved on .NET Framework.
                available = 0;
#endif
                if (available <= 0) available = 8L * 1024 * 1024 * 1024; // conservative fallback
                return footprintBytes > 0.5 * available;
        }
    }

    private Training.IStreamingOptimizer<T> GetOrCreateStreamingOptimizer(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> resolvedOptimizer,
        bool useStreamingDefaults)
    {
        string key = Training.StreamingOptimizerResolver<T>.BuildKey(
            resolvedOptimizer,
            useStreamingDefaults,
            StreamingTrainingWeightDecay);

        if (_streamingOptimizerState is null || _streamingOptimizerKey != key)
        {
            _streamingOptimizerState = Training.StreamingOptimizerResolver<T>.Create(
                resolvedOptimizer,
                useStreamingDefaults,
                StreamingTrainingLearningRate,
                StreamingTrainingWeightDecay);
            _streamingOptimizerKey = key;
        }

        Training.StreamingOptimizerResolver<T>.RefreshLearningRate(
            _streamingOptimizerState,
            resolvedOptimizer,
            useStreamingDefaults,
            StreamingTrainingLearningRate);

        return _streamingOptimizerState;
    }

    /// <summary>
    /// Memory-bounded streaming training step: optimizer-in-backward with 8-bit
    /// quantized optimizer state and topological-min gradient release. Each
    /// parameter's gradient is applied (via <see cref="Training.IStreamingOptimizer{T}"/>)
    /// and freed the instant it is computed, so the full gradient set is never
    /// resident. Used automatically by <see cref="TrainWithTape(Tensor{T}, Tensor{T}, IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}}?)"/>
    /// when the autotuner engages it. The streaming resolver mirrors the user's
    /// configured first-order optimizer while replacing its full-precision state
    /// tensors with per-parameter 8-bit block-quantized state.
    /// </summary>
    protected void TrainWithTapeStreaming(
        Tensor<T> input,
        Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> resolvedOptimizer,
        bool useStreamingDefaults)
    {
        var loss = LossFunction as LossFunctions.LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "LossFunction must derive from LossFunctionBase<T> for tape-based training.");

        // Global grad-norm clipping (MaxGradNorm > 0, the default) needs the TOTAL
        // gradient norm before any parameter is updated, which a single
        // optimizer-in-backward pass cannot provide. We therefore run two streaming
        // passes over a PERSISTENT tape in that case (norm pass, then apply pass) —
        // a persistent tape keeps its recorded graph between backward calls. When
        // clipping is off, one streaming pass suffices and the tape is non-persistent.
        double maxGradNorm = MaxGradNormValue;
        bool clip = maxGradNorm > 0.0;

        using var tape = new GradientTape<T>(
            clip ? new GradientTapeOptions { Persistent = true } : null);
        var output = ForwardForTraining(input);

        // Align target rank to the tape-tracked output (reshape the leaf target,
        // never the tape output) — same policy as the eager TrainWithTape path.
        if (output.Rank > expected.Rank && output.Shape[0] == 1 && output.Length == expected.Length)
        {
            expected = Engine.Reshape(expected, output._shape);
        }
        else if (expected.Rank > output.Rank && expected.Shape[0] == 1 && expected.Length == output.Length)
        {
            expected = Engine.Reshape(expected, output._shape);
        }

        var lossTensor = loss.ComputeTapeLoss(output, expected);
        LastLoss = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;

        // Sources = layer-owned trainable params + network-level extras
        // (cls/pos tokens, etc.), exactly the set the eager path optimizes.
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers, _layerStructureVersion);
        var sources = new List<Tensor<T>>(trainableParams.Count + 4);
        sources.AddRange(trainableParams);
        foreach (var t in GetExtraTrainableTensors())
            if (t is not null && t.Length > 0) sources.Add(t);

        var streamingOptimizer = GetOrCreateStreamingOptimizer(
            resolvedOptimizer,
            useStreamingDefaults);
        streamingOptimizer.BeginStep();

        if (!clip)
        {
            // Memory-bounded optimizer-in-backward: tape.ComputeGradientsStreaming
            // (AiDotNet.Tensors#564) hands each parameter's gradient to the callback
            // at its topological last-use and releases it immediately, so the full
            // gradient set is NEVER resident. The same backward walk also rehydrates
            // streamed (paged-out) input weights before each step and lets the pool
            // evict them again afterward — so neither the full gradient set nor the
            // full weight set has to fit in RAM, which is what makes a model whose
            // gradients exceed memory trainable at all. The 8-bit StreamingAdam
            // epilogue runs inside the callback at each gradient's release point.
            tape.ComputeGradientsStreaming(lossTensor, sources,
                (source, grad) =>
                {
                    if (grad is null || grad.Length == 0) return;
                    streamingOptimizer.Apply(source, grad);
                });
        }
        else
        {
            // Clip set = layer-owned trainable params only (NOT the extras), matching
            // the eager ApplyGradientClipping call site exactly so the global norm and
            // the rescale set are identical across paths.
            var clipSet = new HashSet<Tensor<T>>(
                trainableParams, Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

            // Pass 1 — accumulate the global L2 norm over the clip set, streaming so
            // nothing is retained. (Each gradient is released right after we read it.)
            double totalNormSq = 0.0;
            tape.ComputeGradientsStreaming(lossTensor, sources,
                (source, grad) =>
                {
                    if (grad is null || grad.Length == 0 || !clipSet.Contains(source)) return;
                    var span = grad.Data.Span;
                    int len = grad.Length;
                    for (int i = 0; i < len; i++)
                    {
                        double v = NumOps.ToDouble(span[i]);
                        totalNormSq += v * v;
                    }
                });

            // Match ApplyGradientClipping exactly: only scale down when the norm
            // exceeds the cap; the +1e-6 mirrors PyTorch's clip_grad_norm_ denominator.
            double totalNorm = Math.Sqrt(totalNormSq);
            bool scaleDown = totalNormSq > 0.0 && totalNorm > maxGradNorm;
            T scale = scaleDown
                ? NumOps.FromDouble(maxGradNorm / (totalNorm + 1e-6))
                : NumOps.One;

            // Pass 2 — re-run the streaming backward and fold the clip scale into each
            // clip-set gradient before the optimizer epilogue (extras pass through
            // unscaled, exactly as the eager path leaves them). Peak memory stays
            // bounded in this pass too.
            tape.ComputeGradientsStreaming(lossTensor, sources,
                (source, grad) =>
                {
                    if (grad is null || grad.Length == 0) return;
                    if (scaleDown && clipSet.Contains(source))
                    {
                        var span = grad.Data.Span;
                        int len = grad.Length;
                        for (int i = 0; i < len; i++)
                            span[i] = NumOps.Multiply(span[i], scale);
                    }
                    streamingOptimizer.Apply(source, grad);
                });
        }

        // Post-step hook: first-order optimizers already updated each parameter in place during
        // Apply (no-op here); a full-gradient optimizer like streaming L-BFGS buffered the
        // per-parameter gradients above and performs its global two-loop update + parameter
        // write-back now that the whole gradient is available.
        streamingOptimizer.EndStep();

        // GPU weight-cache coherence after the in-place parameter mutation. The
        // 8-bit StreamingAdam state already updated source tensors in place; without
        // invalidation here, the cached derived weights (CPU SIMD-packed copies,
        // GPU-uploaded copies) would still reference the pre-update parameter values
        // and subsequent forwards would silently produce stale predictions. Other
        // update paths (TrainWithTape, batch eager) call this at the same point.
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
        StepSchedulerIfSupported(resolvedOptimizer);
    }

    protected void TrainWithTape(Tensor<T> input, Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        // Reentrancy guard: a single instance is single-thread-per-step until
        // the multi-thread training story (HOGWILD / DDP-shard) ships. The
        // using-declaration generates the try/finally that releases the
        // sentinel on every return path — early returns, exceptions, normal
        // exit. See AcquireTrainSentinel for the contract.
        using var __reentrancyGuard = AcquireTrainSentinel();

        var configuredOptimizer = optimizer ?? _baseTrainOptimizer;
        bool useStreamingDefaults = configuredOptimizer is null;
        var resolvedOptimizer = configuredOptimizer ?? GetOrCreateBaseOptimizer();

        // Memory-bounded streaming training path (optimizer-in-backward). The
        // autotuner engages it only when the model's estimated full-precision
        // training footprint (weights + grads + optimizer moments) would not
        // comfortably fit in available memory — so for the overwhelming
        // majority of models that already fit, this is a no-op and the classic
        // path below runs unchanged (zero overhead, bit-identical results).
        // StreamingTraining = ForceOn/ForceOff overrides the autotuner.
        if (ShouldUseStreamingTraining())
        {
            TrainWithTapeStreaming(input, expected, resolvedOptimizer, useStreamingDefaults);
            return;
        }

        // Reset the pending fused-miss reason so this call's emission
        // window starts clean. The fused-path try sets it via
        // EmitFusedMissAndFallback when it bails out; the post-commit
        // diagnostic block (after opt.Step + extras update succeed)
        // emits the FusedOptimizerPathEvent then. This preserves the
        // "advance only on success" contract — a miss event for a step
        // that never commits would otherwise drift from the loss/grad
        // events in the diagnostics stream.
        _pendingFusedMissReason = null;

        // Fused-compiled fast path: forward + backward + parameter update all
        // in one compiled kernel, SIMD-accelerated, zero materialized gradient
        // tensors between backward and the optimizer step. Engages when the
        // optimizer is a plain Adam/AdamW/SGD with constant hyperparameters
        // AND the numeric type is float (Tensors-side fused optimizer kernels
        // operate on float* directly). Returns false if any condition isn't
        // met or tracing fails; we then fall through to the eager tape path
        // below with behavior unchanged.
        //
        // Mixed-precision (issue #1354): the fused path does NOT materialize
        // intermediate gradient tensors that the loss-scaler needs to unscale
        // + overflow-check, and does NOT consult _mixedPrecisionContext for
        // forward-weight round-tripping. When MP is enabled we MUST take the
        // eager tape path so the MP wiring below (cast weights to low precision,
        // scale loss, unscale gradients, overflow detection) actually runs.
        if (_mixedPrecisionContext is null
            && TryTrainWithFusedOptimizer(input, expected, resolvedOptimizer, useStreamingDefaults))
            return;

        // Loud one-time fallback warning. The fused path above is the fast path;
        // when it doesn't engage we drop to the eager tape every step — a large,
        // previously-SILENT perf cliff. Surface it once per model so "compiled
        // training does nothing" is never invisible again. Skipped when mixed
        // precision is active (that path is an intentional, documented fallback,
        // not a misconfiguration). Suppressible via AIDOTNET_QUIET.
        // Don't warn when compilation is intentionally disabled — that's an explicit
        // opt-out (TensorCodecOptions.EnableCompilation = false), not an unexpected
        // fallback, so the "training is slower" warning would be noisy and misleading.
        if (!_loggedFusedFallback
            && _mixedPrecisionContext is null
            && AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
        {
            _loggedFusedFallback = true;
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_QUIET")))
            {
                var reason = _pendingFusedMissReason ?? "(unspecified gate)";
                System.Diagnostics.Trace.TraceWarning(
                    $"[AiDotNet] Compiled fused-training fast path is OFF for {GetType().Name} — " +
                    $"every Train() step falls back to the eager autograd tape (reason: {reason}). " +
                    "Training is substantially slower than the compiled path. To re-enable it, use a " +
                    "fused-compatible optimizer (plain Adam/AdamW/SGD, no adaptive-rate/AMSGrad, float/double) " +
                    "and keep TensorCodecOptions.EnableCompilation = true. Set TrainingDiagnosticsConfig.Level " +
                    "= PerStep for per-step path detail, or AIDOTNET_QUIET to silence this warning.");
            }
        }

        // The parameter walk here exists only to size the buffer on first Train()
        // call. On every subsequent call the buffer is already the right size, and
        // GetOrCreateParameterBuffer short-circuits to return it. Skipping the
        // pre-swap walk in the steady state saves a full recursive layer traversal
        // per Train() call — non-trivial on DiT-XL with 28 transformer blocks × the
        // sub-layers in each. Only walk when we actually need sizing info.
        //
        // The ParameterBuffer aliases LAYER-OWNED params only — its
        // CreateAllViews/ValidateBufferAlignment contract requires every
        // tensor in the buffer be a view into the underlying storage.
        // Network-level raw tensors (ViT cls/pos, etc.) returned by
        // GetExtraTrainableTensors are standalone tensors, not buffer
        // views, so they CANNOT participate in the buffer-aliased
        // optimizer step. We update them through a separate lightweight
        // gradient-descent path against the same tape gradients below.
        // ParameterBuffer is a contiguous flat copy of all trainable
        // parameters that replaces each layer's tensor with a view into
        // one giant array. It enables fused-optimizer paths in the
        // Tensors-package internals BUT costs an extra ~N×sizeof(T) bytes
        // resident for the lifetime of the network. For foundation
        // models (Sundial-Base ~300M params × 8 B = 2.4 GB) this mirror
        // collides with Adam's m/v state (another 2× weights) on top of
        // the original weights, blowing past CI's ~7 GB ceiling. Skip
        // the buffer when total parameter memory exceeds the threshold
        // so foundation-scale models stay trainable on CPU CI runners.
        // The optimizers we ship (`AdamOptimizer.Step` line 494,
        // AdamW, SGD, etc.) all iterate `context.Parameters` directly
        // and never read `context.ParamBuffer`, so passing null is safe
        // for the model layer's optimizer path.
        // Lazy-init warmup: materialise weight tensors for any layer whose
        // shape isn't resolved yet, BEFORE we collect parameters or set up
        // the ParameterBuffer. Without this, the subsequent
        // `CollectParameters` call captures `Tensor<T>.Empty()` placeholder
        // refs from layers built with single-arg ctors (FeedForwardLayer,
        // DenseLayer, EmbeddingLayer, MultiHeadAttentionLayer, etc. when
        // input dim is deferred to first forward). The tape's backward pass
        // then writes gradients keyed by the post-Forward real-tensor refs;
        // the filter loop that intersects `allGrads` against `trainableParams`
        // finds zero matches; the optimizer step iterates an empty grad dict
        // and the first Train() call becomes a complete no-op. Test scaffolds
        // (LossStrictlyDecreasesOnMemorizationTask,
        // Training_ShouldChangeParameters) then capture the placeholder-
        // forward baseline as "step 1" and report flat-loss / no-param-change
        // failures across every model with at least one lazy layer.
        //
        // Industry-standard PyTorch contract (nn.LazyLinear / nn.LazyConv*):
        // the user runs a dummy forward to materialise lazy params BEFORE
        // constructing the optimizer. AiDotNet attaches the optimizer at
        // model construction time, so the framework does the warmup itself.
        //
        // EVAL mode (not training) so Dropout/BatchNorm don't consume the
        // per-forward RNG counter or advance running statistics — same
        // contract the fused path's pre-init forward already follows at
        // `CompiledTapeTrainingStep.cs` lines 514-534. Caught exceptions are
        // best-effort: a model whose forward fails in eval mode still falls
        // through to the real training step below, which will surface the
        // failure with the proper diagnostic.
        // Gate on actual weight MATERIALIZATION, not just shape resolution:
        // ResolveLazyLayerShapes()/ResolveShapesOnly (driven by a pre-train
        // ParameterCount / GetParameters call) can flip IsShapeResolved to true
        // WITHOUT allocating the weight tensors, leaving length-0 placeholders
        // that CollectParameters would still capture. So also warm up whenever
        // any trainable parameter is still an unallocated placeholder.
        if (AnyLayerNeedsShapeResolution() || AnyLayerHasUnmaterializedParameters())
        {
            bool wasTraining = IsTrainingMode;
            if (wasTraining) SetTrainingMode(false);
            try { ForwardForTraining(input); }
            catch
            {
                // Swallow: warmup is best-effort. If the model genuinely
                // can't forward in eval mode, the real Train forward below
                // will hit the same path under training mode and surface
                // the failure with the actionable stack trace there.
            }
            finally { if (wasTraining) SetTrainingMode(true); }
        }

        ParameterBuffer<T>? paramBuffer;
        // Fast-path: a previous training step on the SAME layer structure
        // already concluded "skip buffer" — re-applying that decision is
        // O(1). InvalidateParameterCountCache resets _skipParameterBufferVersion
        // to -1 (and clears _parameterBuffer) so a layer-structure change
        // re-tests the threshold from scratch.
        if (_skipParameterBuffer && _skipParameterBufferVersion == _layerStructureVersion)
        {
            paramBuffer = null;
        }
        else if (_parameterBuffer is null)
        {
            var initialParams = Training.TapeTrainingStep<T>.CollectParameters(Layers, structureVersion: -1);
            long totalParamCount = 0L;
            for (int i = 0; i < initialParams.Count; i++)
                totalParamCount += (long)initialParams[i].Length;
            // Lazy-init detection: layers like EmbeddingLayer / DenseLayer
            // (constructed without an input size) hold `_weights = new
            // Tensor<T>([0,0])` and DON'T call RegisterTrainableParameter
            // until EnsureWeightsAllocated / EnsureEmbeddingInitialized
            // fires on the first Forward call. Result: initialParams here
            // doesn't include those layers' weights at all, so the buffer
            // is sized for fewer parameters than the post-Forward state.
            // After Forward materializes them the layer's registered
            // parameter list grows beyond the buffer. Two downstream
            // failure modes result:
            //   1. The next CopyFrom call slices past the buffer storage
            //      end, throwing `ArgumentOutOfRangeException` (observed
            //      on MobileNet, EfficientNet, DenseNet121 — all use
            //      lazy DenseLayers).
            //   2. Any later CollectParameters returns tensors that
            //      aren't buffer views, and TapeStepContext.
            //      ValidateBufferAlignment then throws
            //      "Parameter N is not a view into the provided
            //      ParameterBuffer."
            //
            // Detect by walking trainable layers and checking whether
            // any one has no registered parameters yet — that's the lazy
            // signal. Skip the buffer for THIS step only when present;
            // the eager optimizer path iterates context.Parameters
            // directly without buffer aliasing so correctness is preserved.
            // On step 2+ the lazy layers will have materialized their
            // weights and the buffer-aliased fast path engages.
            bool hasLazyParam = false;
            var trainableLayers = Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers, _layerStructureVersion);
            for (int i = 0; i < trainableLayers.Length; i++)
            {
                if (trainableLayers[i].GetTrainableParameters().Count == 0)
                {
                    hasLazyParam = true;
                    break;
                }
            }
            // ~125 M parameter cutoff: small enough that any model
            // passing this threshold is foundation-class (its weight-
            // mirror would consume ~1 GB at double precision and collide
            // with Adam's m/v state on CI hosts); large enough that
            // every standard CV / NLP / time-series model below ~1 B
            // params keeps the buffer + fused-path benefit. Use parameter
            // COUNT rather than byte size so the threshold doesn't shift
            // when T = float vs double.
            const long ParameterBufferSkipThresholdParams = 125_000_000L;
            if (totalParamCount > ParameterBufferSkipThresholdParams)
            {
                paramBuffer = null;
                // Memoize the foundation-scale skip decision so subsequent
                // training steps don't repeat the CollectParameters +
                // sum-Length scan. Sundial-Base (~300 M params) takes
                // ~120ms per scan; caching makes step 2..N effectively free.
                _skipParameterBuffer = true;
                _skipParameterBufferVersion = _layerStructureVersion;
            }
            else if (hasLazyParam)
            {
                // Skip the buffer for THIS step only — do NOT memoize.
                // Forward will materialize the lazy weights, so on the
                // next training step the lazy-detection scan will return
                // false and the buffer-aliased fast path can engage.
                // Permanently skipping would force the eager tape path
                // forever and lose the fused-optimizer speedup on every
                // model with at least one input-size-inferred DenseLayer
                // / EmbeddingLayer (BERT-class architectures).
                paramBuffer = null;
            }
            else
            {
                paramBuffer = GetOrCreateParameterBuffer(initialParams);
            }
        }
        else
        {
            paramBuffer = _parameterBuffer;
        }

        // Mixed-precision restore state — declared OUTSIDE the try so the
        // finally below can call into the helper even when an exception
        // mid-step prevents the happy-path restore from running. Stays
        // function-LOCAL (no instance field) so TrainWithTape is reentrant
        // across nested / concurrent invocations on the same instance
        // (review #1362).
        List<float[]>? mpFp32Snapshots = null;
        List<Tensor<T>>? mpSnapshotTargets = null;
        bool mpFp32Restored = true;
        MixedPrecisionContext? mpCtxForFinally = null;

        try
        {
            // Re-collect after buffer initialization — references are now views
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers, _layerStructureVersion);
            // Snapshot the network-level extras (ViT cls/pos, etc.) once
            // here so we can both (a) include them in tape source
            // collection so gradients are computed for them, and
            // (b) apply a separate gradient-descent update step after
            // the buffer-aliased optimizer.Step has run on layer params.
            var extraTrainableTensors = new List<Tensor<T>>();
            foreach (var t in GetExtraTrainableTensors())
            {
                if (t is null || t.Length == 0) continue;
                extraTrainableTensors.Add(t);
            }

            var loss = LossFunction as LossFunctions.LossFunctionBase<T>
                ?? throw new InvalidOperationException("LossFunction must derive from LossFunctionBase<T> for tape-based training.");

            // ---------------- Mixed-precision (#1354) — pre-forward setup ----------------
            // When _mixedPrecisionContext != null we are running mixed-precision
            // training. The contract is:
            //   1. master weights stay FP32 (in the layer storage)
            //   2. forward + backward use low-precision (FP16 or BF16) working
            //      weights — on CPU we approximate this by round-tripping each
            //      trainable parameter through the target low precision and
            //      writing the rounded value back in-place before the forward
            //      pass. The tensor reference (which the optimizer's
            //      TapeStepContext tracks) is unchanged.
            //   3. loss is scaled up by LossScaler.Scale BEFORE backward so
            //      gradients are S× larger (preventing FP16 underflow on real
            //      hardware backends); we unscale gradients afterward.
            //   4. master weights are restored to FP32 BEFORE the optimizer
            //      step so the optimizer sees the precise FP32 trajectory.
            //   5. if any unscaled gradient is NaN / Inf, LossScaler reduces
            //      the scale and we skip the optimizer step.
            // We snapshot the FP32 master values here so we can restore them
            // before opt.Step, then on next Train() call the round-trip starts
            // from the updated FP32 masters again.
            // Snapshot order MUST stay aligned with the restore loop below:
            //   [0 .. trainableParams.Count)        -> trainableParams
            //   [trainableParams.Count .. end)      -> extraTrainableTensors
            // The restore loop iterates both lists in the same order.
            MixedPrecisionContext? mpCtx = _mixedPrecisionContext;
            if (mpCtx is not null && typeof(T) != typeof(float))
            {
                // EnableMixedPrecision already throws on T != float, so reaching
                // this branch indicates internal inconsistency (e.g. someone
                // bypassed the public surface). Fail loud rather than silently
                // skipping MP — the alternative is "configured but never
                // consumed", the exact footgun this PR eliminates (review #1362).
                throw new InvalidOperationException(
                    $"MixedPrecisionContext is non-null but T = {typeof(T).Name} (only float is supported). " +
                    $"This indicates the context was attached through a non-public path; use " +
                    $"EnableMixedPrecision / AiModelBuilder.ConfigureMixedPrecision which validate T.");
            }
            if (mpCtx is not null && typeof(T) == typeof(float))
            {
                int totalParams = trainableParams.Count + extraTrainableTensors.Count;
                mpFp32Snapshots = new List<float[]>(totalParams);
                mpSnapshotTargets = new List<Tensor<T>>(totalParams);
                mpFp32Restored = false; // we just took snapshots; restore is now pending
                mpCtxForFinally = mpCtx;
                bool useFp16 = mpCtx.Config.PrecisionType == Enums.MixedPrecisionType.FP16;

                void SnapshotAndRoundTrip(Tensor<T> param)
                {
                    int len = param.Length;
                    // Get-or-rent the per-tensor master snapshot buffer from
                    // the MixedPrecisionContext pool — keyed by tensor
                    // reference identity, reused step after step on the
                    // steady state (review #1362). Eliminates the previous
                    // `new float[len]` per parameter per step.
                    // T is float in this branch; cast the tensor to its
                    // float view for the pool lookup.
                    var paramAsFloat = (object)param as AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
                    if (paramAsFloat is null)
                        throw new InvalidOperationException(
                            "MixedPrecision active but parameter tensor is not Tensor<float> — internal invariant violated.");
                    float[] snapshot = mpCtx.GetOrCreateFp32Snapshot(paramAsFloat);
                    // T == float in this branch (asserted via the
                    // paramAsFloat null-check above). Operate directly
                    // on the underlying Span<float> instead of the
                    // generic Span<T> + Convert.ToSingle / (T)(object)
                    // boxing round-trip, which would re-box every
                    // element on the training hot path (review #1362:
                    // measured non-trivial impact on large-model step
                    // wall — boxing dominated when len was in the
                    // millions of params).
                    var fp32Span = paramAsFloat.Data.Span;
                    for (int i = 0; i < len; i++)
                    {
                        float fp32Value = fp32Span[i];
                        snapshot[i] = fp32Value;
                        float rounded = useFp16
                            ? (float)(Half)fp32Value
                            : MixedPrecision.BitConverterHelper.Bf16RoundTrip(fp32Value);
                        fp32Span[i] = rounded;
                    }
                    mpFp32Snapshots!.Add(snapshot);
                    mpSnapshotTargets!.Add(param);
                }

                for (int p = 0; p < trainableParams.Count; p++)
                    SnapshotAndRoundTrip(trainableParams[p]);
                // Also round-trip GetExtraTrainableTensors() (CLS tokens,
                // positional embeddings, etc.) — these participate in forward
                // and get updated in the extras-update path below; without
                // including them here they would train OUTSIDE mixed precision
                // (review #1362).
                for (int e = 0; e < extraTrainableTensors.Count; e++)
                    SnapshotAndRoundTrip(extraTrainableTensors[e]);
            }

            // Local helper used by the normal restore path: writes FP32
            // master values back to the tape-tracked tensor storage so the
            // optimizer step sees the precise FP32 trajectory rather than
            // the FP16/BF16-rounded working weights from the forward pass.
            void RestoreFp32MastersIfNeeded()
            {
                if (mpFp32Restored) return;
                if (mpSnapshotTargets is null || mpFp32Snapshots is null) return;
                var targets = mpSnapshotTargets;
                var snaps = mpFp32Snapshots;
                for (int p = 0; p < targets.Count && p < snaps.Count; p++)
                {
                    var param = targets[p];
                    var snapshot = snaps[p];
                    int len = Math.Min(snapshot.Length, param.Length);
                    // T == float in this branch (the MP path is gated on
                    // typeof(T) == float upstream; mpFp32Snapshots is only
                    // populated when paramAsFloat is non-null). Cast the
                    // tensor to its Tensor<float> view and use Span<float>
                    // CopyTo to avoid the per-element (T)(object) boxing
                    // that the generic Span<T> write would force (review
                    // #1362 C6NB0 — matches the corresponding fix on the
                    // SnapshotAndRoundTrip path).
                    var paramAsFloatLocal = (object)param as AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
                    if (paramAsFloatLocal is not null)
                    {
                        snapshot.AsSpan(0, len).CopyTo(paramAsFloatLocal.Data.Span.Slice(0, len));
                    }
                }
                mpFp32Restored = true;
            }

            // Activate a TensorArena for the forward/backward/update scope.
            // After the first iteration warms the arena, ALL subsequent TensorAllocator.Rent
            // calls reuse pooled arrays — zero GC allocation in the training hot loop.
            // The arena is thread-static and resets on Dispose, so intermediate tensors
            // (conv outputs, attention scores, gradient buffers) are recycled every iteration.
            using var arena = TensorArena.Create();
            // Persistent tape (the parameterless GradientTape<T> ctor picks
            // up GradientTapeOptions.Default which sets Persistent = true):
            // gates the AutoTrainingCompiler fast path in
            // AiDotNet.Tensors.Engines.Compilation.AutoTrainingCompiler.
            // With Persistent = true, after the first training step the
            // compiler records the forward op pattern; on subsequent steps
            // with a matching pattern, ComputeGradients replays a compiled
            // CompiledBackwardGraph instead of walking the tape entry list +
            // dispatching dictionary-keyed gradient lookups per op. Profiling
            // (dotnet-trace + GC.GetTotalAllocatedBytes) showed gradient-tape
            // backward dominating training-step time on paper-scale CNNs
            // (~838 ms / call out of ~1.3 s VGG11 Train, ~73 % of step
            // wall-time; similar fraction for ResNet50). Per-step allocations
            // also drop because the compiled backward keeps tensors in a flat
            // indexed array rather than the dictionary-of-tensor-refs the
            // tape-walk path uses. Pattern mismatch (different shapes /
            // different loss) gracefully falls back to the tape-walk path,
            // so the change is safe across the model zoo.
            using var tape = new GradientTape<T>();
            var output = ForwardForTraining(input);

            // Align output shape to target: when ranks mismatch, ALWAYS reshape the
            // target (a leaf tensor not on the tape) rather than the network output
            // (which IS on the tape). Reshape's tape-backward path does not always
            // propagate gradients through to its source tensor in the current Tensors
            // engine, so reshaping `output` would break the gradient chain between
            // ForwardForTraining's last op and the loss — leaving the optimizer with
            // no grads for the network's trainable params (issue surfaced by ResNet's
            // GradientFlow_ShouldBeNonZeroAndFinite). Reshaping `expected` instead
            // keeps `output` tape-connected end-to-end. Use the internal _shape field
            // (zero-alloc) rather than Shape.ToArray().
            expected = AlignTargetToOutputShape(output, expected);

            var lossTensor = loss.ComputeTapeLoss(output, expected);

            // ---------------- Mixed-precision (#1354) — scale loss before backward ----------------
            // Multiply the tape-tracked loss by LossScaler.Scale so the backward
            // pass produces gradients that are S× their natural magnitude. We
            // unscale them post-backward and run the overflow check. For BF16
            // the default scale is 1.0 (LossScaler is effectively a no-op),
            // so this multiplication is a tape-safe identity.
            //
            // Important: build a SEPARATE scaledLossForBackward tensor instead
            // of mutating lossTensor in place. The unmodified lossTensor is
            // what feeds LastLoss / the TapeStepContext loss value / training
            // diagnostics; rewriting it would surface a scaled loss to every
            // caller observing training progress (see #1362 review).
            var lossForBackward = lossTensor;
            if (mpCtx is not null && typeof(T) == typeof(float))
            {
                double scale = mpCtx.LossScaler.Scale;
                if (scale != 1.0)
                {
                    lossForBackward = Engine.TensorMultiplyScalar(lossTensor, NumOps.FromDouble(scale));
                }
            }

            // Compute all gradients then filter to trainable params.
            // Passing sources directly can miss parameters when the tape backward
            // can't match view tensor references through the GradFn chain.
            var allGrads = tape.ComputeGradients(lossForBackward, sources: null);
            var grads = new Dictionary<Tensor<T>, Tensor<T>>(
                Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var param in trainableParams)
            {
                if (allGrads.TryGetValue(param, out var grad))
                    grads[param] = grad;
            }

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            // Resolve optimizer
            var opt = optimizer ?? GetOrCreateBaseOptimizer();

            // Re-evaluation callback applies the SAME alignment policy the
            // initial forward used: reshape the target (a leaf tensor not on
            // the tape), never the tape-tracked forward output, so the
            // gradient chain between the tape's last op and the loss stays
            // intact across the optimizer's per-step recomputation. The
            // earlier "Engine.Reshape(fwd, tgt._shape)" form snapped the
            // chain because Reshape's tape-backward path doesn't always
            // propagate gradients through to its source — the same problem
            // that the initial-forward fix at line ~3060 was added to avoid,
            // surfaced again here when the optimizer re-evaluated the loss.
            // Re-evaluation alignment: the optimizer hands us (input, target)
            // pairs from its TapeStepContext. The forward path returns the
            // tape-tracked prediction unchanged; alignment between prediction
            // shape and target shape happens in the loss callback where the
            // target is actually consumed. Reshaping `tgt` inside
            // ComputeForward and returning only `fwd` (the previous behaviour)
            // dropped the reshape on the floor — the loss callback received
            // the closure-captured `expected` instead of the locally-aligned
            // `tgt`. Moving alignment into the loss callback is what the
            // reviewer suggested and it also keeps the gradient chain intact
            // (we reshape the leaf target, never the tape-tracked prediction).
            Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> tgt) => ForwardForTraining(inp);

            Tensor<T> ComputeLossAligned(Tensor<T> pred, Tensor<T> tgt)
            {
                if (pred.Rank > tgt.Rank && pred.Shape[0] == 1 && pred.Length == tgt.Length)
                {
                    tgt = Engine.Reshape(tgt, pred._shape);
                }
                else if (tgt.Rank > pred.Rank && tgt.Shape[0] == 1 && tgt.Length == pred.Length)
                {
                    tgt = Engine.Reshape(tgt, pred._shape);
                }
                return loss.ComputeTapeLoss(pred, tgt);
            }

            // ---------------- Mixed-precision (#1354) — unscale gradients + overflow check ----------------
            // (a) Divide every gradient (including those for extras) by
            //     LossScaler.Scale so the optimizer sees the real magnitudes.
            // (b) Run overflow detection: if ANY gradient is NaN/Inf the
            //     LossScaler reduces its scale and we MUST skip the optimizer
            //     step for this iteration (PyTorch GradScaler semantics).
            // (c) Restore FP32 master weights to the layers BEFORE opt.Step
            //     so the optimizer's update is applied to the precise FP32
            //     trajectory (NOT the rounded values that drove the forward
            //     pass). Restoration happens unconditionally — even on
            //     overflow we don't want the FP16-rounded values to leak into
            //     the next iteration's master state.
            bool mpSkipOptimizerStep = false;
            if (mpCtx is not null && typeof(T) == typeof(float) && mpFp32Snapshots is not null)
            {
                double scale = mpCtx.LossScaler.Scale;
                float inverseScaleF = (float)(1.0 / scale);
                bool hasOverflow = false;

                // Unscale every gradient that BELONGS TO A TRAINABLE PARAMETER —
                // both the layer-trainables (in `grads`) and the network-level
                // extras (in `extraTrainableTensors`). The overflow flag is set
                // ONLY from these gradients; allGrads can include tape entries
                // for inputs/intermediates whose NaN/Inf would false-positive
                // the overflow-skip branch even when every trainable gradient
                // is finite (review #1362 follow-up).
                //
                // T == float in this branch (gated above). Cast each gradient
                // tensor to its Tensor<float> view and operate on Span<float>
                // directly to avoid NumOps.Multiply / NumOps.IsNaN / NumOps.IsInfinity
                // virtual dispatch + boxing on every gradient element on every
                // training step (review #1362 perf).
                void UnscaleAndCheck(Tensor<T>? g)
                {
                    if (g is null || g.Length == 0) return;
                    var gAsFloat = (object)g as AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
                    if (gAsFloat is null) return; // defensive — gated by typeof(T) == float
                    var span = gAsFloat.Data.Span;
                    int len = gAsFloat.Length;
                    for (int i = 0; i < len; i++)
                    {
                        float v = span[i] * inverseScaleF;
                        span[i] = v;
                        if (!hasOverflow && (float.IsNaN(v) || float.IsInfinity(v)))
                        {
                            hasOverflow = true;
                        }
                    }
                }

                foreach (var kvp in grads)
                {
                    UnscaleAndCheck(kvp.Value);
                }
                foreach (var extra in extraTrainableTensors)
                {
                    if (allGrads.TryGetValue(extra, out var extraGrad))
                        UnscaleAndCheck(extraGrad);
                }

                // Drive the LossScaler dynamic-scaling automaton DIRECTLY
                // via the dedicated RecordOverflow / RecordSuccess APIs.
                // Previously this path fabricated a 1-element probe vector
                // (NaN to signal overflow, zero otherwise) and fed it to
                // UnscaleGradientsAndCheck just for the side-effect state
                // update — fragile (relied on probe-sized input behavior,
                // discarded the return value, allocated a throwaway vector
                // per step). The explicit APIs make the state transition
                // unambiguous (review #1362).
                if (hasOverflow)
                    mpCtx.LossScaler.RecordOverflow();
                else
                    mpCtx.LossScaler.RecordSuccess();

                // Restore FP32 master weights to the layer storage. The
                // optimizer step below will then update FP32 values, not
                // FP16-rounded ones. The tensor REFERENCE is preserved so
                // the optimizer (which keyed on references in its own state)
                // continues to work. Restores both trainableParams AND
                // extraTrainableTensors via the snapshot target list
                // (review #1362 fix #2).
                RestoreFp32MastersIfNeeded();

                mpSkipOptimizerStep = hasOverflow;
            }

            // Apply global gradient-norm clipping when configured. Mirrors
            // PyTorch's torch.nn.utils.clip_grad_norm_ — total L2 norm across
            // all trainable-parameter gradients; when it exceeds MaxGradNorm,
            // every gradient is scaled by maxNorm/totalNorm. Without this
            // step, deep networks with batch-norm at small batch sizes (e.g.
            // GraFPrint at batch=1) can produce single-step Adam explosions
            // that drive loss up by orders of magnitude on the very first
            // iteration. Skipped when MaxGradNorm <= 0 (the default).
            double maxGradNorm = MaxGradNormValue;
            if (maxGradNorm > 0.0 && grads.Count > 0)
            {
                // Pass trainableParams as the iteration-order key so the
                // total-norm sum is computed in a deterministic order
                // (Dictionary iteration order is bucket-order, which uses
                // the per-process-randomized identity hash for tensor
                // keys → different sum-order per process → different
                // clip scale per process → non-deterministic training).
                ApplyGradientClipping(grads, maxGradNorm, trainableParams);
            }

            var context = new TapeStepContext<T>(
                trainableParams, grads, lossValue,
                input, expected, ComputeForward,
                ComputeLossAligned,
                paramBuffer);

            if (!mpSkipOptimizerStep)
            {
                opt.Step(context);
            }

            // Apply gradient updates to network-level RAW trainable
            // tensors (ViT cls_token / positional_embeddings etc.).
            // These cannot ride the ParameterBuffer-aliased optimizer
            // path — its alignment validator rejects any tensor that
            // isn't a view into the buffer's storage. Instead we use
            // the optimizer's current learning rate (Adam / AdamW /
            // SGD all expose this) to do a vanilla gradient-descent
            // update against the tape's gradients. This loses Adam's
            // per-parameter m/v adaptive state for the extras, but it
            // is correct, deterministic, and crucially KEEPS THEM
            // TRAINED — the previous behaviour left them frozen at
            // their initial random values forever, which was the
            // actual review concern.
            if (extraTrainableTensors.Count > 0 && !mpSkipOptimizerStep)
            {
                T extrasLr = NumOps.FromDouble(GetOptimizerLearningRate(opt));
                foreach (var extra in extraTrainableTensors)
                {
                    if (!allGrads.TryGetValue(extra, out var extraGrad)) continue;
                    if (extraGrad is null || extraGrad.Length != extra.Length) continue;
                    var update = Engine.TensorMultiplyScalar(extraGrad, extrasLr);
                    Engine.TensorSubtractInPlace(extra, update);
                }
            }

            // GPU weight-cache coherence after the opt.Step + extras-update above.
            // See InvalidateWeightCachesAfterSuccessfulWeightUpdate for the full
            // rationale (cache keys by array reference → stale GPU weights → model
            // never learns).
            if (!mpSkipOptimizerStep)
                InvalidateWeightCachesAfterSuccessfulWeightUpdate();

            // Advance the optimizer's learning-rate scheduler at the
            // tape-batch boundary via the shared helper. Without this,
            // any LR scheduler attached to the optimizer (NoamSchedule,
            // LinearWarmupScheduler, CosineAnnealing, etc.) would never
            // tick and the LR would stay pinned at its initial value.
            // Closes #1270.zKjB (single-source-of-truth helper across
            // every training entry point).
            //
            // Skip the scheduler tick when MP overflow forced us to skip
            // the optimizer step. Advancing the LR schedule on a step
            // where no parameters were updated would desynchronize any
            // per-step schedule during overflow/backoff periods (see
            // #1362 review).
            if (!mpSkipOptimizerStep)
            {
                StepSchedulerIfSupported(opt);
            }

            // ---- Training diagnostics emission point (issue #1328 hook) ----
            // Emitted AFTER opt.Step and the extras-update path commit. If
            // either threw, we never reach here and the step counter does
            // NOT advance — diagnostics will not record a step that didn't
            // commit. Mirrors the fused path's "advance only on success"
            // contract (CompiledTapeTrainingStep increments _fusedStepCount
            // only after plan.Step returns).
            if (Configuration.TrainingDiagnosticsConfig.Level
                > Configuration.TrainingDiagnosticLevel.Silent)
            {
                int stepIdx = Configuration.TrainingDiagnosticsConfig.AdvanceStep();

                // PerStep+ : emit deferred fused-miss event (if any) FIRST
                // so consumers see the path-taken event before the loss/
                // gradient events for the same step. The reason was set
                // by EmitFusedMissAndFallback during the failed fused try
                // earlier in this Train call.
                if (_pendingFusedMissReason is not null
                    && Configuration.TrainingDiagnosticsConfig.Level
                        >= Configuration.TrainingDiagnosticLevel.PerStep)
                {
                    Configuration.TrainingDiagnosticsConfig.Emit(
                        new Configuration.FusedOptimizerPathEvent(
                            StepIndex: stepIdx,
                            Hit: false,
                            Reason: _pendingFusedMissReason));
                }
                _pendingFusedMissReason = null;

                // Minimal-level: per-step loss event.
                if (Configuration.TrainingDiagnosticsConfig.Level
                    >= Configuration.TrainingDiagnosticLevel.Minimal)
                {
                    Configuration.TrainingDiagnosticsConfig.Emit(
                        new Configuration.TrainingLossEvent(
                            StepIndex: stepIdx,
                            LossValue: NumOps.ToDouble(lossValue),
                            OutputRank: output.Rank,
                            OutputLength: output.Length));
                }

                // PerStep-level: per-parameter gradient L2 norm.
                if (Configuration.TrainingDiagnosticsConfig.Level
                    >= Configuration.TrainingDiagnosticLevel.PerStep)
                {
                    // Build a parameter-tensor -> (LayerCategory enum, type name) map by
                    // walking the same path CollectParameters uses (ITrainableLayer<T>.
                    // GetTrainableParameters), which guarantees the same order / dedup
                    // behavior as trainableParams. LayerCategory is read from the
                    // [LayerCategory(...)] attribute on the layer class — type-safe
                    // and matches the existing categorization scheme used by
                    // quantizers / pruners / pipeline schedulers.
                    var paramCategory =
                        new System.Collections.Generic.Dictionary<Tensor<T>, (Interfaces.LayerCategory Cat, string Name)>(
                            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                    var seenForDiag =
                        new System.Collections.Generic.HashSet<Tensor<T>>(
                            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
                    foreach (var trainable in Training.TapeTrainingStep<T>.CollectTrainableLayers(
                        Layers, _layerStructureVersion))
                    {
                        if (trainable is null) continue;
                        var lyrType = trainable.GetType();
                        var attr = (Attributes.LayerCategoryAttribute?)Attribute
                            .GetCustomAttribute(lyrType, typeof(Attributes.LayerCategoryAttribute));
                        var cat = attr?.Category ?? Interfaces.LayerCategory.Other;
                        string name = lyrType.Name;
                        foreach (var p in trainable.GetTrainableParameters())
                        {
                            if (p is null || p.Length == 0) continue;
                            if (seenForDiag.Add(p))
                            {
                                paramCategory[p] = (cat, name);
                            }
                        }
                    }

                    // Iterate trainableParams + extraTrainableTensors so the
                    // diagnostics stream covers raw network-level params
                    // (ViT cls_token / positional_embeddings etc.) too — these
                    // are updated by training via the extras path above but
                    // would otherwise never appear in the per-parameter
                    // diagnostics. Lookup uses allGrads which holds both
                    // layer-owned and network-level gradient entries.
                    int totalDiag = trainableParams.Count + extraTrainableTensors.Count;
                    for (int i = 0; i < totalDiag; i++)
                    {
                        bool isExtra = i >= trainableParams.Count;
                        var param = isExtra
                            ? extraTrainableTensors[i - trainableParams.Count]
                            : trainableParams[i];
                        if (param is null) continue;
                        bool hasGrad = allGrads.TryGetValue(param, out var grad);
                        double l2 = 0;
                        if (hasGrad && grad is not null)
                        {
                            double sumSq = 0;
                            long n = grad.Length;
                            var span = grad.Data.Span;
                            for (long k = 0; k < n; k++)
                            {
                                double v = NumOps.ToDouble(span[(int)k]);
                                sumSq += v * v;
                            }
                            l2 = System.Math.Sqrt(sumSq);
                        }
                        var (cat, name) = paramCategory.TryGetValue(param, out var info)
                            ? info
                            : (Interfaces.LayerCategory.Other, isExtra ? "(network)" : "(unknown)");
                        int[] shape = new int[param._shape.Length];
                        for (int s = 0; s < shape.Length; s++) shape[s] = param._shape[s];
                        Configuration.TrainingDiagnosticsConfig.Emit(
                            new Configuration.GradientNormEvent(
                                StepIndex: stepIdx,
                                ParamIndex: i,
                                ParamShape: shape,
                                ParamLength: param.Length,
                                HasGradient: hasGrad,
                                GradientL2Norm: l2,
                                LayerCategory: cat,
                                LayerTypeName: name));
                    }
                }
            }
        }
        finally
        {
            // Emergency FP32 master restore: if the happy-path unscale-and-
            // restore block didn't run (an exception inside the try block —
            // ForwardForTraining / ComputeTapeLoss / ComputeGradients —
            // skipped past it), restore the FP32 master values now so the
            // layer storage doesn't keep the FP16/BF16-rounded working
            // weights from this aborted step (review #1362 fix #1).
            // The locals (mpFp32Snapshots, mpSnapshotTargets, mpFp32Restored)
            // were declared BEFORE the try block so they remain in scope
            // here. Idempotent — when the happy-path block already ran the
            // flag is true and this is a no-op.
            if (!mpFp32Restored
                && mpFp32Snapshots is not null
                && mpSnapshotTargets is not null)
            {
                for (int p = 0; p < mpSnapshotTargets.Count && p < mpFp32Snapshots.Count; p++)
                {
                    var param = mpSnapshotTargets[p];
                    var snapshot = mpFp32Snapshots[p];
                    int len = Math.Min(snapshot.Length, param.Length);
                    // T == float (gated upstream — snapshots are only
                    // populated for float tensors). Use Span<float> CopyTo
                    // instead of the per-element (T)(object) boxing fallback
                    // (review #1362 C6NB0 — finally-block emergency restore
                    // must match the happy-path's float-direct semantics).
                    var paramAsFloatLocal = (object)param as AiDotNet.Tensors.LinearAlgebra.Tensor<float>;
                    if (paramAsFloatLocal is not null)
                    {
                        snapshot.AsSpan(0, len).CopyTo(paramAsFloatLocal.Data.Span.Slice(0, len));
                    }
                }
                mpFp32Restored = true;
            }
            // The mpCtxForFinally / pool entries are deliberately retained
            // across calls — they ARE the per-tensor master pool, reused on
            // the next TrainWithTape invocation. The mpCtxForFinally local
            // was used by the snapshot-pool lookup in the try-body; it
            // remains in scope for clarity but the pool reuse semantics
            // are managed by MixedPrecisionContext itself (review #1362
            // C6NCl — removed the no-op discard).

            // Restore original tensor references so Clone/serialization see real tensors.
            // Copies updated weights from buffer views back to originals before restoring.
            RestoreOriginalParameters();
        }
    }

    /// <summary>
    /// Best-effort read of the supplied optimizer's current learning
    /// rate, used by the network-level extras update path. Returns the
    /// optimizer-typed value when the optimizer is a recognised
    /// Applies global gradient L2-norm clipping in-place across all gradient
    /// tensors in the supplied dictionary. Computes
    /// <c>totalNorm = sqrt(sum_i sum(grad_i²))</c>; when
    /// <c>totalNorm &gt; maxNorm</c>, scales every gradient by
    /// <c>maxNorm / (totalNorm + 1e-6)</c>. Mirrors PyTorch's
    /// <c>clip_grad_norm_</c> semantics — collective scaling, not per-tensor
    /// — so the relative direction of the gradient field is preserved while
    /// its magnitude is bounded.
    /// </summary>
    private void ApplyGradientClipping(
        Dictionary<Tensor<T>, Tensor<T>> grads, double maxNorm,
        IReadOnlyList<Tensor<T>> iterationOrder)
    {
        // Step 1: total L2 norm across all gradient tensors, iterating in
        // the caller-supplied deterministic order (NOT dict bucket order —
        // that's process-randomized for reference-keyed dicts).
        double totalNormSq = 0.0;
        for (int p = 0; p < iterationOrder.Count; p++)
        {
            if (!grads.TryGetValue(iterationOrder[p], out var g)) continue;
            if (g is null || g.Length == 0) continue;
            var span = g.Data.Span;
            int len = g.Length;
            for (int i = 0; i < len; i++)
            {
                double v = NumOps.ToDouble(span[i]);
                totalNormSq += v * v;
            }
        }
        if (totalNormSq == 0.0) return;
        double totalNorm = Math.Sqrt(totalNormSq);
        if (totalNorm <= maxNorm) return;

        // Step 2: scale every gradient down so the new global norm == maxNorm.
        // The +1e-6 in the denominator matches PyTorch's clip_grad_norm_ to
        // avoid div-by-zero when the gradient magnitudes are vanishingly small.
        T scale = NumOps.FromDouble(maxNorm / (totalNorm + 1e-6));
        for (int p = 0; p < iterationOrder.Count; p++)
        {
            if (!grads.TryGetValue(iterationOrder[p], out var g)) continue;
            if (g is null || g.Length == 0) continue;
            var span = g.Data.Span;
            int len = g.Length;
            for (int i = 0; i < len; i++)
                span[i] = NumOps.Multiply(span[i], scale);
        }
    }

    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}"/>; falls
    /// back to a conservative default for optimizers that don't expose
    /// the rate.
    /// </summary>
    private static double GetOptimizerLearningRate(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> opt)
    {
        if (opt is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> typed)
        {
            return typed.GetCurrentLearningRate();
        }
        // Conservative SGD-default; only hit when the optimizer hides
        // its LR. Logged via the trace path so users see the fallback.
        return 0.001;
    }

    // Removed: MixedPrecisionBf16RoundTrip — deduplicated to
    // MixedPrecision.BitConverterHelper.Bf16RoundTrip (single source of
    // truth, review #1362). Call sites in TrainWithTape's MP path use the
    // helper directly via the using-static import. (Doc-comment kept as
    // a plain comment so it doesn't attach to the next method via the
    // XML doc-cascade — review #1362 C6NCS.)

    /// <summary>
    /// Overload for backward compatibility — accepts a learning rate instead of an optimizer.
    /// Creates a temporary GradientDescent optimizer with the specified rate.
    /// </summary>
    protected void TrainWithTape(Tensor<T> input, Tensor<T> expected, double learningRate)
    {
        // Use the default optimizer (which respects configured LR) rather than creating a throwaway one
        TrainWithTape(input, expected, optimizer: null);
    }

    /// <summary>
    /// Sticky disable for the fused training path on this model instance. Set
    /// when <see cref="TryTrainWithFusedOptimizer"/> falls back BEFORE the
    /// fused path has ever successfully run. Subsequent steps stay on the
    /// eager path. Mixing fused (plan-embedded m/v) and eager (optimizer-
    /// instance state) updates within one training run would reset Adam
    /// moments at the next successful fused step. Stickiness keeps optimizer
    /// state consistent for the rest of the run. Cleared in
    /// <see cref="ResetState"/> and <see cref="InvalidateParameterCountCache"/>.
    /// </summary>
    /// <remarks>
    /// Promoted from <c>private</c> to <c>protected</c> so models with
    /// architecture-specific fused-Adam divergence (e.g. GraFPrint's
    /// 53-layer BatchNorm pyramid hits a CompiledTrainingPlan
    /// per-parameter gradient propagation residual that v0.80.1's #351
    /// two-bug repair didn't fully close) can opt OUT of fused-Adam
    /// while keeping every other compile-mode optimization (ConvBnFusion,
    /// dataflow fusion, algebraic backward, forward CSE, etc.) engaged.
    /// Setting this in the model constructor avoids the much heavier
    /// hammer of globally toggling <c>TensorCodecOptions.EnableCompilation</c>,
    /// which would also disable those orthogonal compile-mode
    /// optimizations and tank training-step performance.
    /// </remarks>
    protected bool _fusedTrainingDisabled;

    /// <summary>
    /// Tracks whether the fused compiled training path has EVER successfully
    /// run on this model. Once true, Adam/AdamW/SGD moment buffers live
    /// exclusively inside the compiled plan — falling back to eager would
    /// silently lose that state (<c>resolvedOptimizer</c> has empty m/v).
    /// So once committed, any condition that would force a fallback
    /// (optimizer-config drift, input-shape change producing a new plan)
    /// throws an explicit exception instead of silently diverging from the
    /// reference Adam trajectory. Cleared in <see cref="ResetState"/> and
    /// <see cref="InvalidateParameterCountCache"/> — both are explicit
    /// reset points where the caller has acknowledged state changes and
    /// both fused and eager optimizer states start fresh.
    /// </summary>
    private bool _fusedTrainingCommitted;

    /// <summary>
    /// Reason string set by <see cref="EmitFusedMissAndFallback"/> when
    /// <see cref="TryTrainWithFusedOptimizer"/> bails out early, consumed
    /// (and cleared) by the post-success diagnostic block in
    /// <see cref="TrainWithTape"/>. Defers emission of the
    /// <see cref="Configuration.FusedOptimizerPathEvent"/> miss event
    /// until the eager fallback has actually committed — preserving the
    /// "advance only on success" contract that the fused path also
    /// follows. Reset at the top of every <see cref="TrainWithTape"/>
    /// call so a prior step's bail-out can't leak into this one.
    /// </summary>
    private string? _pendingFusedMissReason;

    /// <summary>
    /// One-shot guard for the loud fused-fallback warning emitted by
    /// <see cref="TrainWithTape"/>. The compiled fused-training fast path
    /// silently falls back to the eager autograd tape when its gates aren't
    /// met (incompatible optimizer config, non-float/double type, mixed
    /// precision, tracing failure). That fallback is a multi-× perf cliff
    /// with — until now — zero signal at the default diagnostic level: a user
    /// could "enable compilation" and unknowingly train on the slow path
    /// forever (the AIsEval benchmark hit exactly this — the default Adam was
    /// rejected by <see cref="TryMapToFusedOptimizerConfig"/> and every step
    /// fell back, invisibly). We surface it ONCE per model instance via
    /// <see cref="System.Diagnostics.Trace"/> (suppressible with
    /// <c>AIDOTNET_QUIET</c>); the flag keeps the hot training loop quiet
    /// after the first warning. <see cref="Configuration.TrainingDiagnosticsConfig"/>
    /// at PerStep still gives per-step detail for those who want it.
    /// </summary>
    private bool _loggedFusedFallback;

    /// <summary>
    /// Returns <c>true</c> if any layer (including nested sub-layers) reports
    /// <see cref="Layers.LayerBase{T}.IsShapeResolved"/> = <c>false</c>, i.e.,
    /// hasn't yet seen a forward pass and is still carrying placeholder
    /// <c>Tensor&lt;T&gt;.Empty()</c> weight refs. Used by
    /// <see cref="TrainWithTape"/> to gate a one-shot warmup forward that
    /// materialises lazy weights before <c>CollectParameters</c> captures
    /// references the gradient tape will later key by. Short-circuits on the
    /// first unresolved layer; on fully eager networks (most CNNs / fixed-
    /// dim transformers constructed with all input sizes known) every layer
    /// is resolved at construction time so this returns <c>false</c> and
    /// the warmup is skipped — zero overhead for the common case.
    /// </summary>
    private bool AnyLayerNeedsShapeResolution()
    {
        // Walk top-level Layers; LayerBase exposes GetSubLayers() so we
        // recurse to catch composite layers (TransformerEncoder,
        // ResidualBlock, etc.) that wrap their own children.
        return AnyLayerNeedsShapeResolutionRecursive(Layers);
    }

    private static bool AnyLayerNeedsShapeResolutionRecursive(IEnumerable<ILayer<T>> layers)
    {
        foreach (var layer in layers)
        {
            if (layer is Layers.LayerBase<T> baseLayer && !baseLayer.IsShapeResolved)
                return true;
            var subs = layer.GetSubLayers();
            if (subs.Count > 0 && AnyLayerNeedsShapeResolutionRecursive(subs))
                return true;
        }
        return false;
    }

    /// <summary>
    /// True if any (possibly nested) layer still holds an unallocated
    /// placeholder trainable parameter (a <c>Length == 0</c> tensor). Shape
    /// resolution alone (<see cref="AnyLayerNeedsShapeResolution"/>) is not
    /// sufficient to decide whether the lazy-init warmup is needed:
    /// <c>ResolveShapesOnly</c> can flip <c>IsShapeResolved</c> to true without
    /// allocating the weight tensors, so a forward is still required to
    /// materialise them before <c>CollectParameters</c> runs. After a real
    /// forward materialises the weights this returns false, so it does not
    /// re-fire the warmup on subsequent training steps.
    /// </summary>
    private bool AnyLayerHasUnmaterializedParameters()
    {
        return AnyLayerHasUnmaterializedParametersRecursive(Layers);
    }

    private static bool AnyLayerHasUnmaterializedParametersRecursive(IEnumerable<ILayer<T>> layers)
    {
        foreach (var layer in layers)
        {
            if (layer is Layers.LayerBase<T> baseLayer)
            {
                foreach (var p in baseLayer.GetTrainableParameters())
                {
                    if (p is null || p.Length == 0) return true;
                }
            }
            var subs = layer.GetSubLayers();
            if (subs.Count > 0 && AnyLayerHasUnmaterializedParametersRecursive(subs))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Attempts the fused-compiled training path — forward + backward + fused
    /// optimizer update all in one compiled kernel. Engages when conditions
    /// permit, returns <c>false</c> to signal the caller to fall back to the
    /// eager tape path otherwise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Conditions for fused path:
    /// <list type="bullet">
    /// <item><c>TensorCodecOptions.Current.EnableCompilation</c> is true (user
    /// opted into JIT via <see cref="AiModelBuilder{T,TInput,TOutput}.ConfigureJitCompilation"/>).</item>
    /// <item><c>T</c> is <see cref="float"/> — the Tensors-side fused optimizer
    /// operates on <c>float*</c> buffers directly.</item>
    /// <item>The resolved optimizer is a plain <see cref="AdamOptimizer{T,TInput,TOutput}"/>,
    /// <see cref="AdamWOptimizer{T,TInput,TOutput}"/>, or
    /// <see cref="StochasticGradientDescentOptimizer{T,TInput,TOutput}"/> without
    /// adaptive-rate machinery that would mutate hyperparameters between steps.
    /// LR schedulers and adaptive rates fall back to the eager path — the fused
    /// kernel bakes hyperparameters into the compiled plan and reconfiguring
    /// them per step would reset Adam's moment buffers, destroying training.</item>
    /// <item>At least one trainable layer participates in the graph (no-op models fall back).</item>
    /// </list>
    /// </para>
    /// <para>
    /// When all hold, <see cref="Training.CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
    /// runs the compiled fwd+bwd+update kernel and updates <see cref="LastLoss"/>.
    /// Behavior matches eager Adam/AdamW/SGD closely enough that training converges
    /// identically on standard benchmarks (the fused kernels use the same formulas).
    /// </para>
    /// </remarks>
    /// <summary>
    /// Helper for <see cref="TryTrainWithFusedOptimizer"/>'s early-return paths.
    /// Stores <paramref name="reason"/> in <see cref="_pendingFusedMissReason"/>
    /// (consumed and cleared by <see cref="TrainWithTape"/>'s post-success
    /// diagnostic block) and returns <c>false</c> so the caller can fall
    /// through to the eager tape path. Deferring emission to after the
    /// eager path commits preserves the "advance only on success"
    /// contract — if the eager fallback throws (forward / backward /
    /// optimizer / extras update / scheduler), no miss event is emitted
    /// for a step that never committed.
    /// </summary>
    private bool EmitFusedMissAndFallback(string reason)
    {
        _pendingFusedMissReason = reason;
        return false;
    }

    private bool TryTrainWithFusedOptimizer(
        Tensor<T> input,
        Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> resolvedOptimizer,
        bool useStreamingDefaults)
    {
        // Callers can still force the eager path via
        // <c>TensorCodecOptions.EnableCompilation = false</c> (handled
        // below). The dedicated <c>ForceEagerPath</c> diagnostic flag
        // (added as a #1328 workaround) was removed in #1331 once the
        // fused-compiled training path was fixed; the EnableCompilation
        // gate is now the single supported way to bypass fused training.
        if (_fusedTrainingDisabled)
            return EmitFusedMissAndFallback("fused path sticky-disabled from prior fallback");
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return EmitFusedMissAndFallback("TensorCodecOptions.EnableCompilation = false");
        // PR #319 fused-optimizer double-kernel support — paired with the
        // matching gate drop in CompiledTapeTrainingStep.TryStepWithFusedOptimizer
        // (line 232 in that file). Both float and double models can now hit
        // the compile-once-replay-many fast path; other numeric types still
        // fall through to the eager autograd tape.
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            return EmitFusedMissAndFallback($"numeric type {typeof(T).Name} not supported by fused kernel");

        if (!TryMapToFusedOptimizerConfig(
                resolvedOptimizer, out var fusedType, out float lr, out float b1, out float b2, out float eps, out float wd,
                out AiDotNet.Tensors.Engines.Compilation.LrSchedule? lrSched))
            return EmitFusedMissAndFallback($"optimizer {resolvedOptimizer.GetType().Name} not compatible with fused kernel");

        // Use the existing recursive trainable-layer collector instead of the
        // top-level-only scan — composite layers with trainable children (e.g.,
        // residual blocks, transformer layers) expose those children via
        // GetSubLayers() but aren't ITrainableLayer themselves. Without
        // recursion the fused path silently stops updating part of the model.
        var trainableLayers = Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers, _layerStructureVersion);
        if (trainableLayers.Length == 0)
            return EmitFusedMissAndFallback("no trainable layers");

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>;
        if (loss is null)
            return EmitFusedMissAndFallback("loss function not derived from LossFunctionBase<T>");

        // Mirror the eager path's bidirectional shape alignment exactly:
        // (a) forward has extra leading dim → reshape FORWARD to target shape
        // (b) target has extra leading dim → reshape TARGET to forward shape
        // The eager path at TrainWithTape reshapes target down to forward's
        // shape in branch (b); doing it the other way (reshape forward up to
        // target's shape) makes the loss compute in a different space and
        // produces different gradients. Apply the same direction-aware fix
        // inside computeLoss where both tensors are in scope.
        var ran = Training.CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
            trainableLayers,
            input,
            expected,
            forward: inp =>
            {
                var fwd = ForwardForTraining(inp);
                // Branch (a): fwd has extra leading batch dim.
                if (fwd.Rank > expected.Rank && fwd.Shape[0] == 1 && fwd.Length == expected.Length)
                    return Engine.Reshape(fwd, expected._shape);
                return fwd;
            },
            computeLoss: (pred, tgt) =>
            {
                // Branch (b): target has extra leading batch dim → reshape TARGET
                // (matches the eager path's direction at TrainWithTape:2509-2512).
                if (tgt.Rank > pred.Rank && tgt.Shape[0] == 1 && tgt.Length == pred.Length)
                    tgt = Engine.Reshape(tgt, pred._shape);
                return loss.ComputeTapeLoss(pred, tgt);
            },
            optimizerType: fusedType,
            learningRate: lr,
            beta1: b1,
            beta2: b2,
            epsilon: eps,
            weightDecay: wd,
            out T lossValue,
            maxGradNorm: MaxGradNormValue,
            lrSchedule: lrSched,
            // Lets the compiled FP16-activation path (AIDOTNET_FP16_ACTIVATIONS=1) cover
            // fused optimizers beyond the inline Adam/SGD fast paths by applying this
            // optimizer's own master update to the FP16-computed FP32 gradients.
            eagerOptimizer: resolvedOptimizer);

        if (ran)
        {
            LastLoss = lossValue;
            // First successful fused step commits this model to the fused
            // path for the rest of the session — Adam m/v are now
            // inside the compiled plan and transferring them to the eager
            // optimizer isn't possible without API we don't have.
            _fusedTrainingCommitted = true;

            // Weight-cache coherence (CodeRabbit, PR #1488): the fused optimizer
            // kernel mutates the weight tensors IN PLACE exactly like the eager
            // path's opt.Step, so the GPU weight uploads AND the CPU engine's
            // identity-keyed derived caches (pre-packed B panels, transposed conv
            // kernels) are equally stale here. Every other in-place update path
            // flushes via this helper; without it, fused-compatible training reuses
            // frozen pre-step weights on the next forward and never learns.
            InvalidateWeightCachesAfterSuccessfulWeightUpdate();

            // Emit diagnostic events for the fused-path hit. This is the
            // ONLY place we can observe that the fused path ran without
            // running through TrainWithTape's tape-walk hook, so consumers
            // tracing #1328-class regressions can correlate "fused hit"
            // with model misbehaviour.
            if (Configuration.TrainingDiagnosticsConfig.Level
                > Configuration.TrainingDiagnosticLevel.Silent)
            {
                int stepIdx = Configuration.TrainingDiagnosticsConfig.AdvanceStep();
                if (Configuration.TrainingDiagnosticsConfig.Level
                    >= Configuration.TrainingDiagnosticLevel.PerStep)
                {
                    Configuration.TrainingDiagnosticsConfig.Emit(
                        new Configuration.FusedOptimizerPathEvent(
                            StepIndex: stepIdx, Hit: true, Reason: null));
                }
                if (Configuration.TrainingDiagnosticsConfig.Level
                    >= Configuration.TrainingDiagnosticLevel.Minimal)
                {
                    Configuration.TrainingDiagnosticsConfig.Emit(
                        new Configuration.TrainingLossEvent(
                            StepIndex: stepIdx,
                            LossValue: NumOps.ToDouble(lossValue),
                            OutputRank: -1,    // fused path doesn't materialize output here
                            OutputLength: -1));
                }
            }
        }
        else if (_fusedTrainingCommitted)
        {
            // We've previously run fused successfully, so Adam/SGD moments
            // live inside the compiled plan. Falling back to eager now would
            // silently reset optimizer state and produce a trajectory that
            // diverges from the previous fused steps. Surface the problem
            // explicitly rather than corrupt training. Common causes:
            //   - variable {input, target} shape produced a new compiled
            //     plan (strict single-plan policy refused to configure it)
            //   - mutated optimizer hyperparameters between steps
            //     (attached LR scheduler, changed betas, etc.)
            //   - a kernel-level exception caught and swallowed
            //     (CompiledTrainingPlan.Step / ConfigureOptimizer threw)
            // Resolution: call ResetState() or InvalidateParameterCountCache()
            // to fully reset training state, then retrain with stable
            // shapes + fixed hyperparameters. Or disable compilation via
            // AllowNondeterminism / Configure(JitCompilationConfig.Disabled)
            // so training runs entirely on the eager path from the start.
            //
            // AiDotNet#1395: quote the swallowed exception (if any) inline so
            // failing tests don't have to chase Trace output to learn the
            // root cause (Parameter N non-contiguous CPU layout, shape mismatch
            // in a backward kernel, NaN guard trip, etc.). The inner exception
            // is also attached for catch (InvalidOperationException ex) callers
            // that introspect ex.InnerException.
            var fallbackEx = Training.CompiledTapeTrainingStep<T>.GetLastFallbackException();

            // Transient GPU/CUDA fault in the committed fused plan (e.g. CUDA error 700 from the
            // activation-cache deferred-materializer race, or a device fault): the compiled plan
            // can't continue on the GPU. Rather than abort the whole training run, reset the
            // fused/compiled state and fall back to the eager path — the optimizer moments reset
            // (a one-time trajectory perturbation), which is far better than crashing. Non-GPU
            // causes (shape drift, hyperparameter changes) still surface loudly below.
            if (IsGpuTransientFailure(fallbackEx))
            {
                _fusedTrainingDisabled = true;
                _fusedTrainingCommitted = false;
                InvalidateParameterCountCache();
                return false;
            }

            // Out-of-memory in the committed compiled plan: the model is too large
            // (or the host too pressured) to keep the fused plan's buffers + Adam
            // moments resident. This is precisely what the memory-bounded streaming
            // training path exists for — degrade to it rather than aborting the run.
            // TrainWithTapeStreaming drives optimizer-in-backward with topological-min
            // gradient release (tape.ComputeGradientsStreaming) and an 8-bit
            // streaming optimizer that maintains its OWN moment state, so the
            // "the compiled plan's moments can't be transferred" obstruction
            // that forces the throw below does NOT apply here — streaming simply
            // continues training under its own optimizer. Pin StreamingTraining =
            // ForceOn so every subsequent step in this run also takes the streaming
            // path (a single moment-state reset at the switchover, then stable),
            // and disable the OOMing fused plan so it is never re-engaged. This
            // mirrors the IsGpuTransientFailure graceful-degradation policy above:
            // a one-time trajectory perturbation is far better than crashing.
            if (fallbackEx is OutOfMemoryException)
            {
                _fusedTrainingDisabled = true;
                _fusedTrainingCommitted = false;
                StreamingTraining = StreamingTrainingMode.ForceOn;
                InvalidateParameterCountCache();
                TrainWithTapeStreaming(input, expected, resolvedOptimizer, useStreamingDefaults);
                return true; // step handled via the streaming path
            }

            var rootCauseSuffix = fallbackEx is not null
                ? $" Root-cause exception (caught in CompiledTapeTrainingStep): " +
                  $"{fallbackEx.GetType().FullName}: {fallbackEx.Message}"
                : " (No exception was caught — fused path returned false from one of the explicit " +
                  "refuse paths: plan reference changed, optimizer hyperparameters drifted, " +
                  "TensorCodecOptions.EnableCompilation=false, or numeric/optimizer type unsupported.)";
            throw new InvalidOperationException(
                "Fused compiled training has already run successfully, but the current step cannot " +
                "engage the fused path. The plan-embedded Adam/AdamW/SGD state cannot be transferred " +
                "to the eager optimizer, so falling back silently would produce a trajectory that " +
                "diverges from the previous fused steps. Common causes: variable input/target shape " +
                "(new compiled plan), LR scheduler or adaptive-rate changes, attached AMSGrad, or a " +
                "kernel-level exception in plan.Step/ConfigureOptimizer that was caught and swallowed. " +
                "Resolution: keep shapes and optimizer hyperparameters stable across steps, OR call " +
                "ResetState() / InvalidateParameterCountCache() to explicitly reset training state, " +
                "OR disable compilation (AiModelBuilder.ConfigureJitCompilation(JitCompilationConfig.Disabled))." +
                rootCauseSuffix,
                innerException: fallbackEx);
        }
        else
        {
            // Sticky disable: subsequent training steps in this run stay on the
            // eager path so we don't reset Adam moments by re-engaging fused
            // mid-run. Cleared on next ResetState/InvalidateParameterCountCache.
            _fusedTrainingDisabled = true;
        }
        return ran;
    }

    /// <summary>
    /// True if the exception chain indicates a transient GPU/CUDA fault (device error, failed
    /// host/device copy, or the activation-cache deferred-materializer race) — as opposed to a
    /// logical cause (shape drift, hyperparameter change). Used to decide whether a committed
    /// fused step can reset-and-recover on the eager path instead of aborting.
    /// </summary>
    protected static bool IsGpuTransientFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            var message = e.Message;
            if (message.Contains("CUDA error", StringComparison.OrdinalIgnoreCase)
                || message.Contains("cuMem", StringComparison.Ordinal)
                || message.Contains("cuStream", StringComparison.Ordinal)
                || message.Contains("cuLaunch", StringComparison.Ordinal)
                || message.Contains("OpenCL", StringComparison.OrdinalIgnoreCase)
                || message.Contains("released before materialization", StringComparison.OrdinalIgnoreCase)
                || message.Contains("buffer was released", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Inspects a pluggable optimizer and maps it onto the fixed set supported
    /// by the Tensors-side fused kernel (<c>SGD</c>, <c>Adam</c>, <c>AdamW</c>).
    /// Returns <c>false</c> when the optimizer is outside that set OR when
    /// per-step hyperparameter mutation would defeat the fused plan's
    /// configure-once contract: adaptive learning rates, attached LR schedulers,
    /// or AMSGrad mode (which the fused kernel doesn't model).
    /// </summary>
    private static bool TryMapToFusedOptimizerConfig(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        out AiDotNet.Tensors.Engines.Compilation.OptimizerType optimizerType,
        out float learningRate,
        out float beta1,
        out float beta2,
        out float epsilon,
        out float weightDecay,
        out AiDotNet.Tensors.Engines.Compilation.LrSchedule? lrSchedule)
    {
        // Open/closed-compliant dispatch: the optimizer self-describes its fused
        // config (incl. selecting the AMSGrad kernel variant when it opts in, and
        // converting any attached LR scheduler) via IFusedOptimizerSpec. Replaces
        // the old type-switch that had to be edited for every new optimizer and
        // silently rejected AMSGrad. Only optimizers that actually have a fused
        // SIMD kernel implement the interface, so there is no central whitelist to
        // maintain — an optimizer without a kernel simply isn't an
        // IFusedOptimizerSpec and uses the eager tape.
        optimizerType = default;
        learningRate = 0f;
        beta1 = 0f;
        beta2 = 0f;
        epsilon = 0f;
        weightDecay = 0f;
        lrSchedule = null;

        if (optimizer is not Optimizers.Fused.IFusedOptimizerSpec spec
            || !spec.TryGetFusedOptimizerConfig(out var cfg))
            return false;

        optimizerType = cfg.Type;
        learningRate = cfg.LearningRate;
        beta1 = cfg.Beta1;
        beta2 = cfg.Beta2;
        epsilon = cfg.Epsilon;
        weightDecay = cfg.WeightDecay;
        lrSchedule = cfg.Schedule;
        return true;
    }

    /// <summary>
    /// Performs tape-based training with a caller-provided loss function.
    /// Use this for RL agents and other scenarios where the loss is not a standard
    /// predicted-vs-target comparison (e.g., PPO's clipped surrogate objective).
    /// </summary>
    /// <param name="input">The input tensor for the forward pass.</param>
    /// <param name="computeLoss">
    /// Function that receives the forward pass output and computes a scalar loss tensor
    /// using engine ops (tape-tracked). Must return a scalar tensor for gradient computation.
    /// </param>
    /// <param name="optimizer">Optional optimizer override. Uses default Adam if null.</param>
    /// <returns>The scalar loss value for monitoring.</returns>
    public T TrainWithCustomLoss(
        Tensor<T> input,
        Func<Tensor<T>, Tensor<T>> computeLoss,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        SetTrainingMode(true);
        try
        {
            var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
            var opt = optimizer ?? GetOrCreateBaseOptimizer();

            using var tape = new GradientTape<T>();
            var output = ForwardForTraining(input);
            var lossTensor = computeLoss(output);

            var grads = tape.ComputeGradients(lossTensor, trainableParams);

            T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
            LastLoss = lossValue;

            Tensor<T> ComputeForward(Tensor<T> inp, Tensor<T> _) => ForwardForTraining(inp);
            Tensor<T> RecomputeLoss(Tensor<T> pred, Tensor<T> _) => computeLoss(pred);

            var context = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(
                trainableParams, grads, lossValue,
                input, input, ComputeForward, RecomputeLoss);

            opt.Step(context);
            // GPU weight-cache coherence after the custom-loss step.
            // See InvalidateWeightCachesAfterSuccessfulWeightUpdate.
            InvalidateWeightCachesAfterSuccessfulWeightUpdate();

            // Mirror the OnBatchEnd advance from TrainWithTape via the
            // shared helper so a custom-loss caller and a regular Train
            // caller see identical scheduler behaviour. Closes #1269.zFt3
            // (route via shared StepSchedulerIfSupported helper to keep
            // every training entry point consistent — #1270.zKjB).
            StepSchedulerIfSupported(opt);

            return lossValue;
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Variant of <see cref="TrainWithCustomLoss"/> that runs backward+optim
    /// on a loss tensor whose forward pass was already recorded on a
    /// caller-owned <see cref="GradientTape{T}"/>. Use this when the caller
    /// needs to share the forward output across multiple training paths so
    /// the forward only runs once.
    ///
    /// <para>Issue #1390 driver: <c>GenerativeAdversarialNetwork.Train</c>
    /// previously ran the generator forward twice per step — once detached
    /// (<see cref="Predict"/> + <see cref="NoGradScope{T}"/>) to feed the
    /// discriminator step, then a second time tape-tracked
    /// (inside <see cref="TrainWithCustomLoss"/>) for the generator's
    /// adversarial backward. With this entry point, the GAN's <c>Train</c>
    /// can open its own gen tape, run the forward once, detach a
    /// value-snapshot for the discriminator step, and feed the same
    /// tape-tracked output into the generator loss — eliminating the
    /// duplicate forward (~4% of step time on the DCGAN MoreData fixture
    /// per the issue's substep profile).</para>
    ///
    /// <para>Contract: the caller MUST keep the tape alive (not dispose it)
    /// from the time the forward runs until this method returns. The tape
    /// is NOT disposed here — the caller owns its lifecycle.</para>
    /// </summary>
    /// <param name="tape">Open gradient tape on which the forward was recorded.</param>
    /// <param name="lossTensor">Scalar loss tensor recorded on <paramref name="tape"/>.</param>
    /// <param name="optimizer">Optimizer to drive; defaults to the base Adam.</param>
    /// <returns>Scalar loss value (also stored in <see cref="LastLoss"/>).</returns>
    /// <remarks>
    /// Visibility: <c>internal</c>. The codebase contract is "users should only
    /// interact with <c>AiModelBuilder</c> / <c>AiModelResult</c>" — this helper
    /// is training plumbing for in-assembly callers (currently
    /// <see cref="GenerativeAdversarialNetwork{T,TI,TO}.Train"/>) and should not
    /// appear on the public API surface. (PR #1389 CodeRabbit review.)
    /// </remarks>
    internal T BackwardAndStepOnPrecomputedLoss(
        GradientTape<T> tape,
        Tensor<T> lossTensor,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        if (tape is null) throw new ArgumentNullException(nameof(tape));
        if (lossTensor is null) throw new ArgumentNullException(nameof(lossTensor));

        // PR #1389 CodeRabbit fix: reentrancy guard so concurrent callers don't
        // interleave optimizer/tape state. Mirrors TrainWithTape's sentinel
        // discipline; without it, two threads calling this helper on the same
        // model would race on LastLoss + optimizer-internal state.
        using var __reentrancyGuard = AcquireTrainSentinel();

        // PR #1389 CodeRabbit fix: include network-level trainable tensors
        // (raw parameters owned by the model rather than a layer) in the
        // gradient-and-step set. Without this, models that expose params via
        // GetExtraTrainableTensors() would silently skip those updates here
        // while TrainWithTape would still update them — divergent semantics
        // between the two training entry points.
        var layerParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
        var extraTrainableTensors = new List<Tensor<T>>();
        foreach (var t in GetExtraTrainableTensors())
        {
            if (t is not null && t.Length > 0) extraTrainableTensors.Add(t);
        }
        var trainableParams = extraTrainableTensors.Count == 0
            ? layerParams
            : layerParams.Concat(extraTrainableTensors).ToList();
        var opt = optimizer ?? GetOrCreateBaseOptimizer();

        var grads = tape.ComputeGradients(lossTensor, trainableParams);
        T lossValue = lossTensor.Length > 0 ? lossTensor[0] : NumOps.Zero;
        LastLoss = lossValue;

        var context = new AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T>(
            trainableParams, grads, lossValue);

        opt.Step(context);
        // GPU weight-cache coherence after the precomputed-loss step.
        // See InvalidateWeightCachesAfterSuccessfulWeightUpdate.
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();

        // Mirror the OnBatchEnd advance from TrainWithCustomLoss / TrainWithTape
        // so a precomputed-loss caller sees identical scheduler behaviour.
        StepSchedulerIfSupported(opt);

        return lossValue;
    }

    /// <summary>
    /// Gets or lazily creates the default optimizer for tape-based training.
    /// Used when a network doesn't provide its own optimizer.
    /// </summary>
    /// <remarks>
    /// Default is <b>standard Adam</b> (Kingma &amp; Ba 2015) — matching the
    /// industry default in PyTorch (<c>torch.optim.Adam(amsgrad=False)</c>),
    /// TensorFlow/Keras, and Optax. AMSGrad is a niche opt-in variant, not a
    /// default anywhere; callers who want it set <c>UseAMSGrad = true</c> on a
    /// supplied optimizer.
    /// <para>
    /// History: #1350 flipped this default to AMSGrad to suppress
    /// post-convergence drift on a couple of recurrent models'
    /// <c>MoreData_ShouldNotDegrade</c> invariant. That was the wrong lever — it
    /// only partially helped those models (GRU still drifted past tolerance
    /// even with AMSGrad; DBM was unchanged), yet it <b>silently disabled the
    /// compiled fused-training fast path for EVERY model</b> (the fused Adam
    /// kernel implements standard Adam, correctly omitting the non-standard
    /// AMSGrad max-second-moment rule), so all models fell back to the slow
    /// eager tape. The proper drift fix is LR-decay scheduling past convergence
    /// (as #1350 itself noted), applied per-model — not a non-standard global
    /// optimizer default. Reverting to standard Adam restores both the industry
    /// default and the fused fast path.
    /// </para>
    /// </remarks>
    protected virtual IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        // Standard Adam with AMSGrad pinned OFF locally (not relying on the
        // AdamOptimizerOptions default) so the fused compiled-training kernel can map
        // this optimizer and engage — if that external default ever flips, the fused
        // fast path must not silently regress.
        return _baseTrainOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { UseAMSGrad = false });
    }

    /// <summary>
    /// Advances the optimizer's learning-rate scheduler at a training-batch
    /// boundary. Single source of truth for the OnBatchEnd contract — every
    /// training entry point (legacy <c>Train</c>, <see cref="TrainWithTape"/>,
    /// <c>TrainWithCustomLoss</c>) routes through this helper so all paths
    /// keep identical scheduler-step semantics. Optimizers that don't derive
    /// from <see cref="Optimizers.GradientBasedOptimizerBase{T, TInput, TOutput}"/>
    /// are silently ignored — that's the documented contract on
    /// <c>OnBatchEnd</c>: only gradient-based optimizers carry an LR
    /// scheduler. Closes review-comment #1270.zKjB.
    /// </summary>
    /// <param name="optimizer">The optimizer that just finished a batch.</param>
    private static void StepSchedulerIfSupported(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> stepped)
        {
            stepped.OnBatchEnd();
        }
    }

    /// <summary>
    /// Invalidates EVERY identity-keyed weight cache (GPU weight-buffer uploads
    /// AND the CPU engine's derived-weight caches) after a successful in-place
    /// weight update. Single source of truth for the "post-weight-write cache
    /// flush" contract — every in-place parameter-write path (the training
    /// entry points <see cref="TrainWithTape"/>, <c>TrainWithGradientAccumulation</c>,
    /// <c>TrainWithCustomLoss</c>, <c>BackwardAndStepOnPrecomputedLoss</c>, and
    /// the bulk-load paths <see cref="WithParameters"/> / <see cref="SetParameters"/>)
    /// routes through this helper so they all keep the same coherency guarantee.
    /// </summary>
    /// <remarks>
    /// Weight writes mutate the tensors IN PLACE, but DirectGpuTensorEngine's
    /// weight-buffer cache keys by array REFERENCE and returns the cached upload
    /// without re-checking contents. Without flushing here the next forward reads
    /// STALE weights and the model NEVER LEARNS on GPU — loss frozen at ln(V) even
    /// though the gradient is correct (it is recomputed each step). The GPU flush
    /// is a no-op on the CPU engine (GpuEngine is null); the CPU flush below is
    /// always live. Mirrors the OnParametersUpdated contract.
    /// </remarks>
    private void InvalidateWeightCachesAfterSuccessfulWeightUpdate()
    {
        GpuEngine?.InvalidateAllWeightCaches();
        // CPU-side mirror of the same contract: the CPU engine's inference
        // fast paths cache DERIVED weight forms (SgemmWithCachedB's
        // pre-packed B panels, Conv2D's transposed kernels) keyed by the
        // weight ARRAY's object identity, never re-reading its contents.
        // opt.Step mutates the weight tensors IN PLACE, so without this
        // flush the next Predict consumes STALE packed weights — the model
        // appears not to learn on exactly the layers whose GEMMs hit the
        // cached path (root cause of the Issue1296 grad-accum equivalence
        // probe failure: two models with bit-identical parameters predicted
        // differently because one carried packs from its pre-load init).
        AiDotNet.Tensors.Engines.InferenceWeightCache.InvalidateAll();
    }

    /// <summary>
    /// Pre-wires the optimizer instance that <see cref="GetOrCreateBaseOptimizer"/>
    /// will return on subsequent calls. Used by <c>AiModelBuilder.ConfigureOptimizer</c>
    /// to inject the user-configured optimizer (e.g. AdamW with a learning-rate
    /// scheduler, Lion, custom subclass) into the model BEFORE the streaming /
    /// build training loop calls <see cref="Train(Tensor{T}, Tensor{T})"/>.
    /// Without this hook, calling <c>nn.Train(input, target)</c> would resolve
    /// to a freshly-allocated default Adam via the lazy fallback in
    /// <see cref="GetOrCreateBaseOptimizer"/>, silently dropping any builder-
    /// level optimizer configuration on the floor.
    /// </summary>
    /// <param name="optimizer">
    /// The optimizer to install as the model's base training optimizer, or null
    /// to clear the override (the next call to <see cref="GetOrCreateBaseOptimizer"/>
    /// will then re-create the lazy default).
    /// </param>
    /// <remarks>
    /// Virtual so subclasses that maintain a separate private optimizer
    /// field (e.g. <see cref="Transformer{T}"/>'s <c>_optimizer</c>) can
    /// override and keep that field in sync. Without the override, the
    /// subclass field becomes stale after the first builder-driven
    /// <see cref="SetBaseTrainOptimizer"/> call — and any path that still
    /// reads the subclass field (metadata, serialization, custom training
    /// shortcuts) silently reports / persists the wrong optimizer.
    /// </remarks>
    internal virtual void SetBaseTrainOptimizer(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
    {
        _baseTrainOptimizer = optimizer;
    }

    /// <summary>
    /// Persistent optimizer for models using the standard TrainStep pattern.
    /// Lazily initialized on first use (Adam with default settings).
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _baseTrainOptimizer;

    /// <summary>
    /// Contiguous parameter buffer for zero-copy flat parameter access.
    /// Lazily initialized on first training step when trainable parameters are available.
    /// </summary>
    private ParameterBuffer<T>? _parameterBuffer;

    /// <summary>
    /// Once foundation-scale models cross the parameter-buffer skip
    /// threshold we want each subsequent training step to take the
    /// no-buffer path in O(1), not re-scan every parameter tensor with
    /// CollectParameters + sum-Length on each call. We memoize the
    /// "skip buffer" decision keyed by <see cref="_layerStructureVersion"/>
    /// so InvalidateParameterCountCache (which bumps the version) re-tests
    /// the threshold the first time the parameter set actually changed.
    /// </summary>
    private bool _skipParameterBuffer;
    private int _skipParameterBufferVersion = -1;

    /// <summary>
    /// <summary>
    /// Original tensor references saved before buffer view replacement.
    /// Restored after each training step so Clone/serialization see real tensors.
    /// </summary>
    private Dictionary<ILayer<T>, IReadOnlyList<Tensor<T>>>? _savedOriginalParameters;

    /// <summary>
    /// Gets or lazily creates the contiguous parameter buffer from the current trainable parameters.
    /// </summary>
    private ParameterBuffer<T>? GetOrCreateParameterBuffer(IReadOnlyList<Tensor<T>> trainableParams)
    {
        if (trainableParams.Count == 0)
            return null;

        if (_parameterBuffer is not null && _parameterBuffer.Count == trainableParams.Count)
            return _parameterBuffer;

        // Build buffer from current parameter shapes
        var shapes = new int[trainableParams.Count][];
        for (int i = 0; i < trainableParams.Count; i++)
            shapes[i] = trainableParams[i]._shape;

        var buffer = new ParameterBuffer<T>(shapes);

        // Copy current weights into the buffer
        buffer.CopyFrom(trainableParams);

        // Replace layer parameter tensors with views into the contiguous buffer.
        // This makes the buffer the single source of truth — in-place updates by
        // first-order optimizers automatically reflect in the flat vector, and vice versa.
        // Must recurse into sublayers to match CollectRecursive's walk order.
        // Build a map from original parameter tensor → buffer view tensor,
        // using the same deduplication order as CollectRecursive (by tensor reference).
        var views = buffer.CreateAllViews();
        var paramToView = new Dictionary<Tensor<T>, Tensor<T>>(
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        for (int i = 0; i < trainableParams.Count; i++)
            paramToView[trainableParams[i]] = views[i];

        // Save original tensor references before replacement so we can restore later
        _savedOriginalParameters = new Dictionary<ILayer<T>, IReadOnlyList<Tensor<T>>>();
        SaveOriginalParameters(Layers, _savedOriginalParameters, new HashSet<ILayer<T>>());

        // Replace each layer's parameters with their buffer-backed views.
        var seenLayers = new HashSet<ILayer<T>>();
        ReplaceParametersFromMap(Layers, paramToView, seenLayers);

        _parameterBuffer = buffer;
        return buffer;
    }

    /// <summary>
    /// Recursively replaces trainable parameter tensors with buffer-backed views
    /// using a pre-built parameter→view map. This avoids walk-order sensitivity
    /// by looking up each layer's parameters in the map by reference identity.
    /// </summary>
    private static void ReplaceParametersFromMap(
        IEnumerable<ILayer<T>> layers,
        Dictionary<Tensor<T>, Tensor<T>> paramToView,
        HashSet<ILayer<T>> seenLayers)
    {
        foreach (var layer in layers)
        {
            if (!seenLayers.Add(layer)) continue;

            if (layer is ITrainableLayer<T> trainable)
            {
                var layerParams = trainable.GetTrainableParameters();
                var layerViews = new Tensor<T>[layerParams.Count];
                bool anyReplaced = false;
                for (int i = 0; i < layerParams.Count; i++)
                {
                    if (paramToView.TryGetValue(layerParams[i], out var view))
                    {
                        layerViews[i] = view;
                        anyReplaced = true;
                    }
                    else
                    {
                        layerViews[i] = layerParams[i]; // Keep original if not in buffer
                    }
                }
                if (anyReplaced)
                    trainable.SetTrainableParameters(layerViews);
            }

            var subLayers = layer.GetSubLayers();
            if (subLayers.Count > 0)
                ReplaceParametersFromMap(subLayers, paramToView, seenLayers);
        }
    }

    /// <summary>
    /// Saves original tensor references from all trainable layers before buffer view replacement.
    /// </summary>
    private static void SaveOriginalParameters(
        IEnumerable<ILayer<T>> layers,
        Dictionary<ILayer<T>, IReadOnlyList<Tensor<T>>> saved,
        HashSet<ILayer<T>> seen)
    {
        foreach (var layer in layers)
        {
            if (!seen.Add(layer)) continue;

            if (layer is ITrainableLayer<T> trainable)
            {
                var current = trainable.GetTrainableParameters();
                // Clone the list so we have the original references, not the views
                saved[layer] = current.ToArray();
            }

            var subLayers = layer.GetSubLayers();
            if (subLayers.Count > 0)
                SaveOriginalParameters(subLayers, saved, seen);
        }
    }

    /// <summary>
    /// Restores original tensor references on all layers after a training step.
    /// Copies updated data from buffer views back to the original tensors first.
    /// </summary>
    private void RestoreOriginalParameters()
    {
        if (_savedOriginalParameters == null)
            return;

        // ParameterBuffer architecture (TapeStepContext.ValidateBufferAlignment
        // line 358 of AiDotNet.Tensors): when paramBuffer is non-null,
        // EVERY parameter passed to the optimizer step MUST be a view into
        // that buffer's storage. The check is enforced via
        // ReferenceEquals(parameters[i]._storage, bufferStorage). Layers
        // therefore have to keep their _* fields pointing at buffer views
        // during training — that's the design contract.
        //
        // An unconditional SetTrainableParameters(originals) here would
        // VIOLATE that contract: layer fields would get swapped back to
        // standalone originals at end-of-step, the next step's forward
        // would run on originals, the tape would record gradients keyed
        // by originals, and TapeStepContext would reject the params-vs-
        // buffer mismatch on the next iteration's optimizer step (or —
        // worse, before the upstream Tensors validation landed — silently
        // accept the mismatch and the optimizer would update buffer
        // storage that was no longer reachable from any layer's forward
        // pass).
        //
        // Strategy — two-pass walk over the saved layers:
        //   Pass 1: detect structural change AND sync view→original data
        //           for stable layers in a single sweep. Save
        //           (trainable, originals) pairs for stable layers so
        //           pass 2 can restore them if the buffer is going to
        //           be invalidated.
        //   Pass 2: only fires if anyStructureChanged is true. Swap
        //           EVERY recorded stable-layer's fields back to
        //           originals — even layers we'd otherwise leave on
        //           buffer-views need this when the buffer is about to
        //           be invalidated below by InvalidateParameterCountCache,
        //           because the next CollectParameters call must build
        //           a fresh buffer from the originals (any layer still
        //           holding orphaned buffer-view refs would either get
        //           its data lost or trigger a buffer-alignment
        //           mismatch on the rebuild). This makes the restore
        //           order-independent: detecting a structure change
        //           late in pass 1 must NOT leave earlier stable
        //           layers stuck on orphaned views.
        bool anyStructureChanged = false;
        var stableRestoreCandidates =
            new List<(ITrainableLayer<T> Trainable, IReadOnlyList<Tensor<T>> Originals)>();

        foreach (var (layer, originals) in _savedOriginalParameters)
        {
            if (layer is not ITrainableLayer<T> trainable)
            {
                continue;
            }

            var currentViews = trainable.GetTrainableParameters();

            // If parameter count or sizes changed (e.g., DenseLayer or
            // EmbeddingLayer lazy initialization resized weights during
            // the first forward pass), skip restoration for this layer
            // — the pre-init parameters are meaningless and the layer
            // now has the correct shape for the actual input data.
            if (currentViews.Count != originals.Count)
            {
                anyStructureChanged = true;
                continue;
            }

            bool sizeChanged = false;
            for (int i = 0; i < originals.Count; i++)
            {
                if (currentViews[i].Length != originals[i].Length)
                {
                    sizeChanged = true;
                    break;
                }
            }
            if (sizeChanged)
            {
                anyStructureChanged = true;
                continue;
            }

            // Stable layer: copy view→original data so external
            // observers (Clone / Serialize / GetTrainableParameters
            // from user code) see up-to-date weights. Defer the
            // swap-back decision to pass 2 once we know whether ANY
            // layer in this batch changed structure — restore-back is
            // correct only when the buffer is being invalidated.
            for (int i = 0; i < originals.Count; i++)
            {
                Engine.TensorCopy(currentViews[i], originals[i]);
            }
            stableRestoreCandidates.Add((trainable, originals));
        }

        if (anyStructureChanged)
        {
            // Buffer is going to be invalidated below via
            // InvalidateParameterCountCache. Restore EVERY stable
            // layer's field references back to originals so the next
            // CollectParameters call rebuilds buffer + views from a
            // consistent set of standalone tensors. Order-independent:
            // even if the structure-change layer was processed last,
            // earlier stable layers still get their fields swapped
            // back here.
            foreach (var (trainable, originals) in stableRestoreCandidates)
            {
                trainable.SetTrainableParameters(originals);
            }
        }
        // else: keep stable layers' fields pointing at buffer views —
        // the tape, optimizer, and layer all agree on the same tensor
        // reference, the data sync above means the originals also hold
        // the post-step weights, and external observers see correct
        // values.

        _savedOriginalParameters = null;

        if (anyStructureChanged)
        {
            // Reuse the full structural invalidation path so every cache that
            // depends on layer structure (parameter count, parameter buffer,
            // tape collector, layer-info, compiled inference plans, known-bad
            // compile shapes) is reset together. Maintaining a partial path
            // here would let _knownBadCompileShapes / _cachedParameterCount
            // stay stale after a lazy-init resize and lock the model into
            // permanently-eager Predict for shapes that previously failed.
            InvalidateParameterCountCache();
        }
    }

    /// <summary>
    /// Gets the metadata for this neural network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Resets the internal state of the different layers, clearing any remembered information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state (hidden state and cell state) of all layers in the network.
    /// This is useful when starting to process a new, unrelated sequence or when the network's memory
    /// should be cleared before making new predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the neural network's memory to start fresh.
    /// 
    /// Think of this like:
    /// - Wiping the slate clean before starting a new task
    /// - Erasing the neural network's "memory" so past inputs don't influence new predictions
    /// - Starting fresh when processing a completely new sequence
    /// 
    /// For example, if you've been using an neural network to analyze one document and now want to
    /// analyze a completely different document, you would reset the state first to avoid
    /// having the first document influence the analysis of the second one.
    /// </para>
    /// </remarks>
    public virtual void ResetState()
    {
        foreach (var layer in Layers)
        {
            layer.ResetState();
        }
        // Give the fused-training path a fresh chance after ResetState — the
        // user typically calls this between training runs, which is exactly
        // when the sticky-disable from a prior fallback should be cleared.
        // Also clear the fused-commitment: ResetState is an explicit
        // "start training over" signal, so any plan-embedded Adam/SGD state
        // is no longer needed, and the next run can engage fused fresh.
        _fusedTrainingDisabled = false;
        _fusedTrainingCommitted = false;
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the entire neural network, including all layers and parameters,
    /// and saves it to the specified file path. The file includes an AIMF envelope header
    /// that allows automatic model type detection when loading.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This saves your trained neural network to a file on your computer.
    ///
    /// Think of it like saving a document - you can later load the model back from the file
    /// and use it to make predictions without having to retrain it from scratch.
    ///
    /// This is useful when:
    /// - You've finished training and want to save your model
    /// - You want to use the model in a different application
    /// - You need to share the model with others
    /// - You want to deploy the model to production
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        ModelPersistenceGuard.EnforceBeforeSave();
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] serializedData = Serialize();
            byte[] envelopedData = ModelFileHeader.WrapWithHeader(
                serializedData, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary,
                GetDynamicShapeInfo());
            File.WriteAllBytes(filePath, envelopedData);
        }
    }

    /// <summary>
    /// Loads a neural network model from a file.
    /// </summary>
    /// <param name="filePath">The path to the file containing the saved model.</param>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to load a previously saved neural network model
    /// from a file on disk. This is the counterpart to SaveModel and uses the Deserialize method
    /// to reconstruct the network from the saved data.
    /// </para>
    /// </remarks>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        }

        ModelPersistenceGuard.EnforceBeforeLoad();
        using (ModelPersistenceGuard.InternalOperation())
        {
            byte[] data = File.ReadAllBytes(filePath);

            // Extract payload from AIMF envelope
            data = ModelFileHeader.ExtractPayload(data);

            Deserialize(data);
        }
    }

    private const int SerializationMagic = 0x4E444941; // "AIDN" (little-endian int)
    private const int SerializationVersion = 4;

    // Mirrors System.Array.MaxLength (introduced in .NET 6). Hardcoded
    // here so the check still compiles on net471, where Array.MaxLength
    // is not defined. Value is the CLR's actual largest single-dimension
    // byte array length (= int.MaxValue - 56).
    private const int MaxArrayLength = 0X7FFFFFC7;

    /// <summary>
    /// Serializes the neural network to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized neural network.</returns>
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        return SerializeInternalUnchecked();
    }

    /// <summary>
    /// Internal, non-virtual, no-guard serialization used by trusted framework
    /// call sites such as <see cref="DeepCopy"/>. Users cannot call this and
    /// cannot override it, so a malicious/careless subclass override of
    /// <see cref="Serialize"/> cannot intercept this path.
    /// </summary>
    private byte[] SerializeInternalUnchecked()
    {
        // Pre-size the MemoryStream to avoid ensureCapacity doubling near the
        // 2GB array cap on large models. ViLBERT (~174M params × 8 B = 1.4 GB)
        // hits OutOfMemoryException during serialize because at ~1 GB filled
        // the stream tries to double to 2 GB, which exceeds Array.MaxLength.
        // Compute the parameter payload ahead of time; rough overhead for
        // type-name strings, shape ints, metadata etc. is tiny vs the
        // parameter doubles, but budget ~64 KB per layer just in case.
        long paramBytes = 0;
        foreach (var layer in Layers)
        {
            paramBytes += (long)layer.ParameterCount * sizeof(double);
            if (layer is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> extras)
            {
                paramBytes += (long)extras.ExtraParameterCount * sizeof(double);
            }
        }
        long estimatedTotal = paramBytes + (long)Layers.Count * 65536 + 1024;
        // MemoryStream capacity is an int. Cap at MaxArrayLength so we never
        // request a capacity that cannot be allocated as a single byte array.
        // If the model is so large that even this cap won't hold it, the
        // existing grow-on-write logic will still fail at the correct point,
        // which is the right behavior (serialization of a >2 GB blob needs a
        // file-backed stream, not an in-memory byte array).
        int initialCapacity = (int)Math.Min(estimatedTotal, (long)MaxArrayLength);
        using var ms = new MemoryStream(initialCapacity);
        using var writer = new BinaryWriter(ms);

        writer.Write(SerializationMagic);
        writer.Write(SerializationVersion);

        // Write the number of layers
        writer.Write(Layers.Count);

        // Write each layer's type and shape
        foreach (var layer in Layers)
        {
            // Write layer type
            writer.Write(layer.GetType().Name);

            // Write input shape
            var inputShape = layer.GetInputShape();
            writer.Write(inputShape.Length);
            foreach (var dim in inputShape)
            {
                writer.Write(dim);
            }

            // Write output shape
            var outputShape = layer.GetOutputShape();
            writer.Write(outputShape.Length);
            foreach (var dim in outputShape)
            {
                writer.Write(dim);
            }

            // Write constructor-level metadata needed to reconstruct layers during cloning/serialization.
            var metadata = layer is AiDotNet.NeuralNetworks.Layers.LayerBase<T> layerBase
                ? layerBase.GetMetadata()
                : new Dictionary<string, string>(StringComparer.Ordinal);

            writer.Write(metadata.Count);
            foreach (var kvp in metadata)
            {
                writer.Write(kvp.Key ?? string.Empty);
                writer.Write(kvp.Value ?? string.Empty);
            }

            // Write parameters (do not rely on ParameterCount: some layers keep trainable state outside LayerBase.Parameters).
            var parameters = layer.GetParameters();
            writer.Write(parameters.Length);
            foreach (var param in parameters)
            {
                writer.Write(Convert.ToDouble(param));
            }

            // Write any optional extra parameter blocks required to fully reconstruct the layer (e.g., frozen base weights in LoRA adapters).
            if (layer is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> extras)
            {
                var extraParameters = extras.GetExtraParameters();
                writer.Write(extraParameters.Length);
                foreach (var param in extraParameters)
                {
                    writer.Write(Convert.ToDouble(param));
                }
            }
            else
            {
                writer.Write(0);
            }
        }

        // Write network-specific data
        SerializeNetworkSpecificData(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the neural network from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized neural network data.</param>
    public virtual void Deserialize(byte[] data)
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        if (data.Length == 0)
            throw new ArgumentException("Serialized data cannot be empty.", nameof(data));

        ModelPersistenceGuard.EnforceBeforeDeserialize();
        DeserializeInternalUnchecked(data);
    }

    /// <summary>
    /// Internal, non-virtual, no-guard deserialization used by trusted
    /// framework call sites such as <see cref="DeepCopy"/>. Users cannot
    /// call this and cannot override it.
    /// </summary>
    private void DeserializeInternalUnchecked(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Clear existing layers
        ClearLayers();

        // Read format header (versioned). If absent, fall back to legacy format.
        int version = 1;
        int first = reader.ReadInt32();
        int layerCount;

        if (first == SerializationMagic)
        {
            version = reader.ReadInt32();
            layerCount = reader.ReadInt32();
        }
        else
        {
            layerCount = first;
        }

        // Read and recreate each layer
        for (int i = 0; i < layerCount; i++)
        {
            // Read layer type
            string layerType = reader.ReadString();

            // Read input shape
            int inputShapeLength = reader.ReadInt32();
            int[] inputShape = new int[inputShapeLength];
            for (int j = 0; j < inputShapeLength; j++)
            {
                inputShape[j] = reader.ReadInt32();
            }

            // Read output shape
            int outputShapeLength = reader.ReadInt32();
            int[] outputShape = new int[outputShapeLength];
            for (int j = 0; j < outputShapeLength; j++)
            {
                outputShape[j] = reader.ReadInt32();
            }

            Dictionary<string, object>? additionalParams = null;
            if (version >= 3)
            {
                int metadataCount = reader.ReadInt32();
                if (metadataCount > 0)
                {
                    additionalParams = new Dictionary<string, object>(metadataCount, StringComparer.Ordinal);
                    for (int m = 0; m < metadataCount; m++)
                    {
                        string key = reader.ReadString();
                        string value = reader.ReadString();
                        additionalParams[key] = value;
                    }
                }
            }

            // Read parameters
            int paramCount = reader.ReadInt32();
            Vector<T>? parametersVector = null;
            if (paramCount > 0)
            {
                parametersVector = new Vector<T>(paramCount);
                for (int j = 0; j < paramCount; j++)
                {
                    parametersVector[j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Legacy v2 stored metadata after parameters; read it now.
            if (version == 2)
            {
                int metadataCount = reader.ReadInt32();
                if (metadataCount > 0)
                {
                    additionalParams = new Dictionary<string, object>(metadataCount, StringComparer.Ordinal);
                    for (int m = 0; m < metadataCount; m++)
                    {
                        string key = reader.ReadString();
                        string value = reader.ReadString();
                        additionalParams[key] = value;
                    }
                }
            }

            Vector<T>? extraParametersVector = null;
            if (version >= 4)
            {
                int extraCount = reader.ReadInt32();
                if (extraCount > 0)
                {
                    extraParametersVector = new Vector<T>(extraCount);
                    for (int j = 0; j < extraCount; j++)
                    {
                        extraParametersVector[j] = NumOps.FromDouble(reader.ReadDouble());
                    }
                }
            }

            // Create the layer.
            var layer = DeserializationHelper.CreateLayerFromType<T>(layerType, inputShape, outputShape, additionalParams);

            // Lazy layers need their shape resolved before SetParameters can size sub-layer
            // weights. The serialized inputShape is concrete by definition when the source
            // model had run a forward pass; if a layer's saved shape contains -1 placeholders
            // and this is the network's first layer, fall back to the architecture's input
            // shape (the only authoritative concrete shape we have at this point).
            if (layer is LayerBase<T> lb && !lb.IsShapeResolved)
            {
                int[]? candidate = inputShape is { Length: > 0 } && inputShape.All(d => d > 0)
                    ? inputShape
                    : null;
                if (candidate is null && _layers.Count == 0)
                {
                    var archShape = Architecture?.GetInputShape();
                    if (archShape is { Length: > 0 } && archShape.All(d => d > 0))
                        candidate = archShape;
                }
                if (candidate is { Length: > 0 })
                {
                    try { lb.ResolveFromShape(candidate); }
                    catch (ArgumentException) { /* layer rejects this shape; leave lazy */ }
                }
            }

            // Apply parameters if any
            if (parametersVector != null)
            {
                layer.SetParameters(parametersVector);
            }

            if (extraParametersVector != null && layer is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> extrasLayer)
            {
                extrasLayer.SetExtraParameters(extraParametersVector);
            }

            // Add the layer to the network
            _layers.Add(layer);
        }

        // Invalidate caches after loading all layers
        // (InvalidateParameterCountCache already calls InvalidateLayerInfoCache)
        InvalidateParameterCountCache();

        // Deserialized models should be in inference mode by default.
        // This ensures BatchNorm uses running statistics (not batch statistics)
        // and dropout is disabled, matching the behavior of the original model.
        SetTrainingMode(false);

        // Read network-specific data
        DeserializeNetworkSpecificData(reader);
    }

    /// <summary>
    /// Serializes network-specific data that is not covered by the general serialization process.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method is called at the end of the general serialization process to allow derived classes
    /// to write any additional data specific to their implementation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as packing a special compartment in your suitcase. 
    /// While the main serialization method packs the common items (layers, parameters), 
    /// this method allows each specific type of neural network to pack its own unique items 
    /// that other networks might not have.
    /// </para>
    /// </remarks>
    protected abstract void SerializeNetworkSpecificData(BinaryWriter writer);

    /// <summary>
    /// Deserializes network-specific data that was not covered by the general deserialization process.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method is called at the end of the general deserialization process to allow derived classes
    /// to read any additional data specific to their implementation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Continuing the suitcase analogy, this is like unpacking that special 
    /// compartment. After the main deserialization method has unpacked the common items (layers, parameters), 
    /// this method allows each specific type of neural network to unpack its own unique items 
    /// that were stored during serialization.
    /// </para>
    /// </remarks>
    protected abstract void DeserializeNetworkSpecificData(BinaryReader reader);

    /// <summary>
    /// Creates a new neural network with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use for the new network.</param>
    /// <returns>A new neural network with the specified parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new neural network that is a copy of this one, but with different parameter values.
    /// It's useful for creating variations of a model without retraining or for ensemble methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as creating a copy of your neural network but with different
    /// internal settings. It's like having the same blueprint for a house but using different materials.
    /// 
    /// This is useful when you want to:
    /// - Try different variations of a trained model
    /// - Create an ensemble of similar models with different parameters
    /// - Manually adjust model parameters without retraining
    /// 
    /// The new model will have the same structure but different parameter values.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        // In-place update — issue #1221: DeepCopy + UpdateParameters loses
        // gradients for lazy layers because deserialization resets them to
        // placeholder state where ParameterCount=0, so UpdateParameters
        // skips them.
        UpdateParameters(parameters);
        // The in-place write keeps every weight ARRAY's identity — and the
        // CPU engine's inference fast paths cache derived weight forms
        // (pre-packed GEMM panels, transposed conv kernels) keyed by that
        // identity without re-reading contents. Flush them, or the next
        // Predict computes with the PRE-update weights (root cause of the
        // Issue1296 grad-accum probe failure: a WithParameters'd model with
        // bit-identical parameters predicted differently because its MHA /
        // FFN GEMMs consumed packs built from the pre-load random init).
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
        return this;
    }

    /// <summary>
    /// Gets the indices of input features that are actively used by the network.
    /// </summary>
    /// <returns>A collection of indices representing the active features.</returns>
    /// <remarks>
    /// <para>
    /// This method determines which input features have the most influence on the network's output
    /// by analyzing the weights of the first layer. Features with larger absolute weights are
    /// considered more active or important.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This helps you understand which parts of your input data the network
    /// considers most important for making predictions.
    /// 
    /// For example, if your inputs are:
    /// - Age (index 0)
    /// - Income (index 1)
    /// - Education level (index 2)
    /// 
    /// And this method returns [0, 2], it means the network relies heavily on age and education level,
    /// but not so much on income when making its predictions.
    /// 
    /// This can help you:
    /// - Understand what your model is paying attention to
    /// - Potentially simplify your model by removing unused features
    /// - Gain insights about the problem you're solving
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // If the network has no layers, return an empty list
        if (Layers.Count == 0)
            return Array.Empty<int>();

        // Get the first layer for analysis
        var firstLayer = Layers[0];
        var inputShape = firstLayer.GetInputShape();

        // Layers with lazily-resolved input shapes report -1 / 0 / null until the
        // first forward pass. Return an empty set rather than crashing inside
        // Enumerable.Range(0, count) — callers treat the result as "no features
        // can be analyzed yet" which is the right semantic for a pre-warmup model.
        if (inputShape == null || inputShape.Length == 0 || inputShape[0] <= 0)
            return Array.Empty<int>();

        // If the first layer is not a dense or convolutional layer, we can't easily determine active features
        if (firstLayer is not (DenseLayer<T> or ConvolutionalLayer<T>))
        {
            // Return all indices as potentially active (conservative approach)
            return Enumerable.Range(0, inputShape[0]);
        }

        // Get the weights from the first layer
        Vector<T> weights = firstLayer.GetParameters();
        int inputSize = inputShape[0];
        var outputShape = firstLayer.GetOutputShape();
        if (outputShape == null || outputShape.Length == 0 || outputShape[0] <= 0)
            return Array.Empty<int>();
        int outputSize = outputShape[0];

        // Calculate feature importance by summing absolute weights per input feature
        var featureImportance = new Dictionary<int, T>();

        for (int i = 0; i < inputSize; i++)
        {
            T importance = NumOps.Zero;

            // For each neuron in the first layer, add the absolute weight for this feature
            for (int j = 0; j < outputSize; j++)
            {
                // In most layers, weights are organized as [input1-neuron1, input2-neuron1, ..., input1-neuron2, ...]
                int weightIndex = j * inputSize + i;

                if (weightIndex < weights.Length)
                {
                    importance = NumOps.Add(importance, NumOps.Abs(weights[weightIndex]));
                }
            }

            featureImportance[i] = importance;
        }

        // Sort features by importance and get the top 50% (or at least 1)
        int featuresCount = Math.Max(1, inputSize / 2);

        return featureImportance
            .OrderByDescending(pair => Convert.ToDouble(pair.Value))
            .Take(featuresCount)
            .Select(pair => pair.Key);
    }

    /// <summary>
    /// Determines if a specific input feature is actively used by the network.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is actively used; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if a specific input feature has a significant influence on the network's
    /// output based on the weights in the first layer. A feature is considered used if its
    /// associated weights have non-negligible magnitudes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you whether a specific piece of your input data matters
    /// to the neural network's decisions.
    /// 
    /// For example, if your inputs include age, income, and education level, this method can
    /// tell you whether the network is actually using age (or any other specific feature) when
    /// making predictions.
    /// 
    /// This is useful for:
    /// - Understanding what information your model uses
    /// - Simplifying your inputs by removing unused features
    /// - Debugging models that ignore features you think should be important
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        // If feature index is explicitly set as active, return true immediately
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Contains(featureIndex))
        {
            return true;
        }

        // If explicitly set active features exist but don't include this index, it's not used
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return false;
        }

        // If feature index is out of range, it's not used. Guard the input-shape
        // read the same way GetActiveFeatureIndices does: pre-warmup lazy layers
        // report a null/empty/-1 shape, and indexing [0] on that would crash
        // instead of returning the safe default.
        if (Layers.Count == 0 || featureIndex < 0)
            return false;
        int[] inputShape = Layers[0].GetInputShape();
        if (inputShape == null || inputShape.Length == 0 || inputShape[0] <= 0)
            return false;
        if (featureIndex >= inputShape[0])
            return false;

        // Get active feature indices
        var activeIndices = GetActiveFeatureIndices().ToList();

        // Check if the specified index is in the active indices
        return activeIndices.Contains(featureIndex);
    }

    /// <summary>
    /// Creates a deep copy of the neural network.
    /// </summary>
    /// <returns>A new instance that is a deep copy of this neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete independent copy of the network, including all layers
    /// and their parameters. It uses serialization and deserialization to ensure a true deep copy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a completely independent duplicate of your neural network.
    /// 
    /// Think of it like creating an exact clone of your network where:
    /// - The copy has the same structure (layers, connections)
    /// - The copy has the same learned parameters (weights, biases)
    /// - Changes to one network don't affect the other
    /// 
    /// This is useful when you want to:
    /// - Experiment with modifications without risking your original network
    /// - Create multiple variations of a model
    /// - Save a snapshot of your model at a particular point in training
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // DeepCopy is a training-internal in-memory clone, not a user-facing
        // save/load. We route through the private SerializeInternalUnchecked /
        // DeserializeInternalUnchecked helpers rather than the public virtual
        // Serialize / Deserialize APIs. This has two consequences:
        //
        //   1. The ModelPersistenceGuard trial check (which lives on the
        //      public Serialize/Deserialize entry points) is not invoked,
        //      so in-memory clones during training do not consume trial
        //      operations — consistent with the guard's own XML doc
        //      ("Training and inference are never restricted.").
        //
        //   2. A subclass that overrides public Serialize() (whether to
        //      customise the format or, hypothetically, to exfiltrate bytes)
        //      cannot intercept this path. The private helpers are non-
        //      virtual so DeepCopy's serialization contract is locked to the
        //      base implementation.
        //
        // If you need to customise how DeepCopy behaves, override DeepCopy
        // itself — that's the correct extension point.

        // Large-model fast path: when the serialized byte[] would exceed the
        // CLR array limit (~2 GB), skip the serialize/deserialize roundtrip
        // and copy parameters layer-by-layer into a freshly-constructed
        // instance. Foundational VLMs (ViLBERT, VisualBERT, UNITER) and
        // GPT-3-class transformers cross the 256M-params × 8 B = 2 GB line
        // and fail with "Array dimensions exceeded supported range" when
        // MemoryStream tries to grow past MaxArrayLength. The direct path
        // is also strictly faster — no 2 GB allocate/free cycle, no
        // redundant double-conversion loop — so we also take it whenever
        // the new instance has an identically-shaped Layers list.
        long paramBytes = 0;
        bool hasCustomLayer = false;
        for (int i = 0; i < _layers.Count; i++)
        {
            paramBytes += (long)_layers[i].ParameterCount * sizeof(double);
            if (_layers[i] is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> ext)
                paramBytes += (long)ext.ExtraParameterCount * sizeof(double);

            // Detect layers whose concrete type lives outside AiDotNet — the
            // DeserializationHelper's hard-coded type registry only knows the
            // built-in layers, so a custom external layer would fail the
            // serialize/deserialize roundtrip path with NotSupportedException.
            // Force the per-layer copy path for these chains. (We can't
            // enumerate the registry directly from a static helper, so the
            // check is by assembly identity: anything outside the AiDotNet
            // core assembly is treated as custom.)
            var layerAssembly = _layers[i].GetType().Assembly;
            if (layerAssembly != typeof(NeuralNetworkBase<T>).Assembly)
                hasCustomLayer = true;
        }

        if (paramBytes > (long)MaxArrayLength || hasCustomLayer)
        {
            var largeCopy = CreateNewInstance();
            if (largeCopy is NeuralNetworkBase<T> largeBase && largeBase._layers.Count == _layers.Count)
            {
                // A DeepCopy clone is a transient in-memory copy whose weights are materialized
                // resident below. Suppress its independent weight-streaming auto-enable: the
                // process-wide WeightRegistry / streaming pool is a singleton that cannot host two
                // streaming models at once (original + clone). Without this, the clone's first
                // SetTrainingMode / forward re-Configures the singleton and throws "existing streaming
                // pool has entries" / "unknown handle". Must run BEFORE any copy/SetTrainingMode below.
                largeBase.DisableAutoStreaming();

                // Copy layer-by-layer parameters + serialization extras + network-specific data.
                for (int i = 0; i < _layers.Count; i++)
                {
                    var srcLayer = _layers[i];
                    var dstLayer = largeBase._layers[i];

                    // The freshly-constructed destination may carry lazy layers whose
                    // weight tensors aren't materialized yet (ParameterCount == 0 until
                    // a forward pass resolves their shape). The source layer has already
                    // run a forward (its ParameterCount is concrete), so resolve the
                    // destination from the source's input shape before copying — otherwise
                    // the ParameterCount guard below skips the copy and the clone keeps the
                    // destination's random-initialized weights, diverging from the original.
                    if (dstLayer is LayerBase<T> dstLazy && !dstLazy.IsShapeResolved
                        && srcLayer.ParameterCount > 0)
                    {
                        int[] srcInputShape = srcLayer.GetInputShape();
                        if (srcInputShape is { Length: > 0 } && Array.TrueForAll(srcInputShape, d => d > 0))
                        {
                            try { dstLazy.ResolveFromShape(srcInputShape); }
                            catch (ArgumentException) { /* layer rejects this shape; leave lazy */ }
                        }
                    }

                    if (srcLayer.ParameterCount > 0 && dstLayer.ParameterCount == srcLayer.ParameterCount)
                    {
                        dstLayer.SetParameters(srcLayer.GetParameters());
                    }
                    if (srcLayer is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> srcExtras
                        && dstLayer is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> dstExtras
                        && srcExtras.ExtraParameterCount == dstExtras.ExtraParameterCount
                        && srcExtras.ExtraParameterCount > 0)
                    {
                        dstExtras.SetExtraParameters(srcExtras.GetExtraParameters());
                    }
                }
                largeBase.InvalidateParameterCountCache();
                largeBase.SetTrainingMode(false);
                return largeCopy;
            }
        }

        byte[] serialized = SerializeInternalUnchecked();
        var copy = CreateNewInstance();
        if (copy is NeuralNetworkBase<T> copyBase)
        {
            // Suppress the clone's independent weight-streaming auto-enable BEFORE deserialize (whose
            // trailing SetTrainingMode would otherwise re-Configure the process-wide singleton streaming
            // pool already owned by the original, throwing "existing streaming pool has entries" /
            // "unknown handle"). A transient in-memory clone keeps its weights resident — see the
            // matching guard in the large/custom-layer copy path above.
            copyBase.DisableAutoStreaming();
            copyBase.DeserializeInternalUnchecked(serialized);
        }
        else
        {
            // Fallback for copies that aren't NeuralNetworkBase<T> — should
            // not happen given CreateNewInstance contract, but keep the call
            // graph safe. Use InternalOperation so the public Deserialize's
            // guard is suppressed (equivalent pattern to SaveModel/LoadModel).
            using (ModelPersistenceGuard.InternalOperation())
            {
                copy.Deserialize(serialized);
            }
        }
        return copy;
    }

    /// <summary>
    /// Creates a clone of the neural network.
    /// </summary>
    /// <returns>A new instance that is a clone of this neural network.</returns>
    /// <remarks>
    /// <para>
    /// For most neural networks, Clone and DeepCopy perform the same function - creating a complete
    /// independent copy of the network. Some specialized networks might implement this differently.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates an identical copy of your neural network.
    /// 
    /// In most cases, this works the same as DeepCopy and creates a completely independent
    /// duplicate of your network. The duplicate will have the same structure and the same
    /// learned parameters, but changes to one won't affect the other.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        // By default, Clone behaves the same as DeepCopy
        return DeepCopy();
    }

    /// <summary>
    /// Creates a new instance of the same type as this neural network.
    /// </summary>
    /// <returns>A new instance of the same neural network type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a blank version of the same type of neural network.
    /// 
    /// It's used internally by methods like DeepCopy and Clone to create the right type of
    /// network before copying the data into it.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance();

    /// <summary>
    /// Sets which input features should be considered active in the neural network.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative or exceeds the input dimension.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which input features should be considered active
    /// in the neural network, overriding the automatic determination based on weights.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of their weights in the network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method lets you manually select which parts of your input data
    /// the neural network should pay attention to. For example, if your inputs include various
    /// measurements or features, you can tell the network to focus only on specific ones
    /// that you know are important based on your domain knowledge.
    ///
    /// This can be useful for:
    /// - Forcing the network to use features you know are important
    /// - Ignoring features you know are irrelevant or noisy
    /// - Testing how the network performs with different feature subsets
    /// - Implementing feature selection techniques
    /// </para>
    /// </remarks>
    /// <summary>
    /// Serializes the model for embedding in <c>ModelMetadata.ModelData</c>, degrading to an EMPTY payload
    /// when model persistence is not licensed. GetModelMetadata() is a descriptive/inspection call invoked
    /// by <c>AiModelResult</c>'s constructor on every facade build — before this guard, the embedded
    /// <c>this.Serialize()</c> made BUILDING any neural model through the facade throw
    /// <see cref="AiDotNet.Exceptions.LicenseRequiredException"/> without a license, contradicting the
    /// stated policy that training and inference remain fully available. Explicit saves
    /// (<c>SaveModel</c>/<c>Serialize</c>) still enforce the license as before.
    /// </summary>
    /// <summary>
    /// Aligns a training target's shape to the network output before the loss computation.
    /// Reshapes the TARGET (a leaf tensor not on the tape) rather than the output (which IS on the
    /// tape — see the gradient-chain note in TrainWithTape). Handles, in order:
    /// (1) leading singleton batch dims (legacy policy: [1, ...] vs [...]), and
    /// (2) rank-off-by-one trailing singletons — the industry-standard rank-1 regression target
    ///     ([B] vs output [B, 1], or [B, S] vs [B, S, 1], and the symmetric [B, 1] target vs [B]
    ///     output). Without this, regression losses crash in TensorSubtract on the most common
    ///     target shape every other ML stack accepts (sklearn y=(n,), PyTorch broadcasting).
    /// Only fires when element counts match and the extra dimension is exactly 1, so integer
    /// class-index targets against [B, C&gt;1] logits are never touched (the loss one-hots those).
    /// </summary>
    private Tensor<T> AlignTargetToOutputShape(Tensor<T> output, Tensor<T> expected)
    {
        if (output.Rank == expected.Rank || output.Length != expected.Length)
        {
            return expected;
        }

        // Legacy leading-singleton policy.
        if (output.Rank > expected.Rank && output.Shape[0] == 1)
        {
            return Engine.Reshape(expected, output._shape);
        }

        if (expected.Rank > output.Rank && expected.Shape[0] == 1)
        {
            return Engine.Reshape(expected, output._shape);
        }

        // Rank-off-by-one trailing singleton: [B] target vs [B,1] output (or [B,1] vs [B]).
        if (output.Rank == expected.Rank + 1 && output.Shape[output.Rank - 1] == 1)
        {
            return Engine.Reshape(expected, output._shape);
        }

        if (expected.Rank == output.Rank + 1 && expected.Shape[expected.Rank - 1] == 1)
        {
            return Engine.Reshape(expected, output._shape);
        }

        return expected;
    }

    protected byte[] SerializeForMetadata()
    {
        try
        {
            return this.Serialize();
        }
        catch (AiDotNet.Exceptions.LicenseRequiredException)
        {
            return [];
        }
    }

    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Sequence models with EmbeddingLayer don't support feature selection.
        // Their input shape is [1] (single token ID), not a feature vector.
        // Fixes #1113.
        //
        // Models with a non-flat (multi-dimensional) first-layer input shape — CNN
        // (depth×H×W), RNN/GRU/LSTM (seqLen×features), Vision Transformer, etc. — are the
        // same case generalized: "select a subset of input features" is a tabular concept
        // that doesn't apply to spatial/sequence input, and the optimizer feeds flat indices
        // (0..flatSize) that exceed the first layer's input-shape axis 0. Treat as a no-op
        // rather than validating those indices against GetInputShape()[0]. Fixes #1468.
        if (Layers.Count > 0
            && (Layers[0] is Layers.EmbeddingLayer<T> || Layers[0].GetInputShape() is { Length: > 1 }))
        {
            // Clear any stale feature mask so IsFeatureUsed() doesn't
            // answer from a previous dense-feature configuration.
            _explicitlySetActiveFeatures?.Clear();
            return;
        }

        // Initialize the hash set if it doesn't exist
        _explicitlySetActiveFeatures ??= new HashSet<int>();

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Get the input dimension to validate feature indices
        int inputDimension = 0;
        if (Layers.Count > 0)
        {
            inputDimension = Layers[0].GetInputShape()[0];
        }

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} cannot be negative.");
            }

            if (inputDimension > 0 && index >= inputDimension)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} exceeds the input dimension {inputDimension}.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }
    }

    #region IInterpretableModel Implementation

    // Suppress CS0618 (obsolete) warnings for legacy interface implementations that call deprecated helper overloads.
    // The interface methods maintain backwards compatibility while the helper exposes new overloads with required background data.
#pragma warning disable CS0618

    /// <summary>
    /// Set of interpretation methods that are enabled for this neural network model.
    /// Controls which interpretability features (SHAP, LIME, etc.) are available.
    /// </summary>
    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();

    /// <summary>
    /// Indices of features considered sensitive for fairness analysis.
    /// </summary>
    protected Vector<int> _sensitiveFeatures;

    /// <summary>
    /// List of fairness metrics to evaluate for this model.
    /// </summary>
    protected readonly List<FairnessMetric> _fairnessMetrics = new();

    /// <summary>
    /// Base model instance for interpretability delegation.
    /// </summary>
    /// <remarks>
    /// Typed as <see cref="IFullModel{T, TInput, TOutput}"/> to maintain type safety while supporting
    /// the interpretability infrastructure. This field stores models that implement the full model interface,
    /// which includes training, prediction, serialization, and parameterization capabilities.
    /// </remarks>
    protected IFullModel<T, Tensor<T>, Tensor<T>>? _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods, inputs);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync(this, _enabledMethods, input, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync(this, _enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync(this, _enabledMethods, input, desiredOutput, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feature interactions occur when the effect of one feature depends on the value
    /// of another feature. For example, in a house price model, the effect of "number of bathrooms" might
    /// depend on "house size" - adding a bathroom to a large house has a different effect than adding one
    /// to a small house.
    /// </para>
    /// <para>
    /// This method computes the H-statistic, which measures interaction strength from 0 (no interaction)
    /// to 1 (complete dependence).
    /// </para>
    /// </remarks>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync(this, _enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fairness metrics help identify if your model treats different groups of people
    /// fairly. For example, if you have a loan approval model, you want to ensure it doesn't discriminate
    /// based on gender, race, or other sensitive attributes.
    /// </para>
    /// <para>
    /// Key metrics computed include:
    /// - Demographic Parity: Are positive predictions equally distributed across groups?
    /// - Disparate Impact: Ratio of positive prediction rates between groups (should be close to 1)
    /// </para>
    /// </remarks>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync(this, inputs, sensitiveFeatureIndex, _fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(this, _enabledMethods, input, threshold);
    }

    /// <summary>
    /// Gets Integrated Gradients attributions for a neural network prediction.
    /// </summary>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="numSteps">Number of integration steps (default: 50).</param>
    /// <returns>Integrated Gradients explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Integrated Gradients is a theoretically-grounded method
    /// that satisfies completeness (attributions sum to prediction - baseline_prediction)
    /// and sensitivity (important features get non-zero attributions).
    /// </para>
    /// </remarks>
    public virtual async Task<IntegratedGradientsExplanation<T>> GetIntegratedGradientsAsync(
        Tensor<T> input,
        Tensor<T>? baseline = null,
        int numSteps = 50)
    {
        // Use backprop-based version for efficient gradient computation
        return await InterpretableModelHelper.GetIntegratedGradientsWithBackpropAsync(this, _enabledMethods, input, baseline, numSteps);
    }

    /// <summary>
    /// Gets DeepLIFT attributions for a neural network prediction.
    /// </summary>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="useRevealCancel">Use RevealCancel rule instead of Rescale.</param>
    /// <returns>DeepLIFT explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepLIFT compares activations to a reference baseline.
    /// It's faster than Integrated Gradients and handles non-linearities better.
    /// </para>
    /// </remarks>
    public virtual async Task<DeepLIFTExplanation<T>> GetDeepLIFTAsync(
        Tensor<T> input,
        Tensor<T>? baseline = null,
        bool useRevealCancel = false)
    {
        var rule = useRevealCancel ? DeepLIFTRule.RevealCancel : DeepLIFTRule.Rescale;
        // Use backprop-based version for efficient gradient computation
        return await InterpretableModelHelper.GetDeepLIFTWithBackpropAsync(this, _enabledMethods, input, baseline, rule);
    }

    /// <summary>
    /// Gets GradCAM visual explanation for a CNN prediction.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="targetClass">Target class to explain (-1 for predicted class).</param>
    /// <returns>GradCAM explanation with heatmap.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GradCAM creates visual heatmaps showing which parts
    /// of an image were most important for the CNN's prediction.
    /// </para>
    /// </remarks>
    public virtual async Task<GradCAMExplanation<T>> GetGradCAMAsync(
        Tensor<T> input,
        int targetClass = -1)
    {
        // Get input shape from the tensor
        int[] inputShape = input._shape;

        // For GradCAM we need feature map shape, which depends on the network architecture
        // Default to a reasonable size; users can override with the helper method directly
        int[] featureMapShape = inputShape.Length >= 3
            ? new[] { inputShape[0], inputShape[1] / 4, inputShape[2] / 4, 64 }
            : new[] { 7, 7, 64 };

        return await InterpretableModelHelper.GetGradCAMAsync(
            this, _enabledMethods, input, inputShape, featureMapShape, targetClass);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    /// <param name="model">The model to use for interpretability analysis. Must implement IFullModel.</param>
    /// <exception cref="ArgumentNullException">Thrown when model is null.</exception>
    public virtual void SetBaseModel<TInput, TOutput>(IFullModel<T, TInput, TOutput> model)
    {
        Guard.NotNull(model);
        _baseModel = model as IFullModel<T, Tensor<T>, Tensor<T>>;
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        Guard.NotNull(sensitiveFeatures);
        _sensitiveFeatures = sensitiveFeatures;
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

#pragma warning restore CS0618

    #endregion

    #region INeuralNetworkModel Implementation

    /// <summary>
    /// Gets the intermediate activations from each layer when processing the given input with named keys.
    /// </summary>
    public virtual Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Gets the architectural structure of the neural network.
    /// </summary>
    public virtual NeuralNetworkArchitecture<T> GetArchitecture()
    {
        return Architecture;
    }

    #endregion

    /// <summary>
    /// Gets the feature importance scores for the model.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of each input feature by analyzing the weights
    /// in the first layer of the neural network. Features with larger absolute weights are
    /// considered more important to the model's predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you which parts of your input data are most important
    /// for the neural network's decisions.
    ///
    /// For example, if you're predicting house prices with features like size, location, and age,
    /// this method might tell you that "location" has an importance of 0.8, "size" has 0.6,
    /// and "age" has 0.2 - meaning the network relies heavily on location and size, but less on age.
    ///
    /// This is useful for:
    /// - Understanding what your model pays attention to
    /// - Explaining model decisions to others
    /// - Identifying which features matter most
    /// - Simplifying your model by removing unimportant features
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();

        // If the network has no layers, return an empty dictionary
        if (Layers.Count == 0)
            return importance;

        // Get the first layer for analysis
        var firstLayer = Layers[0];

        // If the first layer is not a dense or convolutional layer, we can't easily determine importance
        if (firstLayer is not (DenseLayer<T> or ConvolutionalLayer<T>))
        {
            // Return uniform importance for all features (conservative approach)
            int inputSize = firstLayer.GetInputShape()[0];
            T uniformImportance = NumOps.FromDouble(1.0 / inputSize);

            for (int i = 0; i < inputSize; i++)
            {
                importance[$"Feature_{i}"] = uniformImportance;
            }

            return importance;
        }

        // Get the weights from the first layer
        Vector<T> weights = firstLayer.GetParameters();
        int featureCount = firstLayer.GetInputShape()[0];
        int outputSize = firstLayer.GetOutputShape()[0];

        // Calculate feature importance by summing absolute weights per input feature
        var featureScores = new Dictionary<int, T>();

        for (int i = 0; i < featureCount; i++)
        {
            T score = NumOps.Zero;

            // For each neuron in the first layer, add the absolute weight for this feature
            for (int j = 0; j < outputSize; j++)
            {
                // In most layers, weights are organized as [input1-neuron1, input2-neuron1, ..., input1-neuron2, ...]
                int weightIndex = j * featureCount + i;

                if (weightIndex < weights.Length)
                {
                    score = NumOps.Add(score, NumOps.Abs(weights[weightIndex]));
                }
            }

            featureScores[i] = score;
        }

        // Normalize the scores to sum to 1
        T totalScore = featureScores.Values.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

        if (NumOps.GreaterThan(totalScore, NumOps.Zero))
        {
            foreach (var kvp in featureScores)
            {
                importance[$"Feature_{kvp.Key}"] = NumOps.Divide(kvp.Value, totalScore);
            }
        }
        else
        {
            // If all scores are zero, use uniform importance
            T uniformImportance = NumOps.FromDouble(1.0 / featureCount);
            for (int i = 0; i < featureCount; i++)
            {
                importance[$"Feature_{i}"] = uniformImportance;
            }
        }

        return importance;
    }

    /// <summary>
    /// Sets the parameters of the neural network.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to all layers in the network.
    /// The parameters should be in the same format as returned by GetParameters.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // ParameterCount is long; SetParameters takes a flat Vector<T> whose
        // Length is int. Guard at this boundary: if the model's true
        // parameter count exceeds int.MaxValue the caller can't even
        // construct a Vector<T> big enough to feed in, so report which
        // limit was hit clearly instead of silently truncating.
        long totalParameterCountLong = ParameterCount;
        if (totalParameterCountLong > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Model parameter count ({totalParameterCountLong:N0}) exceeds " +
                $"int32 capacity ({int.MaxValue:N0}); the flat-vector " +
                $"SetParameters path cannot accept a model this large. Walk " +
                $"Layers per-layer and call SetParameters on each, or split " +
                $"this architecture across multiple network instances.");
        }
        int totalParameterCount = (int)totalParameterCountLong;
        if (parameters.Length != totalParameterCount)
        {
            throw new ArgumentException($"Expected {totalParameterCount} parameters, got {parameters.Length}");
        }

        int currentIndex = 0;
        var srcSpan = parameters.AsSpan();
        foreach (var layer in Layers.Where(l => l.ParameterCount > 0))
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            // Bulk copy via Span instead of element-by-element
            var layerParameters = new Vector<T>((int)(layerParameterCount));
            srcSpan.Slice(currentIndex, layerParameterCount)
                .CopyTo(layerParameters.AsWritableSpan());
            layer.SetParameters(layerParameters);
            currentIndex += layerParameterCount;
        }

        // Some ITrainableLayer implementations swap their parameter tensors
        // wholesale during SetParameters rather than mutating in place. When
        // they do, any compiled plan captured against the prior tensor
        // references is stale — replay would write into freed buffers. Drop
        // BOTH the inference compile cache AND the tape-training caches that
        // also key off the captured tensor references (TapeTrainingStep's
        // collected-parameter cache + CompiledTapeTrainingStep's compiled
        // plans). Without invalidating the tape side, the next training step
        // would replay against parameters that no longer exist.
        _compileHost.Invalidate();
        Training.TapeTrainingStep<T>.InvalidateCache();
        // Layers that mutate their parameter tensors IN PLACE keep the same
        // weight-array identities — and both engines cache derived weight
        // forms keyed by exactly that identity. Single source of truth:
        // see InvalidateWeightCachesAfterSuccessfulWeightUpdate.
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    /// <summary>
    /// Adds a layer to the neural network.
    /// </summary>
    /// <param name="layerType">The type of layer to add.</param>
    /// <param name="units">The number of units/neurons in the layer.</param>
    /// <param name="activation">The activation function to use.</param>
    public virtual void AddLayer(LayerType layerType, int units, ActivationFunction activation)
    {
        // Get input size from previous layer or use units as default
        int inputSize = Layers.Count > 0 ? Layers[Layers.Count - 1].GetOutputShape()[0] : units;

        // Create activation function from enum
        var activationFunc = ActivationFunctionFactory<T>.CreateActivationFunction(activation);

        ILayer<T> layer = layerType switch
        {
            LayerType.Dense => new DenseLayer<T>(units, activationFunc),
            _ => throw new NotSupportedException($"Layer type {layerType} not supported in AddLayer method")
        };
        AddLayerToCollection(layer);
    }

    /// <summary>
    /// Adds a convolutional layer to the neural network.
    /// </summary>
    public virtual void AddConvolutionalLayer(int filters, int kernelSize, int stride, ActivationFunction activation)
    {
        throw new InvalidOperationException(
            "AddConvolutionalLayer requires additional parameters that are not provided in this method signature. " +
            "Use ConvolutionalLayer.Configure() with the full input shape, or create the layer directly with " +
            "new ConvolutionalLayer<T>(inputHeight, inputWidth, stride, padding, activation) " +
            "and add it to Layers manually.");
    }

    /// <summary>
    /// Adds an LSTM layer to the neural network.
    /// </summary>
    public virtual void AddLSTMLayer(int units, bool returnSequences = false)
    {
        throw new InvalidOperationException(
            "AddLSTMLayer requires additional parameters that are not provided in this method signature. " +
            "Create the layer directly with new LSTMLayer<T>(inputSize, hiddenSize, inputShape, activation, recurrentActivation) " +
            "and add it to Layers manually.");
    }

    /// <summary>
    /// Adds a dropout layer to the neural network.
    /// </summary>
    public virtual void AddDropoutLayer(double dropoutRate)
    {
        var layer = new DropoutLayer<T>(dropoutRate);
        AddLayerToCollection(layer);
    }

    /// <summary>
    /// Adds a batch normalization layer to the neural network.
    /// </summary>
    /// <param name="featureSize">The number of features to normalize.</param>
    /// <param name="epsilon">A small constant for numerical stability (default: 1e-5).</param>
    /// <param name="momentum">The momentum for running statistics (default: 0.9).</param>
    public virtual void AddBatchNormalizationLayer(int featureSize, double epsilon = 1e-5, double momentum = 0.9)
    {
        var layer = new BatchNormalizationLayer<T>();
        AddLayerToCollection(layer);
    }

    /// <summary>
    /// Adds a pooling layer to the neural network.
    /// </summary>
    /// <param name="inputShape">The input shape (channels, height, width).</param>
    /// <param name="poolingType">The type of pooling operation.</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window (default: same as poolSize).</param>
    public virtual void AddPoolingLayer(int[] inputShape, PoolingType poolingType, int poolSize, int? strides = null)
    {
        // inputShape is now ignored (lazy ctor resolves it on first forward); kept on the
        // signature for backwards compatibility with existing callers.
        var layer = new MaxPoolingLayer<T>(poolSize, strides ?? poolSize);
        AddLayerToCollection(layer);
    }

    /// <summary>
    /// Gets the gradients from all layers in the neural network.
    /// </summary>
    /// <returns>A vector containing all gradients from all layers concatenated together.</returns>
    /// <remarks>
    /// <para>
    /// This method collects the gradients from every layer in the network and combines them
    /// into a single vector. This is useful for optimization algorithms that need access to
    /// all gradients at once.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, each layer calculates how its parameters should change
    /// (the gradients). This method gathers all those gradients from every layer and puts them
    /// into one long list.
    ///
    /// Think of it like:
    /// - Each layer has notes about how to improve (gradients)
    /// - This method collects all those notes into one document
    /// - The optimizer can then use this document to update the entire network
    ///
    /// This is essential for the learning process, as it tells the optimizer how to adjust
    /// all the network's parameters to improve performance.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetGradients()
    {
        var allGradients = new List<T>();

        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            if (layerGradients != null && layerGradients.Length > 0)
            {
                for (int i = 0; i < layerGradients.Length; i++)
                {
                    allGradients.Add(layerGradients[i]);
                }
            }
        }

        return new Vector<T>(allGradients.ToArray());
    }

    /// <summary>
    /// Gets the input shape expected by the neural network.
    /// </summary>
    /// <returns>An array representing the dimensions of the input.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the shape of input data that the network expects. For example,
    /// if the network expects images of size 28x28 pixels, this might return [28, 28].
    /// If it expects a vector of 100 features, it would return [100].
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what size and shape of data the network needs as input.
    /// Think of it like knowing what size batteries a device needs - you need to provide the right
    /// dimensions of data for the network to work properly.
    /// </para>
    /// </remarks>
    public virtual int[] GetInputShape()
    {
        if (Layers.Count > 0)
        {
            return Layers[0].GetInputShape();
        }

        // Layer-only models with no registered layers can't yield a
        // shape — return empty rather than throwing, since this is a
        // query method.
        if (Architecture.IsLayerOnly) return Array.Empty<int>();
        return new[] { Architecture.InputSize };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        if (Layers.Count > 0)
        {
            return Layers[Layers.Count - 1].GetOutputShape();
        }

        return new[] { Architecture.OutputSize };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        // GetInputShape/GetOutputShape return per-sample shapes (no batch dimension).
        // Batch dimension is handled implicitly by the serving layer, not by the model shape.
        return DynamicShapeInfo.None;
    }

    /// <summary>
    /// Gets the activations (outputs) from each layer for a given input.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>A dictionary mapping layer index to layer activation tensors.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the network and captures the output of each layer.
    /// This is useful for visualizing what each layer is detecting, debugging the network, or
    /// implementing techniques like feature extraction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This shows you what each layer in your neural network "sees" or produces
    /// when given an input. It's like following a signal through a circuit and measuring the output
    /// at each component. This helps you understand what patterns each layer is detecting.
    ///
    /// For example, in an image recognition network:
    /// - Early layers might detect edges and simple shapes
    /// - Middle layers might detect parts of objects (like eyes or wheels)
    /// - Later layers might detect whole objects
    ///
    /// This method lets you see all of these intermediate representations.
    /// </para>
    /// </remarks>
    public virtual Dictionary<int, Tensor<T>> GetLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<int, Tensor<T>>();

        if (Layers.Count == 0)
        {
            return activations;
        }

        var currentInput = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var output = layer.Forward(currentInput);
            activations[i] = output;
            currentInput = output;
        }

        return activations;
    }

    /// <summary>
    /// Gets the default loss function for this network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A loss function measures how wrong the network's predictions are.
    /// This is used during training to guide learning.
    /// </para>
    /// </remarks>
    public virtual ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Computes a flattened gradient vector for all trainable parameters in the network.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="lossFunction">Optional override loss function (defaults to the model's configured loss).</param>
    /// <returns>A vector containing the concatenated gradients for all layer parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass under tape recording, computes the loss via
    /// <see cref="LossFunctions.LossFunctionBase{T}.ComputeTapeLoss"/>, runs reverse-mode
    /// autodiff, and concatenates per-parameter gradients into a single vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Gradients are the "direction to change weights" so the model makes fewer mistakes.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        using var tape = new GradientTape<T>();

        // Forward pass under tape recording (NOT Predict which uses NoGradScope).
        // Must happen BEFORE collecting trainable parameters — layers may
        // initialize or resize weights on their first forward pass.
        var prediction = ForwardForTraining(input);

        // Collect parameters AFTER forward so lazy-initialized layers are included
        var trainableParams = Training.TapeTrainingStep<T>.CollectParameters(Layers);
        if (trainableParams.Count == 0)
        {
            throw new InvalidOperationException(
                "No trainable parameters found. ComputeGradients requires at least one " +
                "layer implementing ITrainableLayer<T> with registered parameters.");
        }

        // Align rank-off-by-one regression targets ([B] vs [B,1]) BEFORE the loss: classification
        // index targets are handled inside the loss (EnsureTargetMatchesPredicted one-hots them),
        // but regression losses subtract tensors directly and crash on the industry-standard
        // rank-1 target shape. AlignTargetToOutputShape only fires on a trailing singleton, so
        // class-index targets against [B, C>1] logits are untouched.
        target = AlignTargetToOutputShape(prediction, target);

        // Compute loss via the user's configured loss function.
        // Integer-class → one-hot matching is handled inside each loss function's
        // ComputeTapeLoss via EnsureTargetMatchesPredicted.
        var resolved = lossFunction ?? LossFunction;
        Tensor<T> lossTensor;
        if (resolved is LossFunctions.LossFunctionBase<T> tapeLoss)
        {
            lossTensor = tapeLoss.ComputeTapeLoss(prediction, target);
        }
        else
        {
            // Fallback for custom ILossFunction: use CalculateDerivative to get
            // the loss gradient w.r.t. predictions, then backpropagate manually.
            // This preserves the IGradientComputable contract without silently
            // producing zero gradients from a disconnected scalar tensor.
            var predVec = prediction.ToVector();
            var targetVec = target.ToVector();
            var derivVec = resolved.CalculateDerivative(predVec, targetVec);
            lossTensor = new Tensor<T>(prediction._shape, derivVec);
        }

        // Reverse-mode AD: compute gradients across the FULL tape, then
        // filter to trainable parameters via reference-keyed lookup. The
        // earlier `tape.ComputeGradients(lossTensor, trainableParams)` form
        // passed sources directly, which silently dropped gradients when
        // the tape's backward walk could not match a trainable parameter
        // tensor reference through a view / GradFn chain (the parameter
        // pointer surfaced in the forward pass may differ from the
        // pointer the trainable-parameter walk hands in if any layer
        // wraps its weight in a buffer view, alias, or pooled
        // allocation). The result was finite-difference gradients of
        // magnitude ~5e-2 reported as analytic gradients of ~1e-4 by
        // ComputeGradients — mode-collapsing every gradient-based
        // optimizer (BuildAsync's AdamOptimizer.Optimize loop) on
        // Transformers and other lazy-allocation networks. Per-sample
        // model.Train via TrainWithTape was unaffected because its
        // backward path (NeuralNetworkBase.TrainWithTape line ~5303)
        // already used the compute-then-filter idiom for the same
        // reason — surfaced by ResNet's
        // GradientFlow_ShouldBeNonZeroAndFinite, then locked in here
        // for the IGradientComputable contract.
        // Pass the trainable-param set DIRECTLY to ComputeGradients as the
        // `sources` arg so the tape only computes gradients for those
        // tensors — avoids the previous compute-then-discard cost where
        // gradients accumulated for non-trainable tensors got dropped at
        // the post-hoc filter step (review #1364 C4nM4: filter at tape
        // construction, not after the full backward pass). Frozen /
        // detached tensors that the tape doesn't see still get zero-
        // padded in the flatten loop below to preserve length-alignment.
        var allGrads = tape.ComputeGradients(lossTensor, sources: trainableParams);
        var grads = allGrads;

        // Use GetParameterChunks to keep gradient/parameter ordering
        // aligned (fixes #1245 / #1232). Frozen-or-detached tensors that
        // tape didn't see are zero-padded to preserve length-alignment.
        // ParameterCount is long (#1244 widening) but List<T> ctor takes
        // int. Cap the capacity hint at int.MaxValue — flattening
        // gradients into a single managed list isn't viable past that
        // limit anyway (would need TB of RAM), so this matches the
        // implicit single-host inference contract. Saturating instead
        // of `checked((int)...)` keeps very-large models running with a
        // suboptimal capacity hint rather than crashing on construction.
        var flatGradients = new List<T>((int)Math.Min(ParameterCount, int.MaxValue));
        foreach (var paramTensor in GetParameterChunks())
        {
            if (paramTensor is null || paramTensor.Length == 0) continue;
            if (grads.TryGetValue(paramTensor, out var grad))
            {
                for (int i = 0; i < grad.Length; i++)
                    flatGradients.Add(grad[i]);
            }
            else
            {
                for (int i = 0; i < paramTensor.Length; i++)
                    flatGradients.Add(NumOps.Zero);
            }
        }
        return new Vector<T>(flatGradients.ToArray());
    }

    /// <summary>
    /// Applies a flattened gradient vector to update the network's parameters.
    /// </summary>
    /// <param name="gradients">The concatenated gradients for all parameters.</param>
    /// <param name="learningRate">The learning rate to scale updates.</param>
    /// <remarks>
    /// <para>
    /// This method slices the provided gradient vector per layer, updates each layer's parameters, and writes them back.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate controls how big each update step is. Smaller values are safer but slower.
    /// </para>
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients == null)
            throw new ArgumentNullException(nameof(gradients));

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length > gradients.Length)
                throw new ArgumentException($"Gradient vector is too short for layer parameters.");

            // Vectorized parameter update: params = params - learningRate * gradients
            var layerGradients = gradients.GetSubVector(offset, layerParams.Length);
            var scaledGradients = (Vector<T>)Engine.Multiply(layerGradients, learningRate);
            var updatedParams = (Vector<T>)Engine.Subtract(layerParams, scaledGradients);

            layer.SetParameters(updatedParams);
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write the model state to.</param>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));
        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read the model state from.</param>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(stream));
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        var data = ms.ToArray();
        if (data.Length == 0) throw new InvalidOperationException("Stream contains no data.");
        Deserialize(data);
    }

    /// <summary>
    /// Disposes resources used by the neural network.
    /// </summary>
    /// <remarks>
    /// Ensures that the mixed-precision context is properly disposed if it was enabled.
    /// </remarks>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Protected Dispose pattern implementation.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    /// <remarks>
    /// Cascades Dispose to every child layer that implements <see cref="IDisposable"/>.
    /// Without this, a networkwide <c>using</c> or explicit <c>Dispose()</c> call only
    /// tore down mixed-precision state and left each layer's pool-rented weight
    /// buffer (up to multi-GB for production-scale models like VGG16BN or DiT-XL)
    /// live until GC ran. Cascading lets DenseLayer/ConvolutionalLayer return their
    /// rented weight tensors to the <c>TensorAllocator</c> pool immediately — the
    /// main memory win from the Dispose path.
    /// </remarks>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Release compiled plans first so pooled tensor buffers the plans
            // captured are freed before layers Dispose and return their weights.
            _compileHost.Dispose();

            // Cascade Dispose into every layer that owns releasable state
            // (pool-rented weight tensors, GPU handles, native buffers).
            //
            // Shared-layer graphs can cause the same ILayer instance to
            // appear in multiple networks (or multiple times in one graph).
            // Route each layer through DisposeOnceGuard so the same instance
            // is disposed at most once process-wide, regardless of how many
            // owners cascade into it. Without the guard, non-idempotent
            // Dispose implementations (e.g., DenseLayer returning rented
            // tensors to TensorAllocator) would double-return pooled buffers.
            //
            // Single loop: _layers and Layers reference the same collection.
            // Guard against null for partially-constructed networks (ctor
            // threw before InitializeLayers).
            if (Layers is not null)
            {
                foreach (var layer in Layers)
                {
                    if (layer is IDisposable disposable)
                    {
                        AiDotNet.Helpers.DisposeOnceGuard.TryDispose(disposable);
                    }
                }
            }

            // Release activation pool / gradient checkpoint state before
            // mixed-precision teardown — the memory manager may hold pooled
            // buffers that mixed-precision teardown wants to recycle.
            DisableMemoryManagement();
            DisableMixedPrecision();

            // Cascade to child layers. Guard against null because Layers may not be
            // populated on a partially-constructed network (e.g., if a ctor threw
            // before InitializeLayers ran).
            if (Layers is not null)
            {
                foreach (var layer in Layers)
                {
                    if (layer is IDisposable disposable)
                    {
                        try
                        {
                            disposable.Dispose();
                        }
                        catch (ObjectDisposedException)
                        {
                            // A layer shared between networks may have been disposed
                            // already — not a bug, don't let it abort the cascade.
                        }
                    }
                }
            }
        }
    }

    #region ILayeredModel<T> Implementation

    /// <summary>
    /// Gets the ordered list of layers in this model (explicit interface implementation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This provides read-only access to the model's layers
    /// for tools that need to inspect the model structure without modifying it.</para>
    /// </remarks>
    IReadOnlyList<ILayer<T>> ILayeredModel<T>.Layers => _layers.AsReadOnly();

    /// <summary>
    /// Gets metadata for a specific layer including its parameter offset
    /// within the flat parameter vector.
    /// </summary>
    /// <param name="layerIndex">Zero-based index of the layer.</param>
    /// <returns>Metadata about the layer at the specified index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="layerIndex"/> is negative or greater than or equal to <see cref="LayerCount"/>.
    /// </exception>
    public LayerInfo<T> GetLayerInfo(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index {layerIndex} is out of range. Valid range: 0 to {_layers.Count - 1}.");
        }

        // Use cached layer info to avoid O(n) offset computation per call
        var allInfo = GetAllLayerInfo();
        return allInfo[layerIndex];
    }

    /// <summary>
    /// Cached layer info list, invalidated when layers change.
    /// </summary>
    private IReadOnlyList<LayerInfo<T>>? _cachedLayerInfo;
    private int _cachedLayerCount = -1;

    /// <summary>
    /// Gets metadata for all layers, including parameter offsets, types,
    /// shapes, names, and cost estimates.
    /// </summary>
    /// <returns>An ordered list of layer metadata.</returns>
    public IReadOnlyList<LayerInfo<T>> GetAllLayerInfo()
    {
        // Return cached result if layer count hasn't changed
        if (_cachedLayerInfo is not null && _cachedLayerCount == _layers.Count)
        {
            return _cachedLayerInfo;
        }

        var result = new List<LayerInfo<T>>(_layers.Count);
        int parameterOffset = 0;

        for (int i = 0; i < _layers.Count; i++)
        {
            var layer = _layers[i];
            var layerBase = layer as LayerBase<T>;

            result.Add(new LayerInfo<T>
            {
                Index = i,
                Name = layer.LayerName,
                Category = layerBase?.GetLayerCategory() ?? LayerCategory.Other,
                Layer = layer,
                ParameterOffset = parameterOffset,
                ParameterCount = (int)layer.ParameterCount,
                InputShape = layer.GetInputShape(),
                OutputShape = layer.GetOutputShape(),
                IsTrainable = layer.SupportsTraining && layer.ParameterCount > 0,
                EstimatedFlops = layerBase?.EstimateFlops() ?? 2L * layer.ParameterCount,
                EstimatedActivationMemory = layerBase?.EstimateActivationMemory() ?? 0L,
            });

            parameterOffset += (int)layer.ParameterCount;
        }

        _cachedLayerInfo = result.AsReadOnly();
        _cachedLayerCount = _layers.Count;
        return _cachedLayerInfo;
    }

    /// <summary>
    /// Validates that a partition point between layers is valid for pipeline parallelism.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When splitting a neural network across multiple GPUs,
    /// each GPU handles a different set of layers. The split must happen at a point where
    /// the output of one group of layers is compatible with the input of the next group.</para>
    ///
    /// <para>This method checks that the output shape of the layer at <paramref name="afterLayerIndex"/>
    /// matches the input shape of the next layer, ensuring a valid split point.</para>
    /// </remarks>
    /// <param name="afterLayerIndex">The index of the layer after which to partition.
    /// Must be between 0 and <see cref="LayerCount"/> - 2.</param>
    /// <returns>True if the partition point is valid; false otherwise.</returns>
    public bool ValidatePartitionPoint(int afterLayerIndex)
    {
        if (afterLayerIndex < 0 || afterLayerIndex >= _layers.Count - 1)
        {
            return false;
        }

        var currentLayer = _layers[afterLayerIndex];
        var nextLayer = _layers[afterLayerIndex + 1];

        var outputShape = currentLayer.GetOutputShape();
        var inputShape = nextLayer.GetInputShape();

        // Shapes are compatible if they have the same number of dimensions
        // and each dimension matches
        if (outputShape.Length != inputShape.Length)
        {
            return false;
        }

        for (int i = 0; i < outputShape.Length; i++)
        {
            if (outputShape[i] != inputShape[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Extracts a contiguous sub-model from <paramref name="startLayer"/> to
    /// <paramref name="endLayer"/> (inclusive).
    /// </summary>
    /// <param name="startLayer">Zero-based index of the first layer to include.</param>
    /// <param name="endLayer">Zero-based index of the last layer to include (inclusive).</param>
    /// <returns>A sub-model containing the specified layer range with metadata.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when layer indices are out of range or startLayer > endLayer.
    /// </exception>
    public SubModel<T> ExtractSubModel(int startLayer, int endLayer)
    {
        if (startLayer < 0 || startLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(startLayer),
                $"Start layer index {startLayer} is out of range. Valid range: 0 to {_layers.Count - 1}.");
        }
        if (endLayer < 0 || endLayer >= _layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(endLayer),
                $"End layer index {endLayer} is out of range. Valid range: 0 to {_layers.Count - 1}.");
        }
        if (startLayer > endLayer)
        {
            throw new ArgumentOutOfRangeException(nameof(startLayer),
                $"Start layer index {startLayer} cannot be greater than end layer index {endLayer}.");
        }

        int count = endLayer - startLayer + 1;
        var subLayers = new List<ILayer<T>>(count);
        var subInfos = new List<LayerInfo<T>>(count);

        int localOffset = 0;
        for (int i = startLayer; i <= endLayer; i++)
        {
            var layer = _layers[i];
            var layerBase = layer as LayerBase<T>;

            subLayers.Add(layer);
            subInfos.Add(new LayerInfo<T>
            {
                Index = i - startLayer,
                Name = layer.LayerName,
                Category = layerBase?.GetLayerCategory() ?? LayerCategory.Other,
                Layer = layer,
                ParameterOffset = localOffset,
                ParameterCount = (int)layer.ParameterCount,
                InputShape = layer.GetInputShape(),
                OutputShape = layer.GetOutputShape(),
                IsTrainable = layer.SupportsTraining && layer.ParameterCount > 0,
                EstimatedFlops = layerBase?.EstimateFlops() ?? 2L * layer.ParameterCount,
                EstimatedActivationMemory = layerBase?.EstimateActivationMemory() ?? 0L,
            });

            localOffset += (int)layer.ParameterCount;
        }

        return new SubModel<T>(subLayers, subInfos, startLayer, endLayer);
    }

    #endregion

}

