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
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

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
    /// The maximum allowed norm for gradients during training.
    /// </summary>
    protected T MaxGradNorm;

    /// <summary>
    /// Cached parameter count to avoid repeated Sum() calculations.
    /// Null when invalid (layers modified).
    /// </summary>
    private int? _cachedParameterCount;

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
            gradients[i] = ClipTensorGradient(gradients[i], MaxGradNorm);
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
        return ClipTensorGradient(gradient, MaxGradNorm);
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
        return ClipTensorGradient(Tensor<T>.FromVector(gradient), MaxGradNorm).ToVector();
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

        // Single pass: ResolveLazyLayerShapes above has materialized
        // every lazy layer reachable from the architecture, so each
        // layer's GetParameters() now returns its concrete vector — no
        // need to two-pass to "trigger lazy init via the first pass".
        int totalParameterCount = 0;
        foreach (var layer in Layers)
        {
            totalParameterCount += layer.GetParameters().Length;
        }

        var parameters = new Vector<T>((int)(totalParameterCount));
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
            using var gpuResult = ForwardGpu(input);
            result = gpuResult;
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

                // Accumulate as long so the addition itself doesn't overflow
                // under the checked Linq Sum overload introduced in .NET 7+.
                // If the total exceeds int.MaxValue we MUST fail fast: the
                // rest of the flat-parameter int-indexed API
                // (GetParameters / SetParameters / GetAllLayerInfo) cannot
                // represent that many elements as a single Vector<T>, so
                // returning a saturated int here would silently mis-slice
                // the parameter buffer downstream. Callers hitting this
                // limit either need to walk Layers themselves and stay in
                // long arithmetic, or split the model across instances.
                long total = 0L;
                for (int i = 0; i < Layers.Count; i++)
                {
                    total += Layers[i].ParameterCount;
                }
                if (total > int.MaxValue)
                {
                    throw new InvalidOperationException(
                        $"Total parameter count ({total:N0}) exceeds int32 capacity " +
                        $"({int.MaxValue:N0}). The flat-parameter API in NeuralNetworkBase " +
                        $"cannot represent this many parameters as a single Vector<T>. Walk " +
                        $"Layers per-layer and accumulate in long, or split this " +
                        $"architecture across multiple network instances.");
                }
                _cachedParameterCount = (int)total;
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

        // Check input layer
        if (!IsValidInputLayer(layers[0]))
        {
            errors.Add("The first layer must be a valid input layer.");
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
                    errors.Add($"Layer {i - 1} is not compatible with Layer {i}.");
                }
            }
        }

        // Check output layer
        if (!IsValidOutputLayer(layers[layers.Count - 1]))
        {
            errors.Add("The last layer must be a valid output layer.");
        }

        // Throw exception if any errors were found
        if (errors.Count > 0)
        {
            throw new ArgumentException($"Invalid layer configuration:\n{string.Join("\n", errors)}");
        }
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

        // For some specific tasks, the output might be from other layer types
        // For example, in sequence-to-sequence models, it could be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // For image segmentation tasks, it might be a Convolutional layer
        if (layer is ConvolutionalLayer<T>)
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
        // Lazy layers report InputShape = [-1] (or empty) until first Forward —
        // skip the strict shape-equality check; resolution happens at first
        // forward. Empty-shape layers are shape-agnostic by design (e.g.
        // DropoutLayer, ActivationLayer constructed without an explicit
        // InputShape) and would have been incorrectly rejected by the
        // SequenceEqual check otherwise.
        var currentInputShape = currentLayer.GetInputShape();
        bool currentIsLazy = currentInputShape.Length == 0
                             || currentInputShape.Any(d => d <= 0);
        if (!currentIsLazy && !Enumerable.SequenceEqual(prevLayer.GetOutputShape(), currentInputShape))
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
            return reshapeLayer.GetOutputShape().Aggregate((a, b) => a * b) ==
                   currentLayer.GetInputShape().Aggregate((a, b) => a * b);
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
    /// Returns the architecture's input shape as a positive-dim array,
    /// or null if the architecture doesn't declare a usable shape.
    /// Used by <see cref="ResolveLazyLayerShapes"/>; intentionally
    /// silent on missing data so dynamic-spatial models that depend on
    /// runtime shape feeding still work.
    /// </summary>
    private int[]? TryGetArchitectureInputShape()
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

            // Route through the eager forward instead of PredictCompiled. The
            // compiled-plan cache in CompiledModelHost binds to the trace-time
            // input tensor reference and replay reads stale data when called
            // with a *different* tensor of the same shape (the canonical
            // DifferentInputs / ScaledInput invariant failure: same shape,
            // new values, but the cached plan returns the first call's
            // output).
            //
            // Trade-off: this IS a per-call latency regression vs the prior
            // compiled-by-default behavior. Eager re-runs each forward op
            // through the engine instead of replaying a baked plan. The
            // regression is correctness-driven — compiled replay was
            // returning silently wrong outputs for the very common
            // "same model, same shape, new values" inference pattern.
            // A future Tensors-package release that adds value-aware
            // compiled replay (re-key on a tensor data hash, or
            // explicit re-trace on input change) can restore the
            // compiled fast path as the default; until then,
            // correctness wins. Callers who care about replay latency
            // can opt back in via CompileForward + identical-tensor
            // replay (their responsibility to feed the same tensor
            // reference each call).
            var output = PredictEager(promoted);

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
            return output;
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
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
    public virtual Tensor<T> ForwardForTraining(Tensor<T> input)
    {
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
        if (segmentSize > 0 && Layers.Count > segmentSize)
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

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
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
    protected Tensor<T> PredictCompiled(Tensor<T> input) =>
        _compileHost.Predict(input, _layerStructureVersion, () => PredictEager(input));

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

    /// <summary>
    /// Eager forward pass through all layers. Used as fallback when compilation fails.
    /// </summary>
    protected virtual Tensor<T> PredictEager(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
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
        RegisterTrainableTensorsWithWeightRegistry();
    }

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
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is not LayerBase<T> layer) continue;
            foreach (var tensor in layer.GetTrainableParameters())
            {
                if (tensor is null || tensor.Length == 0) continue;
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
            foreach (var tensor in extra.GetTrainableParameters())
            {
                if (tensor is null || tensor.Length == 0) continue;
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
            WeightRegistry.RegisterWeight(tensor);
        }
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
                // use the legacy per-layer UpdateParameters path
                var opt = GetOrCreateBaseOptimizer();
                opt.UpdateParameters(Layers);
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
    private (Tensor<T> Input, Tensor<T> Target) NormalizeBatchDim(Tensor<T> input, Tensor<T> target)
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
    /// Performs tape-based forward/backward pass and delegates the parameter update to the
    /// provided optimizer via <see cref="IGradientBasedOptimizer{T,TInput,TOutput}.Step"/>.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expected">The target tensor.</param>
    /// <param name="optimizer">The optimizer to apply. If null, uses a default Adam optimizer.</param>
    protected void TrainWithTape(Tensor<T> input, Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        var resolvedOptimizer = optimizer ?? GetOrCreateBaseOptimizer();

        // Fused-compiled fast path: forward + backward + parameter update all
        // in one compiled kernel, SIMD-accelerated, zero materialized gradient
        // tensors between backward and the optimizer step. Engages when the
        // optimizer is a plain Adam/AdamW/SGD with constant hyperparameters
        // AND the numeric type is float (Tensors-side fused optimizer kernels
        // operate on float* directly). Returns false if any condition isn't
        // met or tracing fails; we then fall through to the eager tape path
        // below with behavior unchanged.
        if (TryTrainWithFusedOptimizer(input, expected, resolvedOptimizer))
            return;

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
                // Memoize the skip decision so subsequent training steps
                // don't repeat the CollectParameters + sum-Length scan.
                // Sundial-Base (~300 M params) takes ~120ms per scan;
                // caching makes step 2..N effectively free.
                _skipParameterBuffer = true;
                _skipParameterBufferVersion = _layerStructureVersion;
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

            // Activate a TensorArena for the forward/backward/update scope.
            // After the first iteration warms the arena, ALL subsequent TensorAllocator.Rent
            // calls reuse pooled arrays — zero GC allocation in the training hot loop.
            // The arena is thread-static and resets on Dispose, so intermediate tensors
            // (conv outputs, attention scores, gradient buffers) are recycled every iteration.
            using var arena = TensorArena.Create();
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
            if (output.Rank > expected.Rank && output.Shape[0] == 1 && output.Length == expected.Length)
            {
                expected = Engine.Reshape(expected, output._shape);
            }
            else if (expected.Rank > output.Rank && expected.Shape[0] == 1 && expected.Length == output.Length)
            {
                expected = Engine.Reshape(expected, output._shape);
            }

            var lossTensor = loss.ComputeTapeLoss(output, expected);

            // Compute all gradients then filter to trainable params.
            // Passing sources directly can miss parameters when the tape backward
            // can't match view tensor references through the GradFn chain.
            var allGrads = tape.ComputeGradients(lossTensor, sources: null);
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

            var context = new TapeStepContext<T>(
                trainableParams, grads, lossValue,
                input, expected, ComputeForward,
                ComputeLossAligned,
                paramBuffer);

            opt.Step(context);

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
            if (extraTrainableTensors.Count > 0)
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

            // Advance the optimizer's learning-rate scheduler. This was
            // never being called from the training pipeline — schedulers
            // configured on optimizers (LinearWarmupScheduler, NoamSchedule,
            // CosineAnnealing, etc.) silently never stepped, leaving the LR
            // pinned at its initial value forever. The contract documented
            // on GradientBasedOptimizerBase.OnBatchEnd is "called at the end
            // of each training batch"; TrainWithTape is the canonical batch
            // boundary for tape-based training so it's the right place for
            // the call. Optimizers that don't derive from
            // GradientBasedOptimizerBase fall through unchanged.
            if (opt is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> stepped)
            {
                stepped.OnBatchEnd();
            }
        }
        finally
        {
            // Restore original tensor references so Clone/serialization see real tensors.
            // Copies updated weights from buffer views back to originals before restoring.
            RestoreOriginalParameters();
        }
    }

    /// <summary>
    /// Best-effort read of the supplied optimizer's current learning
    /// rate, used by the network-level extras update path. Returns the
    /// optimizer-typed value when the optimizer is a recognised
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
    private bool _fusedTrainingDisabled;

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
    private bool TryTrainWithFusedOptimizer(
        Tensor<T> input,
        Tensor<T> expected,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> resolvedOptimizer)
    {
        if (_fusedTrainingDisabled) return false;
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return false;
        if (typeof(T) != typeof(float))
            return false;

        if (!TryMapToFusedOptimizerConfig(
                resolvedOptimizer, out var fusedType, out float lr, out float b1, out float b2, out float eps, out float wd))
            return false;

        // Use the existing recursive trainable-layer collector instead of the
        // top-level-only scan — composite layers with trainable children (e.g.,
        // residual blocks, transformer layers) expose those children via
        // GetSubLayers() but aren't ITrainableLayer themselves. Without
        // recursion the fused path silently stops updating part of the model.
        var trainableLayers = Training.TapeTrainingStep<T>.CollectTrainableLayers(Layers, _layerStructureVersion);
        if (trainableLayers.Length == 0) return false;

        var loss = LossFunction as LossFunctions.LossFunctionBase<T>;
        if (loss is null) return false;

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
            out T lossValue);

        if (ran)
        {
            LastLoss = lossValue;
            // First successful fused step commits this model to the fused
            // path for the rest of the training session — Adam m/v are now
            // inside the compiled plan and transferring them to the eager
            // optimizer isn't possible without API we don't have.
            _fusedTrainingCommitted = true;
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
            // Resolution: call ResetState() or InvalidateParameterCountCache()
            // to fully reset training state, then retrain with stable
            // shapes + fixed hyperparameters. Or disable compilation via
            // AllowNondeterminism / Configure(JitCompilationConfig.Disabled)
            // so training runs entirely on the eager path from the start.
            throw new InvalidOperationException(
                "Fused compiled training has already run successfully, but the current step cannot " +
                "engage the fused path. The plan-embedded Adam/AdamW/SGD state cannot be transferred " +
                "to the eager optimizer, so falling back silently would produce a trajectory that " +
                "diverges from the previous fused steps. Common causes: variable input/target shape " +
                "(new compiled plan), LR scheduler or adaptive-rate changes, attached AMSGrad. " +
                "Resolution: keep shapes and optimizer hyperparameters stable across steps, OR call " +
                "ResetState() / InvalidateParameterCountCache() to explicitly reset training state, " +
                "OR disable compilation (AiModelBuilder.ConfigureJitCompilation(JitCompilationConfig.Disabled)).");
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
        out float weightDecay)
    {
        optimizerType = default;
        learningRate = 0f;
        beta1 = 0f;
        beta2 = 0f;
        epsilon = 0f;
        weightDecay = 0f;

        // Reject when an LR scheduler is attached — GetCurrentLearningRate()
        // would change between steps but the fused plan bakes the LR at first
        // ConfigureOptimizer, so subsequent rate changes silently disappear.
        if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradBase
            && gradBase.LearningRateScheduler is not null)
            return false;

        switch (optimizer)
        {
            case Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>> adam:
            {
                if (adam.GetOptions() is not Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> opts)
                    return false;
                if (opts.UseAdaptiveLearningRate) return false;
                optimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType.Adam;
                learningRate = (float)adam.GetCurrentLearningRate();
                beta1 = (float)opts.Beta1;
                beta2 = (float)opts.Beta2;
                epsilon = (float)opts.Epsilon;
                weightDecay = 0f;
                return true;
            }
            case Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>> adamW:
            {
                if (adamW.GetOptions() is not Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> opts)
                    return false;
                if (opts.UseAdaptiveLearningRate) return false;
                // Fused AdamW kernel does not implement AMSGrad's max-of-second-moment
                // update rule. If the user enabled it, fall back to eager so the
                // configured update rule isn't silently swapped for standard AdamW.
                if (adamW.UseAMSGrad) return false;
                optimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType.AdamW;
                learningRate = (float)adamW.GetCurrentLearningRate();
                beta1 = (float)opts.Beta1;
                beta2 = (float)opts.Beta2;
                epsilon = (float)opts.Epsilon;
                weightDecay = (float)opts.WeightDecay;
                return true;
            }
            case Optimizers.StochasticGradientDescentOptimizer<T, Tensor<T>, Tensor<T>> sgd:
            {
                if (sgd.GetOptions() is not Models.Options.StochasticGradientDescentOptimizerOptions<T, Tensor<T>, Tensor<T>> opts)
                    return false;
                if (opts.UseAdaptiveLearningRate) return false;
                optimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType.SGD;
                learningRate = (float)sgd.GetCurrentLearningRate();
                return true;
            }
            default:
                return false;
        }
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
            return lossValue;
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Gets or lazily creates the default optimizer for tape-based training.
    /// Used when a network doesn't provide its own optimizer.
    /// </summary>
    protected virtual IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
    {
        return _baseTrainOptimizer ??= new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    /// Subclasses that maintain a separate private optimizer field (e.g.
    /// <see cref="Transformer{T}"/>'s <c>_optimizer</c>) are responsible for
    /// keeping that field in sync if they want their <c>Train</c> override to
    /// honor builder overrides — typically by routing their override through
    /// <see cref="GetOrCreateBaseOptimizer"/> rather than reading the private
    /// field directly.
    /// </remarks>
    internal void SetBaseTrainOptimizer(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer)
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

        // If the first layer is not a dense or convolutional layer, we can't easily determine active features
        if (firstLayer is not (DenseLayer<T> or ConvolutionalLayer<T>))
        {
            // Return all indices as potentially active (conservative approach)
            return Enumerable.Range(0, firstLayer.GetInputShape()[0]);
        }

        // Get the weights from the first layer
        Vector<T> weights = firstLayer.GetParameters();
        int inputSize = firstLayer.GetInputShape()[0];
        int outputSize = firstLayer.GetOutputShape()[0];

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

        // If feature index is out of range, it's not used
        if (Layers.Count == 0 || featureIndex < 0 || featureIndex >= Layers[0].GetInputShape()[0])
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
        for (int i = 0; i < _layers.Count; i++)
        {
            paramBytes += (long)_layers[i].ParameterCount * sizeof(double);
            if (_layers[i] is AiDotNet.NeuralNetworks.Layers.ILayerSerializationExtras<T> ext)
                paramBytes += (long)ext.ExtraParameterCount * sizeof(double);
        }

        if (paramBytes > (long)MaxArrayLength)
        {
            var largeCopy = CreateNewInstance();
            if (largeCopy is NeuralNetworkBase<T> largeBase && largeBase._layers.Count == _layers.Count)
            {
                // Copy layer-by-layer parameters + serialization extras + network-specific data.
                for (int i = 0; i < _layers.Count; i++)
                {
                    var srcLayer = _layers[i];
                    var dstLayer = largeBase._layers[i];
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
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Sequence models with EmbeddingLayer don't support feature selection.
        // Their input shape is [1] (single token ID), not a feature vector.
        // Fixes #1113.
        if (Layers.Count > 0 && Layers[0] is Layers.EmbeddingLayer<T>)
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

        int totalParameterCount = (int)ParameterCount;
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

        // Compute loss via the user's configured loss function.
        // Shape matching (integer → one-hot, singleton reshape) is handled
        // inside each loss function's ComputeTapeLoss via EnsureTargetMatchesPredicted.
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

        // Reverse-mode AD: compute gradients for all trainable parameters
        var grads = tape.ComputeGradients(lossTensor, trainableParams);

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

