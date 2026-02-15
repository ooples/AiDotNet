using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.Models.Options;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.MixedPrecision;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
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
public abstract class NeuralNetworkBase<T> : INeuralNetworkModel<T>, IInterpretableModel<T>, IInputGradientComputable<T>, IConfigurableModel<T>, IDisposable
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
    public readonly NeuralNetworkArchitecture<T> Architecture;

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
    public virtual bool SupportsTraining => false;

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
    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
    {
        Architecture = architecture;
        _layers = new List<ILayer<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        MaxGradNorm = NumOps.FromDouble(maxGradNorm);
        LossFunction = lossFunction;
        _cachedParameterCount = null;
        _sensitiveFeatures = new Vector<int>(0);
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
        int totalParameterCount = ParameterCount;
        var parameters = new Vector<T>(totalParameterCount);

        int currentIndex = 0;
        foreach (var layer in Layers.Where(l => l.ParameterCount > 0))
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = layer.GetParameters();
            for (int i = 0; i < layerParameterCount; i++)
            {
                parameters[currentIndex + i] = layerParameters[i];
            }

            currentIndex += layerParameterCount;
        }

        return parameters;
    }

    /// <summary>
    /// Performs backpropagation to compute gradients for network parameters.
    /// </summary>
    /// <param name="outputGradients">The gradients of the loss with respect to the network outputs.</param>
    /// <returns>The gradients of the loss with respect to the network inputs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn. After making a prediction, the network
    /// calculates how wrong it was (the error). Then it works backward through the layers to figure out
    /// how each parameter contributed to that error. This method handles that backward flow of information.
    /// </para>
    /// <para>
    /// The "gradients" are numbers that tell us how to adjust each parameter to reduce the error.
    /// </para>
    /// <para>
    /// <b>API Change Note:</b> The signature changed from Vector&lt;T&gt; to Tensor&lt;T&gt; to support multi-dimensional
    /// gradients. This is a breaking change. If you need backward compatibility, consider adding an overload that
    /// accepts Vector&lt;T&gt; and converts it internally to Tensor&lt;T&gt;.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network is not in training mode or doesn't support training.</exception>
    public virtual Tensor<T> Backpropagate(Tensor<T> outputGradients)
    {
        if (!IsTrainingMode)
        {
            throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
        }

        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support backpropagation");
        }

        // Use memory-managed backward if gradient checkpointing is enabled
        if (_memoryManager is not null && _memoryManager.IsCheckpointingEnabled)
        {
            return BackpropagateWithRecompute(outputGradients);
        }

        // Standard backpropagation through layers in reverse order
        var gradientTensor = outputGradients;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        return gradientTensor;
    }

    /// <summary>
    /// Performs backpropagation with activation recomputation for non-checkpointed layers.
    /// </summary>
    /// <param name="outputGradients">Gradients from the loss function.</param>
    /// <returns>Gradients with respect to input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When gradient checkpointing is enabled, we don't store all layer
    /// activations during forward pass (to save memory). During backprop, we need those
    /// activations, so we recompute them from the nearest checkpoint.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> BackpropagateWithRecompute(Tensor<T> outputGradients)
    {
        if (_memoryManager is null)
            throw new InvalidOperationException("Memory manager is not configured.");

        var gradientTensor = outputGradients;

        // Backpropagate through layers in reverse order with recomputation
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            // Use memory manager to handle recomputation from checkpoints
            gradientTensor = _memoryManager.BackwardWithRecompute(Layers[i], gradientTensor, i);
        }

        // Clear checkpoints after backprop to free memory
        _memoryManager.ClearCheckpoints();

        return gradientTensor;
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
    public virtual IGpuTensor<T> ForwardGpu(IGpuTensor<T> input)
    {
        if (!CanTrainOnGpu)
        {
            throw new InvalidOperationException(
                "GPU forward pass is not supported. Check CanTrainOnGpu before calling this method.");
        }

        var current = input;
        foreach (var layer in Layers)
        {
            if (layer is LayerBase<T> layerBase)
            {
                current = layerBase.ForwardGpu(current);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Layer {layer.GetType().Name} does not inherit from LayerBase<T> and cannot be used with GPU training.");
            }
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation through all layers entirely on GPU.
    /// </summary>
    /// <param name="outputGradients">The GPU-resident gradient of loss with respect to network output.</param>
    /// <returns>The GPU-resident gradient with respect to network input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support GPU training.</exception>
    /// <remarks>
    /// <para>
    /// This method backpropagates through all layers on GPU:
    /// - Each layer computes input gradients and stores weight gradients on GPU
    /// - No data is transferred to CPU during backpropagation
    /// - After calling this, call UpdateParametersGpu() to apply the gradients
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Like Backpropagate() but everything stays on GPU.
    /// The weight gradients are computed and stored on GPU, ready for the update step.
    /// </para>
    /// </remarks>
    public virtual IGpuTensor<T> BackpropagateGpu(IGpuTensor<T> outputGradients)
    {
        if (!IsTrainingMode)
        {
            throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
        }

        if (!CanTrainOnGpu)
        {
            throw new InvalidOperationException(
                "GPU backward pass is not supported. Check CanTrainOnGpu before calling this method.");
        }

        var gradientTensor = outputGradients;

        // Backpropagate through layers in reverse order
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (Layers[i] is LayerBase<T> layerBase)
            {
                gradientTensor = layerBase.BackwardGpu(gradientTensor);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Layer {Layers[i].GetType().Name} does not inherit from LayerBase<T> and cannot be used with GPU training.");
            }
        }

        return gradientTensor;
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
    public virtual IGpuTensor<T> BackpropagateGpuDeferred(
        IGpuTensor<T> outputGradients,
        GpuExecutionOptions? options = null)
    {
        var engine = AiDotNetEngine.Current as DirectGpuTensorEngine;
        if (engine?.GetBackend() == null)
        {
            // Fallback to non-deferred if no GPU backend
            return BackpropagateGpu(outputGradients);
        }

        var backend = engine.GetBackend() as IAsyncGpuBackend;
        if (backend == null)
        {
            return BackpropagateGpu(outputGradients);
        }

        return backend.ExecuteDeferred(
            scope => BackpropagateGpu(outputGradients),
            options);
    }

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
        IGpuTensor<T> input,
        IGpuTensor<T> target,
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
                BackpropagateGpu(lossResult.Gradient);

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
        IGpuTensor<T> input,
        IGpuTensor<T> target,
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
                BackpropagateGpu(lossResult.Gradient);

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
    protected bool TryForwardGpuOptimized(Tensor<T> input, out Tensor<T> result)
    {
        result = null!;

        if (Engine is not DirectGpuTensorEngine)
            return false;

        if (!CanUseGpuResidentPath())
            return false;

        try
        {
            using var gpuResult = ForwardGpu(input);
            result = gpuResult.ToTensor();
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
    /// var output = gpuResult.ToTensor(); // Only downloads here
    /// </code>
    /// </remarks>
    public virtual IGpuTensor<T> ForwardGpu(Tensor<T> input)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires DirectGpuTensorEngine. Current engine: " +
                Engine.GetType().Name);
        }

        // Upload input to GPU once
        IGpuTensor<T>? current = null;
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
                        cpuInput = current.ToTensor();
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
                    IGpuTensor<T> current = gpuInput;
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
                                var cpuInput = current.ToTensor();
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
                        var result = current.ToTensor();
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
            return result.ToTensor();
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
                    IGpuTensor<T> current = gpuInput;
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
                                var cpuInput = current.ToTensor();
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
                        var result = current.ToTensor();
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
                return result.ToTensor();
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
    ///         predictions.Add(result.ToTensor());
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
    public virtual IGpuTensor<T> ForwardWithGpuContext(Tensor<T> input)
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
        IGpuTensor<T> current = ctx.Upload(input, GpuTensorRole.Activation);

        try
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];

                if (layer.CanExecuteOnGpu)
                {
                    var next = layer.ForwardGpu(current);

                    // Register output with context (if not already registered by layer)
                    if (next is GpuTensor<T> gpuNext)
                    {
                        ctx.Registry.TryRegister(gpuNext);
                    }

                    current = next;
                }
                else
                {
                    // Layer doesn't support GPU - fall back to CPU
                    var cpuInput = current.ToTensor();
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
    /// Performs a GPU-resident forward pass within a GPU execution context with GPU-resident input.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor managed by the current context.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no GPU context is active.</exception>
    public virtual IGpuTensor<T> ForwardWithGpuContext(IGpuTensor<T> input)
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

        IGpuTensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];

            if (layer.CanExecuteOnGpu)
            {
                var next = layer.ForwardGpu(current);

                // Register output with context (if not already registered by layer)
                if (next is GpuTensor<T> gpuNext)
                {
                    ctx.Registry.TryRegister(gpuNext);
                }

                current = next;
            }
            else
            {
                // Layer doesn't support GPU - fall back to CPU
                var cpuInput = current.ToTensor();
                var cpuOutput = layer.Forward(cpuInput);

                // Upload result back using context
                current = ctx.Upload(cpuOutput, GpuTensorRole.Activation);
            }
        }

        return current;
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
    public virtual int ParameterCount
    {
        get
        {
            if (_cachedParameterCount == null)
            {
                _cachedParameterCount = Layers.Sum(layer => layer.ParameterCount);
            }
            return _cachedParameterCount.Value;
        }
    }

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
    public int GetParameterCount()
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
        InvalidateLayerInfoCache();
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
            // Ensure the dense layer doesn't have any inputs (it's the first layer)
            return denseLayer.GetInputShape().Length == 1 && denseLayer.GetInputShape()[0] > 0;
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
        // Check if the output shape of the previous layer matches the input shape of the current layer
        if (!Enumerable.SequenceEqual(prevLayer.GetOutputShape(), currentLayer.GetInputShape()))
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

        // Check for dimension compatibility in case of Reshape or Flatten layers
        if (prevLayer is ReshapeLayer<T> reshapeLayer)
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
        if (!Architecture.IsInitialized)
        {
            // Initialize from cached data
            Architecture.InitializeFromCachedData();

            // Initialize network-specific layers
            InitializeLayers();
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
    /// </remarks>
    public abstract Tensor<T> Predict(Tensor<T> input);

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
    internal virtual void EnableMixedPrecision(MixedPrecisionConfig? config = null)
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
    /// Trains the neural network on a single input-output pair.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="expectedOutput">The expected output for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training step on the neural network using the provided input and expected output.
    /// It updates the network's parameters to reduce the error between the network's prediction and the expected output.
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
    /// </para>
    /// </remarks>
    public abstract void Train(Tensor<T> input, Tensor<T> expectedOutput);

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
    }

    /// <summary>
    /// Computes gradients of the network output with respect to the network input using backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient signal from the output (typically all ones for gradient computation).</param>
    /// <returns>A tensor containing the gradients with respect to the network input.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a backward pass through all layers to compute how the output changes
    /// with respect to the input. Unlike the standard backward pass which computes gradients for
    /// parameters, this method computes gradients for the input itself.
    /// </para>
    /// <para>
    /// This is essential for techniques like:
    /// - Gradient-based input optimization
    /// - Saliency maps and input attribution
    /// - WGAN-GP gradient penalty computation
    /// - Adversarial example generation
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how sensitive the output is to changes in the input.
    ///
    /// Normally, backpropagation adjusts the network's internal parameters (weights and biases).
    /// This method instead computes how the output would change if we modified the input data.
    ///
    /// Use cases:
    /// - Understanding which input features matter most (interpretability)
    /// - Generating adversarial examples (security research)
    /// - Computing gradient penalties for training stability (WGAN-GP)
    ///
    /// The process:
    /// 1. Assumes a forward pass has already been run (outputs are cached)
    /// 2. Starts with a gradient signal at the output (how much we "care" about each output)
    /// 3. Propagates this gradient backwards through each layer
    /// 4. Returns the gradient with respect to the original input
    /// </para>
    /// </remarks>
    public virtual Tensor<T> BackwardWithInputGradient(Tensor<T> outputGradient)
    {
        if (Layers.Count == 0)
        {
            throw new InvalidOperationException("Cannot compute input gradients for a network with no layers.");
        }

        // Start with the output gradient and propagate backwards through layers
        var currentGradient = outputGradient;

        // Iterate backwards through layers
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            // Each layer's Backward method takes the gradient from the next layer
            // and returns the gradient to pass to the previous layer
            currentGradient = layer.Backward(currentGradient);
        }

        // The final gradient is with respect to the network input
        return currentGradient;
    }

    /// <inheritdoc/>
    public virtual Vector<T> ComputeInputGradient(Vector<T> input, Vector<T> outputGradient)
    {
        // Convert vectors to tensors and use the existing tensor-based implementation
        var inputTensor = Tensor<T>.FromVector(input);
        var gradientTensor = Tensor<T>.FromVector(outputGradient);

        // Run forward pass to cache layer activations
        Predict(inputTensor);

        // Compute input gradient using backpropagation
        var resultTensor = BackwardWithInputGradient(gradientTensor);

        return resultTensor.ToVector();
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ComputeInputGradient(Tensor<T> input, Tensor<T> outputGradient)
    {
        // Run forward pass to cache layer activations
        Predict(input);

        // Compute input gradient using backpropagation
        return BackwardWithInputGradient(outputGradient);
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the entire neural network, including all layers and parameters,
    /// and saves it to the specified file path.
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

        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
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

        byte[] data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    private const int SerializationMagic = 0x4E444941; // "AIDN" (little-endian int)
    private const int SerializationVersion = 4;

    /// <summary>
    /// Serializes the neural network to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized neural network.</returns>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
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
        // Create a deep copy of the current network
        var newNetwork = (NeuralNetworkBase<T>)DeepCopy();

        // Update the parameters of the new network
        newNetwork.UpdateParameters(parameters);

        return newNetwork;
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
        // The most reliable way to create a deep copy is through serialization/deserialization
        byte[] serialized = Serialize();

        // Create a new instance of the same type as this network
        var copy = CreateNewInstance();

        // Load the serialized data into the new instance
        copy.Deserialize(serialized);

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
        int[] inputShape = input.Shape;

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

        int totalParameterCount = ParameterCount;
        if (parameters.Length != totalParameterCount)
        {
            throw new ArgumentException($"Expected {totalParameterCount} parameters, got {parameters.Length}");
        }

        int currentIndex = 0;
        foreach (var layer in Layers.Where(l => l.ParameterCount > 0))
        {
            int layerParameterCount = layer.ParameterCount;
            // Extract parameters for this layer
            var layerParameters = new Vector<T>(layerParameterCount);
            for (int i = 0; i < layerParameterCount; i++)
            {
                layerParameters[i] = parameters[currentIndex + i];
            }

            // Set the layer's parameters
            layer.SetParameters(layerParameters);
            currentIndex += layerParameterCount;
        }
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
            LayerType.Dense => new DenseLayer<T>(inputSize, units, activationFunc),
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
            "new ConvolutionalLayer<T>(inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, activation) " +
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
        var layer = new BatchNormalizationLayer<T>(featureSize, epsilon, momentum);
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
        var layer = new MaxPoolingLayer<T>(inputShape, poolSize, strides ?? poolSize);
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
        if (Layers.Count == 0)
        {
            return Array.Empty<int>();
        }

        return Layers[0].GetInputShape();
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
    /// <param name="lossFunction">Optional override loss function (defaults to <see cref="DefaultLossFunction"/>).</param>
    /// <returns>A vector containing the concatenated gradients for all layer parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass, computes the loss derivative, backpropagates gradients, and then
    /// concatenates the parameter gradients across all layers into a single vector.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Gradients are the "direction to change weights" so the model makes fewer mistakes.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? DefaultLossFunction;

        var prediction = Predict(input);
        var lossDerivative = loss.CalculateDerivative(prediction.ToVector(), target.ToVector());
        var outputGradients = new Tensor<T>(prediction.Shape, lossDerivative);

        Backpropagate(outputGradients);

        var gradients = new List<T>();
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            gradients.AddRange(layerParams.ToArray());
        }

        return new Vector<T>(gradients.ToArray());
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
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Dispose managed resources
            DisableMixedPrecision();
        }
    }

    #region IJitCompilable Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Neural networks support JIT compilation for accelerated inference.
    /// The computation graph represents the forward pass through all layers.
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation optimizes neural networks for faster predictions.
    ///
    /// Instead of executing each layer one by one at runtime, JIT compilation:
    /// - Analyzes the entire network structure
    /// - Combines and optimizes operations
    /// - Generates specialized native code
    /// - Results in 5-10x faster predictions
    ///
    /// This is especially beneficial for:
    /// - Production deployment (real-time predictions)
    /// - Batch inference (processing many examples)
    /// - Edge devices (mobile, embedded systems)
    ///
    /// Note: Not all layer types support JIT compilation yet. The SupportsJitCompilation
    /// property indicates whether this specific network configuration can be JIT compiled.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => Layers.Count == 0 || Layers.All(layer => layer.SupportsJitCompilation);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exports the neural network as a computation graph for JIT compilation.
    /// The graph represents the forward pass through all layers in sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the neural network into a computation graph.
    ///
    /// A computation graph is like a flowchart that describes:
    /// 1. How data flows through each layer
    /// 2. What operations each layer performs
    /// 3. How layer outputs connect to the next layer's inputs
    ///
    /// The JIT compiler uses this graph to:
    /// - Optimize the operations (remove redundancy)
    /// - Fuse operations together (combine multiple steps)
    /// - Generate fast native code
    ///
    /// For example, a simple network:
    /// Input → Dense Layer → ReLU → Dense Layer → Output
    ///
    /// Becomes a graph:
    /// input_node → matmul_node → add_bias_node → relu_node → matmul_node → add_bias_node
    ///
    /// The JIT compiler can then optimize this graph (e.g., fuse bias addition with matmul)
    /// to create highly efficient code.
    /// </para>
    /// </remarks>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Validation: Ensure network has layers
        if (Layers == null || Layers.Count == 0)
        {
            throw new InvalidOperationException("Cannot export computation graph: Network has no layers.");
        }

        // Create input node (placeholder for input data)
        // For neural networks, input shape is typically [batch_size, input_features]
        // We use [1, Architecture.InputSize] as a placeholder
        var inputShape = new int[] { 1, Architecture.InputSize };
        var inputTensor = new Tensor<T>(inputShape);
        var inputNode = new ComputationNode<T>(inputTensor);
        inputNodes.Add(inputNode);

        // Build computation graph by chaining layers
        var currentNode = inputNode;
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            try
            {
                currentNode = ConvertLayerToGraph(layer, currentNode);
            }
            catch (NotSupportedException ex)
            {
                throw new NotSupportedException(
                    $"JIT compilation failed at layer {i} ({layer.GetType().Name}): {ex.Message}. " +
                    $"This layer type is not yet supported for JIT compilation.", ex);
            }
        }

        return currentNode;
    }

    /// <summary>
    /// Converts a single layer to computation graph nodes by delegating to the layer's ExportComputationGraph method.
    /// </summary>
    /// <param name="layer">The layer to convert.</param>
    /// <param name="input">The input node to the layer.</param>
    /// <returns>The output node from the layer.</returns>
    /// <exception cref="NotSupportedException">Thrown when the layer does not support JIT compilation.</exception>
    /// <remarks>
    /// This method follows the Open/Closed Principle by delegating to each layer's own ExportComputationGraph implementation.
    /// New layers can be added without modifying this base class.
    /// </remarks>
    protected virtual ComputationNode<T> ConvertLayerToGraph(ILayer<T> layer, ComputationNode<T> input)
    {
        // Delegate to the layer's ExportComputationGraph implementation
        // Each layer is responsible for converting itself to a computation graph
        var layerInputs = new List<ComputationNode<T>> { input };
        return layer.ExportComputationGraph(layerInputs);
    }


    #endregion

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
                ParameterCount = layer.ParameterCount,
                InputShape = layer.GetInputShape(),
                OutputShape = layer.GetOutputShape(),
                IsTrainable = layer.SupportsTraining && layer.ParameterCount > 0,
                EstimatedFlops = layerBase?.EstimateFlops() ?? 2L * layer.ParameterCount,
                EstimatedActivationMemory = layerBase?.EstimateActivationMemory() ?? 0L,
            });

            parameterOffset += layer.ParameterCount;
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

        // Compute the parameter offset for the first layer in the sub-model
        int baseOffset = 0;
        for (int i = 0; i < startLayer; i++)
        {
            baseOffset += _layers[i].ParameterCount;
        }

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
                ParameterCount = layer.ParameterCount,
                InputShape = layer.GetInputShape(),
                OutputShape = layer.GetOutputShape(),
                IsTrainable = layer.SupportsTraining && layer.ParameterCount > 0,
                EstimatedFlops = layerBase?.EstimateFlops() ?? 2L * layer.ParameterCount,
                EstimatedActivationMemory = layerBase?.EstimateActivationMemory() ?? 0L,
            });

            localOffset += layer.ParameterCount;
        }

        return new SubModel<T>(subLayers, subInfos, startLayer, endLayer);
    }

    #endregion

}

