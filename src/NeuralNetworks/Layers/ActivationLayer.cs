using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer that applies an activation function to transform the input data.
/// <para>
/// Activation functions introduce non-linearity to neural networks. Non-linearity means the output isn't 
/// simply proportional to the input (like y = 2x). Instead, it can follow curves or more complex patterns.
/// severely limiting what it can learn.
/// </para>
/// <para>
/// Common activation functions include:
/// - ReLU: Returns 0 for negative inputs, or the input value for positive inputs
/// - Sigmoid: Squashes values between 0 and 1, useful for probabilities
/// - Tanh: Similar to sigmoid but outputs values between -1 and 1
/// </para>
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (like float, double, etc.)</typeparam>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, TestInputShape = "1, 4", TestConstructorArgs = "new[] { 1, 4 }, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.ReLUActivation<double>()")]
public class ActivationLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Stores the input from the most recent forward pass for use in the backward pass.
    /// </summary>
    /// <remarks>
    /// This field caches the input tensor from the most recent call to Forward(). During backpropagation,
    /// this cached input is used to calculate the gradient of the activation function. The field is nullable
    /// and will be null until Forward() is called at least once.
    /// </remarks>
    private Tensor<T>? _lastInput;

    // GPU-resident cached tensors for GPU training pipeline
    private Tensor<T>? _lastInputGpu;
    private Tensor<T>? _lastOutputGpu; // Post-activation for sigmoid/tanh backward

    /// <summary>
    /// Indicates whether this layer uses a vector activation function instead of a scalar one.
    /// </summary>
    /// <remarks>
    /// When true, the layer applies an activation function that operates on entire vectors at once
    /// (like Softmax). When false, it applies a function that operates on individual scalar values
    /// (like ReLU or Sigmoid).
    /// </remarks>
    private readonly bool _useVectorActivation;

    /// <summary>
    /// Indicates whether this layer has trainable parameters.
    /// <para>
    /// Always returns false because activation layers don't have parameters to train.
    /// Unlike layers such as Dense/Convolutional layers which have weights and biases
    /// that need updating during training, activation layers simply apply a fixed
    /// mathematical function to their inputs.
    /// </para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property overrides the base class property to specify that activation layers do not have trainable parameters.
    /// Trainable parameters are values within a layer that are adjusted during the training process to minimize the loss
    /// function. Since activation layers simply apply a fixed mathematical function to their inputs without any adjustable
    /// parameters, this property always returns false.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you that activation layers don't learn or change during training.
    /// 
    /// While layers like Dense layers have weights that get updated during training,
    /// activation layers just apply a fixed mathematical formula that never changes.
    /// 
    /// Think of it like this:
    /// - Dense layers are like adjustable knobs that the network learns to tune
    /// - Activation layers are like fixed functions (like f(x) = max(0, x) for ReLU)
    /// 
    /// This property helps the training system know that it doesn't need to
    /// update anything in this layer during the training process.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Creates a new activation layer that applies a scalar activation function to each value individually.
    /// <para>
    /// A scalar activation function processes each number in your data independently.
    /// For example, if you have an input tensor with 100 values, the function will be applied
    /// 100 times, once to each value, without considering the other values.
    /// </para>
    /// <para>
    /// This is appropriate for most common activation functions like ReLU, Sigmoid, and Tanh.
    /// </para>
    /// </summary>
    /// <param name="inputShape">The shape (dimensions) of the input data, such as [batchSize, height, width, channels]</param>
    /// <param name="activationFunction">The activation function to apply (like ReLU, Sigmoid, etc.)</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an activation layer that applies a scalar activation function to each value in the input tensor
    /// independently. The input shape and output shape are the same, as activation functions don't change the dimensions of the data.
    /// The activation function is passed to the base class constructor and stored for use during forward and backward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an activation layer that applies a function to each value separately.
    /// 
    /// Use this constructor for common activation functions like:
    /// - ReLU: Which replaces negative values with zero
    /// - Sigmoid: Which squashes values between 0 and 1
    /// - Tanh: Which squashes values between -1 and 1
    /// 
    /// For example:
    /// ```csharp
    /// // Create a ReLU activation layer for 28x28 images
    /// var reluLayer = new ActivationLayer<float>(new[] { 32, 28, 28, 1 }, new ReLUActivation<float>());
    /// ```
    /// 
    /// The inputShape parameter defines the dimensions of your data:
    /// - For images: [batchSize, height, width, channels]
    /// - For sequences: [batchSize, sequenceLength, features]
    /// - For simple data: [batchSize, features]
    /// </para>
    /// </remarks>
    public ActivationLayer(int[] inputShape, IActivationFunction<T> activationFunction)
        : base(inputShape, inputShape, activationFunction)
    {
        _useVectorActivation = false;
    }

    /// <summary>
    /// Creates a new activation layer that applies a vector activation function to the entire tensor at once.
    /// <para>
    /// A vector activation function needs to consider multiple values together when processing.
    /// For example, the Softmax function needs to know all values in a vector to calculate
    /// the normalized probabilities across all elements.
    /// </para>
    /// <para>
    /// This type of activation is typically used for:
    /// - Softmax: Converts a vector of numbers into probabilities that sum to 1
    /// - Attention mechanisms: Where relationships between different positions in a sequence matter
    /// - Normalization functions: That need to consider statistics across multiple values
    /// </para>
    /// </summary>
    /// <param name="inputShape">The shape (dimensions) of the input data, such as [batchSize, height, width, channels]</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an activation layer that applies a vector activation function to the entire input tensor at once.
    /// Vector activation functions need to consider multiple values together, unlike scalar functions that process each value
    /// independently. The input shape and output shape are the same, as activation functions don't change the dimensions of the data.
    /// The vector activation function is passed to the base class constructor and stored for use during forward and backward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an activation layer that needs to look at all values together.
    /// 
    /// Use this constructor for activation functions that need to consider relationships
    /// between different values, such as:
    /// - Softmax: Which converts values to probabilities that sum to 1
    /// - Attention mechanisms: Which determine how much focus to put on different parts of the input
    /// - Normalization functions: Which adjust values based on statistics of the entire input
    /// 
    /// For example:
    /// ```csharp
    /// // Create a Softmax layer for a classification network with 10 classes
    /// var softmaxLayer = new ActivationLayer<float>(new[] { 32, 10 }, new Softmax<float>());
    /// ```
    /// 
    /// The difference from the other constructor is that these functions need to
    /// "see" all the values at once to make their calculations, rather than
    /// processing each value independently.
    /// </para>
    /// </remarks>
    public ActivationLayer(int[] inputShape, IVectorActivationFunction<T> vectorActivationFunction)
        : base(inputShape, inputShape, vectorActivationFunction)
    {
        _useVectorActivation = true;
    }

    /// <summary>
    /// Processes the input data by applying the activation function.
    /// <para>
    /// This is called during the forward pass of the neural network, which is when
    /// data flows from the input layer through all hidden layers to the output layer.
    /// The forward pass is used both during training and when making predictions with a trained model.
    /// </para>
    /// <para>
    /// For example, if using ReLU activation, this method would replace all negative values in the input
    /// with zeros while keeping positive values unchanged.
    /// </para>
    /// </summary>
    /// <param name="input">The input data to process</param>
    /// <returns>The transformed data after applying the activation function</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass for the activation layer. It stores the input tensor for later use in the
    /// backward pass, then applies either a scalar or vector activation function based on the layer's configuration.
    /// For scalar activation, the function is applied to each element independently. For vector activation, the function
    /// is applied to the entire tensor at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the activation function to transform the input data.
    /// 
    /// During the forward pass, data flows through the network from input to output.
    /// This method:
    /// 1. Saves the input for later use in backpropagation
    /// 2. Applies the activation function to transform the data
    /// 3. Returns the transformed data
    /// 
    /// For example, with ReLU activation:
    /// - Input: [-2, 0, 3, -1, 5]
    /// - Output: [0, 0, 3, 0, 5] (negative values become 0)
    /// 
    /// This transformation adds non-linearity to the network, which is essential
    /// for learning complex patterns in the data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return _useVectorActivation ? ApplyVectorActivation(input) : ApplyScalarActivation(input);
    }

    /// <summary>
    /// Applies a scalar activation function to each element of the input tensor.
    /// </summary>
    /// <param name="input">The input tensor to transform</param>
    /// <returns>A new tensor with the activation function applied to each element</returns>
    /// <remarks>
    /// This private helper method applies the scalar activation function to the input tensor.
    /// It delegates to the activation function's Activate(Tensor) method, allowing for potential
    /// optimizations (like GPU acceleration) implemented within the activation function class.
    /// </remarks>
    private Tensor<T> ApplyScalarActivation(Tensor<T> input)
    {
        var activation = ScalarActivation ?? throw new InvalidOperationException("ScalarActivation has not been initialized.");
        return activation.Activate(input);
    }

    /// <summary>
    /// Applies a vector activation function to the entire input tensor.
    /// </summary>
    /// <param name="input">The input tensor to transform</param>
    /// <returns>A new tensor resulting from applying the vector activation function</returns>
    /// <remarks>
    /// This private helper method applies the vector activation function to the entire input tensor at once.
    /// Vector activation functions need to consider multiple values together, unlike scalar functions that
    /// process each value independently.
    /// </remarks>
    private Tensor<T> ApplyVectorActivation(Tensor<T> input)
    {
        var activation = VectorActivation ?? throw new InvalidOperationException("VectorActivation has not been initialized.");
        return activation.Activate(input);
    }

    /// <summary>
    /// Updates the layer's internal parameters during training.
    /// <para>
    /// This method is part of the training process where layers adjust their parameters
    /// (weights and biases) based on the gradients calculated during backpropagation.
    /// </para>
    /// <para>
    /// For activation layers, this method does nothing because they have no trainable parameters.
    /// Unlike layers such as Dense layers which need to update their weights and biases,
    /// activation layers simply apply a fixed mathematical function.
    /// </para>
    /// </summary>
    /// <param name="learningRate">How quickly the network should learn from new data. Higher values mean bigger parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is called during the training process after the forward and backward passes have been completed.
    /// For layers with trainable parameters, this method would update those parameters based on the gradients
    /// calculated during backpropagation and the provided learning rate. However, since activation layers have
    /// no trainable parameters, this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would update the layer's internal values during training, but activation layers have nothing to update.
    /// 
    /// In neural networks, training involves adjusting parameters to reduce errors.
    /// This method is where those adjustments happen, but activation layers don't have
    /// any adjustable parameters, so this method is empty.
    /// 
    /// For comparison:
    /// - In a Dense layer, this would update weights and biases
    /// - In a BatchNorm layer, this would update scale and shift parameters
    /// - In this ActivationLayer, there's nothing to update
    /// 
    /// The learning rate parameter controls how big the updates would be if there
    /// were any parameters to update - higher values mean bigger changes.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Activation layer has no parameters to update
    }

    /// <summary>
    /// Gets all trainable parameters of this layer as a flat vector.
    /// <para>
    /// This method is useful for operations that need to work with all parameters at once,
    /// such as certain optimization algorithms, regularization techniques, or when saving a model.
    /// </para>
    /// <para>
    /// Returns an empty vector since activation layers have no trainable parameters.
    /// Other layer types like Dense layers would return their weights and biases.
    /// </para>
    /// </summary>
    /// <returns>An empty vector representing the layer's parameters</returns>
    /// <remarks>
    /// <para>
    /// This method returns all trainable parameters of the layer as a flat vector. For layers with trainable
    /// parameters, this would involve reshaping multi-dimensional parameters (like weight matrices) into a
    /// one-dimensional vector. However, since activation layers have no trainable parameters, this method
    /// returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the layer's trainable values as a single list, but activation layers have none.
    /// 
    /// Some operations in neural networks need to work with all parameters at once:
    /// - Saving and loading models
    /// - Applying regularization (techniques to prevent overfitting)
    /// - Using advanced optimization algorithms
    /// 
    /// This method provides those parameters as a single vector, but since
    /// activation layers don't have any trainable parameters, it returns an empty vector.
    /// 
    /// For comparison:
    /// - A Dense layer with 100 inputs and 10 outputs would return a vector with 1,010 values
    ///   (1,000 weights + 10 biases)
    /// - This ActivationLayer returns an empty vector with 0 values
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Activation layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Clears the layer's memory of previous inputs.
    /// <para>
    /// Neural networks maintain state between operations, especially during training.
    /// This method resets that state, which is useful in several scenarios:
    /// - When starting to process a new batch of data
    /// - Between training epochs
    /// - When switching from training to evaluation mode
    /// - When you want to ensure the layer behaves deterministically
    /// </para>
    /// <para>
    /// For activation layers, this means forgetting the last input that was processed,
    /// which was stored to help with the backward pass calculations.
    /// </para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input tensor. The activation
    /// layer stores the input from the most recent forward pass to use during the backward pass for calculating
    /// gradients. Resetting this state is useful when starting to process new data or when you want to ensure
    /// the layer behaves deterministically.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of previous calculations.
    /// 
    /// During training, the layer remembers the last input it processed to help with
    /// backpropagation calculations. This method makes the layer "forget" that input.
    /// 
    /// You might need to reset state:
    /// - When starting a new batch of training data
    /// - Between training epochs
    /// - When switching from training to testing
    /// - When you want to ensure consistent behavior
    /// 
    /// For activation layers, this is simple - it just clears the saved input tensor.
    /// Other layer types might have more complex state to reset.
    /// 
    /// This helps ensure that processing one batch doesn't accidentally affect
    /// the processing of the next batch.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;

        // Clear GPU-resident cached tensors
        _lastInputGpu = null;
        _lastOutputGpu = null;
    }

    /// <summary>
    /// Gets whether this layer's activation function supports GPU execution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// GPU execution is supported for common scalar activation functions that have
    /// dedicated GPU kernels: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, and Swish.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the activation function can run on GPU.
    /// Most common activations like ReLU and Sigmoid have GPU support.
    /// Exotic or vector activations (like Softmax) may not support GPU execution yet.
    /// </para>
    /// </remarks>
    protected override bool SupportsGpuExecution => TryGetFusedActivationType(out _);

    /// <summary>
    /// Performs the forward pass on GPU using GPU-accelerated activation kernels.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU tensor with the activation function applied.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when GPU execution is not supported for this activation type.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method applies the activation function entirely on the GPU using optimized kernels.
    /// Supported activations: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Swish.
    /// </para>
    /// <para><b>For Beginners:</b> The GPU version of activation is much faster for large tensors
    /// because GPUs can process thousands of values in parallel.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        if (!TryGetFusedActivationType(out var fusedType))
            throw new InvalidOperationException(
                $"Activation function '{GetActivationTypeName()}' does not support GPU execution. " +
                $"Use the CPU Forward method instead.");

        var input = inputs[0];
        var result = gpuEngine.ActivationGpu<T>(input, fusedType);

        // Cache GPU tensors for BackwardGpu
        if (IsTrainingMode)
        {
            _lastInputGpu = input;
            _lastOutputGpu = result;
        }

        return result;
    }

    /// <summary>
    /// Attempts to map the configured activation function to a FusedActivationType for GPU execution.
    /// </summary>
    private bool TryGetFusedActivationType(out FusedActivationType fusedType)
    {
        // Vector activations (like Softmax) are not supported on GPU yet
        if (_useVectorActivation)
        {
            fusedType = FusedActivationType.None;
            return false;
        }

        IActivationFunction<T>? activation = ScalarActivation;
        if (activation == null)
        {
            fusedType = FusedActivationType.None;
            return false;
        }

        // Map activation function types to GPU kernel types
        fusedType = activation switch
        {
            ReLUActivation<T> => FusedActivationType.ReLU,
            LeakyReLUActivation<T> => FusedActivationType.LeakyReLU,
            SigmoidActivation<T> => FusedActivationType.Sigmoid,
            TanhActivation<T> => FusedActivationType.Tanh,
            GELUActivation<T> => FusedActivationType.GELU,
            SwishActivation<T> or SiLUActivation<T> => FusedActivationType.Swish,
            _ => FusedActivationType.None
        };

        return fusedType != FusedActivationType.None;
    }

    /// <summary>
    /// Gets the name of the activation function type for error messages.
    /// </summary>
    private string GetActivationTypeName()
    {
        if (_useVectorActivation && VectorActivation != null)
            return VectorActivation.GetType().Name;
        if (ScalarActivation != null)
            return ScalarActivation.GetType().Name;
        return "Unknown";
    }
}
