using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents the base class for all neural network layers, providing common functionality and interfaces.
/// </summary>
/// <remarks>
/// <para>
/// LayerBase is an abstract class that serves as the foundation for all neural network layers. It defines 
/// the common structure and functionality that all layers must implement, such as forward and backward 
/// propagation, parameter management, and activation functions. This class handles the core mechanics 
/// of layers in a neural network, allowing derived classes to focus on their specific implementations.
/// </para>
/// <para><b>For Beginners:</b> This is the blueprint that all neural network layers follow.
/// 
/// Think of LayerBase as the common foundation that all layers are built upon:
/// - It defines what every layer must be able to do (process data forward and backward)
/// - It provides shared tools that all layers can use (like activation functions)
/// - It manages the shapes of data flowing in and out of layers
/// - It handles saving and loading layer parameters
/// 
/// All specific layer types (like convolutional, dense, etc.) inherit from this class,
/// which ensures they all work together consistently in a neural network.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class LayerBase<T> : ILayer<T>, IDisposable
{
    /// <summary>
    /// Counter for generating unique instance IDs across all layer instances.
    /// </summary>
    private static int _instanceCounter;

    /// <summary>
    /// The unique instance ID for this layer, used to distinguish multiple instances of the same layer type.
    /// </summary>
    private readonly int _instanceId;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets the element-wise activation function for this layer, if specified.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The scalar activation function applies to individual values in the layer's output tensor.
    /// Common activation functions include ReLU, Sigmoid, and Tanh.
    /// </para>
    /// <para><b>For Beginners:</b> This is the function that adds non-linearity to each value individually.
    /// 
    /// Activation functions:
    /// - Add non-linearity, helping the network learn complex patterns
    /// - Process each number one at a time
    /// - Transform values into more useful ranges (like 0 to 1, or -1 to 1)
    /// 
    /// For example, ReLU turns all negative values to zero while keeping positive values unchanged.
    /// Without activation functions, neural networks couldn't learn complex patterns.
    /// </para>
    /// </remarks>
    public IActivationFunction<T>? ScalarActivation { get; private set; }

    /// <summary>
    /// Gets the vector activation function for this layer, if specified.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vector activation function applies to entire vectors in the layer's output tensor.
    /// This can capture dependencies between different elements of the vectors, such as in Softmax.
    /// </para>
    /// <para><b>For Beginners:</b> This is a more advanced function that processes groups of values together.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - Are used for special purposes like classification (Softmax)
    /// 
    /// For example, Softmax turns a vector of numbers into probabilities that sum to 1,
    /// which is useful for classifying inputs into categories.
    /// </para>
    /// </remarks>
    public IVectorActivationFunction<T>? VectorActivation { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer uses a vector activation function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer is using a vector activation function or an element-wise
    /// activation function. It is used to determine which type of activation to apply during forward and
    /// backward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the layer which type of activation function to use.
    /// 
    /// It's like a switch that determines:
    /// - Whether to process values one by one (scalar activation)
    /// - Or to process groups of values together (vector activation)
    /// 
    /// This helps the layer know which method to use when applying activations.
    /// </para>
    /// </remarks>
    protected bool UsingVectorActivation { get; }

    /// <summary>
    /// Gets the numeric operations provider for type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to numeric operations (like addition, multiplication, etc.) that work
    /// with the generic type T. This allows the layer to perform mathematical operations regardless of
    /// whether T is float, double, or another numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a toolkit for math operations that works with different number types.
    /// 
    /// It provides:
    /// - Basic math operations (add, subtract, multiply, etc.)
    /// - Ways to convert between different number formats
    /// - Special math functions needed by neural networks
    /// 
    /// This allows the layer to work with different types of numbers (float, double, etc.)
    /// without needing different code for each type.
    /// </para>
    /// </remarks>
    protected INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the thread-safe random number generator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the centralized thread-safe random number generator,
    /// which is used for initializing weights and other parameters that require randomization.
    /// </para>
    /// <para><b>For Beginners:</b> This provides random numbers for initializing the layer.
    ///
    /// Random numbers are needed to:
    /// - Set starting values for weights and biases
    /// - Add randomness to avoid symmetry problems
    /// - Help the network learn diverse patterns
    ///
    /// Good initialization with proper randomness is important for neural networks to learn effectively.
    /// </para>
    /// </remarks>
    protected static Random Random => RandomHelper.ThreadSafeRandom;

    /// <summary>
    /// The trainable parameters of this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains all trainable parameters for the layer, such as weights and biases.
    /// The specific interpretation of these parameters depends on the layer type.
    /// </para>
    /// <para><b>For Beginners:</b> These are the values that the layer learns during training.
    /// 
    /// Parameters include:
    /// - Weights that determine how important each input is
    /// - Biases that provide a baseline or starting point
    /// - Other learnable values specific to certain layer types
    /// 
    /// During training, these values are adjusted to make the network's predictions better.
    /// </para>
    /// </remarks>
    protected Vector<T> Parameters;

    /// <summary>
    /// The gradients of the trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the gradients of all trainable parameters for the layer. These gradients
    /// indicate how each parameter should be adjusted during training to reduce the error.
    /// </para>
    /// <para><b>For Beginners:</b> These values show how to adjust the parameters during training.
    /// 
    /// Parameter gradients:
    /// - Tell the network which direction to change each parameter
    /// - Show how sensitive the error is to each parameter
    /// - Guide the learning process
    /// 
    /// A larger gradient means a parameter has more influence on the error and
    /// needs a bigger adjustment during training.
    /// </para>
    /// </remarks>
    protected Vector<T>? ParameterGradients;

    /// <summary>
    /// Gets the input shape for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the shape of the input tensor that the layer expects. For example,
    /// a 2D convolutional layer might expect an input shape of [batchSize, channels, height, width].
    /// </para>
    /// <para><b>For Beginners:</b> This defines the shape of data this layer expects to receive.
    /// 
    /// The input shape:
    /// - Tells the layer how many dimensions the input data has
    /// - Specifies the size of each dimension
    /// - Helps the layer organize its operations properly
    /// 
    /// For example, if processing images that are 28x28 pixels with 1 color channel,
    /// the input shape might be [1, 28, 28] (channels, height, width).
    /// </para>
    /// </remarks>
    protected int[] InputShape { get; private set; }

    /// <summary>
    /// Gets the input shapes for this layer, supporting multiple inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the shapes of all input tensors that the layer expects, for layers that
    /// accept multiple inputs (such as merge layers).
    /// </para>
    /// <para><b>For Beginners:</b> This defines the shapes of all input sources for layers that take multiple inputs.
    /// 
    /// For layers that combine multiple data sources:
    /// - Each input may have a different shape
    /// - This array stores all those shapes
    /// - Helps the layer handle multiple inputs properly
    /// 
    /// For example, a layer that combines features from two different sources
    /// would need to know the shape of each source.
    /// </para>
    /// </remarks>
    protected int[][] InputShapes { get; private set; }

    protected void UpdateInputShape(int[] inputShape)
    {
        if (inputShape == null || inputShape.Length == 0)
        {
            throw new ArgumentException("Input shape must be non-empty.", nameof(inputShape));
        }

        InputShape = inputShape;
        InputShapes = [inputShape];
    }

    /// <summary>
    /// Gets the output shape for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the shape of the output tensor that the layer produces. For example,
    /// a 2D convolutional layer with 16 filters might produce an output shape of [batchSize, 16, height, width].
    /// </para>
    /// <para><b>For Beginners:</b> This defines the shape of data this layer produces as output.
    /// 
    /// The output shape:
    /// - Tells the next layer what shape of data to expect
    /// - Shows how this layer transforms the data dimensions
    /// - Helps verify the network is structured correctly
    /// 
    /// For example, if a layer reduces image size from 28x28 to 14x14 and produces 16 feature maps,
    /// the output shape might be [16, 14, 14] (channels, height, width).
    /// </para>
    /// </remarks>
    protected int[] OutputShape { get; private set; }

    /// <summary>
    /// Gets or sets a value indicating whether the layer is in training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This flag indicates whether the layer is currently in training mode or inference (evaluation) mode.
    /// Some layers behave differently during training versus inference, such as Dropout or BatchNormalization.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the layer whether it's currently training or being used for predictions.
    ///
    /// This mode flag:
    /// - Affects how certain layers behave
    /// - Can turn on/off special training features
    /// - Helps the network switch between learning and using what it learned
    ///
    /// For example, dropout layers randomly turn off neurons during training to improve
    /// generalization, but during inference they don't drop anything.
    /// </para>
    /// </remarks>
    protected bool IsTrainingMode = true;

    /// <summary>
    /// Tracks whether Dispose has been called.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Collection of tensors that have been registered as persistent with the engine.
    /// These will be unregistered when the layer is disposed.
    /// </summary>
    private readonly List<object> _registeredTensors = new();

    /// <summary>
    /// Gets or sets the initialization strategy for this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The initialization strategy controls when and how the layer's weights are allocated
    /// and initialized. Lazy initialization defers weight allocation until the first forward
    /// pass, which significantly speeds up network construction.
    /// </para>
    /// <para><b>For Beginners:</b> This controls when the layer sets up its internal weights.
    ///
    /// Lazy initialization:
    /// - Defers weight allocation until the layer is actually used
    /// - Makes network construction much faster
    /// - Useful for tests and when comparing network architectures
    ///
    /// Eager initialization:
    /// - Allocates weights immediately at construction time
    /// - Traditional behavior, weights are ready immediately
    /// </para>
    /// </remarks>
    public IInitializationStrategy<T>? InitializationStrategy { get; set; }

    /// <summary>
    /// Gets a value indicating whether this layer has been initialized.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For layers with lazy initialization, this indicates whether the weights have been
    /// allocated and initialized. For eager initialization, this is always true after construction.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the layer's weights are ready to use.
    ///
    /// A value of true means:
    /// - Weights have been allocated
    /// - The layer is ready for forward/backward passes
    ///
    /// A value of false means:
    /// - Weights are not yet allocated (lazy initialization)
    /// - The first Forward() call will initialize them
    /// </para>
    /// </remarks>
    public virtual bool IsInitialized => true;

    /// <summary>
    /// Object used for thread-safe lazy initialization.
    /// </summary>
    protected readonly object InitializationLock = new();

    /// <summary>
    /// Ensures that the layer is initialized. Call this at the start of Forward() for lazy initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For layers that support lazy initialization, this method should be called at the start
    /// of Forward() to ensure weights are allocated before use. The default implementation
    /// does nothing (for layers without lazy initialization support).
    /// </para>
    /// <para><b>For Beginners:</b> This makes sure the layer is ready before processing data.
    ///
    /// For lazy initialization:
    /// - First call allocates and initializes weights
    /// - Subsequent calls do nothing (weights already initialized)
    /// - Thread-safe for parallel execution
    /// </para>
    /// </remarks>
    protected virtual void EnsureInitialized()
    {
        // Default implementation does nothing.
        // Layers with lazy initialization override this to allocate weights.
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if the layer has trainable parameters and supports backpropagation; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// Layers with trainable parameters such as weights and biases typically return true, while layers
    /// that only perform fixed transformations (like pooling or activation layers) typically return false.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has parameters that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// A value of false means:
    /// - The layer doesn't have any adjustable parameters
    /// - It performs the same operation regardless of training
    /// - It doesn't need to learn (but may still be useful)
    /// </para>
    /// </remarks>
    public abstract bool SupportsTraining { get; }

    /// <summary>
    /// Gets or sets a value indicating whether this layer uses automatic differentiation for backward passes.
    /// </summary>
    /// <value>
    /// <c>true</c> if the layer should use autodiff; <c>false</c> if it uses manual backward implementation. Default is <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property controls whether the layer uses the automatic differentiation system (autodiff) or
    /// manual backward pass implementations during training. Manual backward passes are typically faster
    /// but require explicit gradient computation code. Autodiff is more flexible and can be useful for:
    /// - Custom layer implementations where manual gradients are complex
    /// - Research and experimentation with novel architectures
    /// - Rapid prototyping of new layer types
    /// </para>
    /// <para><b>For Beginners:</b> This controls how the layer computes gradients during training.
    ///
    /// Two modes are available:
    /// - <b>Manual (default, false):</b> Uses hand-written, optimized gradient code. Faster but requires careful implementation.
    /// - <b>Autodiff (true):</b> Uses automatic differentiation to compute gradients. Slower but more flexible and less error-prone.
    ///
    /// Most users should leave this as false (default) for best performance. Set to true only for:
    /// - Custom layers with complex gradients
    /// - Experimental or research purposes
    /// - When you need guaranteed correct gradients for a new operation
    ///
    /// <b>Note:</b> Autodiff support must be implemented by the specific layer type. Not all layers support autodiff mode yet.
    /// </para>
    /// </remarks>
    public bool UseAutodiff { get; set; } = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with the specified input and output shapes.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer with the specified input and output shapes. It initializes
    /// an empty parameter vector and sets up the single input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with the specified data shapes.
    /// 
    /// When creating a layer, you need to define:
    /// - The shape of data coming in (inputShape)
    /// - The shape of data going out (outputShape)
    /// 
    /// This helps the layer organize its operations and connect properly with other layers.
    /// </para>
    /// </remarks>
    protected LayerBase(int[] inputShape, int[] outputShape)
    {
        _instanceId = Interlocked.Increment(ref _instanceCounter);
        InputShape = inputShape;
        InputShapes = [inputShape];
        OutputShape = outputShape;
        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with the specified shapes and element-wise activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="scalarActivation">The element-wise activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer with the specified input and output shapes and element-wise activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with a standard activation function.
    /// 
    /// In addition to the shapes, this also sets up:
    /// - A scalar activation function that processes each value independently
    /// - The foundation for a layer that transforms data in a specific way
    /// 
    /// For example, you might create a layer with a ReLU activation function,
    /// which turns all negative values to zero while keeping positive values.
    /// </para>
    /// </remarks>
    protected LayerBase(int[] inputShape, int[] outputShape, IActivationFunction<T> scalarActivation)
        : this(inputShape, outputShape)
    {
        ScalarActivation = scalarActivation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with the specified shapes and vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="vectorActivation">The vector activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer with the specified input and output shapes and vector activation function.
    /// Vector activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with an advanced vector-based activation.
    /// 
    /// This constructor:
    /// - Sets up the layer's input and output shapes
    /// - Configures a vector activation that processes groups of values together
    /// - Marks the layer as using vector activation
    /// 
    /// Vector activations like Softmax are important for specific tasks like
    /// classification, where outputs need to be interpreted as probabilities.
    /// </para>
    /// </remarks>
    protected LayerBase(int[] inputShape, int[] outputShape, IVectorActivationFunction<T> vectorActivation)
        : this(inputShape, outputShape)
    {
        VectorActivation = vectorActivation;
        UsingVectorActivation = true;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with multiple input shapes and a specified output shape.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer that accepts multiple inputs with different shapes. This is
    /// useful for layers that combine multiple inputs, such as concatenation or addition layers.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a layer that can handle multiple input sources.
    /// 
    /// When creating a layer that combines different data sources:
    /// - You need to specify the shape of each input source
    /// - The layer needs to know how to handle multiple inputs
    /// - The output shape defines what comes out after combining them
    /// 
    /// For example, a layer that combines features from images and text would
    /// need to know the shape of both the image and text data.
    /// </para>
    /// </remarks>
    protected LayerBase(int[][] inputShapes, int[] outputShape)
    {
        _instanceId = Interlocked.Increment(ref _instanceCounter);
        InputShapes = inputShapes;
        // For multi-input layers, use the first input shape as the primary input shape
        // This ensures GetInputShape() always returns a valid (non-empty) shape
        InputShape = inputShapes.Length > 0 ? inputShapes[0] : [];
        OutputShape = outputShape;
        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with multiple input shapes, a specified output shape, and an element-wise activation function.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="scalarActivation">The element-wise activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer that accepts multiple inputs with different shapes and applies
    /// an element-wise activation function to the output.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a layer that handles multiple inputs and applies a standard activation.
    /// 
    /// This constructor:
    /// - Sets up the layer to accept multiple input sources
    /// - Defines the shape of the combined output
    /// - Adds a scalar activation function that processes each output value independently
    /// 
    /// This is useful for creating complex networks that merge data from different sources.
    /// </para>
    /// </remarks>
    protected LayerBase(int[][] inputShapes, int[] outputShape, IActivationFunction<T> scalarActivation)
        : this(inputShapes, outputShape)
    {
        ScalarActivation = scalarActivation;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LayerBase{T}"/> class with multiple input shapes, a specified output shape, and a vector activation function.
    /// </summary>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <param name="outputShape">The shape of the output tensor.</param>
    /// <param name="vectorActivation">The vector activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Layer that accepts multiple inputs with different shapes and applies
    /// a vector activation function to the output.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a layer that handles multiple inputs and applies a vector-based activation.
    /// 
    /// This constructor:
    /// - Sets up the layer to accept multiple input sources
    /// - Defines the shape of the combined output
    /// - Adds a vector activation function that processes groups of output values together
    /// - Marks the layer as using vector activation
    /// 
    /// This combines the flexibility of multiple inputs with the power of vector activations.
    /// </para>
    /// </remarks>
    protected LayerBase(int[][] inputShapes, int[] outputShape, IVectorActivationFunction<T> vectorActivation)
        : this(inputShapes, outputShape)
    {
        VectorActivation = vectorActivation;
        UsingVectorActivation = true;
    }

    /// <summary>
    /// Sets whether the layer is in training mode or inference mode.
    /// </summary>
    /// <param name="isTraining"><c>true</c> to set the layer to training mode; <c>false</c> to set it to inference mode.</param>
    /// <remarks>
    /// <para>
    /// This method sets the layer's mode to either training or inference (evaluation). Some layers behave
    /// differently during training versus inference, such as Dropout or BatchNormalization. This method
    /// only has an effect if the layer supports training.
    /// </para>
    /// <para><b>For Beginners:</b> This method switches the layer between learning mode and prediction mode.
    /// 
    /// Setting this mode:
    /// - Tells the layer whether to optimize for learning or for making predictions
    /// - Changes behavior in layers like Dropout (which randomly ignores neurons during training)
    /// - Has no effect in layers that don't support training
    /// 
    /// It's important to set this correctly before using a network - training mode for learning,
    /// inference mode for making predictions.
    /// </para>
    /// </remarks>
    public virtual void SetTrainingMode(bool isTraining)
    {
        // Always set training mode, regardless of SupportsTraining.
        // Layers like GaussianNoiseLayer and DropoutLayer need training mode
        // to control their behavior even though they have no trainable parameters.
        IsTrainingMode = isTraining;
    }

    /// <summary>
    /// Gets the gradients of all trainable parameters in this layer.
    /// </summary>
    /// <returns>A vector containing the gradients of all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the gradients of all trainable parameters in the layer. If the gradients
    /// haven't been calculated yet, it initializes a new vector of the appropriate size.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides the current adjustment values for all parameters.
    /// 
    /// The parameter gradients:
    /// - Show how each parameter should be adjusted during training
    /// - Are calculated during the backward pass
    /// - Guide the optimization process
    /// 
    /// These gradients are usually passed to an optimizer like SGD or Adam,
    /// which uses them to update the parameters in a way that reduces errors.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameterGradients()
    {
        if (ParameterGradients == null || ParameterGradients.Length != ParameterCount)
        {
            ParameterGradients = new Vector<T>(ParameterCount);
        }

        return ParameterGradients;
    }

    /// <summary>
    /// Clears all parameter gradients in this layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets all parameter gradients to zero. This is typically called at the beginning of
    /// each batch during training to ensure that gradients from previous batches don't affect the current batch.
    /// </para>
    /// <para><b>For Beginners:</b> This method resets all adjustment values to zero to start fresh.
    /// 
    /// Clearing gradients:
    /// - Erases all previous adjustment information
    /// - Prepares the layer for a new training batch
    /// - Prevents old adjustments from interfering with new ones
    /// 
    /// This is typically done at the start of processing each batch of training data
    /// to ensure clean, accurate gradient calculations.
    /// </para>
    /// </remarks>
    public virtual void ClearGradients()
    {
        if (ParameterGradients != null)
        {
            ParameterGradients.Fill(NumOps.Zero);
        }
    }

    /// <summary>
    /// Gets the input shape for this layer.
    /// </summary>
    /// <returns>The input shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the input shape of the layer. If the layer has multiple input shapes,
    /// it returns the first one.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what shape of data the layer expects.
    /// 
    /// The input shape:
    /// - Shows the dimensions of data this layer processes
    /// - Is needed to connect this layer with previous layers
    /// - Helps verify the network structure is correct
    /// 
    /// For layers with multiple inputs, this returns just the first input shape.
    /// </para>
    /// </remarks>
    public virtual int[] GetInputShape() =>
        InputShape != null && InputShape.Length > 0 ? InputShape : InputShapes[0];

    /// <summary>
    /// Gets all input shapes for this layer.
    /// </summary>
    /// <returns>An array of input shapes.</returns>
    /// <remarks>
    /// <para>
    /// This method returns all input shapes of the layer. This is particularly useful for layers that
    /// accept multiple inputs with different shapes.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you the shapes of all data sources this layer can accept.
    /// 
    /// For layers that combine multiple inputs:
    /// - This returns all the input shapes in an array
    /// - Each shape defines the dimensions of one input source
    /// - Helpful for understanding complex network connections
    /// 
    /// This is most useful for layers like concatenation or merge layers.
    /// </para>
    /// </remarks>
    public virtual int[][] GetInputShapes() => InputShapes;

    /// <summary>
    /// Gets the output shape for this layer.
    /// </summary>
    /// <returns>The output shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the output shape of the layer, which defines the dimensions of the tensor
    /// that will be produced when data flows through this layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what shape of data the layer produces.
    /// 
    /// The output shape:
    /// - Shows the dimensions of data after this layer processes it
    /// - Is needed to connect this layer with the next layer
    /// - Helps verify that data flows correctly through the network
    /// 
    /// For example, a convolutional layer might change the number of channels in the data,
    /// which would be reflected in the output shape.
    /// </para>
    /// </remarks>
    public int[] GetOutputShape() => OutputShape;


    /// <summary>
    /// Gets the weight matrix for layers that have trainable weights.
    /// </summary>
    /// <returns>The weight matrix, or null if the layer has no weights.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the layer's weight matrix for layers that use weights
    /// during computation. Layers without weights (like pooling or activation layers) return null.
    /// </para>
    /// <para><b>For Beginners:</b> Weights are the learnable parameters that define how a layer transforms data.
    ///
    /// For example:
    /// - Dense layers use a weight matrix to transform inputs
    /// - Convolutional layers use filters (which are weights) to detect patterns
    /// - Pooling layers have no weights, so they return null
    ///
    /// This method lets you inspect or modify the weights after training.
    /// </para>
    /// </remarks>
    public virtual Tensor<T>? GetWeights() => null;

    /// <summary>
    /// Gets the bias tensor for layers that have trainable biases.
    /// </summary>
    /// <returns>The bias tensor, or null if the layer has no biases.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the layer's bias tensor for layers that use biases
    /// during computation. Layers without biases return null.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are learnable offsets added to the layer's output.
    ///
    /// Think of biases as a starting point:
    /// - Without bias: output = weights × input
    /// - With bias: output = weights × input + bias
    ///
    /// Biases help the network learn more flexible patterns by shifting the activation function.
    /// </para>
    /// </remarks>
    public virtual Tensor<T>? GetBiases() => null;

    /// <summary>
    /// Exports the layer's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the layer's operation.</returns>
    /// <remarks>
    /// <para>
    /// This method constructs a computation graph representation of the layer's forward pass
    /// that can be JIT compiled for faster inference. All layers MUST implement this method
    /// to support JIT compilation.
    /// </para>
    /// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation converts the layer's operations
    /// into optimized native code for 5-10x faster inference.
    ///
    /// To support JIT compilation, a layer must:
    /// 1. Implement this method to export its computation graph
    /// 2. Set SupportsJitCompilation to true
    /// 3. Use ComputationNode and TensorOperations to build the graph
    ///
    /// All layers are required to implement this method, even if they set SupportsJitCompilation = false.
    /// </para>
    /// </remarks>
    public abstract ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <value>True if the layer can be JIT compiled, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer has implemented ExportComputationGraph()
    /// and can benefit from JIT compilation. All layers MUST implement this property.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation can make inference 5-10x faster by converting
    /// the layer's operations into optimized native code.
    ///
    /// Layers should return false if they:
    /// - Have not yet implemented a working ExportComputationGraph()
    /// - Use dynamic operations that change based on input data
    /// - Are too simple to benefit from JIT compilation
    ///
    /// When false, the layer will use the standard Forward() method instead.
    /// </para>
    /// </remarks>
    public abstract bool SupportsJitCompilation { get; }
    /// <summary>
    /// Performs the forward pass of the layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to define the forward pass of the layer.
    /// The forward pass transforms the input tensor according to the layer's operation and activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the layer.
    /// 
    /// The forward pass:
    /// - Takes input data from the previous layer or the network input
    /// - Applies the layer's specific transformation (like convolution or matrix multiplication)
    /// - Applies any activation function
    /// - Passes the result to the next layer
    /// 
    /// This is where the actual data processing happens during both training and prediction.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Forward(Tensor<T> input);

    #region Mixed Precision Support

    /// <summary>
    /// Gets the name of this layer for mixed-precision policy lookup.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This name is used to determine whether the layer should
    /// use full precision (FP32) or reduced precision (FP16/FP8) during mixed-precision training.
    ///
    /// Layers like BatchNorm, LayerNorm, and Softmax typically need full precision for stability.
    /// The name is matched against patterns in <see cref="MixedPrecision.LayerPrecisionPolicy"/>.
    /// </para>
    /// </remarks>
    public virtual string LayerName => $"{GetType().Name}_{_instanceId}";

    /// <summary>
    /// Gets whether mixed-precision training is currently active.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Check this property to see if the network is currently
    /// running in mixed-precision mode. When true, you may need to handle precision
    /// conversions in your layer implementation.
    /// </para>
    /// </remarks>
    protected static bool IsMixedPrecisionActive => MixedPrecision.MixedPrecisionScope.Current != null;

    /// <summary>
    /// Gets whether this layer should use full precision (FP32) even during mixed-precision training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some layers need higher precision to work correctly:
    /// - Normalization layers (BatchNorm, LayerNorm) compute mean/variance
    /// - Softmax involves exponentials that can overflow in low precision
    /// - Loss computation needs accuracy
    ///
    /// This property checks the current mixed-precision policy to determine if this layer
    /// is one that should stay in full precision.
    /// </para>
    /// </remarks>
    protected bool ShouldUseFP32
    {
        get
        {
            var scope = MixedPrecision.MixedPrecisionScope.Current;
            return scope?.ShouldUseFP32(LayerName) ?? false;
        }
    }

    /// <summary>
    /// Gets the precision type this layer should use based on the current mixed-precision policy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns the specific precision type (FP16, FP32, FP8, etc.)
    /// that this layer should use according to the current training configuration.
    /// </para>
    /// </remarks>
    protected Enums.MixedPrecisionType CurrentPrecision
    {
        get
        {
            var scope = MixedPrecision.MixedPrecisionScope.Current;
            return scope?.GetLayerPrecision(LayerName) ?? Enums.MixedPrecisionType.None;
        }
    }

    /// <summary>
    /// Performs a forward pass with automatic mixed-precision handling.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing with appropriate precision.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method wraps the standard Forward pass with
    /// automatic precision handling for mixed-precision training.
    ///
    /// When mixed-precision is active:
    /// - If the layer should use FP32 (like BatchNorm), it processes in full precision
    /// - If the layer can use lower precision, the scope tracks tensor versions
    /// - The precision policy determines which layers need full precision
    ///
    /// This enables faster training on modern GPUs while maintaining numerical stability.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> ForwardWithPrecisionCheck(Tensor<T> input)
    {
        var scope = MixedPrecision.MixedPrecisionScope.Current;

        // If no scope is active, use standard forward pass
        if (scope == null)
        {
            return Forward(input);
        }

        // Check if this layer requires full precision
        bool requiresFP32 = scope.ShouldUseFP32(LayerName);

        if (requiresFP32)
        {
            // Layer needs FP32 - register the input so it can be retrieved if needed
            // For T=float, this is a no-op in terms of precision but tracks the tensor
            string tensorName = $"{LayerName}_input";
            if (typeof(T) == typeof(float) && !scope.HasTensor(tensorName))
            {
                // Only register if it's a float tensor (standard case)
                var floatInput = input as Tensor<float>;
                if (floatInput != null)
                {
                    scope.RegisterAndCastToFP16(tensorName, floatInput);
                }
            }
        }

        // Perform the forward pass
        // The actual precision handling is done at the network level
        // This method allows layers to be aware of the precision context
        return Forward(input);
    }

    #endregion

    /// <summary>
    /// Maps the layer's activation function to a <see cref="FusedActivationType"/> for GPU-fused operations.
    /// </summary>
    /// <returns>
    /// The corresponding <see cref="FusedActivationType"/> for the layer's activation function,
    /// or <see cref="FusedActivationType.None"/> if no activation is configured or the activation
    /// type is not supported for GPU fusion.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is used by GPU-optimized layers to determine which fused activation kernel to use.
    /// Fused operations combine matrix multiplication, bias addition, and activation into a single
    /// GPU kernel, reducing memory bandwidth and improving performance.
    /// </para>
    /// <para><b>For Beginners:</b> When running on a GPU, combining multiple operations (like
    /// matrix multiply and activation) into one step is faster than doing them separately.
    /// This method tells the GPU which activation function to include in the combined operation.
    /// </para>
    /// </remarks>
    protected FusedActivationType MapActivationToFused()
    {
        // Check scalar activation first, then vector activation
        if (ScalarActivation is not null)
        {
            return MapActivationInstanceToFused(ScalarActivation);
        }

        if (VectorActivation is not null)
        {
            return MapActivationInstanceToFused(VectorActivation);
        }

        return FusedActivationType.None;
    }

    /// <summary>
    /// Maps an activation function instance to its corresponding <see cref="FusedActivationType"/>.
    /// </summary>
    /// <param name="activation">The activation function instance to map.</param>
    /// <returns>The corresponding fused activation type.</returns>
    private static FusedActivationType MapActivationInstanceToFused(object activation)
    {
        return activation switch
        {
            ReLUActivation<T> => FusedActivationType.ReLU,
            GELUActivation<T> => FusedActivationType.GELU,
            SigmoidActivation<T> => FusedActivationType.Sigmoid,
            TanhActivation<T> => FusedActivationType.Tanh,
            LeakyReLUActivation<T> => FusedActivationType.LeakyReLU,
            SwishActivation<T> => FusedActivationType.Swish,
            ELUActivation<T> => FusedActivationType.ELU,
            SELUActivation<T> => FusedActivationType.SELU,
            SoftPlusActivation<T> => FusedActivationType.Softplus,
            SoftmaxActivation<T> => FusedActivationType.Softmax,
            MishActivation<T> => FusedActivationType.Mish,
            HardSwishActivation<T> => FusedActivationType.HardSwish,
            HardSigmoidActivation<T> => FusedActivationType.HardSigmoid,
            HardTanhActivation<T> => FusedActivationType.HardTanh,
            IdentityActivation<T> => FusedActivationType.None,
            _ => FusedActivationType.None
        };
    }

    /// <summary>
    /// Performs the backward pass of the layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to define the backward pass of the layer.
    /// The backward pass propagates error gradients from the output of the layer back to its input,
    /// and calculates gradients for any trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output contributed to errors
    /// 2. It calculates how its parameters should change to reduce errors
    /// 3. It calculates how its input should change, which will be used by earlier layers
    /// 
    /// This is the core of how neural networks learn from their mistakes during training.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Backward(Tensor<T> outputGradient);

    #region GPU Training Infrastructure

    /// <summary>
    /// Gets whether this layer has a GPU execution implementation for inference.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Override this to return true when the layer implements <see cref="ForwardGpu"/>.
    /// The actual <see cref="CanExecuteOnGpu"/> property combines this with engine availability.
    /// </para>
    /// <para><b>For Beginners:</b> This flag indicates if the layer has GPU code for the forward pass.
    /// Set this to true in derived classes that implement ForwardGpu.
    /// </para>
    /// </remarks>
    protected virtual bool SupportsGpuExecution => false;

    /// <summary>
    /// Gets whether this layer has full GPU training support (forward, backward, and parameter updates).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can perform its entire training cycle on GPU
    /// without downloading data to CPU. A layer has full GPU training support when:
    /// <list type="bullet">
    /// <item><description>ForwardGpu is implemented</description></item>
    /// <item><description>BackwardGpu is implemented</description></item>
    /// <item><description>UpdateParametersGpu is implemented (for layers with trainable parameters)</description></item>
    /// <item><description>GPU weight/bias/gradient buffers are properly managed</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if training can happen entirely on GPU.
    /// 
    /// GPU-resident training is much faster because:
    /// - Data stays on GPU between forward and backward passes
    /// - No expensive CPU-GPU transfers during each training step
    /// - GPU kernels handle all gradient computation
    /// 
    /// Only layers that return true here can participate in fully GPU-resident training.
    /// </para>
    /// </remarks>
    public virtual bool SupportsGpuTraining => false;

    /// <summary>
    /// Gets whether this layer can execute its forward pass on GPU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns true when both the layer supports GPU execution AND a GPU engine is currently active.
    /// Use this to check at runtime whether GPU forward pass is available.
    /// </para>
    /// <para><b>For Beginners:</b> Check this before calling ForwardGpu.
    /// It combines "does the layer have GPU code?" with "is the GPU engine active?"
    /// </para>
    /// </remarks>
    public virtual bool CanExecuteOnGpu => SupportsGpuExecution && Engine is DirectGpuTensorEngine;

    /// <summary>
    /// Gets whether this layer can execute GPU training (forward, backward, parameter update).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns true when both the layer supports GPU training AND a GPU engine is currently active.
    /// </para>
    /// <para><b>For Beginners:</b> Check this before attempting GPU-resident training.
    /// If false, training will fall back to CPU operations.
    /// </para>
    /// </remarks>
    public virtual bool CanTrainOnGpu => SupportsGpuTraining && Engine is DirectGpuTensorEngine;

    /// <summary>
    /// Performs the forward pass of the layer on GPU.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensor(s).</param>
    /// <returns>The GPU-resident output tensor.</returns>
    /// <exception cref="NotSupportedException">Thrown when the layer does not support GPU execution.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the layer's forward computation entirely on GPU. The input and output
    /// tensors remain in GPU memory, avoiding expensive CPU-GPU transfers.
    /// </para>
    /// <para><b>For Beginners:</b> This is like Forward() but runs on the graphics card.
    /// 
    /// The key difference:
    /// - Forward() uses CPU tensors that may be copied to/from GPU
    /// - ForwardGpu() keeps everything on GPU the whole time
    /// 
    /// Override this in derived classes that support GPU acceleration.
    /// </para>
    /// </remarks>
    public virtual IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        throw new NotSupportedException(
            $"GPU execution is not supported by {GetType().Name}. Use Forward() instead or check CanExecuteOnGpu first.");
    }

    /// <summary>
    /// Performs the backward pass of the layer on GPU.
    /// </summary>
    /// <param name="outputGradient">The GPU-resident gradient of the loss with respect to the layer's output.</param>
    /// <returns>The GPU-resident gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="NotSupportedException">Thrown when the layer does not support GPU training.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the layer's backward computation entirely on GPU, including:
    /// <list type="bullet">
    /// <item><description>Computing input gradients to pass to previous layers</description></item>
    /// <item><description>Computing and storing weight gradients on GPU (for layers with trainable parameters)</description></item>
    /// <item><description>Computing and storing bias gradients on GPU</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This is like Backward() but runs entirely on GPU.
    /// 
    /// During GPU training:
    /// 1. Output gradients come in (on GPU)
    /// 2. Input gradients are computed (stay on GPU)
    /// 3. Weight/bias gradients are computed and stored (on GPU)
    /// 4. Input gradients are returned for the previous layer
    /// 
    /// All data stays on GPU - no CPU round-trips needed!
    /// </para>
    /// </remarks>
    public virtual IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        throw new NotSupportedException(
            $"GPU backward pass is not supported by {GetType().Name}. Use Backward() instead or check CanTrainOnGpu first.");
    }

    /// <summary>
    /// Updates the layer's parameters on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration specifying the update algorithm and hyperparameters.</param>
    /// <exception cref="NotSupportedException">Thrown when the layer does not support GPU training.</exception>
    /// <remarks>
    /// <para>
    /// This method updates weights and biases directly on GPU using the optimizer specified in the config.
    /// Supported optimizers include SGD, Adam, AdamW, RMSprop, Adagrad, NAG, LARS, and LAMB.
    /// </para>
    /// <para><b>For Beginners:</b> This updates the layer's learned values entirely on GPU.
    /// 
    /// The config determines which optimizer algorithm to use:
    /// - SGD: Simple gradient descent with optional momentum
    /// - Adam: Adaptive learning rates with moment estimates (most popular)
    /// - AdamW: Adam with proper weight decay (recommended for transformers)
    /// 
    /// Using this method keeps all training computation on the GPU for maximum speed.
    /// </para>
    /// </remarks>
    public virtual void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        throw new NotSupportedException(
            $"GPU parameter updates are not supported by {GetType().Name}. Use UpdateParameters() instead or check CanTrainOnGpu first.");
    }

    /// <summary>
    /// Uploads the layer's weights and biases to GPU memory for GPU-resident training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this before starting GPU training to initialize GPU weight buffers.
    /// The CPU weights are copied to GPU and remain there until DownloadWeightsFromGpu is called.
    /// </para>
    /// <para><b>For Beginners:</b> This copies the layer's learned values to the GPU.
    /// 
    /// Call this once at the start of training to:
    /// - Create GPU buffers for weights and biases
    /// - Copy current values from CPU to GPU
    /// - Create GPU buffers for gradients and optimizer states (momentum, etc.)
    /// 
    /// After this, all training can happen on GPU without CPU involvement.
    /// </para>
    /// </remarks>
    public virtual void UploadWeightsToGpu()
    {
        // Default implementation does nothing - layers without trainable parameters don't need this.
        // Layers with parameters should override to upload their specific weight tensors.
    }

    /// <summary>
    /// Downloads the layer's weights and biases from GPU memory back to CPU.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this after GPU training to sync weights back to CPU for:
    /// <list type="bullet">
    /// <item><description>Model checkpointing / saving</description></item>
    /// <item><description>CPU inference</description></item>
    /// <item><description>Inspection of trained weights</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This copies learned values back from GPU to CPU.
    /// 
    /// During GPU training, weights are modified on GPU and the CPU copy is stale.
    /// Call this to:
    /// - Save the model to disk
    /// - Switch to CPU inference
    /// - Examine what the layer learned
    /// 
    /// This is relatively expensive, so only do it when necessary (not every batch).
    /// </para>
    /// </remarks>
    public virtual void DownloadWeightsFromGpu()
    {
        // Default implementation does nothing - layers without trainable parameters don't need this.
        // Layers with parameters should override to download their specific weight tensors.
    }

    /// <summary>
    /// Resets the GPU gradient accumulators to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Call this at the start of each training batch to clear accumulated gradients from the previous batch.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the "how to improve" information from the last batch.
    /// 
    /// Each batch computes new gradients. Before processing a new batch, you need to:
    /// - Clear the old gradients
    /// - Compute fresh gradients for the current batch
    /// - Update weights based on the new gradients
    /// 
    /// If you forget to zero gradients, they accumulate and training goes wrong!
    /// </para>
    /// </remarks>
    public virtual void ZeroGradientsGpu()
    {
        // Default implementation does nothing.
        // Layers with GPU training should override to zero their gradient buffers.
    }

    #endregion

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to define how the layer's parameters
    /// are updated during training. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - The weights, biases, or other parameters are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to extract useful patterns from inputs.
    /// </para>
    /// </remarks>
    public abstract void UpdateParameters(T learningRate);

    /// <summary>
    /// Gets the types of activation functions used by this layer.
    /// </summary>
    /// <returns>An enumerable of activation function types.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the types of activation functions used by this layer. This is useful for
    /// serialization and debugging purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you what kinds of activation functions the layer uses.
    /// 
    /// This information:
    /// - Helps identify what non-linearities are applied in the layer
    /// - Is useful for saving/loading models
    /// - Helps with debugging and visualization
    /// 
    /// The information is returned as standardized activation types (like ReLU, Sigmoid, etc.)
    /// rather than the actual function objects.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<ActivationFunction> GetActivationTypes()
    {
        if (ScalarActivation != null)
        {
            yield return GetActivationTypeFromFunction(ScalarActivation);
        }

        if (VectorActivation != null)
        {
            yield return GetActivationTypeFromFunction(VectorActivation);
        }
    }

    /// <summary>
    /// Gets the standardized activation function type from an activation function object.
    /// </summary>
    /// <param name="activationFunction">The activation function object.</param>
    /// <returns>The standardized activation function type.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the standardized type of an activation function from its runtime type.
    /// This allows the layer to expose a consistent interface for activation function types.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts a specific activation function to its standard type.
    /// 
    /// This conversion:
    /// - Takes an actual activation function object
    /// - Identifies what kind of function it is (ReLU, Sigmoid, etc.)
    /// - Returns a standardized identifier for that function type
    /// 
    /// This helps with saving models and making the network structure more understandable.
    /// </para>
    /// </remarks>
    private static ActivationFunction GetActivationTypeFromFunction(object activationFunction)
    {
        return activationFunction switch
        {
            SoftmaxActivation<T> => ActivationFunction.Softmax,
            SigmoidActivation<T> => ActivationFunction.Sigmoid,
            ReLUActivation<T> => ActivationFunction.ReLU,
            TanhActivation<T> => ActivationFunction.Tanh,
            IdentityActivation<T> => ActivationFunction.Identity,
            LeakyReLUActivation<T> => ActivationFunction.LeakyReLU,
            ELUActivation<T> => ActivationFunction.ELU,
            SELUActivation<T> => ActivationFunction.SELU,
            SoftPlusActivation<T> => ActivationFunction.Softplus,
            SoftSignActivation<T> => ActivationFunction.SoftSign,
            SwishActivation<T> => ActivationFunction.Swish,
            GELUActivation<T> => ActivationFunction.GELU,
            LiSHTActivation<T> => ActivationFunction.LiSHT,
            _ => ActivationFunction.Identity
        };
    }

    /// <summary>
    /// Performs the forward pass of the layer with multiple input tensors.
    /// </summary>
    /// <param name="inputs">The input tensors to process.</param>
    /// <returns>The output tensor after processing.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensors are provided or when input tensors have incompatible shapes.</exception>
    /// <remarks>
    /// <para>
    /// This method implements a default forward pass for layers that accept multiple inputs. By default,
    /// it concatenates the inputs along the channel dimension. Derived classes can override this method
    /// to implement more specific behavior for multiple inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method handles processing multiple inputs through the layer.
    /// 
    /// When a layer needs to combine multiple data sources:
    /// - This method takes all the input tensors
    /// - By default, it combines them by stacking them along the channel dimension
    /// - It checks that the inputs are compatible (same shape except for channels)
    /// - It then passes the combined data forward
    /// 
    /// For example, if combining features from two sources each with 10 channels,
    /// this would create a tensor with 20 channels by default.
    /// 
    /// Specialized layers can override this to combine inputs in different ways.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
        {
            throw new ArgumentException("At least one input tensor is required.");
        }

        if (inputs.Length == 1)
        {
            // If there's only one input, use the standard Forward method
            return Forward(inputs[0]);
        }

        // Default behavior: concatenate along the channel dimension (assuming NCHW format)
        int channelDimension = 1;

        // Ensure all input tensors have the same shape except for the channel dimension
        for (int i = 1; i < inputs.Length; i++)
        {
            if (inputs[i].Rank != inputs[0].Rank)
            {
                throw new ArgumentException($"All input tensors must have the same rank. Tensor at index {i} has a different rank.");
            }

            for (int dim = 0; dim < inputs[i].Rank; dim++)
            {
                if (dim != channelDimension && inputs[i].Shape[dim] != inputs[0].Shape[dim])
                {
                    throw new ArgumentException($"Input tensors must have the same dimensions except for the channel dimension. Mismatch at dimension {dim} for tensor at index {i}.");
                }
            }
        }

        // Calculate the total number of channels
        int totalChannels = inputs.Sum(t => t.Shape[channelDimension]);

        // Create the output shape
        int[] outputShape = new int[inputs[0].Rank];
        Array.Copy(inputs[0].Shape, outputShape, inputs[0].Rank);
        outputShape[channelDimension] = totalChannels;

        // Create the output tensor
        Tensor<T> output = new Tensor<T>(outputShape);

        // Copy data from input tensors to the output tensor
        int channelOffset = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            int channels = inputs[i].Shape[channelDimension];
            for (int c = 0; c < channels; c++)
            {
                var slice = inputs[i].Slice(channelDimension, c, c + 1);
                output.SetSlice(channelDimension, channelOffset + c, slice);
            }

            channelOffset += channels;
        }

        return output;
    }

    /// <summary>
    /// Applies the activation function to a tensor.
    /// </summary>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    protected Tensor<T> ApplyActivation(Tensor<T> input)
    {
        if (VectorActivation != null)
        {
            return VectorActivation.Activate(input);
        }

        if (ScalarActivation != null)
        {
            return ScalarActivation.Activate(input);
        }

        return input;
    }

    /// <summary>
    /// Applies the activation function to a vector.
    /// </summary>
    /// <param name="input">The input vector to activate.</param>
    /// <returns>The activated vector.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the layer's activation function to a vector. It uses the vector activation function
    /// if one is specified, or applies the scalar activation function element-wise if no vector activation is available.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the activation function to a vector of values.
    /// 
    /// This method:
    /// - First checks if a vector activation function is available (processes all elements together)
    /// - If not, uses the scalar activation function (processes each element independently)
    /// - If neither is available, returns the input unchanged (identity function)
    /// 
    /// This flexibility allows the layer to use the most appropriate activation method 
    /// based on what was specified during creation.
    /// </para>
    /// </remarks>
    protected Vector<T> ApplyActivation(Vector<T> input)
    {
        if (VectorActivation != null)
        {
            // Use centralized ActivationHelper for optimized activation dispatch
            return ActivationHelper.ApplyActivation(VectorActivation, input, Engine);
        }
        else if (ScalarActivation != null)
        {
            return input.Transform(ScalarActivation.Activate);
        }
        else
        {
            return input; // Identity activation
        }
    }

    /// <summary>
    /// Applies a scalar activation function to each element of a tensor.
    /// </summary>
    /// <param name="activation">The scalar activation function to apply.</param>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    /// <remarks>
    /// <para>
    /// This helper method applies a scalar activation function to each element of a tensor. If the activation
    /// function is null, it returns the input tensor unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies an activation function to each value in a tensor.
    /// 
    /// Activation functions:
    /// - Transform values in specific ways (like sigmoid squeezes values between 0 and 1)
    /// - Add non-linearity, which helps neural networks learn complex patterns
    /// - Are applied individually to each number in the data
    /// 
    /// If no activation function is provided, the values pass through unchanged.
    /// </para>
    /// </remarks>
    protected Tensor<T> ActivateTensor(IActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return input;
        }

        return input.Transform((x, _) => activation.Activate(x));
    }

    /// <summary>
    /// Applies a vector activation function to a tensor.
    /// </summary>
    /// <param name="activation">The vector activation function to apply.</param>
    /// <param name="input">The input tensor to activate.</param>
    /// <returns>The activated tensor.</returns>
    /// <remarks>
    /// <para>
    /// This helper method applies a vector activation function to a tensor. If the activation function is null,
    /// it returns the input tensor unchanged. Vector activation functions operate on entire tensors at once,
    /// which can be more efficient than element-wise operations.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies an activation function to an entire tensor at once.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of values simultaneously
    /// - Can be more efficient than processing one value at a time
    /// - Provide the same mathematical result but often faster
    /// 
    /// If no activation function is provided, the values pass through unchanged.
    /// </para>
    /// </remarks>
    protected Tensor<T> ActivateTensor(IVectorActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return input;
        }

        return activation.Activate(input);
    }

    /// <summary>
    /// Calculates a standard input shape for 2D data with batch size of 1.
    /// </summary>
    /// <param name="inputDepth">The depth (number of channels) of the input.</param>
    /// <param name="height">The height of the input.</param>
    /// <param name="width">The width of the input.</param>
    /// <returns>An array representing the input shape [batch, depth, height, width].</returns>
    /// <remarks>
    /// <para>
    /// This helper method calculates a standard input shape for 2D data (like images) with a batch size of 1.
    /// The shape follows the NCHW (batch, channels, height, width) format.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a standard shape for image-like data.
    /// 
    /// When working with images or similar 2D data:
    /// - This creates a standard shape array in the format [batch, channels, height, width]
    /// - The batch dimension is set to 1 (processing one item at a time)
    /// - The other dimensions come from the parameters
    /// 
    /// For example, for a 28x28 grayscale image, you might use inputDepth=1, height=28, width=28,
    /// resulting in a shape of [1, 1, 28, 28].
    /// </para>
    /// </remarks>
    protected static int[] CalculateInputShape(int inputDepth, int height, int width)
    {
        return [1, inputDepth, height, width];
    }

    /// <summary>
    /// Calculates a standard output shape for 2D data with batch size of 1.
    /// </summary>
    /// <param name="outputDepth">The depth (number of channels) of the output.</param>
    /// <param name="outputHeight">The height of the output.</param>
    /// <param name="outputWidth">The width of the output.</param>
    /// <returns>An array representing the output shape [batch, depth, height, width].</returns>
    /// <remarks>
    /// <para>
    /// This helper method calculates a standard output shape for 2D data (like images) with a batch size of 1.
    /// The shape follows the NCHW (batch, channels, height, width) format.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a standard shape for image-like output data.
    /// 
    /// When defining the output shape for 2D data:
    /// - This creates a standard shape array in the format [batch, channels, height, width]
    /// - The batch dimension is set to 1 (producing one output at a time)
    /// - The other dimensions come from the parameters
    /// 
    /// For example, if a convolutional layer produces 16 feature maps of size 14x14,
    /// you might use outputDepth=16, outputHeight=14, outputWidth=14.
    /// </para>
    /// </remarks>
    protected static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [1, outputDepth, outputHeight, outputWidth];
    }

    /// <summary>
    /// Creates a copy of this layer.
    /// </summary>
    /// <returns>A new instance of the layer with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a shallow copy of the layer with deep copies of the input/output shapes and
    /// activation functions. Derived classes should override this method to properly copy any additional
    /// fields they define.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a duplicate of this layer.
    /// 
    /// When copying a layer:
    /// - Basic properties like shapes are duplicated
    /// - Activation functions are cloned
    /// - The new layer works independently from the original
    /// 
    /// This is useful for:
    /// - Creating similar layers with small variations
    /// - Implementing complex network architectures with repeated patterns
    /// - Saving a layer's state before making changes
    /// </para>
    /// </remarks>
    public virtual LayerBase<T> Clone()
    {
        var copy = (LayerBase<T>)this.MemberwiseClone();

        // Deep copy any reference type members
        copy.InputShape = (int[])InputShape.Clone();
        copy.OutputShape = (int[])OutputShape.Clone();

        // Copy activation functions (use same instance if not cloneable since they're typically stateless)
        if (ScalarActivation != null)
        {
            copy.ScalarActivation = ScalarActivation is ICloneable cloneable
                ? (IActivationFunction<T>)cloneable.Clone()
                : ScalarActivation;
        }
        if (VectorActivation != null)
        {
            copy.VectorActivation = VectorActivation is ICloneable vectorCloneable
                ? (IVectorActivationFunction<T>)vectorCloneable.Clone()
                : VectorActivation;
        }

        return copy;
    }

    /// <summary>
    /// Calculates the derivative of a scalar activation function for each element of a tensor.
    /// </summary>
    /// <param name="activation">The scalar activation function.</param>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tensor containing the derivatives.</returns>
    /// <remarks>
    /// <para>
    /// This helper method calculates the derivative of a scalar activation function for each element of a tensor.
    /// If the activation function is null, it returns a tensor filled with ones, representing the derivative of
    /// the identity function.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how sensitive each value is to changes.
    /// 
    /// The derivative:
    /// - Measures how much the output changes when the input changes slightly
    /// - Is essential for the backpropagation algorithm during training
    /// - Helps determine how to adjust weights to reduce errors
    /// 
    /// If no activation function is provided, it assumes the identity function (y = x),
    /// which has a derivative of 1 everywhere.
    /// </para>
    /// </remarks>
    protected Tensor<T> DerivativeTensor(IActivationFunction<T>? activation, Tensor<T> input)
    {
        if (activation == null)
        {
            return Tensor<T>.CreateDefault(input.Shape, NumOps.One);
        }

        return activation.Derivative(input);
    }

    /// <summary>
    /// Applies the derivative of the activation function to a single value.
    /// </summary>
    /// <param name="input">The input value.</param>
    /// <param name="outputGradient">The output gradient.</param>
    /// <returns>The input gradient after applying the activation derivative.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the layer's activation function to a single value during the
    /// backward pass. It multiplies the derivative of the activation function at the input value by the
    /// output gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how a small change in one value affects the output.
    /// 
    /// During backpropagation:
    /// - We need to know how sensitive each value is to changes
    /// - This method calculates that sensitivity for a single value
    /// - It multiplies the activation derivative by the incoming gradient
    /// 
    /// This helps determine how much each individual value should be adjusted during training.
    /// </para>
    /// </remarks>
    protected T ApplyActivationDerivative(T input, T outputGradient)
    {
        if (ScalarActivation != null)
        {
            return NumOps.Multiply(ScalarActivation.Derivative(input), outputGradient);
        }
        else
        {
            // Identity activation: derivative is just the output gradient
            return outputGradient;
        }
    }

    /// <summary>
    /// Applies the derivative of the activation function to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="outputGradient">The output gradient tensor.</param>
    /// <returns>The input gradient tensor after applying the activation derivative.</returns>
    /// <exception cref="ArgumentException">Thrown when the input and output gradient tensors have different ranks.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the layer's activation function to a tensor during the
    /// backward pass. It multiplies the derivative of the activation function at each point in the input tensor
    /// by the corresponding output gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how small changes in values affect the output.
    /// 
    /// During backpropagation:
    /// - This method handles tensors (multi-dimensional arrays of values)
    /// - It applies the correct derivative calculation based on the activation type
    /// - For vector activations, it uses the specialized derivative method
    /// - For scalar activations, it applies the derivative to each value independently
    /// 
    /// This is a key part of the math that allows neural networks to learn through backpropagation.
    /// </para>
    /// </remarks>
    protected Tensor<T> ApplyActivationDerivative(Tensor<T> input, Tensor<T> outputGradient)
    {
        if (input.Rank != outputGradient.Rank)
            throw new ArgumentException("Input and output gradient tensors must have the same rank.");

        if (VectorActivation != null)
        {
            var derivative = VectorActivation.Derivative(input);

            if (derivative.Rank == input.Rank)
            {
                return derivative.ElementwiseMultiply(outputGradient);
            }

            if (derivative.Rank != input.Rank + 1)
            {
                throw new ArgumentException("Vector activation derivative tensor has an unexpected rank.");
            }

            int vectorLength = input.Shape[input.Shape.Length - 1];
            int batchElements = input.Length / vectorLength;
            var flatDerivative = derivative.Reshape(new[] { batchElements, vectorLength, vectorLength });
            var flatOutputGrad = outputGradient.Reshape(new[] { batchElements, vectorLength });
            var flatInputGrad = new Tensor<T>(new[] { batchElements, vectorLength });

            for (int i = 0; i < batchElements; i++)
            {
                for (int j = 0; j < vectorLength; j++)
                {
                    T sum = NumOps.Zero;
                    for (int k = 0; k < vectorLength; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(flatDerivative[i, j, k], flatOutputGrad[i, k]));
                    }
                    flatInputGrad[i, j] = sum;
                }
            }

            return flatInputGrad.Reshape(input.Shape);
        }
        else if (ScalarActivation != null)
        {
            // Element-wise application of scalar activation derivative
            // Optimized to use Tensor operations
            return ScalarActivation.Derivative(input).ElementwiseMultiply(outputGradient);
        }
        else
        {
            // Identity activation: derivative is just the output gradient
            return outputGradient;
        }
    }

    /// <summary>
    /// Computes the Jacobian matrix of the activation function for a given input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The Jacobian matrix of the activation function at the input.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the Jacobian matrix of the activation function, which represents how each
    /// output element changes with respect to each input element. For vector activation functions,
    /// it uses the function's derivative method. For scalar activation functions, it creates a diagonal
    /// matrix with the derivatives.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates a matrix that shows how changes in inputs affect outputs.
    /// 
    /// The Jacobian matrix:
    /// - Shows how each output value depends on each input value
    /// - For scalar activations, it's a diagonal matrix (each output depends only on the corresponding input)
    /// - For vector activations, it can have off-diagonal elements (outputs depend on multiple inputs)
    /// 
    /// This is an advanced concept used in certain optimization techniques and for 
    /// precise gradient calculations.
    /// </para>
    /// </remarks>
    protected Matrix<T> ComputeActivationJacobian(Vector<T> input)
    {
        if (VectorActivation != null)
        {
            return VectorActivation.Derivative(input);
        }
        else if (ScalarActivation != null)
        {
            // Create a diagonal matrix with the derivatives
            Vector<T> derivatives = input.Transform(ScalarActivation.Derivative);
            return Matrix<T>.CreateDiagonal(derivatives);
        }
        else
        {
            // Identity function: Jacobian is the identity matrix
            return Matrix<T>.CreateIdentity(input.Length);
        }
    }

    /// <summary>
    /// Applies the derivative of the activation function to a vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="outputGradient">The output gradient vector.</param>
    /// <returns>The input gradient vector after applying the activation derivative.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the activation function to a vector during the backward pass.
    /// It computes the Jacobian matrix of the activation function and multiplies it by the output gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how changes in a vector of values affect the output.
    /// 
    /// For vector operations:
    /// - This method computes the full matrix of relationships between inputs and outputs
    /// - It then multiplies this matrix by the incoming gradient
    /// - The result shows how each input value should be adjusted
    /// 
    /// This is a more comprehensive approach than the element-wise method,
    /// accounting for cases where each output depends on multiple inputs.
    /// </para>
    /// </remarks>
    protected Vector<T> ApplyActivationDerivative(Vector<T> input, Vector<T> outputGradient)
    {
        Matrix<T> jacobian = ComputeActivationJacobian(input);
        return jacobian.Multiply(outputGradient);
    }

    /// <summary>
    /// Updates the parameters of the layer with the given vector of parameter values.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all the parameters of the layer from a single vector of parameters.
    /// The parameters vector must have the correct length to match the total number of parameters in the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer at once.
    /// 
    /// When updating parameters:
    /// - The input must be a vector with the correct length
    /// - This replaces all the current parameters with the new ones
    /// - Throws an error if the input doesn't match the expected number of parameters
    /// 
    /// This is useful for:
    /// - Optimizers that work with all parameters at once
    /// - Applying parameters from another source
    /// - Setting parameters to specific values for testing
    /// </para>
    /// </remarks>
    public virtual void UpdateParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        SetParameters(parameters);
    }

    /// <summary>
    /// Gets the total number of parameters in this layer.
    /// </summary>
    /// <value>
    /// The total number of trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total number of trainable parameters in the layer. By default, it returns
    /// the length of the Parameters vector, but derived classes can override this to calculate the number
    /// of parameters differently.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many learnable values the layer has.
    /// 
    /// The parameter count:
    /// - Shows how complex the layer is
    /// - Indicates how many values need to be learned during training
    /// - Can help estimate memory usage and computational requirements
    /// 
    /// Layers with more parameters can potentially learn more complex patterns
    /// but may also require more data to train effectively.
    /// </para>
    /// </remarks>
    public virtual int ParameterCount => Parameters.Length;

    /// <summary>
    /// Serializes the layer's parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the layer's parameters to a binary writer, which can be used to save the layer's
    /// state to a file or other storage medium. It writes the parameter count followed by each parameter value.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the layer's learned values to storage.
    /// 
    /// When serializing a layer:
    /// - The number of parameters is written first
    /// - Then each parameter value is written
    /// - All values are converted to doubles for consistent storage
    /// 
    /// This allows you to save a trained layer and reload it later without
    /// having to retrain it from scratch.
    /// </para>
    /// </remarks>
    public virtual void Serialize(BinaryWriter writer)
    {
        writer.Write(ParameterCount);
        for (int i = 0; i < ParameterCount; i++)
        {
            writer.Write(Convert.ToDouble(Parameters[i]));
        }
    }

    /// <summary>
    /// Deserializes the layer's parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the layer's parameters from a binary reader, which can be used to load the layer's
    /// state from a file or other storage medium. It reads the parameter count followed by each parameter value.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the layer's learned values from storage.
    /// 
    /// When deserializing a layer:
    /// - The number of parameters is read first
    /// - Then each parameter value is read
    /// - All values are converted from doubles to the appropriate numeric type
    /// 
    /// This allows you to load a previously trained layer without
    /// having to retrain it from scratch.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        Parameters = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            Parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to provide access to all trainable
    /// parameters of the layer as a single vector. This is useful for optimization algorithms that operate
    /// on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include weights, biases, and other learnable values
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public abstract Vector<T> GetParameters();

    /// <summary>
    /// Sets the trainable parameters of the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all the trainable parameters of the layer from a single vector of parameters.
    /// The parameters vector must have the correct length to match the total number of parameters in the layer.
    /// By default, it simply assigns the parameters vector to the Parameters field, but derived classes
    /// may override this to handle the parameters differently.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The layer parses this vector to set all its internal parameters
    /// - Throws an error if the input doesn't match the expected number of parameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}");
        }

        Parameters = parameters;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to reset any internal state the layer
    /// maintains between forward and backward passes. This is useful when starting to process a new sequence
    /// or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Cached inputs and outputs are cleared
    /// - Any temporary calculations are discarded
    /// - The layer is ready to process new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated sequence
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// </para>
    /// </remarks>
    public abstract void ResetState();

    /// <summary>
    /// Gets diagnostic information about this layer's state and behavior.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics for this layer. Base implementation provides
    /// common metrics like layer type, input/output shapes, and parameter count. Derived classes
    /// can override this method to add layer-specific diagnostics.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The base implementation provides the following diagnostics:
    /// <list type="bullet">
    /// <item><description><c>layer.type</c>: The concrete type name of the layer</description></item>
    /// <item><description><c>layer.input_shape</c>: The shape of input tensors</description></item>
    /// <item><description><c>layer.output_shape</c>: The shape of output tensors</description></item>
    /// <item><description><c>layer.parameter_count</c>: The total number of trainable parameters</description></item>
    /// <item><description><c>layer.supports_training</c>: Whether the layer has trainable parameters</description></item>
    /// <item><description><c>layer.activation</c>: The activation function type, if any</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> This method returns a report card with useful information about the layer.
    ///
    /// The diagnostics help you understand:
    /// - What type of layer this is (Dense, Convolutional, etc.)
    /// - What size of data it expects (input shape)
    /// - What size of data it produces (output shape)
    /// - How many parameters it's learning
    /// - What activation function it uses
    ///
    /// Derived classes (specific layer types) can add more detailed information:
    /// - Attention layers might report attention weights statistics
    /// - Batch normalization layers might report running mean/variance
    /// - Dropout layers might report dropout rate
    ///
    /// Example usage:
    /// <code>
    /// var diagnostics = layer.GetDiagnostics();
    /// foreach (var (key, value) in diagnostics)
    /// {
    ///     Console.WriteLine($"{key}: {value}");
    /// }
    /// </code>
    /// </para>
    /// <para>
    /// <b>Override Guidelines:</b>
    /// When overriding in derived classes:
    /// <list type="number">
    /// <item><description>Call base.GetDiagnostics() first to get common metrics</description></item>
    /// <item><description>Add your layer-specific diagnostics to the returned dictionary</description></item>
    /// <item><description>Use consistent key naming (e.g., "activation.mean", "gradient.norm")</description></item>
    /// <item><description>Provide human-readable string values</description></item>
    /// <item><description>Keep computations lightweight to avoid impacting performance</description></item>
    /// </list>
    ///
    /// Example override:
    /// <code>
    /// public override Dictionary&lt;string, string&gt; GetDiagnostics()
    /// {
    ///     var diagnostics = base.GetDiagnostics();
    ///
    ///     if (_lastActivations != null)
    ///     {
    ///         diagnostics["activation.mean"] = ComputeMean(_lastActivations).ToString();
    ///         diagnostics["activation.std"] = ComputeStd(_lastActivations).ToString();
    ///     }
    ///
    ///     return diagnostics;
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            ["layer.type"] = GetType().Name,
            ["layer.input_shape"] = $"[{string.Join(", ", InputShape)}]",
            ["layer.output_shape"] = $"[{string.Join(", ", OutputShape)}]",
            ["layer.parameter_count"] = ParameterCount.ToString(),
            ["layer.supports_training"] = SupportsTraining.ToString()
        };

        // Add activation function information
        if (UsingVectorActivation && VectorActivation != null)
        {
            diagnostics["layer.activation"] = VectorActivation.GetType().Name + " (vector)";
        }
        else if (ScalarActivation != null)
        {
            diagnostics["layer.activation"] = ScalarActivation.GetType().Name + " (scalar)";
        }
        else
        {
            diagnostics["layer.activation"] = "None";
        }

        return diagnostics;
    }

    /// <summary>
    /// Applies the layer's configured activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply activation to.</param>
    /// <returns>The computation node with activation applied.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="NotSupportedException">Thrown if activation does not support JIT.</exception>
    /// <remarks>
    /// <para>
    /// This helper method delegates to the activation's ApplyToGraph method,
    /// following the Open/Closed Principle. Adding new activations does not require
    /// modifying layer code.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds the activation function to the computation graph.
    ///
    /// Instead of the layer code checking what type of activation is configured (which would
    /// require changing the layer every time a new activation is added), this method simply
    /// asks the activation to add itself to the graph. This makes the code more maintainable
    /// and extensible.
    /// </para>
    /// </remarks>
    protected ComputationNode<T> ApplyActivationToGraph(ComputationNode<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        // Check scalar activation first
        if (ScalarActivation is not null)
        {
            if (!ScalarActivation.SupportsJitCompilation)
            {
                throw new NotSupportedException(
                    $"Activation {ScalarActivation.GetType().Name} does not support JIT compilation. " +
                    $"Either the gradient computation is not implemented yet, or the activation " +
                    $"uses operations not compatible with computation graphs.");
            }

            return ScalarActivation.ApplyToGraph(input);
        }

        // Check vector activation
        if (VectorActivation is not null)
        {
            if (!VectorActivation.SupportsJitCompilation)
            {
                throw new NotSupportedException(
                    $"Activation {VectorActivation.GetType().Name} does not support JIT compilation. " +
                    $"Either the gradient computation is not implemented yet, or the activation " +
                    $"uses operations not compatible with computation graphs.");
            }

            return VectorActivation.ApplyToGraph(input);
        }

        // No activation configured (identity)
        return input;
    }

    /// <summary>
    /// Checks if the layer's current activation function supports JIT compilation.
    /// </summary>
    /// <returns>True if the activation can be JIT compiled, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether the layer's configured activation function supports
    /// JIT compilation by querying the activation's SupportsJitCompilation property.
    /// If no activation is configured, returns true (identity function is always JIT-compatible).
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the activation is ready for JIT compilation.
    ///
    /// The layer uses this to determine if it can export a computation graph for faster inference.
    /// If the activation does not support JIT yet (because gradients are not implemented), the
    /// layer will fall back to the standard execution path.
    /// </para>
    /// </remarks>
    protected bool CanActivationBeJitted()
    {
        if (ScalarActivation is not null)
            return ScalarActivation.SupportsJitCompilation;

        if (VectorActivation is not null)
            return VectorActivation.SupportsJitCompilation;

        // No activation (identity) always supports JIT
        return true;
    }

    /// <summary>
    /// Returns layer-specific metadata for serialization purposes.
    /// </summary>
    /// <returns>A dictionary of metadata key-value pairs.</returns>
    /// <remarks>
    /// This is intentionally internal to avoid expanding the public API surface area. Derived layers can
    /// override to provide constructor-level settings that are not inferable from shapes/parameters alone
    /// (e.g., attention head count, masking mode, configuration flags).
    /// </remarks>
    internal virtual Dictionary<string, string> GetMetadata()
    {
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

        // Use AssemblyQualifiedName so deserialization can recreate the exact type.
        if (ScalarActivation != null)
        {
            metadata["ScalarActivationType"] = ScalarActivation.GetType().AssemblyQualifiedName ?? ScalarActivation.GetType().FullName ?? string.Empty;
        }

        if (VectorActivation != null)
        {
            metadata["VectorActivationType"] = VectorActivation.GetType().AssemblyQualifiedName ?? VectorActivation.GetType().FullName ?? string.Empty;
        }

        return metadata;
    }

    #region GPU Persistent Tensor Registration

    /// <summary>
    /// Registers a trainable parameter tensor with the engine for GPU memory optimization.
    /// </summary>
    /// <param name="tensor">The tensor to register (typically weights or biases).</param>
    /// <param name="role">The role of the tensor for optimization hints.</param>
    /// <remarks>
    /// <para>
    /// This method hints to the engine that the tensor will be reused across many operations
    /// and should be kept resident in GPU memory when a GPU engine is active. This avoids
    /// expensive CPU-GPU data transfers on every forward pass.
    /// </para>
    /// <para><b>Performance Impact:</b></para>
    /// <para>
    /// Without registration: Layer weights (e.g., 285MB for a large Dense layer) are
    /// transferred to GPU on every forward pass.
    /// </para>
    /// <para>
    /// With registration: Weights are transferred once and cached on GPU. Only activations
    /// (much smaller) are transferred per pass. Expected speedup: 100-1000x for large layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells the GPU to keep certain data (like learned weights)
    /// in its fast memory instead of copying it back and forth every time. Think of it like keeping
    /// frequently used books on your desk instead of walking to the library each time.
    /// </para>
    /// <para><b>Usage Pattern:</b></para>
    /// <para>
    /// Call this method in the layer's constructor after initializing weight tensors:
    /// <code>
    /// public DenseLayer(int inputSize, int outputSize)
    /// {
    ///     _weights = new Tensor&lt;T&gt;(outputSize, inputSize);
    ///     _biases = new Tensor&lt;T&gt;(outputSize);
    ///     InitializeWeights();
    ///
    ///     // Register for GPU persistence
    ///     RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
    ///     RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    protected void RegisterTrainableParameter(Tensor<T> tensor, PersistentTensorRole role)
    {
        if (tensor is null)
            throw new ArgumentNullException(nameof(tensor));

        Engine.RegisterPersistentTensor(tensor, role);
        _registeredTensors.Add(tensor);
    }

    /// <summary>
    /// Notifies the engine that a registered persistent tensor's data has changed.
    /// </summary>
    /// <param name="tensor">The tensor whose data has been modified.</param>
    /// <remarks>
    /// <para>
    /// Call this method after modifying a registered tensor's data (e.g., during parameter updates).
    /// The engine will re-upload the data to GPU on the next operation that uses the tensor.
    /// </para>
    /// <para><b>For Beginners:</b> When you change the values in a registered tensor (like updating
    /// weights during training), you need to tell the GPU that the copy it has is outdated.
    /// This method does that - it tells the GPU "hey, this data changed, please get a fresh copy."
    /// </para>
    /// <para><b>Usage Pattern:</b></para>
    /// <para>
    /// Call after UpdateParameters modifies weights:
    /// <code>
    /// public override void UpdateParameters(T learningRate)
    /// {
    ///     // Update weights using gradients
    ///     _weights = _weights.Subtract(_weightGradients.Multiply(learningRate));
    ///
    ///     // Notify engine that GPU copy is stale
    ///     InvalidateTrainableParameter(_weights);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    protected void InvalidateTrainableParameter(Tensor<T> tensor)
    {
        if (tensor is null)
            throw new ArgumentNullException(nameof(tensor));

        Engine.InvalidatePersistentTensor(tensor);
    }

    #region OCP-Compliant Activation GPU Methods

    /// <summary>
    /// Checks if the layer's scalar activation function supports GPU training.
    /// </summary>
    /// <returns>True if the activation function has GPU kernels; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Not all activation functions have GPU implementations yet.
    /// This method checks whether the layer's activation can run entirely on the GPU.
    /// If false, the layer must fall back to CPU computation for the activation.
    /// </para>
    /// </remarks>
    protected bool HasGpuActivation()
    {
        return ScalarActivation?.SupportsGpuTraining ?? false;
    }

    /// <summary>
    /// Applies the layer's activation function forward pass on GPU using the activation's own GPU method.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <returns>True if the activation was applied on GPU; false if no activation or GPU not supported.</returns>
    /// <remarks>
    /// <para>
    /// This method follows the Open/Closed Principle by delegating to the activation function's
    /// own GPU implementation rather than using a switch statement on activation types.
    /// Each activation function knows how to apply itself on GPU.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Instead of having one giant switch statement that handles every
    /// possible activation type, each activation function has its own ForwardGpu method.
    /// This makes it easy to add new activation functions without modifying this code.
    /// </para>
    /// </remarks>
    protected bool ApplyActivationForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        if (ScalarActivation is not { SupportsGpuTraining: true })
        {
            return false;
        }

        ScalarActivation.ForwardGpu(backend, input, output, size);
        return true;
    }

    /// <summary>
    /// Applies the layer's activation function backward pass on GPU using the activation's own GPU method.
    /// </summary>
    /// <param name="backend">The GPU backend to use for execution.</param>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="input">The input buffer from the forward pass (needed for some activations).</param>
    /// <param name="output">The output buffer from the forward pass (needed for some activations).</param>
    /// <param name="gradInput">The buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <returns>True if the backward pass was applied on GPU; false if no activation or GPU not supported.</returns>
    /// <remarks>
    /// <para>
    /// This method follows the Open/Closed Principle by delegating to the activation function's
    /// own GPU backward implementation. Each activation function knows what it needs:
    /// - ReLU, GELU, Swish, LeakyReLU, SiLU, Mish, etc.: Need the input from forward pass
    /// - Sigmoid, Tanh: Need the output from forward pass
    /// - ELU: Needs both input and output from forward pass
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, we need to compute how the activation affects
    /// the gradients. Each activation function handles this differently, and by delegating
    /// to the activation's BackwardGpu method, we don't need to know the details here.
    /// </para>
    /// </remarks>
    protected bool ApplyActivationBackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
    {
        if (ScalarActivation is not { SupportsGpuTraining: true })
        {
            return false;
        }

        ScalarActivation.BackwardGpu(backend, gradOutput, input, output, gradInput, size);
        return true;
    }

    #endregion

    /// <summary>
    /// Gets the fused activation type for IEngine fused operations.
    /// </summary>
    /// <returns>The FusedActivationType enum value for the current activation function.</returns>
    /// <remarks>
    /// <para>
    /// This method maps the layer's activation function to a FusedActivationType enum value,
    /// allowing IEngine to use optimized fused GPU kernels (e.g., GEMM+Bias+ReLU in one kernel).
    /// </para>
    /// <para><b>For Beginners:</b> GPU operations are faster when combined.
    /// Instead of doing MatMul, then adding bias, then applying ReLU as separate steps,
    /// fused operations do all three in one GPU kernel - this is 20-50% faster.
    /// This method tells the GPU which activation to fuse with other operations.
    /// </para>
    /// <para><b>Supported Activations:</b></para>
    /// <list type="bullet">
    /// <item><description>ReLU → FusedActivationType.ReLU</description></item>
    /// <item><description>Sigmoid → FusedActivationType.Sigmoid</description></item>
    /// <item><description>Tanh → FusedActivationType.Tanh</description></item>
    /// <item><description>GELU → FusedActivationType.GELU</description></item>
    /// <item><description>LeakyReLU → FusedActivationType.LeakyReLU</description></item>
    /// <item><description>Swish/SiLU → FusedActivationType.Swish</description></item>
    /// <item><description>Other/None → FusedActivationType.None (activation applied separately)</description></item>
    /// </list>
    /// </remarks>
    protected FusedActivationType GetFusedActivationType()
    {
        return ScalarActivation switch
        {
            ReLUActivation<T> => FusedActivationType.ReLU,
            SigmoidActivation<T> => FusedActivationType.Sigmoid,
            TanhActivation<T> => FusedActivationType.Tanh,
            GELUActivation<T> => FusedActivationType.GELU,
            LeakyReLUActivation<T> => FusedActivationType.LeakyReLU,
            SwishActivation<T> => FusedActivationType.Swish,
            SiLUActivation<T> => FusedActivationType.Swish, // SiLU is the same as Swish
            _ => FusedActivationType.None
        };
    }

    /// <summary>
    /// Applies the specified activation function on GPU using the direct backend operations.
    /// </summary>
    /// <param name="backend">The GPU backend to use for activation.</param>
    /// <param name="input">The input GPU buffer.</param>
    /// <param name="output">The output GPU buffer.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <param name="activation">The type of activation function to apply.</param>
    /// <remarks>
    /// <para>
    /// This method is primarily used for fused kernel operations where the activation type
    /// is specified via the <see cref="FusedActivationType"/> enum. It maps enum values to
    /// the corresponding backend activation kernels.
    /// </para>
    /// <para>
    /// <b>Note:</b> For new code, prefer using <see cref="ApplyActivationForwardGpu"/> which
    /// follows the Open/Closed Principle by delegating to each activation function's own
    /// GPU implementation. This allows new activation functions to be added without modifying
    /// this switch statement.
    /// </para>
    /// <para>
    /// This static method only supports common activations (ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Swish).
    /// For other activations, use the OCP-compliant method instead.
    /// </para>
    /// </remarks>
    protected static void ApplyGpuActivation(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size, FusedActivationType activation)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                backend.Relu(input, output, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(input, output, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(input, output, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(input, output, size);
                break;
            case FusedActivationType.LeakyReLU:
                backend.LeakyRelu(input, output, 0.01f, size);
                break;
            case FusedActivationType.Swish:
                backend.Swish(input, output, size);
                break;
            default:
                backend.Copy(input, output, size);
                break;
        }
    }

    /// <summary>
    /// Applies the backward pass of the specified activation function on GPU.
    /// </summary>
    /// <param name="backend">The GPU backend to use for activation backward.</param>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The input from the forward pass (needed for ReLU, LeakyReLU, GELU, Swish).</param>
    /// <param name="output">The output from the forward pass (needed for Sigmoid, Tanh).</param>
    /// <param name="gradInput">The buffer to store the input gradient.</param>
    /// <param name="size">The number of elements to process.</param>
    /// <param name="activation">The type of activation function.</param>
    /// <param name="alpha">Alpha parameter for LeakyReLU (default 0.01).</param>
    /// <returns>True if the backward was handled on GPU, false if CPU fallback is needed.</returns>
    /// <remarks>
    /// <para>
    /// This method is primarily used for fused kernel operations where the activation type
    /// is specified via the <see cref="FusedActivationType"/> enum.
    /// </para>
    /// <para>
    /// <b>Note:</b> For new code, prefer using <see cref="ApplyActivationBackwardGpu"/> which
    /// follows the Open/Closed Principle by delegating to each activation function's own
    /// GPU backward implementation. This allows new activation functions to be added without
    /// modifying this switch statement.
    /// </para>
    /// <para>
    /// Different activation functions require different cached values from forward pass:
    /// <list type="bullet">
    /// <item><description>ReLU, LeakyReLU, GELU, Swish: Need the input from forward pass</description></item>
    /// <item><description>Sigmoid, Tanh: Need the output from forward pass</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected static bool ApplyGpuActivationBackward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer? input,
        IGpuBuffer? output,
        IGpuBuffer gradInput,
        int size,
        FusedActivationType activation,
        float alpha = 0.01f)
    {
        switch (activation)
        {
            case FusedActivationType.ReLU:
                if (input == null) return false;
                backend.ReluBackward(gradOutput, input, gradInput, size);
                return true;

            case FusedActivationType.Sigmoid:
                if (output == null) return false;
                backend.SigmoidBackward(gradOutput, output, gradInput, size);
                return true;

            case FusedActivationType.Tanh:
                if (output == null) return false;
                backend.TanhBackward(gradOutput, output, gradInput, size);
                return true;

            case FusedActivationType.GELU:
                if (input == null) return false;
                backend.GeluBackward(gradOutput, input, gradInput, size);
                return true;

            case FusedActivationType.LeakyReLU:
                if (input == null) return false;
                backend.LeakyReluBackward(gradOutput, input, gradInput, alpha, size);
                return true;

            case FusedActivationType.Swish:
                if (input == null) return false;
                backend.SwishBackward(gradOutput, input, gradInput, size);
                return true;

            case FusedActivationType.None:
                // No activation means gradient passes through unchanged
                backend.Copy(gradOutput, gradInput, size);
                return true;

            default:
                // Unsupported activation - caller should use CPU fallback
                return false;
        }
    }

    #endregion

    /// <summary>
    /// Releases all resources used by this layer, including any GPU resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method releases any resources allocated by the layer, including GPU memory for
    /// persistent tensors. All layers that allocate resources should override Dispose(bool)
    /// to properly release them.
    /// </para>
    /// <para><b>For Beginners:</b> GPU memory is limited and precious.
    ///
    /// When you're done with a layer:
    /// - Call Dispose() or use a 'using' statement
    /// - This frees up GPU memory for other operations
    /// - Failing to dispose can cause memory leaks
    ///
    /// Example:
    /// <code>
    /// using var layer = new DenseLayer&lt;float&gt;(784, 128);
    /// // ... use layer ...
    /// // Automatically disposed when out of scope
    /// </code>
    /// </para>
    /// </remarks>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by this layer.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    /// <remarks>
    /// <para>
    /// Override this method in derived classes to release layer-specific resources.
    /// Always call base.Dispose(disposing) after releasing your resources.
    /// </para>
    /// <para><b>For Beginners:</b> When creating a custom layer with resources:
    ///
    /// <code>
    /// protected override void Dispose(bool disposing)
    /// {
    ///     if (disposing)
    ///     {
    ///         // Release your managed resources here
    ///         _myGpuHandle?.Dispose();
    ///         _myGpuHandle = null;
    ///     }
    ///     base.Dispose(disposing);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            // Unregister all persistent tensors from the engine
            // This releases GPU memory that was cached for these tensors
            foreach (var tensorObj in _registeredTensors)
            {
                // Use reflection to call the generic UnregisterPersistentTensor method
                // since we stored tensors as object to support multiple generic types
                if (tensorObj is Tensor<T> tensor)
                {
                    Engine.UnregisterPersistentTensor(tensor);
                }
            }
            _registeredTensors.Clear();
        }

        _disposed = true;
    }

    // Note: No finalizer in base class - derived classes should implement their own
    // finalizer only if they have unmanaged resources that need cleanup.
    // Calling virtual methods from finalizers is unsafe because the derived class
    // may have already been finalized.

    #region IWeightLoadable Implementation

    /// <summary>
    /// Standard parameter name for weight tensors.
    /// </summary>
    protected const string WeightParameterName = "weight";

    /// <summary>
    /// Standard parameter name for bias tensors.
    /// </summary>
    protected const string BiasParameterName = "bias";

    /// <summary>
    /// Sets the weight tensor for this layer.
    /// </summary>
    /// <param name="weights">The weight tensor to set.</param>
    /// <exception cref="InvalidOperationException">Thrown if the layer does not support weights.</exception>
    /// <remarks>
    /// <para>
    /// Derived classes with trainable weights should override this method to update their internal weight storage.
    /// The default implementation throws an exception since LayerBase doesn't know the layer's weight structure.
    /// </para>
    /// </remarks>
    protected virtual void SetWeights(Tensor<T> weights)
    {
        throw new InvalidOperationException(
            $"Layer type {GetType().Name} does not support setting weights. " +
            $"Override SetWeights(Tensor<T>) in derived class to enable weight loading.");
    }

    /// <summary>
    /// Sets the bias tensor for this layer.
    /// </summary>
    /// <param name="biases">The bias tensor to set.</param>
    /// <exception cref="InvalidOperationException">Thrown if the layer does not support biases.</exception>
    /// <remarks>
    /// <para>
    /// Derived classes with trainable biases should override this method to update their internal bias storage.
    /// The default implementation throws an exception since LayerBase doesn't know the layer's bias structure.
    /// </para>
    /// </remarks>
    protected virtual void SetBiases(Tensor<T> biases)
    {
        throw new InvalidOperationException(
            $"Layer type {GetType().Name} does not support setting biases. " +
            $"Override SetBiases(Tensor<T>) in derived class to enable bias loading.");
    }

    /// <summary>
    /// Gets all parameter names in this layer.
    /// </summary>
    /// <returns>A collection of parameter names ("weight", "bias", or both depending on layer type).</returns>
    /// <remarks>
    /// <para>
    /// The default implementation returns "weight" and/or "bias" based on whether
    /// GetWeights() and GetBiases() return non-null values.
    /// </para>
    /// </remarks>
    public virtual IEnumerable<string> GetParameterNames()
    {
        var names = new List<string>();

        if (GetWeights() is not null)
        {
            names.Add(WeightParameterName);
        }

        if (GetBiases() is not null)
        {
            names.Add(BiasParameterName);
        }

        return names;
    }

    /// <summary>
    /// Tries to get a parameter tensor by name.
    /// </summary>
    /// <param name="name">The parameter name ("weight" or "bias").</param>
    /// <param name="tensor">The parameter tensor if found.</param>
    /// <returns>True if the parameter was found, false otherwise.</returns>
    public virtual bool TryGetParameter(string name, out Tensor<T>? tensor)
    {
        if (string.Equals(name, WeightParameterName, StringComparison.OrdinalIgnoreCase))
        {
            tensor = GetWeights();
            return tensor is not null;
        }

        if (string.Equals(name, BiasParameterName, StringComparison.OrdinalIgnoreCase))
        {
            tensor = GetBiases();
            return tensor is not null;
        }

        tensor = null;
        return false;
    }

    /// <summary>
    /// Sets a parameter tensor by name.
    /// </summary>
    /// <param name="name">The parameter name ("weight" or "bias").</param>
    /// <param name="value">The tensor value to set.</param>
    /// <returns>True if the parameter was set successfully, false if the name was not found.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor shape doesn't match expected shape.</exception>
    public virtual bool SetParameter(string name, Tensor<T> value)
    {
        var expectedShape = GetParameterShape(name);
        if (expectedShape is null)
        {
            return false;
        }

        // Validate shape
        var actualShape = value.Shape;
        if (actualShape.Length != expectedShape.Length)
        {
            throw new ArgumentException(
                $"Shape mismatch for '{name}': expected rank {expectedShape.Length}, got {actualShape.Length}");
        }

        for (int i = 0; i < expectedShape.Length; i++)
        {
            if (actualShape[i] != expectedShape[i])
            {
                throw new ArgumentException(
                    $"Shape mismatch for '{name}': expected [{string.Join(", ", expectedShape)}], " +
                    $"got [{string.Join(", ", actualShape)}]");
            }
        }

        if (string.Equals(name, WeightParameterName, StringComparison.OrdinalIgnoreCase))
        {
            SetWeights(value);
            return true;
        }

        if (string.Equals(name, BiasParameterName, StringComparison.OrdinalIgnoreCase))
        {
            SetBiases(value);
            return true;
        }

        return false;
    }

    /// <summary>
    /// Gets the expected shape for a parameter.
    /// </summary>
    /// <param name="name">The parameter name ("weight" or "bias").</param>
    /// <returns>The expected shape, or null if the parameter doesn't exist.</returns>
    public virtual int[]? GetParameterShape(string name)
    {
        if (string.Equals(name, WeightParameterName, StringComparison.OrdinalIgnoreCase))
        {
            var weights = GetWeights();
            return weights?.Shape;
        }

        if (string.Equals(name, BiasParameterName, StringComparison.OrdinalIgnoreCase))
        {
            var biases = GetBiases();
            return biases?.Shape;
        }

        return null;
    }

    /// <summary>
    /// Gets the total number of named parameters.
    /// </summary>
    public virtual int NamedParameterCount
    {
        get
        {
            int count = 0;
            if (GetWeights() is not null) count++;
            if (GetBiases() is not null) count++;
            return count;
        }
    }

    /// <summary>
    /// Validates that a set of weight names can be loaded into this layer.
    /// </summary>
    /// <param name="weightNames">Names of weights to validate.</param>
    /// <param name="mapping">Optional weight name mapping function.</param>
    /// <returns>Validation result with matched and unmatched names.</returns>
    public virtual WeightLoadValidation ValidateWeights(
        IEnumerable<string> weightNames,
        Func<string, string?>? mapping = null)
    {
        var result = new WeightLoadValidation();
        var paramNames = new HashSet<string>(GetParameterNames(), StringComparer.OrdinalIgnoreCase);
        var matchedParams = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var weightName in weightNames)
        {
            var targetName = mapping?.Invoke(weightName) ?? weightName;
            if (targetName is null)
            {
                result.UnmatchedWeights.Add(weightName);
                continue;
            }

            if (paramNames.Contains(targetName))
            {
                result.Matched.Add(targetName);
                matchedParams.Add(targetName);
            }
            else
            {
                result.UnmatchedWeights.Add(weightName);
            }
        }

        // Find missing parameters
        foreach (var paramName in paramNames)
        {
            if (!matchedParams.Contains(paramName))
            {
                result.MissingParameters.Add(paramName);
            }
        }

        return result;
    }

    /// <summary>
    /// Loads weights from a dictionary of tensors using optional name mapping.
    /// </summary>
    /// <param name="weights">Dictionary of weight name to tensor.</param>
    /// <param name="mapping">Optional function to map source names to target names.</param>
    /// <param name="strict">If true, fails when any mapped weight fails to load.</param>
    /// <returns>Load result with statistics.</returns>
    public virtual WeightLoadResult LoadWeights(
        Dictionary<string, Tensor<T>> weights,
        Func<string, string?>? mapping = null,
        bool strict = false)
    {
        var result = new WeightLoadResult { Success = true };

        foreach (var kvp in weights)
        {
            var sourceName = kvp.Key;
            var tensor = kvp.Value;

            var targetName = mapping?.Invoke(sourceName) ?? sourceName;
            if (targetName is null)
            {
                result.SkippedCount++;
                continue;
            }

            try
            {
                if (SetParameter(targetName, tensor))
                {
                    result.LoadedCount++;
                    result.LoadedParameters.Add(targetName);
                }
                else
                {
                    if (strict)
                    {
                        result.FailedCount++;
                        result.FailedParameters.Add((targetName, "Parameter not found"));
                        result.Success = false;
                    }
                    else
                    {
                        result.SkippedCount++;
                    }
                }
            }
            catch (ArgumentException ex)
            {
                result.FailedCount++;
                result.FailedParameters.Add((targetName, ex.Message));
                if (strict)
                {
                    result.Success = false;
                    result.ErrorMessage = $"Failed to load '{targetName}': {ex.Message}";
                }
            }
        }

        return result;
    }

    #endregion
}
