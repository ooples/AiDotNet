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
public abstract class LayerBase<T> : ILayer<T>
{
    /// <summary>
    /// Gets the name of this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a label or identifier for the layer, making it easier to track
    /// and debug which layer is which in a neural network with many layers.
    /// </remarks>
    public virtual string Name { get; protected set; } = "Layer";

    /// <summary>
    /// Gets the size of the input dimension for this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many input values this layer expects.
    /// For example, if the layer expects a vector of 784 values (like flattened 28x28 images),
    /// this property would return 784.
    /// </remarks>
    public virtual int InputSize => InputShape?.Aggregate((a, b) => a * b) ?? 0;

    /// <summary>
    /// Gets the size of the output dimension for this layer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many output values this layer produces.
    /// For example, a dense layer with 10 neurons will output 10 values,
    /// so this property would return 10.
    /// </remarks>
    public virtual int OutputSize => OutputShape?.Aggregate((a, b) => a * b) ?? 0;

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
    protected IActivationFunction<T>? ScalarActivation { get; private set; }

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
    /// Vector<double> activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - Are used for special purposes like classification (Softmax)
    /// 
    /// For example, Softmax turns a vector of numbers into probabilities that sum to 1,
    /// which is useful for classifying inputs into categories.
    /// </para>
    /// </remarks>
    protected IVectorActivationFunction<T>? VectorActivation { get; private set; }

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
    /// Gets a random number generator.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to a random number generator, which is used for initializing weights
    /// and other parameters that require randomization.
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
    protected Random Random => new();

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
    /// Vector<double> activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new layer with an advanced vector-based activation.
    /// 
    /// This constructor:
    /// - Sets up the layer's input and output shapes
    /// - Configures a vector activation that processes groups of values together
    /// - Marks the layer as using vector activation
    /// 
    /// Vector<double> activations like Softmax are important for specific tasks like
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
        InputShapes = inputShapes;
        InputShape = inputShapes.Length == 1 ? inputShapes[0] : [];
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
        if (SupportsTraining)
        {
            IsTrainingMode = isTraining;
        }
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
    public virtual int[] GetInputShape() => InputShape ?? InputShapes[0];

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
                throw new ArgumentException($"All input tensors must have the same rank. Tensor<double> at index {i} has a different rank.");
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
    /// Applies the activation function to each element of the input tensor while preserving its shape.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>A new tensor with the same shape as the input, where the activation function has been applied to each element.</returns>
    /// <exception cref="ArgumentNullException">Thrown when the input tensor is null.</exception>
    /// <remarks>
    /// This method efficiently applies the activation function to all elements while preserving the tensor's shape.
    /// For large tensors, parallel processing is used to improve performance.
    /// </remarks>
    protected Tensor<T> ApplyActivation(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var result = new Tensor<T>(input.Shape);

        // For small tensors, use simple iteration
        if (input.Length < 10000) // Threshold can be tuned based on benchmarks
        {
            // Use indexers to process any rank tensor
            var indices = new int[input.Rank];
            ApplyActivationRecursive(input, result, indices, 0);
        }
        // For larger tensors, use parallel processing
        else
        {
            // For rank-2 tensors, parallelize over the first dimension
            if (input.Rank == 2)
            {
                Parallel.For(0, input.Shape[0], i =>
                {
                    for (int j = 0; j < input.Shape[1]; j++)
                    {
                        result[i, j] = Activation(input[i, j]);
                    }
                });
            }
            // For other ranks, use vector processing
            else
            {
                // Use a thread-safe implementation
                var inputArray = input.ToArray();
                var resultVector = new Vector<T>(inputArray.Length);

                Parallel.For(0, inputArray.Length, i =>
                {
                    result[i] = Activation(inputArray[i]);
                });

                result = Tensor<T>.FromVector(resultVector, input.Shape);
            }
        }

        return result;
    }

    /// <summary>
    /// Recursively applies the activation function to each element of the tensor.
    /// </summary>
    private void ApplyActivationRecursive(Tensor<T> input, Tensor<T> result, int[] indices, int dimension)
    {
        if (dimension == input.Rank)
        {
            // We have full indices, apply activation
            result[indices] = Activation(input[indices]);
            return;
        }

        for (int i = 0; i < input.Shape[dimension]; i++)
        {
            indices[dimension] = i;
            ApplyActivationRecursive(input, result, indices, dimension + 1);
        }
    }

    /// <summary>
    /// Applies activation function to a single value.
    /// </summary>
    private T Activation(T value)
    {
        return ScalarActivation != null ? ScalarActivation.Activate(value) : value;
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
            return VectorActivation.Activate(input);
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
    /// it returns the input tensor unchanged. Vector<double> activation functions operate on entire tensors at once,
    /// which can be more efficient than element-wise operations.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies an activation function to an entire tensor at once.
    /// 
    /// Vector<double> activation functions:
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

        // Copy activation functions
        if (ScalarActivation != null)
        {
            copy.ScalarActivation = (IActivationFunction<T>)((ICloneable)ScalarActivation).Clone();
        }
        if (VectorActivation != null)
        {
            copy.VectorActivation = (IVectorActivationFunction<T>)((ICloneable)VectorActivation).Clone();
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

        return input.Transform((x, _) => activation.Derivative(x));
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
            // Use the vector activation function's derivative method
            return VectorActivation.Derivative(input).Multiply(outputGradient);
        }
        else if (ScalarActivation != null)
        {
            // Element-wise application of scalar activation derivative
            return input.Transform((x, _) => ScalarActivation.Derivative(x)).ElementwiseMultiply(outputGradient);
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

        Parameters = parameters;
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
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    /// <returns>The total count of trainable parameters.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method returns the same information as the ParameterCount property,
    /// but as a method call instead of a property access. It's provided for consistency with code
    /// that expects to call a GetParameterCount() method.
    ///
    /// The number of parameters tells you how much the layer can learn. For example:
    /// - A dense layer with 100 inputs and 50 outputs: 100×50 weights + 50 biases = 5,050 parameters
    /// - A pooling layer (no learning): 0 parameters
    ///
    /// This is useful when analyzing model complexity or debugging parameter updates.
    /// </remarks>
    public virtual int GetParameterCount()
    {
        return ParameterCount;
    }

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
    /// Creates a new instance of the layer with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use for the new instance.</param>
    /// <returns>A new model instance with the specified parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new layer instance with the same configuration as the current layer
    /// but with different parameter values. This is required by the IParameterizable interface
    /// but is not typically used for layers since they are mutable.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        // Layers are not full models - they are components of models.
        // This method is required by IParameterizable but doesn't make sense for layers.
        // Most layers should use SetParameters instead for mutable updates.
        throw new NotSupportedException(
            "WithParameters is not supported for layers. Use SetParameters to update layer parameters in place. " +
            "Layers are components of models, not full models themselves.");
    }

    /// <summary>
    /// Serializes the layer to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized layer data.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the layer's configuration and parameters to a byte array
    /// that can be saved to disk or transmitted over a network.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        
        // Write layer type name
        writer.Write(GetType().FullName ?? GetType().Name);
        
        // Write input and output shapes
        var inputShape = GetInputShape();
        var outputShape = GetOutputShape();
        
        writer.Write(inputShape.Length);
        foreach (var dim in inputShape)
        {
            writer.Write(dim);
        }
        
        writer.Write(outputShape.Length);
        foreach (var dim in outputShape)
        {
            writer.Write(dim);
        }
        
        // Write parameter count and parameters
        writer.Write(ParameterCount);
        if (ParameterCount > 0)
        {
            var parameters = GetParameters();
            for (int i = 0; i < parameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(parameters[i]));
            }
        }
        
        // Write activation function info
        var activationTypes = GetActivationTypes().ToList();
        writer.Write(activationTypes.Count);
        foreach (var activationType in activationTypes)
        {
            writer.Write((int)activationType);
        }
        
        // Note: Training mode is typically managed externally
        // and set via SetTrainingMode, so we don't serialize it
        
        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the layer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized layer data.</param>
    /// <remarks>
    /// <para>
    /// This method restores the layer's configuration and parameters from a byte array
    /// that was previously created by the Serialize method.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        
        // Read layer type name (for validation)
        var typeName = reader.ReadString();
        
        // Read input shape
        var inputShapeLength = reader.ReadInt32();
        var inputShape = new int[inputShapeLength];
        for (int i = 0; i < inputShapeLength; i++)
        {
            inputShape[i] = reader.ReadInt32();
        }
        
        // Read output shape
        var outputShapeLength = reader.ReadInt32();
        var outputShape = new int[outputShapeLength];
        for (int i = 0; i < outputShapeLength; i++)
        {
            outputShape[i] = reader.ReadInt32();
        }
        
        // Read parameters
        var paramCount = reader.ReadInt32();
        if (paramCount > 0)
        {
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                parameters[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            SetParameters(parameters);
        }
        
        // Read activation function info
        var activationCount = reader.ReadInt32();
        for (int i = 0; i < activationCount; i++)
        {
            var activationType = (ActivationFunction)reader.ReadInt32();
            // Note: Activation functions are typically set in constructor,
            // so we just read and discard here
        }
        
        // Note: Training mode is typically managed externally
        // and set via SetTrainingMode, so we don't deserialize it
    }
}