using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Initialization;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a convolutional layer in a neural network that applies filters to input data.
/// </summary>
/// <remarks>
/// <para>
/// A convolutional layer applies a set of learnable filters to input data to extract features. 
/// Each filter slides across the input data, performing element-wise multiplication and summing
/// the results. This operation is called convolution and is particularly effective for processing
/// grid-like data such as images.
/// </para>
/// <para><b>For Beginners:</b> A convolutional layer is like a spotlight that scans over data
/// looking for specific patterns.
/// 
/// Think of it like examining a photo with a small magnifying glass:
/// - You move the magnifying glass across the image, one step at a time
/// - At each position, you note what you see in that small area
/// - After scanning the whole image, you have a collection of observations
/// 
/// For example, in image recognition:
/// - One filter might detect vertical edges
/// - Another might detect horizontal edges
/// - Together, they help the network recognize complex shapes
/// 
/// Convolutional layers are fundamental for recognizing patterns in images, audio, and other
/// grid-structured data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, Cost = ComputeCost.High, TestInputShape = "1, 1, 8, 8", TestConstructorArgs = "2, 3")]
public partial class ConvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets the depth (number of channels) of the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input depth represents the number of channels in the input data. For example, RGB images have
    /// a depth of 3 (red, green, and blue channels), while grayscale images have a depth of 1.
    /// </para>
    /// <para><b>For Beginners:</b> Input depth is the number of "layers" in your input data.
    /// 
    /// Think of it like:
    /// - A color photo has 3 layers (red, green, blue)
    /// - A black and white photo has 1 layer
    /// 
    /// Each layer contains different information about the same data.
    /// </para>
    /// </remarks>
    public int InputDepth { get; private set; }

    /// <summary>
    /// Gets the depth (number of filters) of the output data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The output depth represents the number of filters applied to the input data. Each filter looks for
    /// a different pattern in the input, resulting in a different output channel.
    /// </para>
    /// <para><b>For Beginners:</b> Output depth is how many different patterns this layer will look for.
    /// 
    /// For example:
    /// - If output depth is 16, the layer will look for 16 different patterns
    /// - Each pattern creates its own output "layer" or channel
    /// - More output channels means the network can recognize more complex features
    /// 
    /// A higher number usually means the network can learn more details, but also requires more processing power.
    /// </para>
    /// </remarks>
    public int OutputDepth { get; private set; }

    /// <summary>
    /// Gets the size of each filter (kernel) used in the convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The kernel size determines the area of the input that is examined at each position. A larger kernel
    /// size means a larger area is considered for each output value, potentially capturing more complex patterns.
    /// </para>
    /// <para><b>For Beginners:</b> Kernel size is how big the "spotlight" or "magnifying glass" is.
    /// 
    /// For example:
    /// - A kernel size of 3 means a 3×3 area (9 pixels in an image)
    /// - A kernel size of 5 means a 5×5 area (25 pixels)
    /// 
    /// Smaller kernels (like 3×3) are good for detecting fine details.
    /// Larger kernels (like 7×7) can see broader patterns but may miss small details.
    /// </para>
    /// </remarks>
    public int KernelSize { get; private set; }

    /// <summary>
    /// Gets the step size for moving the kernel across the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The stride determines how many positions to move the kernel for each step during the convolution
    /// operation. A stride of 1 means the kernel moves one position at a time, examining every possible
    /// position. A larger stride means fewer positions are examined, resulting in a smaller output.
    /// </para>
    /// <para><b>For Beginners:</b> Stride is how far you move the spotlight each time.
    /// 
    /// Think of it like:
    /// - Stride of 1: Move one step at a time (examine every position)
    /// - Stride of 2: Skip one position between each examination (move two steps each time)
    /// 
    /// Using a larger stride:
    /// - Makes the output smaller (reduces dimensions)
    /// - Speeds up processing
    /// - But might miss some information
    /// </para>
    /// </remarks>
    public int Stride { get; private set; }

    /// <summary>
    /// Gets the amount of zero-padding added to the input data before convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Padding involves adding extra values (typically zeros) around the input data before performing
    /// the convolution. This allows the kernel to slide beyond the edges of the original input,
    /// preserving the spatial dimensions in the output.
    /// </para>
    /// <para><b>For Beginners:</b> Padding is like adding an extra border around your data.
    /// 
    /// Imagine adding a frame around a photo:
    /// - The frame is filled with zeros (blank data)
    /// - This allows the spotlight to analyze edges without going "off the picture"
    /// 
    /// Benefits of padding:
    /// - Maintains the size of your data through the layer
    /// - Ensures border information isn't lost
    /// - Without padding, each layer would make your data smaller
    /// </para>
    /// </remarks>
    public int Padding { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> for convolutional layers, as they contain trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation. Convolutional
    /// layers have trainable parameters (kernel weights and biases), so they support training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// For convolutional layers:
    /// - The value is always true
    /// - This means the layer can adjust its pattern detectors (filters) during training
    /// - It will improve its pattern recognition as it processes more data
    /// </para>
    /// </remarks>
    /// <summary>
    /// Gets the filter kernels of the convolutional layer.
    /// </summary>
    /// <returns>The filter tensor used for convolution operations.</returns>
    public Tensor<T> GetFilters()
    {
        return _kernels;
    }

    /// <summary>
    /// Gets the biases tensor of the convolutional layer.
    /// </summary>
    /// <returns>The bias values added to each output channel.</returns>
    public override Tensor<T> GetBiases()
    {
        return _biases;
    }

    public override bool SupportsTraining => true;

    /// <summary>
    /// The collection of filter kernels used for the convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the weight values for all kernels used in the layer. It has dimensions
    /// [OutputDepth, InputDepth, KernelSize, KernelSize], where each kernel is a set of weights
    /// that define a specific pattern to detect.
    /// </para>
    /// <para><b>For Beginners:</b> _kernels are the "pattern detectors" that the layer uses.
    /// 
    /// Each kernel:
    /// - Is a grid of numbers (weights)
    /// - Looks for a specific pattern in the input
    /// - Is learned during training
    /// 
    /// The layer has multiple kernels to detect different patterns, and these kernels
    /// are what actually get updated when the network learns.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _kernels;

    /// <summary>
    /// The bias values added to the convolution results for each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the bias values for each output channel. _biases are constants that are
    /// added to the convolution results before applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> _biases are like "adjustment factors" for each pattern detector.
    ///
    /// Think of biases as:
    /// - A starting point or baseline value
    /// - Added to the result after applying the pattern detector
    /// - Helping the network be more flexible in what it can learn
    ///
    /// For example, biases help the network detect patterns even when the input doesn't
    /// perfectly match what the kernel is looking for.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _biases;

    /// <summary>
    /// Cached reshape of _biases to [1, OutputDepth, 1, 1] for broadcast addition.
    /// Avoids allocating a new view tensor on every forward pass.
    /// </summary>
    private Tensor<T>? _biasReshaped4D;

    /// <summary>
    /// Pre-allocated output buffer for Conv2DInto. Reused every forward pass.
    /// </summary>
    private Tensor<T>? _preAllocatedOutput;

    /// <summary>
    /// The execution engine for GPU-accelerated convolution operations.
    /// </summary>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-016 - Layer GPU Acceleration</b></para>
    /// <para>
    /// This engine provides hardware-accelerated Conv2D operations, replacing manual 6-nested loops.
    /// Using IEngine.Conv2D enables:
    /// - CPU: Optimized BLAS libraries for convolution
    /// - GPU: Massive parallelism for 50-500x speedup on large feature maps
    /// </para>
    /// </remarks>

    /// <summary>
    /// Gradient of the kernels computed during backpropagation via autodiff.
    /// </summary>
    private Tensor<T>? _kernelsGradient;

    /// <summary>
    /// Gradient of the biases computed during backpropagation via autodiff.
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Tracks whether lazy initialization has been completed.
    /// </summary>
    private bool _isInitialized;

    /// <inheritdoc />
    public override bool IsInitialized => _isInitialized;

    /// <summary>
    /// Stored input data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the input data from the forward
    /// pass to calculate the gradients for the kernels and the input. This tensor stores that input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the network's "short-term memory" of what it just saw.
    /// 
    /// The layer remembers:
    /// - The last data it processed
    /// - So it can figure out how to improve when learning
    /// 
    /// This is similar to looking at a problem you got wrong and the answer you gave,
    /// so you can understand where you made a mistake.
    /// </para>
    /// </remarks>
    private Tensor<T> _lastInput;

    /// <summary>
    /// Stored output data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the output data from the forward
    /// pass to calculate the gradients for the activation function. This tensor stores that output data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the network's memory of what answer it produced.
    /// 
    /// The layer remembers:
    /// - What output it produced for the last input
    /// - So it can calculate how to improve
    /// 
    /// This allows the network to compare what it predicted with the correct answer
    /// and adjust its internal values to make better predictions next time.
    /// </para>
    /// </remarks>
    private Tensor<T> _lastOutput;

    // GPU-resident cached tensors for GPU training pipeline
    private Tensor<T>? _lastInputGpu;
    private Tensor<T>? _lastOutputGpu;
    private int[]? _gpuInputShape4D;

    /// <summary>
    /// Tracks whether a batch dimension was added during the forward pass.
    /// </summary>
    private bool _addedBatchDimension;

    /// <summary>
    /// Stores the original input shape for restoring higher-rank tensor output.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Random number generator used for weight initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This random number generator is used to initialize the kernel weights with random values.
    /// Random initialization helps the network break symmetry and learn different patterns in
    /// different kernels.
    /// </para>
    /// <para><b>For Beginners:</b> This creates random starting points for the pattern detectors.
    /// 
    /// The random generator:
    /// - Creates different starting weights each time
    /// - Ensures different kernels learn different patterns
    /// - Gives the network a better chance of learning successfully
    /// 
    /// Without randomness, all pattern detectors might end up looking for the same thing.
    /// </para>
    /// </remarks>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionalLayer{T}"/> class with the specified parameters
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input data.</param>
    /// <param name="inputHeight">The height of the input data.</param>
    /// <param name="inputWidth">The width of the input data.</param>
    /// <param name="outputDepth">The number of filters (output channels) to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a convolutional layer with the specified configuration. The input shape is determined
    /// by the inputDepth, inputHeight, and inputWidth parameters, while the output shape is calculated based on
    /// these values along with the kernel size, stride, and padding. The kernels and biases are initialized with
    /// random values.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method creates a new convolutional layer with specific settings.
    ///
    /// When creating the layer, you specify:
    /// - Input details: How many channels and the dimensions of your data
    /// - How many patterns to look for (outputDepth)
    /// - How big each pattern detector is (kernelSize)
    /// - How to move the detector across the data (stride)
    /// - Whether to add an extra border (padding)
    /// - What mathematical function to apply to the results (activationFunction)
    ///
    /// The layer then creates all the necessary pattern detectors with random starting values
    /// that will be improved during training.
    /// </para>
    /// </remarks>
    public ConvolutionalLayer(int outputDepth, int kernelSize, int stride = 1, int padding = 0,
                              IActivationFunction<T>? activationFunction = null,
                              IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, -1, -1 }, new[] { outputDepth, -1, -1 }, activationFunction ?? new ReLUActivation<T>())
    {
        if (outputDepth <= 0) throw new ArgumentOutOfRangeException(nameof(outputDepth), "outputDepth must be positive.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), "padding cannot be negative.");

        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        InputDepth = -1; // resolved in OnFirstForward from input.Shape

        // Store the initialization strategy (defaults to LazyInitializationStrategy semantics
        // since shape is always deferred to first forward in this layer now).
        InitializationStrategy = initializationStrategy;

        // Always start fully deferred: shape, channel count, and weights resolve on first Forward.
        _kernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
        _lastInput = new Tensor<T>([0, 0, 0, 0]);
        _lastOutput = new Tensor<T>([0, 0, 0, 0]);
        _random = RandomHelper.CreateSecureRandom();
        _isInitialized = false;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvolutionalLayer{T}"/> class with the specified parameters
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input data.</param>
    /// <param name="inputHeight">The height of the input data.</param>
    /// <param name="inputWidth">The width of the input data.</param>
    /// <param name="outputDepth">The number of filters (output channels) to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply (required to disambiguate from IActivationFunction overload).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a convolutional layer with the specified configuration and a vector activation function,
    /// which operates on entire vectors rather than individual elements. This can be useful when applying more complex
    /// activation functions or when performance is a concern.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method is similar to the previous one, but uses a different type of
    /// activation function.
    ///
    /// A vector activation function:
    /// - Works on entire groups of numbers at once
    /// - Can be more efficient for certain types of calculations
    /// - Otherwise works the same as the regular activation function
    ///
    /// You would choose this option if you have a specific mathematical operation that
    /// needs to be applied to groups of outputs rather than individual values.
    /// </para>
    /// </remarks>
    public ConvolutionalLayer(int outputDepth, int kernelSize, int stride, int padding,
                              IVectorActivationFunction<T> vectorActivationFunction,
                              IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, -1, -1 }, new[] { outputDepth, -1, -1 }, vectorActivationFunction)
    {
        if (outputDepth <= 0) throw new ArgumentOutOfRangeException(nameof(outputDepth), "outputDepth must be positive.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), "padding cannot be negative.");

        OutputDepth = outputDepth;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        InputDepth = -1; // resolved in OnFirstForward from input.Shape

        // Store the initialization strategy
        InitializationStrategy = initializationStrategy;

        // Always start fully deferred: shape, channel count, and weights resolve on first Forward.
        _kernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
        _lastInput = new Tensor<T>([0, 0, 0, 0]);
        _lastOutput = new Tensor<T>([0, 0, 0, 0]);
        _random = RandomHelper.CreateSecureRandom();
        _isInitialized = false;
    }

    /// <summary>
    /// Creates a convolutional layer with the specified configuration using a fluent interface.
    /// </summary>
    /// <param name="inputShape">The shape of the input data as [depth, height, width].</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="numberOfFilters">The number of filters (output channels) to create.</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to ReLU if not specified.</param>
    /// <returns>A new instance of the <see cref="ConvolutionalLayer{T}"/> class.</returns>
    /// <exception cref="ArgumentException">Thrown when the input shape does not have exactly 3 dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This static method provides a more convenient way to create a convolutional layer by specifying the input shape
    /// as an array rather than individual dimensions. It extracts the depth, height, and width from the input shape
    /// array and passes them to the constructor.
    /// </para>
    /// <para><b>For Beginners:</b> This is a simpler way to create a convolutional layer when you already know
    /// your input data's shape.
    /// 
    /// Instead of providing separate numbers for depth, height, and width, you can:
    /// - Pass all three dimensions in a single array
    /// - Specify the other settings in a more intuitive way
    /// 
    /// For example, if your input is 3-channel images that are 28×28 pixels:
    /// - You would use inputShape = [3, 28, 28]
    /// - Rather than listing all dimensions separately
    /// 
    /// This makes your code cleaner and easier to read.
    /// </para>
    /// </remarks>
    public static ConvolutionalLayer<T> Configure(int[] inputShape, int kernelSize, int numberOfFilters, int stride = 1, int padding = 0,
        IActivationFunction<T>? activationFunction = null)
    {
        if (inputShape.Length != 3)
        {
            throw new ArgumentException("Input shape must have 3 dimensions: depth, height, width");
        }

        int inputDepth = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        return new ConvolutionalLayer<T>(
            outputDepth: numberOfFilters,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            activationFunction: activationFunction
        );
    }

    /// <summary>
    /// Creates a convolutional layer with the specified configuration and a vector activation function using a fluent interface.
    /// </summary>
    /// <param name="inputShape">The shape of the input data as [depth, height, width].</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="numberOfFilters">The number of filters (output channels) to create.</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply. Defaults to ReLU if not specified.</param>
    /// <returns>A new instance of the <see cref="ConvolutionalLayer{T}"/> class with a vector activation function.</returns>
    /// <exception cref="ArgumentException">Thrown when the input shape does not have exactly 3 dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This static method provides a more convenient way to create a convolutional layer with a vector activation function
    /// by specifying the input shape as an array rather than individual dimensions. It is similar to the Configure method
    /// with a scalar activation function, but uses a vector activation function instead.
    /// </para>
    /// <para><b>For Beginners:</b> This is similar to the previous Configure method, but uses a vector activation function.
    /// 
    /// This method:
    /// - Makes it easier to create a layer with an input shape array
    /// - Uses a vector activation function (works on groups of numbers)
    /// - Is otherwise identical to the previous Configure method
    /// 
    /// You would choose this if you need a specific type of mathematical operation
    /// applied to groups of values rather than individual numbers.
    /// </para>
    /// </remarks>
    public static ConvolutionalLayer<T> Configure(int[] inputShape, int kernelSize, int numberOfFilters, int stride = 1, int padding = 0,
        IVectorActivationFunction<T>? vectorActivationFunction = null)
    {
        if (inputShape.Length != 3)
        {
            throw new ArgumentException("Input shape must have 3 dimensions: depth, height, width");
        }

        int inputDepth = inputShape[0];
        int inputHeight = inputShape[1];
        int inputWidth = inputShape[2];

        // Use the appropriate constructor based on whether vectorActivationFunction is provided
        if (vectorActivationFunction is not null)
        {
            return new ConvolutionalLayer<T>(
                outputDepth: numberOfFilters,
                kernelSize: kernelSize,
                stride: stride,
                padding: padding,
                vectorActivationFunction: vectorActivationFunction
            );
        }
        else
        {
            return new ConvolutionalLayer<T>(
                outputDepth: numberOfFilters,
                kernelSize: kernelSize,
                stride: stride,
                padding: padding
            );
        }
    }

    /// <summary>
    /// Saves the layer's configuration and parameters to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to save to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the layer's configuration (input depth, output depth, kernel size, stride, padding)
    /// and parameters (kernel weights and biases) to a binary writer. This allows the layer to be saved to
    /// a file and loaded later.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the layer's settings and learned patterns to a file.
    /// 
    /// When saving a layer:
    /// - First, it saves the basic configuration (size, stride, etc.)
    /// - Then it saves all the learned pattern detectors (kernels)
    /// - Finally, it saves the bias values
    /// 
    /// This allows you to:
    /// - Save a trained model to use later
    /// - Share your trained model with others
    /// - Store multiple versions of your model
    /// 
    /// Think of it like taking a snapshot of everything the model has learned.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputDepth);
        writer.Write(OutputDepth);
        writer.Write(KernelSize);
        writer.Write(Stride);
        writer.Write(Padding);

        // Serialize _kernels — flat span iteration replaces 4-nested indexing loops
        var kernelSpan = _kernels.Data.Span;
        for (int i = 0; i < kernelSpan.Length; i++)
            writer.Write(Convert.ToDouble(kernelSpan[i]));

        // Serialize _biases — flat span iteration
        var biasSpan = _biases.Data.Span;
        for (int i = 0; i < biasSpan.Length; i++)
            writer.Write(Convert.ToDouble(biasSpan[i]));
    }

    /// <summary>
    /// Loads the layer's configuration and parameters from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to load from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the layer's configuration (input depth, output depth, kernel size, stride, padding)
    /// and parameters (kernel weights and biases) from a binary reader. This allows a previously saved layer
    /// to be loaded from a file.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved layer from a file.
    /// 
    /// When loading a layer:
    /// - First, it reads the basic configuration
    /// - Then it recreates all the pattern detectors (kernels)
    /// - Finally, it loads the bias values
    /// 
    /// This allows you to:
    /// - Continue using a model you trained earlier
    /// - Use a model someone else trained
    /// - Compare different versions of your model
    /// 
    /// It's like restoring a snapshot of a trained model exactly as it was.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputDepth = reader.ReadInt32();
        OutputDepth = reader.ReadInt32();
        KernelSize = reader.ReadInt32();
        Stride = reader.ReadInt32();
        Padding = reader.ReadInt32();

        // Deserialize _kernels — flat span iteration replaces 4-nested indexing loops
        _kernels = TensorAllocator.RentUninitialized<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        var kernelSpan = _kernels.Data.Span;
        for (int i = 0; i < kernelSpan.Length; i++)
            kernelSpan[i] = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize _biases — flat span iteration
        _biases = new Tensor<T>([OutputDepth]);
        var biasSpan = _biases.Data.Span;
        for (int i = 0; i < biasSpan.Length; i++)
            biasSpan[i] = NumOps.FromDouble(reader.ReadDouble());

        // Reinitialize _lastInput and _lastOutput
        _lastInput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        _lastOutput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);

        // Re-register the freshly-created tensors as trainable parameters so
        // optimizers and tape training target these objects (not the stale ones
        // from a prior EnsureInitialized or constructor call). Without this,
        // the registered list points at the old tensors while Forward uses the
        // new ones — gradient updates silently go to dead references.
        ClearRegisteredParameters();
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        // Mark as initialized so EnsureInitialized() doesn't re-randomize the
        // just-deserialized weights on the next Forward/GetParameters call.
        // Also ensures Dispose returns the rented _kernels to TensorAllocator.
        _isInitialized = true;
    }

    /// <summary>
    /// Calculates the output dimension after applying a convolution operation.
    /// </summary>
    /// <param name="inputDim">The input dimension (height or width).</param>
    /// <param name="kernelSize">The size of the kernel (filter).</param>
    /// <param name="stride">The stride (step size) of the convolution.</param>
    /// <param name="padding">The amount of padding added to the input.</param>
    /// <returns>The calculated output dimension.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output dimension (height or width) after applying a convolution operation
    /// with the specified parameters. The formula used is (inputDim - kernelSize + 2 * padding) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how big the output will be after applying the layer.
    /// 
    /// The output size depends on:
    /// - How big your input is
    /// - How big your pattern detector (kernel) is
    /// - How much you move the detector each step (stride)
    /// - How much extra border you add (padding)
    /// 
    /// Generally:
    /// - Larger stride = smaller output
    /// - More padding = larger output
    /// - Larger kernel = smaller output
    /// 
    /// This method uses a standard formula to calculate the exact output size.
    /// </para>
    /// </remarks>
    private static int CalculateOutputDimension(int inputDim, int kernelSize, int stride, int padding)
    {
        if (inputDim + 2 * padding < kernelSize)
            throw new ArgumentException("Input dimensions with padding must be at least kernel size.");

        return (inputDim - kernelSize + 2 * padding) / stride + 1;
    }

    /// <summary>
    /// Initializes the kernel weights and biases with random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the kernel weights using the He initialization method, which scales the random
    /// values based on the number of input and output connections. This helps improve training convergence.
    /// The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the pattern detectors.
    /// 
    /// When initializing weights:
    /// - Random values are created for each pattern detector
    /// - The values are carefully scaled to work well for training
    /// - _biases start at zero
    /// 
    /// Good initialization is important because:
    /// - It helps the network learn faster
    /// - It prevents certain mathematical problems during training
    /// - It gives each pattern detector a different starting point
    ///
    /// This uses a technique called "He initialization" which works well
    /// with modern neural networks.
    /// </para>
    /// </remarks>

    /// <summary>
    /// Ensures that kernels are allocated and initialized for lazy initialization.
    /// </summary>
    /// <summary>
    /// Resolves spatial dims (H/W) and channel count from the actual input on the first forward
    /// call (PyTorch <c>LazyConv2d</c>-style). Sets <see cref="InputDepth"/>, computes output H/W
    /// via the standard convolution arithmetic, and propagates the resolved shapes to LayerBase.
    /// Weight allocation still happens in <see cref="EnsureInitialized"/> immediately afterward.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int inDepth, inH, inW;
        if (rank == 4)
        {
            // Batched [B, C, H, W]
            inDepth = input.Shape[1];
            inH = input.Shape[2];
            inW = input.Shape[3];
        }
        else if (rank == 3)
        {
            // Unbatched [C, H, W]
            inDepth = input.Shape[0];
            inH = input.Shape[1];
            inW = input.Shape[2];
        }
        else
        {
            throw new ArgumentException(
                $"ConvolutionalLayer expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {rank}.",
                nameof(input));
        }

        if (inH + 2 * Padding < KernelSize || inW + 2 * Padding < KernelSize)
        {
            throw new ArgumentException(
                $"Input spatial dims after padding ({inH}+2*{Padding}, {inW}+2*{Padding}) must be >= kernelSize ({KernelSize}).",
                nameof(input));
        }

        InputDepth = inDepth;
        int outH = CalculateOutputDimension(inH, KernelSize, Stride, Padding);
        int outW = CalculateOutputDimension(inW, KernelSize, Stride, Padding);

        ResolveShapes(
            new[] { inDepth, inH, inW },
            new[] { OutputDepth, outH, outW });
    }

    protected override void EnsureInitialized()
    {
        if (_isInitialized) return;

        // Cannot eager-initialize a lazy layer that has not yet seen any input — the
        // PyTorch LazyConv2d contract is identical: GetParameters / SetParameters /
        // ParameterCount on an uninitialized lazy module throws because the weight
        // shapes aren't yet known. Callers must run a forward first.
        if (!IsShapeResolved || InputDepth <= 0)
        {
            throw new InvalidOperationException(
                "ConvolutionalLayer is in deferred-shape mode and has not yet seen any input. " +
                "Run a Forward(input) before calling GetParameters / SetParameters / ParameterCount, " +
                "or construct the layer with a concrete input shape.");
        }

        lock (InitializationLock)
        {
            if (_isInitialized) return;

            // Use correct input/output shapes as placeholders (batch=1, replaced in Forward())
            _lastInput = new Tensor<T>([1, InputShape[0], InputShape[1], InputShape[2]]);
            _lastOutput = new Tensor<T>([1, OutputShape[0], OutputShape[1], OutputShape[2]]);

            // Allocate kernels and biases with proper shapes before initializing weights.
            // The lazy path sets _kernels to [0,0,0,0], so we must resize here.
            _kernels = TensorAllocator.RentUninitialized<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
            _biases = new Tensor<T>([OutputDepth]);

            // Initialize weights (fills _kernels and _biases with He-uniform values)
            InitializeWeights();

            // Register trainable parameters with the engine for GPU persistence
            RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

            _isInitialized = true;
        }
    }

    private void InitializeWeights()
    {
        // He-uniform initialization: U(-bound, bound) where bound = sqrt(6 / fanIn)
        // per He et al. 2015 "Delving Deep into Rectifiers".
        int fanIn = InputDepth * KernelSize * KernelSize;
        double bound = Math.Sqrt(6.0 / fanIn);

        // Use SimdRandom for vectorized He-uniform initialization
        var rng = new SimdRandom();
        var span = _kernels.Data.Span;
        int total = span.Length;
        if (total == 0)
        {
            // Zero-sized kernel tensor: no weights to fill, but still zero biases so
            // initialization behavior doesn't depend on tensor-constructor allocator
            // semantics.
            _biases.Fill(NumOps.Zero);
            return;
        }

        // Write via a temp array + array-level reinterpret so the SIMD-batched
        // xoshiro256** fill path still applies. See MultiHeadAttentionLayer for full
        // rationale (Span<T> can't be reinterpreted across generic T without a struct
        // constraint, which we don't have, and CreateSpan isn't on net471).
        if (typeof(T) == typeof(double))
        {
            var buffer = new double[total];
            rng.NextDoubles(buffer.AsSpan());
            for (int i = 0; i < total; i++)
                buffer[i] = (buffer[i] * 2.0 - 1.0) * bound;
            var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<double[], T[]>(ref buffer);
            reinterpreted.AsSpan(0, total).CopyTo(span);
        }
        else if (typeof(T) == typeof(float))
        {
            var buffer = new float[total];
            rng.NextFloats(buffer.AsSpan());
            float boundF = (float)bound;
            for (int i = 0; i < total; i++)
                buffer[i] = (buffer[i] * 2f - 1f) * boundF;
            var reinterpreted = System.Runtime.CompilerServices.Unsafe.As<float[], T[]>(ref buffer);
            reinterpreted.AsSpan(0, total).CopyTo(span);
        }
        else
        {
            const int batchSize = 4096;
            var tempBuf = new double[Math.Min(total, batchSize)];
            int offset = 0;
            while (offset < total)
            {
                int chunk = Math.Min(batchSize, total - offset);
                rng.NextDoubles(tempBuf.AsSpan(0, chunk));
                for (int j = 0; j < chunk; j++)
                    span[offset + j] = NumOps.FromDouble((tempBuf[j] * 2.0 - 1.0) * bound);
                offset += chunk;
            }
        }

        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Processes the input data through the convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process, with shape [batchSize, inputDepth, height, width].</param>
    /// <returns>The output tensor after convolution and activation, with shape [batchSize, outputDepth, outputHeight, outputWidth].</returns>
    /// <remarks>
    /// <para>
    /// This method performs the forward pass of the convolutional layer. For each position of the kernel on the
    /// input data, it computes the element-wise product of the kernel weights and the corresponding input values,
    /// sums the results, adds the bias, and applies the activation function. The result is a tensor where each
    /// channel represents the activation of a different filter.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the pattern detectors to your input data.
    /// 
    /// During the forward pass:
    /// - Each pattern detector (kernel) slides across the input
    /// - At each position, it looks for its specific pattern
    /// - If it finds a match, it produces a high value in the output
    /// - The activation function then adjusts these values
    /// 
    /// Think of it like a series of spotlights scanning across your data,
    /// each one lighting up when it finds the pattern it's looking for.
    /// The result shows where each pattern was found in the input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Resolve deferred shape (PyTorch LazyConv2d-style) and allocate weights on first call.
        EnsureInitializedFromInput(input);

        // Accept any rank and canonicalize to 4D [B, C, H, W]. The library's
        // design principle is rank-agnostic ops — a flat feature vector
        // (rank 1) and a batched feature vector (rank 2) are legitimate
        // conv inputs once we pad singleton spatial dims, which is the
        // standard interpretation used by PyTorch/Keras when a caller
        // feeds rank < 4 into a 2D conv (they treat H=W=1). Higher ranks
        // flatten leading dims into the batch axis.
        Tensor<T> input4D;
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        if (rank == 1)
        {
            // [F] -> [1, F, 1, 1]: single-sample flat feature treated as C
            _addedBatchDimension = true;
            input4D = Engine.Reshape(input, [1, input.Shape[0], 1, 1]);
        }
        else if (rank == 2)
        {
            // [B, F] -> [B, F, 1, 1]: batch of flat features treated as C
            _addedBatchDimension = false;
            input4D = Engine.Reshape(input, [input.Shape[0], input.Shape[1], 1, 1]);
        }
        else if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            _addedBatchDimension = true;
            input4D = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        }
        else if (rank == 4)
        {
            // 4D [B, C, H, W] - no reshaping needed
            _addedBatchDimension = false;
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            _addedBatchDimension = false;
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            input4D = Engine.Reshape(input, [flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // Validate input channels
        int actualInputChannels = input4D.Shape[1];
        if (actualInputChannels != InputDepth)
        {
            throw new ArgumentException(
                $"Expected input depth {InputDepth}, but got {actualInputChannels}.");
        }

        _lastInput = input4D;

        // === Zero-Allocation Convolution ===
        // Pre-allocate output buffer on first forward pass, then reuse via Conv2DInto
        int outputHeight = (_lastInput.Shape[2] + 2 * Padding - KernelSize) / Stride + 1;
        int outputWidth = (_lastInput.Shape[3] + 2 * Padding - KernelSize) / Stride + 1;
        int batchSize_conv = _lastInput.Shape[0];
        int[] expectedShape = [batchSize_conv, OutputDepth, outputHeight, outputWidth];

        if (_preAllocatedOutput is null ||
            _preAllocatedOutput.Shape[0] != batchSize_conv ||
            _preAllocatedOutput.Shape[2] != outputHeight ||
            _preAllocatedOutput.Shape[3] != outputWidth)
        {
            _preAllocatedOutput = TensorAllocator.Rent<T>(expectedShape);
        }

        // === Try FusedConv2D: Conv + Bias + Activation in single kernel ===
        // Eliminates 2 intermediate allocations and enables kernel-level optimization
        var fusedActivation = GetFusedActivationType();
        Tensor<T> result;
        if (fusedActivation != FusedActivationType.None)
        {
            // Single fused call: output = activation(conv(input, kernel) + bias)
            // Reshape bias to [1, C, 1, 1] for proper broadcasting with conv output [B, C, H, W]
            _biasReshaped4D ??= Engine.Reshape(_biases, [1, OutputDepth, 1, 1]);
            result = Engine.FusedConv2D(_lastInput, _kernels, _biasReshaped4D,
                Stride, Stride, Padding, Padding, 1, 1, fusedActivation);
        }
        else if (AiDotNet.Tensors.Engines.Autodiff.GradientTape<T>.Current is not null
                 && !AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed)
        {
            // Tape-tracked path: zero-alloc Into/InPlace variants bypass the gradient
            // tape, so while a tape is active we must use the non-in-place Engine ops
            // (Conv2D + TensorBroadcastAdd) so the backward pass can follow the
            // gradient chain back to the kernel and bias tensors.
            //
            // Check the tape directly rather than IsTrainingMode because not every
            // caller flips IsTrainingMode before invoking the forward pass —
            // DiffusionModelBase.Train opens a GradientTape without ever calling
            // SetTrainingMode, which caused this branch to be silently skipped.
            //
            // CRITICAL: reshape the bias fresh each training step instead of reusing
            // the _biasReshaped4D cache. The cache is typically primed during the
            // first Predict call (under NoGradScope) and holds a reshape tensor
            // with no GradFn pointing back to _biases. Reusing that cached handle
            // would make the gradient walk hit a dead end at _biasReshaped4D,
            // leaving _biases with zero gradient on every training step.
            var conv = Engine.Conv2D(_lastInput, _kernels, Stride, Padding, dilation: 1);
            var biasReshapedForTape = Engine.Reshape(_biases, [1, OutputDepth, 1, 1]);
            result = Engine.TensorBroadcastAdd(conv, biasReshapedForTape);
        }
        else
        {
            // Inference fast path: separate Conv2DInto + in-place bias + activation.
            // Safe because no tape is active during inference (NoGradScope).
            Engine.Conv2DInto(_preAllocatedOutput, _lastInput, _kernels, Stride, Padding, dilation: 1);
            var output = _preAllocatedOutput;

            _biasReshaped4D ??= Engine.Reshape(_biases, [1, OutputDepth, 1, 1]);
            Engine.TensorBroadcastAddInPlace(output, _biasReshaped4D);

            result = ApplyActivation(output);
        }

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        // Return with matching dimensions to preserve original tensor rank
        if (_originalInputShape != null && _originalInputShape.Length > 4)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 3] = OutputDepth;
            outputShape[_originalInputShape.Length - 2] = result.Shape[2];
            outputShape[_originalInputShape.Length - 1] = result.Shape[3];
            return Engine.Reshape(result, outputShape);
        }
        if (_addedBatchDimension)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            // Remove the batch dimension we added
            return Engine.Reshape(result, [OutputDepth, result.Shape[2], result.Shape[3]]);
        }

        return result;
    }

    /// <summary>
    /// Gets whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs a GPU-resident forward pass using fused Conv2D + Bias + Activation.
    /// </summary>
    /// <param name="input">GPU-resident input tensor.</param>
    /// <returns>GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.</para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        EnsureInitializedFromInput(inputs[0]);

        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];

        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (input.Shape.Length < 3)
        {
            throw new ArgumentException(
                $"Conv2D input requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");
        }

        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Reshape input to 4D [B, C, H, W] for convolution
        Tensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            _addedBatchDimension = true;
            input4D = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        }
        else if (rank == 4)
        {
            // 4D [B, C, H, W] - no reshaping needed
            _addedBatchDimension = false;
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            _addedBatchDimension = false;
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
            {
                flatBatch *= input.Shape[d];
            }
            input4D = Engine.Reshape(input, [flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // Validate input channels
        int actualInputChannels = input4D.Shape[1];
        if (actualInputChannels != InputDepth)
        {
            throw new ArgumentException(
                $"Expected input depth {InputDepth}, but got {actualInputChannels}.");
        }

        // Map activation function to FusedActivationType
        var fusedActivation = MapActivationToFused();

        // Execute GPU-fused Conv2D + Bias + Activation
        var result = gpuEngine.FusedConv2DGpu(
            input4D,
            _kernels,
            _biases,
            Stride, Stride,      // strideH, strideW
            Padding, Padding,    // padH, padW
            1, 1,                // dilationH, dilationW
            fusedActivation);

        // Cache state for backward pass only during training - KEEP ON GPU for GPU-resident training
        if (IsTrainingMode)
        {
            // Store GPU-resident tensors for BackwardGpu (no CPU roundtrip)
            _lastInputGpu = input4D;
            _lastOutputGpu = result;
            _gpuInputShape4D = input4D._shape;

            // Also download to CPU for hybrid CPU/GPU backward compatibility
            _lastInput = input4D;
            _lastOutput = result;
        }

        // Restore original shape if needed
        if (_originalInputShape != null && _originalInputShape.Length > 4)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
            {
                outputShape[d] = _originalInputShape[d];
            }
            outputShape[_originalInputShape.Length - 3] = OutputDepth;
            outputShape[_originalInputShape.Length - 2] = result.Shape[2];
            outputShape[_originalInputShape.Length - 1] = result.Shape[3];
            return Engine.Reshape(result, outputShape);
        }

        if (_addedBatchDimension)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            return Engine.Reshape(result, [OutputDepth, result.Shape[2], result.Shape[3]]);
        }

        return result;
    }

    /// <summary>
    /// Computes activation gradient for convolutional layer using GPU-resident backward operations.
    /// </summary>
    private Tensor<T> ComputeConvActivationGradientGpu(DirectGpuTensorEngine gpuEngine, Tensor<T> gradOutput, FusedActivationType activation)
    {
        // For convolutional layers, we need to reshape to 2D for activation backward, then reshape back
        // Most activations are element-wise, so we can flatten the tensor
        int totalElements = gradOutput.Length;
        var flat2DShape = new[] { totalElements, 1 };
        var flatGrad = gradOutput.Reshape(flat2DShape);
        var lastOutputGpu = _lastOutputGpu ?? throw new InvalidOperationException("_lastOutputGpu has not been initialized.");
        var flatOutput = lastOutputGpu.Reshape(flat2DShape);

        Tensor<T> flatResult = activation switch
        {
            FusedActivationType.ReLU => gpuEngine.ReluBackwardGpu<T>(flatGrad, flatOutput), // ReLU uses pre-activation, but we only have post-activation here
            FusedActivationType.Sigmoid => gpuEngine.SigmoidBackwardGpu<T>(flatGrad, flatOutput),
            FusedActivationType.Tanh => gpuEngine.TanhBackwardGpu<T>(flatGrad, flatOutput),
            FusedActivationType.GELU => gpuEngine.GeluBackwardGpu<T>(flatGrad, flatOutput),
            FusedActivationType.Swish => gpuEngine.SwishBackwardGpu<T>(flatGrad, flatOutput),
            FusedActivationType.LeakyReLU => gpuEngine.LeakyReluBackwardGpu<T>(flatGrad, flatOutput, 0.01f),
            _ => flatGrad
        };

        // Reshape back to 4D
        return flatResult.Reshape(gradOutput._shape);
    }

    /// <summary>
    /// Applies scalar activation function using autodiff operations.
    /// </summary>
    private Autodiff.ComputationNode<T> ApplyScalarActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        return ScalarActivation switch
        {
            ReLUActivation<T> => Autodiff.TensorOperations<T>.ReLU(input),
            SigmoidActivation<T> => Autodiff.TensorOperations<T>.Sigmoid(input),
            TanhActivation<T> => Autodiff.TensorOperations<T>.Tanh(input),
            ELUActivation<T> elu => Autodiff.TensorOperations<T>.ELU(input, Convert.ToDouble(elu.Alpha)),
            LeakyReLUActivation<T> leaky => Autodiff.TensorOperations<T>.LeakyReLU(input, Convert.ToDouble(leaky.Alpha)),
            GELUActivation<T> => Autodiff.TensorOperations<T>.GELU(input),
            SwishActivation<T> => Autodiff.TensorOperations<T>.Swish(input),
            SiLUActivation<T> => Autodiff.TensorOperations<T>.Swish(input), // SiLU is same as Swish
            SELUActivation<T> => Autodiff.TensorOperations<T>.SELU(input),
            SoftSignActivation<T> => Autodiff.TensorOperations<T>.SoftSign(input),
            IdentityActivation<T> => input, // Identity just returns input as-is
            _ => throw new NotSupportedException($"Activation {ScalarActivation?.GetType().Name} not supported in autodiff mode. " +
                "Supported: ReLU, Sigmoid, Tanh, ELU, LeakyReLU, GELU, Swish, SiLU, SELU, SoftSign, Identity")
        };
    }

    private Tensor<T>? _kernelsVelocity;
    private Tensor<T>? _biasesVelocity;

    /// <summary>
    /// Updates the layer's parameters (kernel weights and biases) using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (kernel weights and biases) based on the gradients
    /// calculated during the backward pass. The learning rate controls the step size of the update,
    /// with a smaller learning rate resulting in smaller, more cautious updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the lessons learned during training.
    /// 
    /// When updating parameters:
    /// - The learning rate controls how big each adjustment is
    /// - Small learning rate = small, careful changes
    /// - Large learning rate = big, faster changes (but might overshoot)
    /// 
    /// Think of it like adjusting your position in a game:
    /// - If you're far from the target, you might take big steps
    /// - As you get closer, you take smaller, more precise steps
    /// 
    /// The learning rate helps balance between learning quickly and learning accurately.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_kernelsGradient == null || _biasesGradient == null)
            return;

        if (Engine is DirectGpuTensorEngine gpuEngine)
        {
            float lr = (float)NumOps.ToDouble(learningRate);

            // Initialize velocity tensors if needed (for SGD momentum, even if 0 here)
            if (_kernelsVelocity == null)
            {
                _kernelsVelocity = new Tensor<T>(_kernels._shape);
                _kernelsVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_kernelsVelocity, PersistentTensorRole.OptimizerState);
            }
            if (_biasesVelocity == null)
            {
                _biasesVelocity = new Tensor<T>(_biases._shape);
                _biasesVelocity.Fill(NumOps.Zero);
                gpuEngine.RegisterPersistentTensor(_biasesVelocity, PersistentTensorRole.OptimizerState);
            }

            // Perform GPU-resident SGD update
            // Momentum = 0, WeightDecay = 0 to match CPU implementation
            gpuEngine.SgdMomentumUpdateGpu(_kernels, _kernelsGradient, _kernelsVelocity, lr, 0.0f, 0.0f);
            gpuEngine.SgdMomentumUpdateGpu(_biases, _biasesGradient, _biasesVelocity, lr, 0.0f, 0.0f);
        }
        else
        {
            // CPU SGD using in-place ops to preserve tensor identity (cached references like _biasReshaped4D)
            var scaledKernelGrad = Engine.TensorMultiplyScalar(_kernelsGradient, learningRate);
            Engine.TensorSubtractInPlace(_kernels, scaledKernelGrad);
            var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
            Engine.TensorSubtractInPlace(_biases, scaledBiasGrad);
        }

        // Notify engine that parameters have changed (for GPU cache invalidation if needed)
        // Note: SgdMomentumUpdateGpu updates in-place on GPU, so cache is valid but CPU is stale.
        // We keep using GPU buffers for forward pass.
        if (!(Engine is DirectGpuTensorEngine))
        {
            Engine.InvalidatePersistentTensor(_kernels);
            Engine.InvalidatePersistentTensor(_biases);
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all kernel weights and biases.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all trainable parameters (kernel weights and biases) from the layer
    /// and returns them as a single vector. This is useful for optimization algorithms that operate
    /// on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the learned values from the layer.
    /// 
    /// The parameters include:
    /// - All values from all pattern detectors (kernels)
    /// - All bias values
    ///
    /// These are combined into a single long list (vector), which can be used for:
    /// - Saving the model
    /// - Sharing parameters between layers
    /// - Advanced optimization techniques
    ///
    /// This provides access to all the "knowledge" the layer has learned.
    /// </para>
    /// </remarks>
    public override int ParameterCount => _isInitialized
        ? _kernels.Length + _biases.Shape[0]
        : InputDepth > 0
            ? OutputDepth * InputDepth * KernelSize * KernelSize + OutputDepth
            : 0; // Deferred: input channel count unknown until first Forward

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        EnsureInitialized();
        // Bulk copy from contiguous tensor storage — replaces 4-nested scalar loops
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_kernels.Data),
            Vector<T>.FromMemory(_biases.Data));
    }

    /// <summary>
    /// Gets all parameter gradients of the layer as a single vector.
    public override void ClearGradients()
    {
        base.ClearGradients();
        _kernelsGradient = null;
        _biasesGradient = null;
    }

    /// </summary>
    /// <returns>A vector containing all parameter gradients (kernel gradients followed by bias gradients).</returns>
    public override Vector<T> GetParameterGradients()
    {
        // If gradients haven't been computed yet, return zero gradients without
        // forcing initialization — ParameterCount already computes the correct
        // size from constructor-time shapes when the layer is uninitialized,
        // so there's no need to allocate/randomize the full weight tensors
        // just to return a zero vector.
        if (_kernelsGradient == null || _biasesGradient == null)
        {
            return new Vector<T>(ParameterCount);
        }
        EnsureInitialized();

        // Bulk copy from contiguous tensor storage — replaces 4-nested scalar loops
        return Vector<T>.Concatenate(
            Vector<T>.FromMemory(_kernelsGradient.Data),
            Vector<T>.FromMemory(_biasesGradient.Data));
    }

    /// <summary>
    /// Sets all trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (kernel weights and biases) of the layer from a single
    /// vector. The vector must have the exact length required for all parameters of the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learned values at once.
    /// 
    /// When setting parameters:
    /// - The vector must have exactly the right number of values
    /// - The values are assigned to the kernels and biases in a specific order
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Copying parameters from another model
    /// - Setting parameters that were optimized externally
    /// 
    /// It's like replacing all the "knowledge" in the layer with new information.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        EnsureInitialized();
        int kernelLen = _kernels.Length;
        int biasLen = _biases.Shape[0];
        if (parameters.Length != kernelLen + biasLen)
        {
            throw new ArgumentException($"Expected {kernelLen + biasLen} parameters, but got {parameters.Length}");
        }

        // Bulk copy into contiguous tensor storage in-place — replaces 4-nested scalar loops
        // Preserves tensor identity so engine persistent tensor references remain valid
        var src = parameters.AsSpan();
        src.Slice(0, kernelLen).CopyTo(_kernels.Data.Span);
        src.Slice(kernelLen, biasLen).CopyTo(_biases.Data.Span);

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached input and output values from the most recent forward pass.
    /// This is useful when starting to process a new sequence or when implementing stateful layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The layer forgets the last input it processed
    /// - It forgets the last output it produced
    /// 
    /// This is useful for:
    /// - Processing a new, unrelated set of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass (CPU). Use zero-sized placeholders when
        // the layer hasn't yet seen input (InputDepth still -1) — same lazy-init pattern
        // as the constructor.
        if (InputDepth > 0)
        {
            _lastInput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
            _lastOutput = new Tensor<T>([OutputDepth, InputDepth, KernelSize, KernelSize]);
        }
        else
        {
            _lastInput = new Tensor<T>([0, 0, 0, 0]);
            _lastOutput = new Tensor<T>([0, 0, 0, 0]);
        }
        _addedBatchDimension = false;

        // Clear GPU-resident cached tensors
        _lastInputGpu = null;
        _lastOutputGpu = null;
        _gpuInputShape4D = null;
    }


    /// <summary>
    /// Returns layer-specific metadata for serialization purposes.
    /// </summary>
    /// <returns>A dictionary of metadata key-value pairs including kernel size, stride, and padding.</returns>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["FilterSize"] = KernelSize.ToString();
        metadata["Stride"] = Stride.ToString();
        metadata["Padding"] = Padding.ToString();
        // Serialize activation type so deserialization restores it correctly
        // (default is ReLU, but MobileNetV3 uses Identity)
        if (ScalarActivation is not null)
        {
            metadata["ScalarActivationType"] = ScalarActivation.GetType().AssemblyQualifiedName
                ?? ScalarActivation.GetType().FullName ?? string.Empty;
        }
        return metadata;
    }

    /// <summary>
    /// Releases resources used by this layer, including GPU tensor handles.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // Release GPU handles for persistent tensors
            Engine.InvalidatePersistentTensor(_kernels);
            Engine.InvalidatePersistentTensor(_biases);

            // Return rented kernel tensor to the TensorAllocator pool so it can
            // be reused by subsequent layer constructors. Check Length > 0
            // rather than _isInitialized — EnsureInitialized rents BEFORE
            // flipping the flag, so a partial init failure (e.g., exception
            // during weight population) leaves Length > 0 but _isInitialized
            // == false. The old `_isInitialized &&` guard would leak the
            // rented tensor in that window. Length > 0 is sufficient:
            // lazy-init placeholders sit at [0, 0, 0, 0] with Length == 0
            // and aren't pool-rented.
            if (_kernels.Length > 0)
            {
                TensorAllocator.Return(_kernels);
            }

            // Return the rented forward-pass output buffer. Without this,
            // disposing many ConvolutionalLayer instances (one per conv in
            // a deep UNet) leaks one rented activation per layer to the pool
            // free list — dozens of MB per disposed model at SD scale.
            if (_preAllocatedOutput is not null)
            {
                TensorAllocator.Return(_preAllocatedOutput);
                _preAllocatedOutput = null;
            }

            // Clear other managed resources (CPU)
            _kernelsGradient = null;
            _biasesGradient = null;

            // Clear GPU-resident cached tensors
            _lastInputGpu = null;
            _lastOutputGpu = null;
            _gpuInputShape4D = null;
        }

        base.Dispose(disposing);
    }
}
