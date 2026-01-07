using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a separable convolutional layer that decomposes standard convolution into depthwise and pointwise operations.
/// </summary>
/// <remarks>
/// <para>
/// A separable convolutional layer splits the standard convolution operation into two simpler operations: 
/// a depthwise convolution followed by a pointwise convolution. This factorization significantly reduces 
/// computational complexity and number of parameters while maintaining similar model expressiveness.
/// </para>
/// <para><b>For Beginners:</b> This layer processes images or other grid-like data more efficiently than standard convolution.
/// 
/// Think of it like a two-step process:
/// - First step (depthwise): Applies filters to each input channel separately to extract features
/// - Second step (pointwise): Combines these features across all channels to create new feature maps
/// 
/// Benefits include:
/// - Fewer calculations needed (faster processing)
/// - Fewer parameters to learn (uses less memory)
/// - Often similar performance to standard convolution
/// 
/// For example, in image processing, the depthwise convolution might detect edges in each color channel separately,
/// while the pointwise convolution would combine these edges into more complex features like shapes or textures.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SeparableConvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Kernels for the depthwise convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These kernels are applied separately to each input channel during the depthwise convolution step.
    /// The shape is [inputDepth, kernelSize, kernelSize, 1], where each input channel has its own spatial filter.
    /// </para>
    /// <para><b>For Beginners:</b> These are filters that detect patterns (like edges or textures)
    /// within each individual channel of the input data. Each filter only looks at one channel at a time,
    /// which makes the computation more efficient than standard convolution.
    /// </para>
    /// </remarks>
    private Tensor<T> _depthwiseKernels;

    /// <summary>
    /// Kernels for the pointwise convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These 1×1 kernels are applied after the depthwise convolution to combine information across channels.
    /// The shape is [inputDepth, 1, 1, outputDepth], which means they only operate across the channel dimension.
    /// </para>
    /// <para><b>For Beginners:</b> These are tiny filters (just 1×1 in size) that combine information
    /// from all the input channels to create new feature maps. They don't look at spatial patterns
    /// (that's what the depthwise kernels do) but instead mix information between channels.
    /// </para>
    /// </remarks>
    private Tensor<T> _pointwiseKernels;

    /// <summary>
    /// Bias values added to each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These bias terms are added to each output channel after both the depthwise and pointwise convolutions.
    /// There is one bias value per output channel.
    /// </para>
    /// <para><b>For Beginners:</b> These are adjustable values added to each output channel.
    /// They help the network by shifting the activation values up or down, making it easier
    /// for the network to fit the data properly.
    /// </para>
    /// </remarks>
    private Tensor<T> _biases;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    /// It needs to remember this input during training so it can calculate how to improve.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the output from the layer during the forward pass, which is needed during the backward pass
    /// to compute activation function derivatives. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what output it produced.
    /// The layer needs this during training to understand how changes to the output
    /// affect the overall network performance.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Gradient of the loss with respect to the depthwise kernels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each parameter in the
    /// depthwise kernels. These gradients are computed during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each value in the depthwise kernels should be changed
    /// to improve the network's performance. Positive values mean the parameter should decrease,
    /// negative values mean it should increase.
    /// </para>
    /// </remarks>
    private Tensor<T>? _depthwiseKernelsGradient;

    /// <summary>
    /// Gradient of the loss with respect to the pointwise kernels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each parameter in the
    /// pointwise kernels. These gradients are computed during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each value in the pointwise kernels should be changed
    /// to improve the network's performance. It works the same way as the depthwise gradient but
    /// for the parameters that mix information between channels.
    /// </para>
    /// </remarks>
    private Tensor<T>? _pointwiseKernelsGradient;

    /// <summary>
    /// Gradient of the loss with respect to the biases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the gradient (rate of change) of the loss function with respect to each bias parameter.
    /// These gradients are computed during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each bias value should be changed to improve
    /// the network's performance. Biases are often easier to update than other parameters
    /// because they directly shift the output values.
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Number of channels in the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the depth (number of channels) of the input tensor. For RGB images, this would typically be 3.
    /// It determines the first dimension of the depthwise and pointwise kernels.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of different "types" of information in each location of the input.
    /// For example, an RGB image has 3 channels (red, green, blue), while a grayscale image has just 1 channel.
    /// </para>
    /// </remarks>
    private readonly int _inputDepth;

    /// <summary>
    /// Number of channels in the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the depth (number of channels) of the output tensor. It determines the number of feature maps
    /// produced by the layer and the last dimension of the pointwise kernels.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of different "features" or patterns the layer will detect.
    /// More output channels means the network can recognize more different patterns,
    /// but requires more memory and computation.
    /// </para>
    /// </remarks>
    private readonly int _outputDepth;

    /// <summary>
    /// Size of the convolution kernel (assumed to be square).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the spatial size of the convolution kernel, which is assumed to be square (same height and width).
    /// It determines the receptive field size for the depthwise convolution operation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the "window" that slides over the input data.
    /// Larger values (like 5×5 or 7×7) can detect bigger patterns, while smaller values (like 3×3)
    /// focus on fine details. Most common CNN layers use 3×3 kernels.
    /// </para>
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// Stride of the convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents how many pixels to move the kernel when sliding over the input. A larger stride results in
    /// a smaller output spatial dimensions and reduces computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is how far the sliding window moves each step.
    /// A stride of 1 moves one pixel at a time, creating an output nearly the same size as the input.
    /// A stride of 2 moves two pixels at a time, making the output about half the size,
    /// which is useful for reducing the data size as it moves through the network.
    /// </para>
    /// </remarks>
    private readonly int _stride;

    /// <summary>
    /// Padding applied to the input before convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the number of zeros added to each side of the input before convolution. Padding helps preserve
    /// the spatial dimensions and ensures that border pixels are properly processed.
    /// </para>
    /// <para><b>For Beginners:</b> This is like adding a frame of zeros around the input data.
    /// It helps the layer process the edges of the input properly. Without padding,
    /// the output would get smaller with each layer, and edge information would be lost.
    /// </para>
    /// </remarks>
    private readonly int _padding;

    /// <summary>
    /// Stores the velocity for momentum-based updates of depthwise kernels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the accumulated momentum for updating the depthwise kernels. It is used to speed up training
    /// and help escape local minima. The velocity is updated during each parameter update.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a memory of how the depthwise kernels have been changing.
    /// It helps training by making changes in consistent directions faster and smoother,
    /// similar to how a ball rolling downhill gathers momentum.
    /// </para>
    /// </remarks>
    private Tensor<T>? _depthwiseKernelsVelocity;

    /// <summary>
    /// Stores the velocity for momentum-based updates of pointwise kernels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the accumulated momentum for updating the pointwise kernels. It is used to speed up training
    /// and help escape local minima. The velocity is updated during each parameter update.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a memory of how the pointwise kernels have been changing.
    /// It works the same way as the depthwise velocity but for the parameters that
    /// mix information between channels.
    /// </para>
    /// </remarks>
    private Tensor<T>? _pointwiseKernelsVelocity;

    /// <summary>
    /// Stores the velocity for momentum-based updates of biases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the accumulated momentum for updating the biases. It is used to speed up training
    /// and help escape local minima. The velocity is updated during each parameter update.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a memory of how the bias values have been changing.
    /// It helps make the training process smoother and more effective by considering
    /// the history of previous updates.
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasesVelocity;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> as separable convolutional layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the separable convolutional layer can be trained using backpropagation.
    /// The layer contains trainable parameters (kernels and biases) that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains numbers (parameters) that can be adjusted during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// Think of it like a student who can improve by studying - this layer can get better at its job
    /// through a process called backpropagation, which adjusts its internal values based on errors it makes.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SeparableConvolutionalLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [batch, height, width, channels].</param>
    /// <param name="outputDepth">The number of output channels (feature maps).</param>
    /// <param name="kernelSize">The size of the convolution kernel (assumed to be square).</param>
    /// <param name="stride">The stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">The padding applied to the input. Defaults to 0 (no padding).</param>
    /// <param name="scalarActivation">The activation function to apply after convolution. Defaults to identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a separable convolutional layer with the specified parameters and a scalar activation function
    /// that operates on individual elements. The input shape should be a 4D tensor with dimensions [batch, height, width, channels].
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new separable convolutional layer with basic settings.
    /// 
    /// The parameters control how the layer processes data:
    /// - inputShape: The size and structure of the incoming data (like image dimensions)
    /// - outputDepth: How many different features the layer will look for
    /// - kernelSize: The size of the "window" that slides over the input (e.g., 3×3 or 5×5)
    /// - stride: How many pixels to move the window each step (smaller = more overlap)
    /// - padding: Whether to add extra space around the input edges
    /// - scalarActivation: A function that adds non-linearity (helping the network learn complex patterns)
    /// 
    /// For example, with images, larger kernels can detect bigger patterns, while more output channels
    /// can detect more varieties of patterns.
    /// </para>
    /// </remarks>
    public SeparableConvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0, IActivationFunction<T>? scalarActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding), scalarActivation ?? new IdentityActivation<T>())
    {
        _inputDepth = inputShape[3];
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize depthwise kernels
        _depthwiseKernels = new Tensor<T>([_inputDepth, _kernelSize, _kernelSize, 1]);

        // Initialize pointwise kernels
        _pointwiseKernels = new Tensor<T>([_inputDepth, 1, 1, _outputDepth]);

        // Initialize biases
        _biases = new Tensor<T>([_outputDepth]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_depthwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_pointwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SeparableConvolutionalLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor [batch, height, width, channels].</param>
    /// <param name="outputDepth">The number of output channels (feature maps).</param>
    /// <param name="kernelSize">The size of the convolution kernel (assumed to be square).</param>
    /// <param name="stride">The stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">The padding applied to the input. Defaults to 0 (no padding).</param>
    /// <param name="vectorActivation">The vector activation function to apply after convolution. Defaults to identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a separable convolutional layer with the specified parameters and a vector activation function
    /// that operates on entire vectors rather than individual elements. The input shape should be a 4D tensor with 
    /// dimensions [batch, height, width, channels].
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new separable convolutional layer with advanced settings.
    /// 
    /// Similar to the basic constructor, but with one key difference:
    /// - It uses a vector activation function instead of a scalar one
    /// 
    /// A vector activation function:
    /// - Works on entire groups of numbers at once, not just one at a time
    /// - Can capture relationships between different elements in the output
    /// - Is useful for more complex AI tasks
    /// 
    /// This constructor is for advanced users who need more sophisticated activation patterns
    /// for their neural networks.
    /// </para>
    /// </remarks>
    public SeparableConvolutionalLayer(int[] inputShape, int outputDepth, int kernelSize, int stride = 1, int padding = 0, IVectorActivationFunction<T>? vectorActivation = null)
        : base(inputShape, CalculateOutputShape(inputShape, outputDepth, kernelSize, stride, padding), vectorActivation ?? new IdentityActivation<T>())
    {
        _inputDepth = inputShape[3];
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        // Initialize depthwise kernels
        _depthwiseKernels = new Tensor<T>([_inputDepth, _kernelSize, _kernelSize, 1]);

        // Initialize pointwise kernels
        _pointwiseKernels = new Tensor<T>([_inputDepth, 1, 1, _outputDepth]);

        // Initialize biases
        _biases = new Tensor<T>([_outputDepth]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_depthwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_pointwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Calculates the output shape of the separable convolutional layer based on input shape and parameters.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="outputDepth">The number of output channels.</param>
    /// <param name="kernelSize">The size of the convolution kernel.</param>
    /// <param name="stride">The stride of the convolution.</param>
    /// <param name="padding">The padding applied to the input.</param>
    /// <returns>The calculated output shape for the separable convolutional layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output dimensions of the separable convolutional layer based on the input dimensions
    /// and the layer's parameters. The calculation follows the standard formula for convolutional layers:
    /// outputDim = (inputDim - kernelSize + 2 * padding) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// It uses a standard formula to calculate how the dimensions will change:
    /// - The batch size stays the same
    /// - The height and width get smaller based on the kernel size, stride, and padding
    /// - The depth changes to match the outputDepth parameter
    /// 
    /// For example, with a 28×28 image input, a 3×3 kernel, stride of 1, and no padding:
    /// - Output height: (28 - 3 + 0) / 1 + 1 = 26
    /// - Output width: (28 - 3 + 0) / 1 + 1 = 26
    /// - So the output shape would be [batch, 26, 26, outputDepth]
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int outputDepth, int kernelSize, int stride, int padding)
    {
        int outputHeight = (inputShape[1] - kernelSize + 2 * padding) / stride + 1;
        int outputWidth = (inputShape[2] - kernelSize + 2 * padding) / stride + 1;

        return [inputShape[0], outputHeight, outputWidth, outputDepth];
    }

    /// <summary>
    /// Initializes the layer's parameters (kernels and biases).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the depthwise and pointwise kernels using He initialization, which scales the random values
    /// based on the fan-in (number of input connections). Biases are initialized to zero. Proper initialization helps
    /// the network converge faster during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's learnable parameters.
    /// 
    /// The initialization process:
    /// - Sets up the kernels (filters) with small random values using "He initialization"
    /// - Sets all biases to zero
    /// 
    /// Good initialization is important because:
    /// - It helps the network learn more quickly
    /// - It prevents problems like vanishing or exploding gradients
    /// - It gives the network a good starting point for learning
    /// 
    /// The "He initialization" method is specially designed to work well with modern neural networks
    /// and helps them learn effectively from the beginning of training.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Use He initialization for kernels
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize * _inputDepth)));
        InitializeTensor(_depthwiseKernels, scale);
        InitializeTensor(_pointwiseKernels, scale);

        // Initialize biases to zero
        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given tensor with random values between -0.5 and 0.5, scaled by the provided scale factor.
    /// This type of initialization helps prevent large initial values that could cause instability during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a tensor with appropriate random starting values.
    /// 
    /// It works by:
    /// - Generating random numbers between -0.5 and 0.5
    /// - Multiplying each by a scale factor to get the right size
    /// - Setting each element in the tensor to these scaled random values
    /// 
    /// Having good starting values is like giving the neural network a head start in the right direction
    /// before training begins.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        var rand = Vector<T>.CreateRandom(tensor.Length, -0.5, 0.5);
        var randTensor = new Tensor<T>(tensor.Shape, rand);
        var scaled = Engine.TensorMultiplyScalar(randTensor, scale);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = scaled[i];
        }
    }

    /// <summary>
    /// Performs the forward pass of the separable convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after separable convolution and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the separable convolutional layer. It performs a depthwise convolution
    /// followed by a pointwise convolution. The depthwise convolution applies a separate filter to each input channel,
    /// and the pointwise convolution applies a 1x1 convolution to combine the channels. The result is passed through
    /// an activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through the layer.
    /// 
    /// The forward pass happens in three steps:
    /// 1. Depthwise convolution: Applies separate filters to each input channel
    ///    - Like having a specialized detector for each input feature
    ///    - Captures spatial patterns within each channel independently
    ///
    /// 2. Pointwise convolution: Combines results across all channels
    ///    - Uses 1×1 filters to mix information between channels
    ///    - Creates new feature maps that combine information from all inputs
    ///    - Adds bias values to each output channel
    ///
    /// 3. Activation: Applies a non-linear function to the results
    ///    - Helps the network learn more complex patterns
    ///
    /// The method also saves the input and output for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Convert input from NHWC [batch, H, W, channels] to NCHW [batch, channels, H, W]
        var inputNCHW = input.Transpose([0, 3, 1, 2]);

        var strideArr = new int[] { _stride, _stride };
        var paddingArr = new int[] { _padding, _padding };

        // Convert depthwise kernels from [inputDepth, kernelSize, kernelSize, 1] to [inputDepth, 1, kernelSize, kernelSize]
        var depthwiseKernelNCHW = ConvertDepthwiseKernelToNCHW(_depthwiseKernels);

        // Step 1: Depthwise convolution using Engine
        var depthwiseOutputNCHW = Engine.DepthwiseConv2D(inputNCHW, depthwiseKernelNCHW, strideArr, paddingArr);

        // Convert pointwise kernels from [inputDepth, 1, 1, outputDepth] to [outputDepth, inputDepth, 1, 1]
        var pointwiseKernelNCHW = ConvertPointwiseKernelToNCHW(_pointwiseKernels);

        // Step 2: Pointwise convolution (1x1 conv) with fused bias using Engine
        // Note: Can't fuse activation because it's applied after NCHW->NHWC transpose
        var pointwiseOutputNCHW = Engine.FusedConv2D(
            depthwiseOutputNCHW, pointwiseKernelNCHW, _biases,
            1, 1,   // stride
            0, 0,   // padding
            1, 1,   // dilation
            FusedActivationType.None);  // Activation applied after transpose

        // Convert output from NCHW back to NHWC
        var output = pointwiseOutputNCHW.Transpose([0, 2, 3, 1]);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass of the separable convolutional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the separable convolutional layer, which is used during training to propagate
    /// error gradients back through the network. It computes gradients for both depthwise and pointwise kernels, as well as biases,
    /// and returns the gradient with respect to the input for further backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's inputs
    /// and parameters should change to reduce errors.
    ///
    /// The backward pass:
    /// 1. Starts with gradients (error signals) from the next layer
    /// 2. Computes how to adjust the layer's parameters (kernels and biases)
    /// 3. Calculates how to adjust the input that was received
    ///
    /// This happens in reverse order compared to the forward pass:
    /// - First backpropagates through the pointwise convolution
    /// - Then backpropagates through the depthwise convolution
    ///
    /// The calculated gradients are stored for later use when updating the parameters,
    /// and the input gradient is returned to continue the backpropagation process.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Apply activation derivative
        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Convert gradients from NHWC to NCHW for Engine operations
        var deltaNCHW = delta.Transpose([0, 3, 1, 2]);
        var inputNCHW = _lastInput.Transpose([0, 3, 1, 2]);

        var strideArr = new int[] { _stride, _stride };
        var paddingArr = new int[] { _padding, _padding };

        // Convert kernels to NCHW format for backward computation
        var depthwiseKernelNCHW = ConvertDepthwiseKernelToNCHW(_depthwiseKernels);
        var pointwiseKernelNCHW = ConvertPointwiseKernelToNCHW(_pointwiseKernels);

        // Recompute depthwise output (needed for pointwise backward)
        var depthwiseOutputNCHW = Engine.DepthwiseConv2D(inputNCHW, depthwiseKernelNCHW, strideArr, paddingArr);

        // Calculate bias gradient: sum over batch, height, width (axes 0, 2, 3 in NCHW)
        _biasesGradient = Engine.ReduceSum(deltaNCHW, new[] { 0, 2, 3 }, keepDims: false);

        // Calculate pointwise kernel gradient (1x1 conv backward)
        var pointwiseKernelGradNCHW = Engine.Conv2DBackwardKernel(deltaNCHW, depthwiseOutputNCHW, pointwiseKernelNCHW.Shape, new int[] { 1, 1 }, new int[] { 0, 0 }, new int[] { 1, 1 });
        _pointwiseKernelsGradient = ConvertPointwiseKernelFromNCHW(pointwiseKernelGradNCHW);

        // Calculate depthwise output gradient (backward through pointwise)
        var depthwiseGradNCHW = Engine.Conv2DBackwardInput(deltaNCHW, pointwiseKernelNCHW, depthwiseOutputNCHW.Shape, new int[] { 1, 1 }, new int[] { 0, 0 }, new int[] { 1, 1 });

        // Calculate depthwise kernel gradient
        var depthwiseKernelGradNCHW = Engine.DepthwiseConv2DBackwardKernel(depthwiseGradNCHW, inputNCHW, depthwiseKernelNCHW.Shape, strideArr, paddingArr);
        _depthwiseKernelsGradient = ConvertDepthwiseKernelFromNCHW(depthwiseKernelGradNCHW);

        // Calculate input gradient (backward through depthwise)
        var inputGradientNCHW = Engine.DepthwiseConv2DBackwardInput(depthwiseGradNCHW, depthwiseKernelNCHW, inputNCHW.Shape, strideArr, paddingArr);

        // Convert input gradient from NCHW back to NHWC
        return inputGradientNCHW.Transpose([0, 2, 3, 1]);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// Production-grade pattern: Uses cached _lastOutput for activation derivative computation,
    /// Engine.TensorMultiply for GPU/CPU acceleration, and minimal autodiff graph.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Production-grade: Compute activation derivative using cached output
        Tensor<T> preActivationGradient;
        if (VectorActivation != null)
        {
            var actDeriv = VectorActivation.Derivative(_lastOutput);
            preActivationGradient = Engine.TensorMultiply(outputGradient, actDeriv);
        }
        else if (ScalarActivation != null && ScalarActivation is not IdentityActivation<T>)
        {
            var activation = ScalarActivation;
            var activationDerivative = _lastOutput.Transform((x, _) => activation.Derivative(x));
            preActivationGradient = Engine.TensorMultiply(outputGradient, activationDerivative);
        }
        else
        {
            preActivationGradient = outputGradient;
        }

        // Convert from NHWC [batch, H, W, channels] to NCHW [batch, channels, H, W] using Tensor.Transpose
        var inputNCHW = _lastInput.Transpose([0, 3, 1, 2]);
        var preActivationGradientNCHW = preActivationGradient.Transpose([0, 3, 1, 2]);

        // Convert kernels to NCHW format (kernel format requires specialized conversion)
        var depthwiseKernelNCHW = ConvertDepthwiseKernelToNCHW(_depthwiseKernels);
        var pointwiseKernelNCHW = ConvertPointwiseKernelToNCHW(_pointwiseKernels);

        // Create computation nodes
        var inputNode = Autodiff.TensorOperations<T>.Variable(inputNCHW, "input", requiresGradient: true);
        var depthwiseKernelNode = Autodiff.TensorOperations<T>.Variable(depthwiseKernelNCHW, "depthwise_kernel", requiresGradient: true);
        var pointwiseKernelNode = Autodiff.TensorOperations<T>.Variable(pointwiseKernelNCHW, "pointwise_kernel", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_biases, "bias", requiresGradient: true);

        // Build minimal autodiff graph for linear operations (activation derivative already applied)
        // Step 1: Depthwise convolution (no bias)
        var depthwiseOutput = Autodiff.TensorOperations<T>.DepthwiseConv2D(
            inputNode,
            depthwiseKernelNode,
            bias: null,
            stride: new int[] { _stride, _stride },
            padding: new int[] { _padding, _padding });

        // Step 2: Pointwise convolution (1x1 conv with bias)
        var preActivationNode = Autodiff.TensorOperations<T>.Conv2D(
            depthwiseOutput,
            pointwiseKernelNode,
            biasNode,
            stride: new int[] { 1, 1 },
            padding: new int[] { 0, 0 });

        // Set gradient on pre-activation node (activation derivative already applied)
        preActivationNode.Gradient = preActivationGradientNCHW;

        // Inline topological sort and backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((preActivationNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        // Extract gradients with format conversions for kernels
        if (depthwiseKernelNode.Gradient != null)
            _depthwiseKernelsGradient = ConvertDepthwiseKernelFromNCHW(depthwiseKernelNode.Gradient);

        if (pointwiseKernelNode.Gradient != null)
            _pointwiseKernelsGradient = ConvertPointwiseKernelFromNCHW(pointwiseKernelNode.Gradient);

        if (biasNode.Gradient != null)
            _biasesGradient = biasNode.Gradient;

        // Convert input gradient from NCHW back to NHWC using Transpose
        var inputGradientNCHW = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
        return inputGradientNCHW.Transpose([0, 2, 3, 1]);
    }

    /// <summary>
    /// Converts depthwise kernel from [inputDepth, kernelSize, kernelSize, 1] to [inputDepth, 1, kernelSize, kernelSize] format.
    /// </summary>
    private Tensor<T> ConvertDepthwiseKernelToNCHW(Tensor<T> kernel) =>
        // [inputDepth, kernelH, kernelW, 1] -> [inputDepth, 1, kernelH, kernelW]
        kernel.Transpose([0, 3, 1, 2]);

    /// <summary>
    /// Converts depthwise kernel from [inputDepth, 1, kernelSize, kernelSize] back to [inputDepth, kernelSize, kernelSize, 1] format.
    /// </summary>
    private Tensor<T> ConvertDepthwiseKernelFromNCHW(Tensor<T> kernel) =>
        // [inputDepth, 1, kernelH, kernelW] -> [inputDepth, kernelH, kernelW, 1]
        kernel.Transpose([0, 2, 3, 1]);

    /// <summary>
    /// Converts pointwise kernel from [inputDepth, 1, 1, outputDepth] to [outputDepth, inputDepth, 1, 1] format.
    /// </summary>
    private Tensor<T> ConvertPointwiseKernelToNCHW(Tensor<T> kernel) =>
        // [inputDepth, 1, 1, outputDepth] -> [outputDepth, inputDepth, 1, 1]
        kernel.Transpose([3, 0, 1, 2]);

    /// <summary>
    /// Converts pointwise kernel from [outputDepth, inputDepth, 1, 1] back to [inputDepth, 1, 1, outputDepth] format.
    /// </summary>
    private Tensor<T> ConvertPointwiseKernelFromNCHW(Tensor<T> kernel) =>
        // [outputDepth, inputDepth, 1, 1] -> [inputDepth, 1, 1, outputDepth]
        kernel.Transpose([1, 2, 3, 0]);

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients and momentum.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the depthwise kernels, pointwise kernels, and biases of the layer based on the gradients
    /// calculated during the backward pass. It uses momentum and L2 regularization to improve training stability and
    /// prevent overfitting. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. Momentum is used to speed up learning
    ///    - Like a ball rolling downhill, gaining speed in consistent directions
    ///    - Helps overcome small obstacles and reach better solutions faster
    /// 
    /// 2. L2 regularization helps prevent overfitting
    ///    - Slightly reduces the size of parameters over time
    ///    - Encourages the network to learn simpler patterns
    ///    - Helps the model generalize better to new data
    /// 
    /// 3. The learning rate controls how big each update step is
    ///    - Smaller learning rates: slower but more stable learning
    ///    - Larger learning rates: faster but potentially unstable learning
    /// 
    /// This process is repeated many times during training, gradually improving
    /// the layer's performance on the task.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_depthwiseKernelsGradient == null || _pointwiseKernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T momentum = NumOps.FromDouble(0.9); // Momentum factor
        T l2RegularizationFactor = NumOps.FromDouble(0.0001); // L2 regularization factor

        // Initialize velocity tensors if they don't exist
        if (_depthwiseKernelsVelocity == null)
            _depthwiseKernelsVelocity = new Tensor<T>(_depthwiseKernels.Shape);
        if (_pointwiseKernelsVelocity == null)
            _pointwiseKernelsVelocity = new Tensor<T>(_pointwiseKernels.Shape);
        if (_biasesVelocity == null)
            _biasesVelocity = new Tensor<T>([_outputDepth]);

        // Update depthwise kernels using Engine operations
        var dwL2 = Engine.TensorMultiplyScalar(_depthwiseKernels, l2RegularizationFactor);
        var dwGradWithL2 = Engine.TensorAdd(_depthwiseKernelsGradient, dwL2);
        _depthwiseKernelsVelocity = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_depthwiseKernelsVelocity, momentum),
            Engine.TensorMultiplyScalar(dwGradWithL2, learningRate));
        _depthwiseKernels = Engine.TensorSubtract(_depthwiseKernels, _depthwiseKernelsVelocity);

        // Update pointwise kernels using Engine operations
        var pwL2 = Engine.TensorMultiplyScalar(_pointwiseKernels, l2RegularizationFactor);
        var pwGradWithL2 = Engine.TensorAdd(_pointwiseKernelsGradient, pwL2);
        _pointwiseKernelsVelocity = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_pointwiseKernelsVelocity, momentum),
            Engine.TensorMultiplyScalar(pwGradWithL2, learningRate));
        _pointwiseKernels = Engine.TensorSubtract(_pointwiseKernels, _pointwiseKernelsVelocity);

        // Update biases using Engine operations (no L2 on biases)
        _biasesVelocity = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_biasesVelocity, momentum),
            Engine.TensorMultiplyScalar(_biasesGradient, learningRate));
        _biases = Engine.TensorSubtract(_biases, _biasesVelocity);

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_depthwiseKernels);
        Engine.InvalidatePersistentTensor(_pointwiseKernels);
        Engine.InvalidatePersistentTensor(_biases);

        // Clear gradients
        _depthwiseKernelsGradient = null;
        _pointwiseKernelsGradient = null;
        _biasesGradient = null;
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the layer (depthwise kernels, pointwise kernels, and biases)
    /// and combines them into a single vector. This is useful for optimization algorithms that operate on all parameters
    /// at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include depthwise kernels, pointwise kernels, and biases
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(_depthwiseKernels.ToVector(), _pointwiseKernels.ToVector()),
            _biases.ToVector());
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the layer (depthwise kernels, pointwise kernels, and biases)
    /// from a single vector. It expects the vector to contain the parameters in the same order as they are retrieved
    /// by GetParameters(). This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer from a single list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with exactly the right number of values
    /// - The values are distributed to the appropriate places (depthwise kernels, pointwise kernels, and biases)
    /// - The order must match how they were stored in GetParameters()
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        int dwLen = _depthwiseKernels.Length;
        int pwLen = _pointwiseKernels.Length;
        int biasLen = _biases.Length;

        var dwVec = parameters.Slice(0, dwLen);
        var pwVec = parameters.Slice(dwLen, pwLen);
        var biasVec = parameters.Slice(dwLen + pwLen, biasLen);

        _depthwiseKernels = Tensor<T>.FromVector(dwVec, [_inputDepth, _kernelSize, _kernelSize, 1]);
        _pointwiseKernels = Tensor<T>.FromVector(pwVec, [_inputDepth, 1, 1, _outputDepth]);
        _biases = Tensor<T>.FromVector(biasVec, [_outputDepth]);

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_depthwiseKernels);
        Engine.InvalidatePersistentTensor(_pointwiseKernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Resets the internal state of the separable convolutional layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the separable convolutional layer, including the cached inputs and outputs,
    /// gradients, and velocity tensors. This is useful when starting to process a new batch or when implementing
    /// stateful networks that need to be reset between sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous passes are cleared
    /// - Calculated gradients are cleared
    /// - Momentum (velocity) information is cleared
    /// 
    /// This is important for:
    /// - Processing a new batch of unrelated data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like erasing the whiteboard before starting a new calculation -
    /// it ensures that old information doesn't interfere with new processing.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _depthwiseKernelsGradient = null;
        _pointwiseKernelsGradient = null;
        _biasesGradient = null;

        // Optionally reset velocity tensors
        _depthwiseKernelsVelocity = null;
        _pointwiseKernelsVelocity = null;
        _biasesVelocity = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-accelerated execution.
    /// </summary>
    /// <value>
    /// <c>true</c> when kernels and biases are initialized and the engine is a DirectGpuTensorEngine.
    /// </value>
    /// <remarks>
    /// <para>
    /// GPU execution for separable convolution uses DepthwiseConv2DGpu for the depthwise step
    /// and FusedConv2DGpu for the pointwise step with fused bias and activation.
    /// </para>
    /// </remarks>
    protected override bool SupportsGpuExecution =>
        _depthwiseKernels is not null &&
        _pointwiseKernels is not null &&
        _biases is not null &&
        Engine is DirectGpuTensorEngine;

    /// <summary>
    /// Performs the forward pass on GPU, keeping all tensors GPU-resident.
    /// </summary>
    /// <param name="inputs">The input GPU tensors in NCHW format [batch, channels, height, width].</param>
    /// <returns>The output GPU tensor in NCHW format.</returns>
    /// <remarks>
    /// <para>
    /// This method executes separable convolution entirely on GPU:
    /// 1. Depthwise convolution: Each input channel is convolved with its own filter
    /// 2. Pointwise convolution: 1x1 conv combines channels with fused bias and activation
    /// </para>
    /// <para><b>Performance Notes:</b></para>
    /// <list type="bullet">
    /// <item>Input tensors remain GPU-resident throughout computation</item>
    /// <item>Intermediate depthwise output is disposed after use</item>
    /// <item>Kernels are converted to NCHW format for GPU operations</item>
    /// <item>Activation is fused into the pointwise convolution when possible</item>
    /// </list>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

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
                $"SeparableConv2D input requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");
        }

        var originalInputShape = input.Shape;
        int rank = input.Shape.Length;
        bool addedBatchDimension = false;

        // Reshape input to 4D [B, C, H, W] for convolution
        IGpuTensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            addedBatchDimension = true;
            input4D = input.CreateView(0, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        }
        else if (rank == 4)
        {
            // 4D [B, C, H, W] - no reshaping needed
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
            {
                flatBatch *= input.Shape[d];
            }
            input4D = input.CreateView(0, [flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // Validate input channels
        int actualInputChannels = input4D.Shape[1];
        if (actualInputChannels != _inputDepth)
        {
            throw new ArgumentException(
                $"Expected input depth {_inputDepth}, but got {actualInputChannels}.");
        }

        // Convert depthwise kernels from [inputDepth, kernelSize, kernelSize, 1]
        // to [inputDepth, 1, kernelSize, kernelSize] for GPU operations
        var depthwiseKernelNCHW = ConvertDepthwiseKernelToNCHW(_depthwiseKernels);

        // Step 1: GPU-fused depthwise convolution (no bias, no activation)
        var depthwiseOutput = gpuEngine.DepthwiseConv2DGpu(
            input4D,
            depthwiseKernelNCHW,
            null,                    // no bias for depthwise step
            _stride, _stride,        // strideH, strideW
            _padding, _padding,      // padH, padW
            FusedActivationType.None); // no activation for depthwise step

        // Convert pointwise kernels from [inputDepth, 1, 1, outputDepth]
        // to [outputDepth, inputDepth, 1, 1] for GPU operations
        var pointwiseKernelNCHW = ConvertPointwiseKernelToNCHW(_pointwiseKernels);

        // Step 2: GPU-fused pointwise (1x1) convolution + bias + activation
        // Use MapActivationToFused() from base class to avoid code duplication
        var fusedActivation = MapActivationToFused();
        var result = gpuEngine.FusedConv2DGpu(
            depthwiseOutput,
            pointwiseKernelNCHW,
            _biases,
            1, 1,                    // stride 1x1
            0, 0,                    // no padding for 1x1 conv
            1, 1,                    // dilation 1x1
            fusedActivation);

        // Dispose intermediate depthwise output (no longer needed)
        depthwiseOutput.Dispose();

        // Restore original shape if needed
        if (originalInputShape.Length > 4)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[originalInputShape.Length];
            for (int d = 0; d < originalInputShape.Length - 3; d++)
            {
                outputShape[d] = originalInputShape[d];
            }
            outputShape[originalInputShape.Length - 3] = _outputDepth;
            outputShape[originalInputShape.Length - 2] = result.Shape[2];
            outputShape[originalInputShape.Length - 1] = result.Shape[3];
            return result.CreateView(0, outputShape);
        }

        if (addedBatchDimension)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            return result.CreateView(0, [_outputDepth, result.Shape[2], result.Shape[3]]);
        }

        return result;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when kernels are initialized and activation function supports JIT.
    /// </value>
    /// <remarks>
    /// <para>
    /// Separable convolutional layers support JIT compilation using DepthwiseConv2D and Conv2D
    /// operations from TensorOperations. The layer performs depthwise convolution followed by
    /// pointwise (1x1) convolution.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _depthwiseKernels != null && _pointwiseKernels != null && _biases != null &&
        CanActivationBeJitted();

    /// <summary>
    /// Exports the separable convolutional layer's forward pass as a JIT-compilable computation graph.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the separable convolution output.</returns>
    /// <remarks>
    /// <para>
    /// The separable convolution computation graph implements:
    /// 1. Depthwise convolution: Applies separate filters to each input channel
    /// 2. Pointwise convolution: 1x1 convolution to combine channels
    /// 3. Activation function
    /// </para>
    /// <para><b>For Beginners:</b> This creates an optimized version of the separable convolution.
    /// It's more efficient than standard convolution by splitting the operation into two steps.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_depthwiseKernels == null || _pointwiseKernels == null || _biases == null)
            throw new InvalidOperationException("Kernels and biases not initialized.");

        if (InputShape == null || InputShape.Length < 4)
            throw new InvalidOperationException("Layer input shape not configured. Expected [batch, height, width, channels].");

        // Validate activation can be JIT compiled
        if (!CanActivationBeJitted())
        {
            var activationType = (ScalarActivation?.GetType() ?? VectorActivation?.GetType())?.Name ?? "Unknown";
            throw new NotSupportedException(
                $"Activation function '{activationType}' is not supported for JIT compilation. " +
                "Supported activations: ReLU, Sigmoid, Tanh, Softmax, Identity");
        }

        // Create symbolic input node in NHWC format [batch, height, width, channels]
        var symbolicInput = new Tensor<T>(new int[] { 1, InputShape[1], InputShape[2], InputShape[3] });
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "separable_input");
        inputNodes.Add(inputNode);

        // Convert depthwise kernels from [inputDepth, kernelSize, kernelSize, 1] to [inputDepth, 1, kernelSize, kernelSize]
        var depthwiseKernelNCHW = ConvertDepthwiseKernelToNCHW(_depthwiseKernels);
        var depthwiseKernelNode = Autodiff.TensorOperations<T>.Constant(depthwiseKernelNCHW, "depthwise_kernel");

        // Convert pointwise kernels from [inputDepth, 1, 1, outputDepth] to [outputDepth, inputDepth, 1, 1]
        var pointwiseKernelNCHW = ConvertPointwiseKernelToNCHW(_pointwiseKernels);
        var pointwiseKernelNode = Autodiff.TensorOperations<T>.Constant(pointwiseKernelNCHW, "pointwise_kernel");

        // Bias is already a Tensor<T>
        var biasNode = Autodiff.TensorOperations<T>.Constant(_biases, "bias");

        // Step 1: Depthwise convolution (no bias)
        var depthwiseOutput = Autodiff.TensorOperations<T>.DepthwiseConv2D(
            inputNode,
            depthwiseKernelNode,
            bias: null,
            stride: new int[] { _stride, _stride },
            padding: new int[] { _padding, _padding });

        // Step 2: Pointwise convolution (1x1 conv with bias)
        var pointwiseOutput = Autodiff.TensorOperations<T>.Conv2D(
            depthwiseOutput,
            pointwiseKernelNode,
            biasNode,
            stride: new int[] { 1, 1 },
            padding: new int[] { 0, 0 });

        // Step 3: Apply activation function using base class helper
        var output = ApplyActivationToGraph(pointwiseOutput);

        return output;
    }
}
