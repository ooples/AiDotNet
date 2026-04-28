#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 8, 8, 1", TestConstructorArgs = "2, 3, 1, 0, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class SeparableConvolutionalLayer<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

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
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    private int _inputDepth;

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

    #region GPU Training Fields
    private Tensor<T>? _gpuLastInput;
    private Tensor<T>? _gpuLastOutput;

    // GPU weight buffers
    private Tensor<T>? _gpuDepthwiseKernels;
    private Tensor<T>? _gpuPointwiseKernels;
    private Tensor<T>? _gpuBiases;

    // GPU gradient buffers
    private Tensor<T>? _gpuDepthwiseKernelsGradient;
    private Tensor<T>? _gpuPointwiseKernelsGradient;
    private Tensor<T>? _gpuBiasesGradient;

    // GPU velocity buffers (SGD momentum)
    private Tensor<T>? _gpuDepthwiseKernelsVelocity;
    private Tensor<T>? _gpuPointwiseKernelsVelocity;
    private Tensor<T>? _gpuBiasesVelocityGpu;

    // GPU Adam first moment buffers
    private Tensor<T>? _gpuDepthwiseKernelsM;
    private Tensor<T>? _gpuPointwiseKernelsM;
    private Tensor<T>? _gpuBiasesM;

    // GPU Adam second moment buffers
    private Tensor<T>? _gpuDepthwiseKernelsV;
    private Tensor<T>? _gpuPointwiseKernelsV;
    private Tensor<T>? _gpuBiasesV;
    #endregion

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
    public override int ParameterCount => _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;
    public override bool SupportsTraining => true;

    public override Vector<T> GetParameterGradients()
    {
        if (_depthwiseKernelsGradient == null || _pointwiseKernelsGradient == null || _biasesGradient == null)
            return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(new Vector<T>(_depthwiseKernelsGradient.ToArray()), new Vector<T>(_pointwiseKernelsGradient.ToArray())),
            new Vector<T>(_biasesGradient.ToArray()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _depthwiseKernelsGradient = null; _pointwiseKernelsGradient = null; _biasesGradient = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-resident training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

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
    public SeparableConvolutionalLayer(int outputDepth, int kernelSize, int stride = 1, int padding = 0, IActivationFunction<T>? scalarActivation = null)
        : base(new[] { -1, -1, -1 }, new[] { -1, -1, outputDepth },
               scalarActivation ?? new IdentityActivation<T>())
    {
        if (outputDepth <= 0) throw new ArgumentOutOfRangeException(nameof(outputDepth));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _inputDepth = -1;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([0, 0, 0, 0]);
        _pointwiseKernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
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
    public SeparableConvolutionalLayer(int outputDepth, int kernelSize, int stride, int padding, IVectorActivationFunction<T> vectorActivation)
        : base(new[] { -1, -1, -1 }, new[] { -1, -1, outputDepth },
               vectorActivation ?? new IdentityActivation<T>())
    {
        if (outputDepth <= 0) throw new ArgumentOutOfRangeException(nameof(outputDepth));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _inputDepth = -1;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([0, 0, 0, 0]);
        _pointwiseKernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
    }

    /// <summary>
    /// Resolves input depth from input.Shape (NHWC last axis) and output spatial dims on first forward.
    /// Output dim per axis: (input - kernelSize + 2 * padding) / stride + 1.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        int b, h, w, c;
        if (rank == 4) { b = input.Shape[0]; h = input.Shape[1]; w = input.Shape[2]; c = input.Shape[3]; }
        else if (rank == 3) { b = 1; h = input.Shape[0]; w = input.Shape[1]; c = input.Shape[2]; }
        else throw new ArgumentException(
            $"SeparableConvolutionalLayer requires rank-3 [H,W,C] or rank-4 [B,H,W,C] input; got rank {rank}.",
            nameof(input));

        _inputDepth = c;
        int outH = (h - _kernelSize + 2 * _padding) / _stride + 1;
        int outW = (w - _kernelSize + 2 * _padding) / _stride + 1;

        _depthwiseKernels = new Tensor<T>([_inputDepth, _kernelSize, _kernelSize, 1]);
        _pointwiseKernels = new Tensor<T>([_inputDepth, 1, 1, _outputDepth]);
        _biases = new Tensor<T>([_outputDepth]);
        InitializeParameters();
        RegisterTrainableParameter(_depthwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_pointwiseKernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        ResolveShapes(new[] { b, h, w, c }, new[] { b, outH, outW, _outputDepth });
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
        var randTensor = new Tensor<T>(tensor._shape, rand);
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
        EnsureInitializedFromInput(input);
        _lastInput = input;

        // Convert input from NHWC [batch, H, W, channels] to NCHW [batch, channels, H, W]
        var inputNCHW = Engine.TensorPermute(input, [0, 3, 1, 2]);

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
        // Reshape bias from [outputDepth] to [1, outputDepth, 1, 1] for NCHW broadcast
        var biasReshaped4D = Engine.Reshape(_biases, [1, _outputDepth, 1, 1]);
        var pointwiseOutputNCHW = Engine.FusedConv2D(
            depthwiseOutputNCHW, pointwiseKernelNCHW, biasReshaped4D,
            1, 1,   // stride
            0, 0,   // padding
            1, 1,   // dilation
            FusedActivationType.None);  // Activation applied after transpose

        // Convert output from NCHW back to NHWC
        var output = Engine.TensorPermute(pointwiseOutputNCHW, [0, 2, 3, 1]);

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
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
            _depthwiseKernelsVelocity = new Tensor<T>(_depthwiseKernels._shape);
        if (_pointwiseKernelsVelocity == null)
            _pointwiseKernelsVelocity = new Tensor<T>(_pointwiseKernels._shape);
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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
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

        var originalInputShape = input._shape;
        int rank = input.Shape.Length;
        bool addedBatchDimension = false;

        // Reshape input to 4D [B, C, H, W] for convolution
        Tensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            addedBatchDimension = true;
            input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
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
            input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
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

        // Cache inputs and outputs for backward pass during training
        if (IsTrainingMode)
        {
            _gpuLastInput = input4D;
            _gpuLastOutput = result;
        }

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
            return result.Reshape(outputShape);
        }

        if (addedBatchDimension)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            return result.Reshape([_outputDepth, result.Shape[2], result.Shape[3]]);
        }

        return result;
    }

    #region GPU Training Methods

    /// <summary>
    /// Updates parameters on GPU using the configured optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Ensure GPU weight buffers exist
        _gpuDepthwiseKernels ??= GpuTensorHelper.UploadToGpu<T>(backend, _depthwiseKernels, GpuTensorRole.Weight);
        _gpuPointwiseKernels ??= GpuTensorHelper.UploadToGpu<T>(backend, _pointwiseKernels, GpuTensorRole.Weight);
        _gpuBiases ??= GpuTensorHelper.UploadToGpu<T>(backend, _biases, GpuTensorRole.Weight);

        // Ensure optimizer state exists
        EnsureSeparableConvOptimizerState(config, backend);

        // Apply updates for depthwise kernels
        if (_gpuDepthwiseKernelsGradient is not null)
        {
            var dwState = BuildSeparableConvOptimizerState("depthwise");
            config.ApplyUpdate(backend, _gpuDepthwiseKernels.Buffer, _gpuDepthwiseKernelsGradient.Buffer, dwState, _depthwiseKernels.Length);
        }

        // Apply updates for pointwise kernels
        if (_gpuPointwiseKernelsGradient is not null)
        {
            var pwState = BuildSeparableConvOptimizerState("pointwise");
            config.ApplyUpdate(backend, _gpuPointwiseKernels.Buffer, _gpuPointwiseKernelsGradient.Buffer, pwState, _pointwiseKernels.Length);
        }

        // Apply updates for biases
        if (_gpuBiasesGradient is not null)
        {
            var biasState = BuildSeparableConvOptimizerState("biases");
            config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasesGradient.Buffer, biasState, _biases.Length);
        }

        // Download updated weights back to CPU tensors
        _depthwiseKernels = _gpuDepthwiseKernels;
        _pointwiseKernels = _gpuPointwiseKernels;
        _biases = _gpuBiases;

        // Notify engine that tensor data has changed
        Engine.InvalidatePersistentTensor(_depthwiseKernels);
        Engine.InvalidatePersistentTensor(_pointwiseKernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    private void EnsureSeparableConvOptimizerState(IGpuOptimizerConfig config, IDirectGpuBackend backend)
    {
        var optimizerType = config.OptimizerType;

        // Ensure velocity buffers for SGD momentum, NAG, LARS
        if (optimizerType == GpuOptimizerType.Sgd || optimizerType == GpuOptimizerType.Nag || optimizerType == GpuOptimizerType.Lars)
        {
            _gpuDepthwiseKernelsVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_depthwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuPointwiseKernelsVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_pointwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesVelocityGpu ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }

        // Ensure Adam moment buffers
        if (optimizerType == GpuOptimizerType.Adam || optimizerType == GpuOptimizerType.AdamW)
        {
            _gpuDepthwiseKernelsM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_depthwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuDepthwiseKernelsV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_depthwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuPointwiseKernelsM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_pointwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuPointwiseKernelsV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_pointwiseKernels.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
            _gpuBiasesV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([_biases.Length], NumOps.Zero), GpuTensorRole.OptimizerState);
        }
    }

    private GpuOptimizerState BuildSeparableConvOptimizerState(string paramName)
    {
        return paramName switch
        {
            "depthwise" => new GpuOptimizerState
            {
                Velocity = _gpuDepthwiseKernelsVelocity?.Buffer,
                M = _gpuDepthwiseKernelsM?.Buffer,
                V = _gpuDepthwiseKernelsV?.Buffer
            },
            "pointwise" => new GpuOptimizerState
            {
                Velocity = _gpuPointwiseKernelsVelocity?.Buffer,
                M = _gpuPointwiseKernelsM?.Buffer,
                V = _gpuPointwiseKernelsV?.Buffer
            },
            "biases" => new GpuOptimizerState
            {
                Velocity = _gpuBiasesVelocityGpu?.Buffer,
                M = _gpuBiasesM?.Buffer,
                V = _gpuBiasesV?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }

    #endregion
}
