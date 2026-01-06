using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a subpixel convolutional layer that performs convolution followed by pixel shuffling for upsampling.
/// </summary>
/// <remarks>
/// <para>
/// A subpixel convolutional layer combines convolution with a pixel shuffling operation to efficiently increase 
/// spatial resolution of feature maps. It first applies convolution to produce an output with more channels, then 
/// rearranges these channels into a higher resolution output with fewer channels. This approach is particularly 
/// useful for super-resolution tasks and generative models where upsampling is required.
/// </para>
/// <para><b>For Beginners:</b> This layer helps make images larger and more detailed in neural networks.
/// 
/// Think of it like rearranging a small mosaic to create a larger picture:
/// - First, the layer creates many detailed patterns from the input (convolution step)
/// - Then, it rearranges these patterns to form a larger, higher-resolution output (pixel shuffling step)
/// 
/// For example, if you're working with a low-resolution image that's 32×32 pixels, this layer can help
/// transform it into a higher-resolution image of 64×64 or 128×128 pixels by intelligently filling in 
/// the details between the original pixels.
/// 
/// This is often used in applications like:
/// - Making blurry images clearer (super-resolution)
/// - Generating detailed images from rough sketches
/// - Converting low-quality videos to higher quality
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SubpixelConvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The number of channels in the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the depth dimension of the input tensor, which represents the number of channels or features
    /// at each spatial location in the input.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many different types of information the input contains.
    /// 
    /// For example:
    /// - A grayscale image would have 1 channel (just brightness)
    /// - A color RGB image would have 3 channels (red, green, and blue)
    /// - A feature map from a previous layer might have many more channels (like 64 or 128)
    ///   representing different patterns detected in the data
    /// </para>
    /// </remarks>
    private readonly int _inputDepth;

    /// <summary>
    /// The number of channels in the output tensor after upscaling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the depth dimension of the output tensor, which represents the number of channels or features
    /// at each spatial location in the output after the upscaling operation.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how many different types of information the output will contain.
    /// 
    /// After upscaling, the layer typically reduces the number of channels because:
    /// - Some of the channel information is used to increase the spatial dimensions
    /// - The final output needs to have a specific number of channels (often 3 for RGB images)
    /// 
    /// For example, if upscaling a feature map to create a color image, the output depth would be 3
    /// (for the red, green, and blue channels).
    /// </para>
    /// </remarks>
    private readonly int _outputDepth;

    /// <summary>
    /// The factor by which to increase spatial dimensions (height and width).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the scaling factor used to increase the spatial dimensions (height and width) of the input.
    /// A value of 2 means the output height and width will be twice the input dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how much larger the output will be compared to the input.
    /// 
    /// For example:
    /// - With an upscale factor of 2: A 32×32 image becomes 64×64
    /// - With an upscale factor of 3: A 32×32 image becomes 96×96
    /// - With an upscale factor of 4: A 32×32 image becomes 128×128
    /// 
    /// Higher upscale factors create larger outputs but require more channels in the intermediate step,
    /// which makes the layer more computationally intensive.
    /// </para>
    /// </remarks>
    private readonly int _upscaleFactor;

    /// <summary>
    /// The size of the convolutional kernel (filter).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the size of the square convolutional kernel used in the layer. A common value is 3,
    /// which creates a 3×3 kernel that examines a 3×3 region of the input for each output value.
    /// </para>
    /// <para><b>For Beginners:</b> This determines the area size that the network looks at when analyzing patterns.
    /// 
    /// The kernel is like a small window that slides over the input:
    /// - A 3×3 kernel looks at each pixel and its 8 neighbors
    /// - A 5×5 kernel examines a wider area around each pixel
    /// - Larger kernels can detect larger patterns but require more computation
    /// 
    /// Most commonly, a 3×3 kernel is used as it provides a good balance between detecting
    /// useful patterns and computational efficiency.
    /// </para>
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// The convolutional kernels (filters) used by the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor holds the learnable weights for the convolutional operation. Its shape is 
    /// [outputDepth * upscaleFactor * upscaleFactor, inputDepth, kernelSize, kernelSize], where the first
    /// dimension accounts for both the output channels and the upscaling factor.
    /// </para>
    /// <para><b>For Beginners:</b> These are the pattern detectors that the network learns during training.
    /// 
    /// Think of each kernel as a pattern template:
    /// - During training, the network learns which patterns are important
    /// - Each kernel specializes in detecting a specific type of pattern
    /// - The layer uses many kernels to capture different aspects of the input
    /// 
    /// The number of kernels is larger than the final output channels because the extra channels
    /// are used to store information that helps create the higher resolution output.
    /// </para>
    /// </remarks>
    private Tensor<T> _kernels;

    /// <summary>
    /// The bias values added after the convolution operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector holds the learnable bias values that are added to the output of the convolutional operation.
    /// There is one bias value for each output channel in the convolution step (before pixel shuffling).
    /// </para>
    /// <para><b>For Beginners:</b> These are adjustable values that help the network make better predictions.
    ///
    /// Biases work like this:
    /// - Each output channel has its own bias value
    /// - The bias is added to every position in that channel
    /// - They help the network adjust the "baseline" of each feature detector
    ///
    /// Think of them as brightness adjustments that can make certain patterns more or less prominent
    /// across the entire feature map.
    /// </para>
    /// </remarks>
    private Tensor<T> _biases;

    /// <summary>
    /// The cached input from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what it last processed.
    /// 
    /// During training, the layer needs to remember:
    /// - What input it received
    /// - How it processed that input
    /// 
    /// This stored input helps the layer figure out how to improve its kernels and biases
    /// during the backward pass (the learning phase).
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The cached output from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients with respect to the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what result it produced last time.
    /// 
    /// Storing the output is necessary because:
    /// - During training, the layer needs to know how its output differed from the expected result
    /// - Some activation functions need their original output to calculate how to improve
    /// - It helps the layer adjust its parameters more efficiently
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the original input shape before any reshaping.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Indicates whether a batch dimension was added during forward pass.
    /// </summary>
    private bool _addedBatchDimension;

    /// <summary>
    /// The gradients of the loss with respect to the kernels, computed during backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor holds the gradients of the loss function with respect to the convolutional kernels, calculated
    /// during the backward pass. These gradients indicate how the kernels should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each kernel should change to improve the network's performance.
    /// 
    /// During training:
    /// - The backward pass calculates how much each kernel contributed to any errors
    /// - This gradient tensor stores that information (how much and in which direction to change each value)
    /// - Larger gradient values indicate parameters that need bigger adjustments
    /// 
    /// Think of these as detailed instructions for how to adjust each pattern detector
    /// to make the network perform better next time.
    /// </para>
    /// </remarks>
    private Tensor<T>? _kernelGradients;

    /// <summary>
    /// The gradients of the loss with respect to the biases, computed during backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector holds the gradients of the loss function with respect to the bias values, calculated during
    /// the backward pass. These gradients indicate how the biases should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how each bias should change to improve the network's performance.
    ///
    /// Similar to kernel gradients:
    /// - These indicate how much and in what direction to adjust each bias value
    /// - They help fine-tune the "baseline" level of each feature detector
    /// - The update step uses these to modify the biases during training
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasGradients;

    /// <summary>
    /// The accumulated momentum for kernel updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the accumulated momentum for updating the kernels, which helps stabilize and accelerate
    /// the optimization process during training.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the network learn more smoothly and efficiently.
    /// 
    /// Momentum works like this:
    /// - It keeps track of the general direction parameters have been changing
    /// - It helps parameters continue moving in a consistent direction
    /// - It can help overcome small bumps and variations in the gradient
    /// 
    /// Think of it like a ball rolling down a hill - momentum keeps it moving even through
    /// small ups and downs, helping it reach the bottom (optimal solution) faster.
    /// </para>
    /// </remarks>
    private Tensor<T>? _kernelMomentum;

    /// <summary>
    /// The accumulated momentum for bias updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the accumulated momentum for updating the biases, which helps stabilize and accelerate
    /// the optimization process during training.
    /// </para>
    /// <para><b>For Beginners:</b> This works the same way as kernel momentum, but for bias values.
    ///
    /// It helps bias parameters:
    /// - Learn more consistently
    /// - Avoid getting stuck in suboptimal values
    /// - Converge faster to good values
    ///
    /// Like kernel momentum, it acts as a memory of previous update directions
    /// to smooth out the learning process.
    /// </para>
    /// </remarks>
    private Tensor<T>? _biasMomentum;

    /// <summary>
    /// The factor controlling how much previous gradients influence current updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the momentum factor used in the parameter update step. A typical value is 0.9, which means
    /// 90% of the update comes from previous accumulated gradients and 10% from the current gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network remembers about previous learning steps.
    /// 
    /// A value closer to 1 means:
    /// - The network strongly considers the direction it was previously moving
    /// - Updates are more stable but might be slower to change direction
    /// - Good for avoiding oscillations in the learning process
    /// 
    /// The default value of 0.9 provides a good balance between stability and adaptability
    /// for most training scenarios.
    /// </para>
    /// </remarks>
    private readonly T _momentumFactor;

    /// <summary>
    /// The coefficient for L2 weight regularization (weight decay).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the weight decay coefficient used to prevent overfitting by penalizing large weight values.
    /// A typical value is a small number like 0.0001.
    /// </para>
    /// <para><b>For Beginners:</b> This helps prevent the network from becoming too specialized to the training data.
    /// 
    /// Weight decay:
    /// - Slightly reduces the size of weights during each update
    /// - Encourages the network to find simpler solutions with smaller weights
    /// - Helps the network generalize better to new data it hasn't seen before
    /// 
    /// Think of it like a tax on complex solutions - it pushes the network to find
    /// the simplest pattern that explains the data rather than memorizing specific examples.
    /// </para>
    /// </remarks>
    private readonly T _weightDecay;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it contains trainable parameters (kernels and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the subpixel convolutional layer can be trained through backpropagation.
    /// Since this layer has trainable parameters (kernels and biases), it supports training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (kernels and biases) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// For this layer, the value is always true because it needs to learn which patterns
    /// are most important for upscaling the input effectively.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SubpixelConvolutionalLayer{T}"/> class with scalar activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input tensor.</param>
    /// <param name="outputDepth">The number of channels in the output tensor after upscaling.</param>
    /// <param name="upscaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="activation">The activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a subpixel convolutional layer with the specified dimensions and parameters.
    /// It initializes the convolutional kernels and biases with appropriate values for training.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new subpixel convolutional layer.
    /// 
    /// The parameters you provide determine:
    /// - inputDepth: How many channels the input has (like RGB for images would be 3)
    /// - outputDepth: How many channels the output will have after upscaling
    /// - upscaleFactor: How much larger the output will be (2 means twice as wide and tall)
    /// - kernelSize: How large an area the layer examines for each calculation (3 is common)
    /// - inputHeight/inputWidth: The dimensions of the input data
    /// - activation: What mathematical function to apply to the results (ReLU is default)
    /// 
    /// These settings help the layer know exactly what kind of data it's working with and
    /// how to transform it into a higher-resolution output.
    /// </para>
    /// </remarks>
    public SubpixelConvolutionalLayer(int inputDepth, int outputDepth, int upscaleFactor, int kernelSize, int inputHeight, int inputWidth,
                                    IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputDepth, inputHeight * upscaleFactor, inputWidth * upscaleFactor),
            activation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _upscaleFactor = upscaleFactor;
        _kernelSize = kernelSize;
        _momentumFactor = NumOps.FromDouble(0.9);
        _weightDecay = NumOps.FromDouble(0.0001);

        _kernels = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor]);

        InitializeWeights();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SubpixelConvolutionalLayer{T}"/> class with vector activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input tensor.</param>
    /// <param name="outputDepth">The number of channels in the output tensor after upscaling.</param>
    /// <param name="upscaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="vectorActivation">The vector activation function to apply after processing. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a subpixel convolutional layer with the specified dimensions and parameters.
    /// It uses a vector activation function, which operates on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the previous one, but uses vector activations.
    /// 
    /// Vector activations:
    /// - Process entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements
    /// - Allow for more complex transformations
    /// 
    /// This version is useful when you need more sophisticated processing that considers
    /// how different features relate to each other, rather than treating each feature independently.
    /// </para>
    /// </remarks>
    public SubpixelConvolutionalLayer(int inputDepth, int outputDepth, int upscaleFactor, int kernelSize, int inputHeight, int inputWidth,
                                    IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
            CalculateOutputShape(outputDepth, inputHeight * upscaleFactor, inputWidth * upscaleFactor),
            vectorActivation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _upscaleFactor = upscaleFactor;
        _kernelSize = kernelSize;
        _momentumFactor = NumOps.FromDouble(0.9);
        _weightDecay = NumOps.FromDouble(0.0001);

        _kernels = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor, _inputDepth, _kernelSize, _kernelSize]);
        _biases = new Tensor<T>([_outputDepth * _upscaleFactor * _upscaleFactor]);

        InitializeWeights();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes the weights and biases of the layer using Xavier initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the convolutional kernels using Xavier initialization, which scales the random values
    /// based on the input and output dimensions to maintain proper variance in the activations throughout the network.
    /// Biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets the starting values for the layer's pattern detectors.
    /// 
    /// Xavier initialization:
    /// - Creates random values that are neither too large nor too small
    /// - Scales these values based on the number of inputs and outputs
    /// - Helps signals flow properly through the network from the beginning
    /// 
    /// Good initialization is important because:
    /// - Starting with all zeros would make learning impossible
    /// - Starting with values that are too large or too small can slow down learning
    /// - Proper scaling helps the network train more efficiently from the start
    /// 
    /// The biases start at zero, letting the network focus on learning the patterns first.
    /// </para>
    /// </remarks>
    private void InitializeWeights()
    {
        // Xavier initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputDepth * _kernelSize * _kernelSize + _outputDepth * _upscaleFactor * _upscaleFactor)));

        // Vectorized random init in [-0.5, 0.5], scaled by Xavier factor
        var randVec = Vector<T>.CreateRandom(_kernels.Length, -0.5, 0.5);
        var randTensor = new Tensor<T>(_kernels.Shape, randVec);
        _kernels = Engine.TensorMultiplyScalar(randTensor, scale);

        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of the subpixel convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after convolution, pixel shuffling, and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the subpixel convolutional layer. It first applies convolution
    /// to produce a tensor with more channels, then performs pixel shuffling to rearrange these channels into
    /// a higher resolution output with fewer channels. Finally, it applies the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through the upscaling steps.
    /// 
    /// The process works in three main steps:
    /// 
    /// 1. Convolution: 
    ///    - The input is processed using the learned pattern detectors (kernels)
    ///    - This creates a version with many more channels than the final output needs
    ///    - These extra channels contain the information needed to create a larger image
    /// 
    /// 2. Pixel Shuffling:
    ///    - The many channels are rearranged into a larger spatial grid
    ///    - This effectively increases the resolution of the image
    ///    - The proper arrangements of pixels creates the higher resolution output
    /// 
    /// 3. Activation:
    ///    - A mathematical function is applied to introduce non-linearity
    ///    - This helps the network learn more complex patterns
    ///    - The final output has higher resolution but fewer channels
    /// 
    /// For example, with upscaleFactor=2:
    /// - A 32×32×64 input might become 32×32×256 after convolution
    /// - Then become 64×64×64 after pixel shuffling (4 times more pixels, 1/4 the channels)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Support any rank >= 3: last 3 dims are interpreted as [C, H, W]
        // Convert to NCHW for Engine.Conv2D
        Tensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            _addedBatchDimension = true;
            input4D = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
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
            input4D = input.Reshape(flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]);
        }

        _lastInput = input4D;

        // Perform fused convolution + bias (without activation since PixelShuffle must come before activation)
        // Note: Can't use FusedConv2D with activation because PixelShuffle must occur between conv and activation
        int padSize = _kernelSize / 2;
        var convOutputNCHW = Engine.FusedConv2D(
            input4D, _kernels, _biases,
            1, 1,           // stride
            padSize, padSize, // padding
            1, 1,           // dilation
            FusedActivationType.None);  // No activation - applied after PixelShuffle

        // Perform pixel shuffle using Engine (must come before activation)
        var shuffledNCHW = Engine.PixelShuffle(convOutputNCHW, _upscaleFactor);

        // Apply activation after pixel shuffle
        var result = ApplyActivation(shuffledNCHW);

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
            outputShape[_originalInputShape.Length - 3] = _outputDepth;
            outputShape[_originalInputShape.Length - 2] = result.Shape[2];
            outputShape[_originalInputShape.Length - 1] = result.Shape[3];
            return result.Reshape(outputShape);
        }

        if (_addedBatchDimension)
        {
            // 3D input [C, H, W] should produce 3D output [OutputDepth, outH, outW]
            return result.Reshape(_outputDepth, result.Shape[2], result.Shape[3]);
        }

        return result;
    }

    /// <summary>
    /// Performs the GPU-resident forward pass of the subpixel convolutional layer.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor after convolution, pixel shuffle, and activation.</returns>
    /// <remarks>
    /// All computations stay on GPU: Conv2D → PixelShuffle (reshape+permute+reshape) → Activation.
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var gpuEngine = Engine as DirectGpuTensorEngine;
        if (gpuEngine == null)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;

        // Ensure 4D [B, C, H, W] format
        IGpuTensor<T> input4D;
        bool addedBatch = false;
        if (shape.Length == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            addedBatch = true;
            input4D = gpuEngine.ReshapeGpu(input, new[] { 1, shape[0], shape[1], shape[2] });
        }
        else
        {
            input4D = input;
        }

        int batch = input4D.Shape[0];
        int height = input4D.Shape[2];
        int width = input4D.Shape[3];

        // Step 1: Convolution without activation (pixel shuffle must come before activation)
        int padSize = _kernelSize / 2;
        var convOutput = gpuEngine.FusedConv2DGpu(
            input4D,
            _kernels,
            _biases,
            1, 1,           // stride
            padSize, padSize, // padding
            1, 1,           // dilation
            FusedActivationType.None);  // No activation yet

        // Step 2: PixelShuffle [B, C*r², H, W] → [B, C, H*r, W*r]
        int r = _upscaleFactor;
        int outChannels = _outputDepth;
        int outHeight = height * r;
        int outWidth = width * r;

        // PixelShuffle: [B, C*r², H, W] → [B, C, r, r, H, W] → [B, C, H, r, W, r] → [B, C, H*r, W*r]
        var reshaped1 = gpuEngine.ReshapeGpu(convOutput, new[] { batch, outChannels, r, r, height, width });
        var permuted = gpuEngine.PermuteGpu(reshaped1, new[] { 0, 1, 4, 2, 5, 3 }); // [B, C, H, r, W, r]
        var shuffled = gpuEngine.ReshapeGpu(permuted, new[] { batch, outChannels, outHeight, outWidth });

        // Step 3: Apply activation
        var fusedActivation = GetFusedActivationType();
        IGpuTensor<T> result;
        if (fusedActivation != FusedActivationType.None)
        {
            result = gpuEngine.ActivationGpu(shuffled, fusedActivation);
        }
        else
        {
            result = shuffled;
        }

        // Remove batch dimension if it was added
        if (addedBatch)
        {
            result = gpuEngine.ReshapeGpu(result, new[] { outChannels, outHeight, outWidth });
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass of the subpixel convolutional layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the subpixel convolutional layer, which is used during training to
    /// propagate error gradients back through the network. It calculates gradients for the input and for all trainable
    /// parameters (kernels and biases).
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// and parameters should change to reduce errors.
    ///
    /// During the backward pass, we reverse the steps from the forward pass:
    ///
    /// 1. First, calculate how the activation function affects the gradient
    ///
    /// 2. Reverse the pixel shuffling:
    ///    - Convert the gradient from high resolution back to the lower resolution with more channels
    ///    - This helps determine how each output channel contributed to the errors
    ///
    /// 3. Calculate three types of gradients:
    ///    - How the input should change (inputGradient)
    ///    - How the kernels should change (kernelGradients)
    ///    - How the biases should change (biasGradients)
    ///
    /// These gradients tell the network how to adjust its parameters during the update step
    /// to improve its performance on the next forward pass.
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
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle any-rank gradient input by reshaping to 4D to match _lastOutput
        Tensor<T> gradient4D;
        int gradRank = outputGradient.Shape.Length;
        if (gradRank == 3)
        {
            // 3D gradient [C, H, W] -> 4D [1, C, H, W]
            gradient4D = outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2]);
        }
        else if (gradRank == 4)
        {
            gradient4D = outputGradient;
        }
        else
        {
            // Higher-rank: flatten leading dimensions
            int flatBatch = 1;
            for (int d = 0; d < gradRank - 3; d++)
                flatBatch *= outputGradient.Shape[d];
            gradient4D = outputGradient.Reshape(flatBatch, outputGradient.Shape[gradRank - 3],
                outputGradient.Shape[gradRank - 2], outputGradient.Shape[gradRank - 1]);
        }

        // Apply activation derivative (both _lastOutput and gradient4D are in NCHW format)
        var delta = ApplyActivationDerivative(_lastOutput, gradient4D);

        int padSize = _kernelSize / 2;
        var strideArr = new int[] { 1, 1 };
        var paddingArr = new int[] { padSize, padSize };
        var dilationArr = new int[] { 1, 1 };

        // Reverse pixel shuffle using Engine
        int numOutChannels = _outputDepth * _upscaleFactor * _upscaleFactor;
        var convGradShape = new int[] { _lastInput.Shape[0], numOutChannels, _lastInput.Shape[2], _lastInput.Shape[3] };
        var convOutputGradientNCHW = Engine.PixelShuffleBackward(delta, convGradShape, _upscaleFactor);

        // Calculate bias gradient: sum over batch, height, width (axes 0, 2, 3 in NCHW)
        _biasGradients = Engine.ReduceSum(convOutputGradientNCHW, new[] { 0, 2, 3 }, keepDims: false);

        // Calculate kernel gradient using Engine
        _kernelGradients = Engine.Conv2DBackwardKernel(convOutputGradientNCHW, _lastInput, _kernels.Shape, strideArr, paddingArr, dilationArr);

        // Calculate input gradient using Engine (NCHW format)
        var inputGradient = Engine.Conv2DBackwardInput(convOutputGradientNCHW, _kernels, _lastInput.Shape, strideArr, paddingArr, dilationArr);

        // Restore original input shape for higher-rank tensors
        if (_originalInputShape != null && _originalInputShape.Length > 4)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        if (_addedBatchDimension && _originalInputShape != null)
        {
            // 3D input [C, H, W] should produce 3D gradient [C, H, W]
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Currently, subpixel convolution operations
    /// are not yet available in TensorOperations, so this method falls back to the manual implementation.
    /// </para>
    /// <para>
    /// Once subpixel convolution operations are added to TensorOperations, this method will provide:
    /// - Automatic gradient computation through the computation graph
    /// - Verification of manual gradient implementations
    /// - Support for rapid prototyping with custom modifications
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert input to computation node
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Apply pixel shuffle operation
        var outputNode = Autodiff.TensorOperations<T>.PixelShuffle(inputNode, _upscaleFactor);

        // Set gradient on output node
        outputNode.Gradient = outputGradient;

        // Inline topological sort and backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

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

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }

    /// <summary>
    /// Computes the gradient with respect to the activation function.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <param name="lastOutput">The output from the last forward pass.</param>
    /// <returns>The gradient after accounting for the activation function.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradient with respect to the activation function by multiplying the output gradient
    /// with the derivative of the activation function. It handles both scalar and vector activation functions.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the error gradient based on the activation function used.
    /// 
    /// Different activation functions affect how errors flow backward:
    /// - For example, ReLU only allows gradients to flow through positive values
    /// - Sigmoid squeezes gradients when inputs are very positive or very negative
    /// 
    /// This step is necessary because:
    /// - During the forward pass, the activation function changed the values
    /// - During the backward pass, we need to account for those changes
    /// - The activation's derivative tells us how much the function amplified or reduced signals
    /// 
    /// This method handles both standard activations (applied to each value separately)
    /// and vector activations (applied to groups of values together).
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeActivationGradient(Tensor<T> outputGradient, Tensor<T> lastOutput)
    {
        if (UsingVectorActivation)
        {
            return VectorActivation!.Derivative(lastOutput).PointwiseMultiply(outputGradient);
        }
        else
        {
            var result = new Tensor<T>(outputGradient.Shape);
            for (int i = 0; i < outputGradient.Length; i++)
            {
                result[i] = NumOps.Multiply(ScalarActivation!.Derivative(lastOutput[i]), outputGradient[i]);
            }

            return result;
        }
    }

    /// <summary>
    /// Calculates the input shape for the layer based on specified dimensions.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input.</param>
    /// <param name="inputHeight">The height of the input.</param>
    /// <param name="inputWidth">The width of the input.</param>
    /// <returns>The calculated input shape as a jagged array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the expected shape of the input tensor for the layer. The shape is specified as a jagged
    /// array where the innermost array represents the dimensions of each input sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that should go into this layer.
    /// 
    /// The input shape is organized as:
    /// - Height: How tall the input feature map is (in pixels or units)
    /// - Width: How wide the input feature map is
    /// - Depth: How many channels or features each position has
    /// 
    /// For example, a 32×32 RGB image would have a shape of [32, 32, 3].
    /// This helps the layer know exactly what kind of data it should expect.
    /// </para>
    /// </remarks>
    private new static int[][] CalculateInputShape(int inputDepth, int inputHeight, int inputWidth)
    {
        return [[inputHeight, inputWidth, inputDepth]];
    }

    /// <summary>
    /// Calculates the output shape for the layer based on specified dimensions.
    /// </summary>
    /// <param name="outputDepth">The number of channels in the output.</param>
    /// <param name="outputHeight">The height of the output after upscaling.</param>
    /// <param name="outputWidth">The width of the output after upscaling.</param>
    /// <returns>The calculated output shape as an array of integers.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the expected shape of the output tensor for the layer. The shape is specified as an
    /// array representing the dimensions of each output sample.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that will come out of this layer.
    /// 
    /// The output shape is organized as:
    /// - Height: How tall the output will be (upscaled from input)
    /// - Width: How wide the output will be (upscaled from input)
    /// - Depth: How many channels the output will have
    /// 
    /// For example, if upscaleFactor=2, a 32×32×64 input might produce a 64×64×32 output.
    /// The spatial dimensions increase by the upscale factor, while the number of channels
    /// is determined by the outputDepth parameter.
    /// </para>
    /// </remarks>
    private new static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [outputHeight, outputWidth, outputDepth];
    }

    /// <summary>
    /// Updates the parameters of the layer using calculated gradients and momentum.
    /// </summary>
    /// <param name="learningRate">The learning rate to control the size of parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when trying to update parameters before calculating gradients.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the convolutional kernels and biases based on the gradients calculated during the backward pass.
    /// It uses momentum to stabilize the updates and applies weight decay to the kernels to prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the layer's pattern detectors to improve performance.
    /// 
    /// During parameter updates:
    /// 1. Momentum is calculated:
    ///    - 90% of the previous update direction (momentum)
    ///    - 10% of the current gradient direction
    ///    - This helps maintain a steady learning direction
    /// 
    /// 2. Weights are updated using:
    ///    - The momentum-adjusted gradient (for direction)
    ///    - The learning rate (for step size)
    ///    - Weight decay (to prevent overfitting by keeping weights small)
    /// 
    /// 3. Biases are updated using:
    ///    - The momentum-adjusted gradient
    ///    - The learning rate
    ///    - No weight decay (biases typically don't cause overfitting)
    /// 
    /// Think of it like navigating a mountain: momentum helps you keep moving in a consistent
    /// direction despite small bumps, while weight decay prevents you from taking extreme paths.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_kernelGradients == null || _biasGradients == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Initialize momentum if not already done
        int numOutChannels = _outputDepth * _upscaleFactor * _upscaleFactor;
        _kernelMomentum ??= new Tensor<T>(_kernels.Shape);
        _biasMomentum ??= new Tensor<T>([numOutChannels]);

        T oneMinusMomentum = NumOps.Subtract(NumOps.One, _momentumFactor);

        // Update kernel momentum: momentum = momentum_factor * momentum + (1 - momentum_factor) * gradient
        _kernelMomentum = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_kernelMomentum, _momentumFactor),
            Engine.TensorMultiplyScalar(_kernelGradients, oneMinusMomentum));

        // Update kernels with momentum and weight decay
        T lrTimesWd = NumOps.Multiply(learningRate, _weightDecay);
        var kernelUpdate = Engine.TensorMultiplyScalar(_kernelMomentum, learningRate);
        var weightDecayTerm = Engine.TensorMultiplyScalar(_kernels, lrTimesWd);
        _kernels = Engine.TensorSubtract(Engine.TensorSubtract(_kernels, kernelUpdate), weightDecayTerm);

        // Update bias momentum
        _biasMomentum = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(_biasMomentum, _momentumFactor),
            Engine.TensorMultiplyScalar(_biasGradients, oneMinusMomentum));

        // Update biases with momentum (no weight decay for biases)
        _biases = Engine.TensorSubtract(_biases, Engine.TensorMultiplyScalar(_biasMomentum, learningRate));

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);

        // Clear gradients after update
        _kernelGradients = null;
        _biasGradients = null;
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (kernels and biases) of the layer and combines them into a
    /// single vector. This is useful for optimization algorithms that operate on all parameters at once, or for
    /// saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all kernels and biases from the layer
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
        return Vector<T>.Concatenate(new Vector<T>(_kernels.ToArray()), new Vector<T>(_biases.ToArray()));
    }

    /// <summary>
    /// Exports this layer's computation as a differentiable computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which input variable nodes should be added.</param>
    /// <returns>The output computation node representing this layer's operation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when weights/biases are not initialized or activation is not supported.</exception>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representation of the subpixel convolution operation.
    /// Subpixel convolution combines convolution with pixel shuffling (depth-to-space rearrangement).
    /// </para>
    /// <para><b>For Beginners:</b> This creates an optimized version for faster inference.
    ///
    /// For subpixel convolutional layers:
    /// - Creates placeholders for input, convolution kernels, and biases
    /// - Applies convolution operation
    /// - Applies pixel shuffle (depth-to-space) rearrangement
    /// - Applies activation function
    /// - Returns a computation graph for efficient execution
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_kernels == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized. Call Initialize() or train the layer first.");

        if (!CanActivationBeJitted())
        {
            var activationType = ScalarActivation?.GetType().Name ?? VectorActivation?.GetType().Name ?? "unknown";
            throw new NotSupportedException(
                $"Activation function '{activationType}' is not supported for JIT compilation yet. " +
                "Supported activations: ReLU, Sigmoid, Tanh, Softmax");
        }

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        // Input shape: [batch, height, width, channels] (NHWC format)
        var symbolicInput = new Tensor<T>(new int[] { 1, InputShape[0], InputShape[1], InputShape[2] });
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "subpixel_input");
        inputNodes.Add(inputNode);

        // Create constant nodes for kernels and biases (biases is already a Tensor<T>)
        var kernelNode = TensorOperations<T>.Constant(_kernels, "subpixel_kernels");
        var biasNode = TensorOperations<T>.Constant(_biases, "subpixel_biases");

        // Step 1: Apply 2D convolution
        // Conv2D expects NCHW format, so we may need to transpose if our layer uses NHWC
        // For simplicity, we assume the input is compatible with Conv2D operation
        var convOutput = TensorOperations<T>.Conv2D(inputNode, kernelNode, stride: new[] { 1, 1 }, padding: new[] { _kernelSize / 2, _kernelSize / 2 });

        // Step 2: Add bias (broadcast across spatial dimensions)
        var withBias = TensorOperations<T>.Add(convOutput, biasNode);

        // Step 3: Apply PixelShuffle (depth-to-space) for upscaling
        var shuffled = TensorOperations<T>.PixelShuffle(withBias, _upscaleFactor);

        // Step 4: Apply activation function using base class helper
        var output = ApplyActivationToGraph(shuffled);

        return output;
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <value>True, as all required operations (Conv2D, PixelShuffle) are available.</value>
    /// <remarks>
    /// <para>
    /// Subpixel convolutional layers support JIT compilation using Conv2D and PixelShuffle
    /// operations from TensorOperations. The layer requires both convolution and pixel shuffling
    /// operations which are available in the computation graph.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Resets the internal state of the layer and reinitializes weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes,
    /// resetting momentum, and reinitializing the weights and biases. This is useful when starting new training
    /// or when implementing networks that need to reset their state between sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory and starts fresh.
    ///
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Calculated gradients are cleared
    /// - Momentum is reset to zero
    /// - Weights and biases are reinitialized to new random values
    ///
    /// This is useful for:
    /// - Starting a new training session
    /// - Getting out of a "stuck" state where learning has plateaued
    /// - Testing how the layer performs with different initializations
    ///
    /// Think of it like wiping a whiteboard clean and starting over with a fresh approach.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
        _kernelGradients = null;
        _biasGradients = null;

        // Reset momentum if using momentum
        _kernelMomentum = null;
        _biasMomentum = null;

        // Reinitialize weights
        InitializeWeights();
    }
}
