namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a depthwise separable convolutional layer that performs convolution as two separate operations.
/// </summary>
/// <remarks>
/// <para>
/// A depthwise separable convolutional layer splits the standard convolution operation into two parts:
/// a depthwise convolution, which applies a single filter per input channel, and a pointwise convolution,
/// which uses 1×1 convolutions to combine the outputs. This approach dramatically reduces the number of
/// parameters and computational cost compared to standard convolution.
/// </para>
/// <para><b>For Beginners:</b> A depthwise separable convolution is like a more efficient way to filter an image.
/// 
/// Think of it as a two-step process:
/// - First step (depthwise): Apply separate filters to each input channel (like filtering red, green, and blue separately)
/// - Second step (pointwise): Mix these filtered channels together (like combining the filtered colors)
/// 
/// For example, in image processing:
/// - Standard convolution might use 100,000 calculations for a single operation
/// - Depthwise separable convolution might do the same job with only 10,000 calculations
/// 
/// This makes your neural network faster and smaller while still capturing important patterns.
/// It's commonly used in mobile and edge devices where efficiency is critical.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DepthwiseSeparableConvolutionalLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The filter kernels used for the depthwise convolution step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the weights for the depthwise convolution. It has dimensions
    /// [InputDepth, 1, KernelSize, KernelSize], where each input channel has its own
    /// spatial filter but no channel mixing occurs.
    /// </para>
    /// <para><b>For Beginners:</b> These are the filters that process each input channel separately.
    /// 
    /// For example, if your input has 3 channels (like RGB):
    /// - The red channel gets filtered by its own dedicated filter
    /// - The green channel gets filtered by a different dedicated filter
    /// - The blue channel gets filtered by yet another dedicated filter
    /// 
    /// Each filter looks for specific patterns within its own channel, like edges or textures.
    /// </para>
    /// </remarks>
    private Tensor<T> _depthwiseKernels = default!;

    /// <summary>
    /// The filter kernels used for the pointwise convolution step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the weights for the pointwise convolution. It has dimensions
    /// [OutputDepth, InputDepth, 1, 1], which applies 1×1 convolutions to mix channels
    /// without spatial filtering.
    /// </para>
    /// <para><b>For Beginners:</b> These are the filters that combine channels after they've been processed.
    /// 
    /// After each channel has been filtered separately:
    /// - These 1×1 filters mix the channels together
    /// - They learn how to combine the filtered channels to create useful output features
    /// - They don't look at spatial patterns (no width or height, just combining channels)
    /// 
    /// Think of it like a recipe that determines how much of each filtered ingredient to use.
    /// </para>
    /// </remarks>
    private Tensor<T> _pointwiseKernels = default!;

    /// <summary>
    /// The bias values added to each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains a bias value for each output channel. Biases are added after
    /// both convolution steps are completed and before the activation function is applied.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like base values added to each output channel.
    /// 
    /// Think of biases as:
    /// - A starting point or offset value
    /// - Added after all the filtering and mixing is done
    /// - Helping the network be more flexible in what patterns it detects
    /// 
    /// For example, a positive bias might make a feature detector more sensitive,
    /// triggering more easily even with weaker input signals.
    /// </para>
    /// </remarks>
    private Vector<T> _biases = default!;

    /// <summary>
    /// Stored input data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// During the backward pass (training), the layer needs access to the input data from the forward
    /// pass to calculate the gradients for the kernels. This tensor stores that input data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the network's memory of what input it just processed.
    /// 
    /// The layer remembers:
    /// - The exact input it received
    /// - So it can calculate exactly how to improve its filters
    /// 
    /// Just like a student remembering the exact problems they got wrong on a test,
    /// this helps the layer learn precisely from its mistakes.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored output from the depthwise convolution step, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the intermediate output after applying the depthwise convolution
    /// but before the pointwise convolution. It's needed during backpropagation to calculate
    /// the gradients for the pointwise kernels.
    /// </para>
    /// <para><b>For Beginners:</b> This is the network's memory of the intermediate results.
    /// 
    /// The layer remembers:
    /// - The output after the first filtering step (depthwise)
    /// - Before the second mixing step (pointwise)
    /// 
    /// This helps the layer understand how each step contributed to any errors,
    /// so it can improve both steps independently.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastDepthwiseOutput;

    /// <summary>
    /// Stored final output data from the most recent forward pass, used for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the final output after both convolution steps, bias addition, and
    /// activation. It's used during backpropagation to calculate the derivatives through
    /// the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This is the network's memory of the final result it produced.
    /// 
    /// The layer remembers:
    /// - The final output after all processing steps
    /// - So it can calculate how to improve based on any errors
    /// 
    /// Like keeping track of your final answer to see exactly where you went wrong.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Calculated gradients for the depthwise kernels during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the calculated gradients for the depthwise kernels during the backward pass.
    /// These gradients indicate how the kernels should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> These are the suggested improvements for the first-step filters.
    /// 
    /// During training:
    /// - The network calculates how each depthwise filter should change
    /// - These suggestions are stored here temporarily
    /// - They're used to update the actual filters during the update step
    /// 
    /// Think of it like a to-do list of adjustments for each filter.
    /// </para>
    /// </remarks>
    private Tensor<T>? _depthwiseKernelsGradient;

    /// <summary>
    /// Calculated gradients for the pointwise kernels during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the calculated gradients for the pointwise kernels during the backward pass.
    /// These gradients indicate how the kernels should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> These are the suggested improvements for the second-step filters.
    /// 
    /// During training:
    /// - The network calculates how each channel-mixing filter should change
    /// - These suggestions are stored here temporarily
    /// - They're used to update the actual filters during the update step
    /// 
    /// Similar to the depthwise gradients, but for the mixing step rather than the filtering step.
    /// </para>
    /// </remarks>
    private Tensor<T>? _pointwiseKernelsGradient;

    /// <summary>
    /// Calculated gradients for the biases during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the calculated gradients for the biases during the backward pass.
    /// These gradients indicate how the biases should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> These are the suggested improvements for the base values.
    /// 
    /// During training:
    /// - The network calculates how each bias should change
    /// - These suggestions are stored here temporarily
    /// - They're used to update the actual biases during the update step
    /// 
    /// Adjusting biases can help fine-tune the sensitivity of feature detectors.
    /// </para>
    /// </remarks>
    private Vector<T>? _biasesGradient;

    /// <summary>
    /// The number of channels in the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of channels in the input data, which determines the number
    /// of depthwise filters and the input dimension for the pointwise convolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different types of information the input has.
    /// 
    /// For example:
    /// - An RGB image has 3 channels (red, green, blue)
    /// - A feature map might have many channels (16, 32, 64, etc.)
    /// - Each channel represents a different aspect or feature of the data
    /// 
    /// The layer needs to know this to create the right number of filters.
    /// </para>
    /// </remarks>
    private readonly int _inputDepth;

    /// <summary>
    /// The number of channels in the output data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of channels in the output data, which determines the number
    /// of pointwise filters and output features produced by the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different types of features the layer will detect.
    /// 
    /// For example:
    /// - If outputDepth is 16, the layer creates 16 different feature detectors
    /// - Each output channel looks for a different pattern in the input
    /// - More output channels mean more patterns can be detected
    /// 
    /// This is typically larger than the input depth as networks learn more complex features.
    /// </para>
    /// </remarks>
    private readonly int _outputDepth;

    /// <summary>
    /// The size of each depthwise filter kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the spatial size of the depthwise filter kernels. For example, a kernel size
    /// of 3 means 3×3 kernels are used for the depthwise convolution.
    /// </para>
    /// <para><b>For Beginners:</b> This is how big each filter is when looking for patterns.
    /// 
    /// For example:
    /// - A kernelSize of 3 means each filter is a 3×3 grid (9 points)
    /// - A kernelSize of 5 means each filter is a 5×5 grid (25 points)
    /// 
    /// Smaller kernels (3×3) look for simple patterns like edges.
    /// Larger kernels (5×5 or 7×7) can detect more complex patterns but use more memory.
    /// </para>
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// The step size for moving the kernel across the input data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the stride parameter, which determines how many positions to move the kernel
    /// for each step during the convolution operation. A stride of 2 means the kernel moves 2 positions
    /// at a time, reducing the output dimensions by roughly half.
    /// </para>
    /// <para><b>For Beginners:</b> This is how far the filter moves each step when scanning the input.
    /// 
    /// Think of it like:
    /// - Stride of 1: Move one pixel at a time (examine every position)
    /// - Stride of 2: Move two pixels each time (skip positions, make output smaller)
    /// 
    /// A larger stride:
    /// - Makes the output smaller
    /// - Speeds up processing
    /// - But might miss some details
    /// </para>
    /// </remarks>
    private readonly int _stride;

    /// <summary>
    /// The amount of zero-padding added to the input data before convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the padding parameter, which determines how much zero-padding is added around the
    /// input data before performing the convolution. Padding helps preserve the spatial dimensions and
    /// ensures border information isn't lost.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much extra space is added around the input data's edges.
    /// 
    /// Imagine adding a frame of zeros around your data:
    /// - Padding of 0: No extra border (output will be smaller than input)
    /// - Padding of 1: One row/column of zeros on each side
    /// - Padding of 2: Two rows/columns of zeros on each side
    /// 
    /// Benefits of padding:
    /// - Helps preserve the size of your data
    /// - Ensures the edges of your data receive proper attention
    /// - Prevents the output from shrinking too much
    /// </para>
    /// </remarks>
    private readonly int _padding;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> for depthwise separable convolutional layers, as they contain trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation. Depthwise separable
    /// convolutional layers have trainable parameters (kernel weights and biases), so they support training.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// For depthwise separable convolutional layers:
    /// - The value is always true
    /// - This means the layer can adjust its filters and biases during training
    /// - It will improve its pattern recognition as it processes more data
    /// 
    /// Some other layer types might not have trainable parameters and would return false here.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DepthwiseSeparableConvolutionalLayer{T}"/> class with the specified 
    /// parameters and a scalar activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input data.</param>
    /// <param name="outputDepth">The number of output channels to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="inputHeight">The height of the input data.</param>
    /// <param name="inputWidth">The width of the input data.</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="activation">The activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a depthwise separable convolutional layer with the specified configuration.
    /// It initializes both depthwise and pointwise kernels with appropriate scaling factors to help with
    /// training convergence. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This setup method creates a new depthwise separable convolutional layer with specific settings.
    /// 
    /// When creating the layer, you specify:
    /// - Input details: How many channels and the dimensions of your data
    /// - How many patterns to look for (outputDepth)
    /// - How big each filter is (kernelSize)
    /// - How to move the filter across the data (stride)
    /// - Whether to add an extra border (padding)
    /// - What mathematical function to apply to the results (activation)
    /// 
    /// The layer then creates all the necessary filters with random starting values
    /// that will be improved during training. This more efficient approach requires
    /// fewer parameters than a standard convolutional layer.
    /// </para>
    /// </remarks>
    public DepthwiseSeparableConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth,
                                                int stride = 1, int padding = 0, IActivationFunction<T>? activation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding),
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)),
               activation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([inputDepth, 1, kernelSize, kernelSize]);
        _pointwiseKernels = new Tensor<T>([outputDepth, inputDepth, 1, 1]);
        _biases = new Vector<T>(outputDepth);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DepthwiseSeparableConvolutionalLayer{T}"/> class with the specified 
    /// parameters and a vector activation function.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input data.</param>
    /// <param name="outputDepth">The number of output channels to create.</param>
    /// <param name="kernelSize">The size of each filter kernel (width and height).</param>
    /// <param name="inputHeight">The height of the input data.</param>
    /// <param name="inputWidth">The width of the input data.</param>
    /// <param name="stride">The step size for moving the kernel. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add around the input. Defaults to 0.</param>
    /// <param name="vectorActivation">The vector activation function to apply. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a depthwise separable convolutional layer with the specified configuration
    /// and a vector activation function. Vector<double> activation functions operate on entire vectors at once,
    /// which can be more efficient for certain operations.
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
    public DepthwiseSeparableConvolutionalLayer(int inputDepth, int outputDepth, int kernelSize, int inputHeight, int inputWidth,
                                                int stride = 1, int padding = 0, IVectorActivationFunction<T>? vectorActivation = null)
        : base(CalculateInputShape(inputDepth, inputHeight, inputWidth),
               CalculateOutputShape(outputDepth, CalculateOutputDimension(inputHeight, kernelSize, stride, padding),
                   CalculateOutputDimension(inputWidth, kernelSize, stride, padding)),
               vectorActivation ?? new ReLUActivation<T>())
    {
        _inputDepth = inputDepth;
        _outputDepth = outputDepth;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;

        _depthwiseKernels = new Tensor<T>([inputDepth, 1, kernelSize, kernelSize]);
        _pointwiseKernels = new Tensor<T>([outputDepth, inputDepth, 1, 1]);
        _biases = new Vector<T>(outputDepth);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the kernel weights and biases with appropriate random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the depthwise and pointwise kernel weights using different scaling factors.
    /// The depthwise kernels are scaled based on the kernel size, while the pointwise kernels are scaled
    /// based on the input depth. This helps improve training convergence. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for all the filters.
    /// 
    /// When initializing parameters:
    /// - Random values are created for both types of filters
    /// - Different scaling is used for each type of filter
    /// - Biases start at zero
    /// 
    /// Good initialization is important because:
    /// - It helps the network learn faster
    /// - It prevents certain mathematical problems during training
    /// - It gives each filter a different starting point
    /// 
    /// This implementation uses carefully chosen scaling factors for the
    /// random values that work well for depthwise separable convolutions.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T depthwiseScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize)));
        T pointwiseScale = NumOps.Sqrt(NumOps.FromDouble(2.0 / _inputDepth));

        InitializeTensor(_depthwiseKernels, depthwiseScale);
        InitializeTensor(_pointwiseKernels, pointwiseScale);

        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a tensor with random values scaled by the specified factor.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scaling factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This helper method initializes a tensor with random values between -0.5 and 0.5, scaled by
    /// the specified factor. This ensures that the initial values are appropriately sized to avoid
    /// issues during training.
    /// </para>
    /// <para><b>For Beginners:</b> This helper method fills a tensor with carefully scaled random values.
    /// 
    /// For each value in the tensor:
    /// - Generate a random number between -0.5 and 0.5
    /// - Multiply it by the scaling factor
    /// - This keeps the initial values in a good range for learning
    /// 
    /// The scaling helps prevent the network from starting with values that are
    /// too large or too small, which can cause problems during training.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Shape[0]; i++)
        {
            for (int j = 0; j < tensor.Shape[1]; j++)
            {
                for (int k = 0; k < tensor.Shape[2]; k++)
                {
                    for (int l = 0; l < tensor.Shape[3]; l++)
                    {
                        tensor[i, j, k, l] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Calculates the input shape for the layer based on input dimensions.
    /// </summary>
    /// <param name="inputDepth">The number of channels in the input data.</param>
    /// <param name="inputHeight">The height of the input data.</param>
    /// <param name="inputWidth">The width of the input data.</param>
    /// <returns>An array representing the input shape as [inputHeight, inputWidth, inputDepth].</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the input shape for the layer based on the provided dimensions.
    /// The input shape is represented as [height, width, channels], which is the standard format
    /// for input to convolutional layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a description of how big the input data is.
    /// 
    /// The input shape:
    /// - Describes the exact dimensions of the data the layer will process
    /// - Includes height, width, and the number of channels
    /// - Is used to properly set up the layer
    /// 
    /// For example, a color image might have a shape of [224, 224, 3], meaning:
    /// - 224 pixels tall
    /// - 224 pixels wide
    /// - 3 channels (red, green, blue)
    /// </para>
    /// </remarks>
    private new static int[] CalculateInputShape(int inputDepth, int inputHeight, int inputWidth)
    {
        return [inputHeight, inputWidth, inputDepth];
    }

    /// <summary>
    /// Calculates the output shape for the layer based on output dimensions.
    /// </summary>
    /// <param name="outputDepth">The number of channels in the output data.</param>
    /// <param name="outputHeight">The height of the output data after convolution.</param>
    /// <param name="outputWidth">The width of the output data after convolution.</param>
    /// <returns>An array representing the output shape as [outputHeight, outputWidth, outputDepth].</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape for the layer based on the provided dimensions.
    /// The output shape is represented as [height, width, channels], matching the format of the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a description of how big the output data will be.
    /// 
    /// The output shape:
    /// - Describes the dimensions of the data after processing
    /// - Includes height, width, and the number of output channels
    /// - Is used by the next layer in the network
    /// 
    /// For example, if your layer creates 64 feature maps that are 112×112, 
    /// the output shape would be [112, 112, 64], meaning:
    /// - 112 pixels tall
    /// - 112 pixels wide
    /// - 64 channels (one for each feature detector)
    /// </para>
    /// </remarks>
    private new static int[] CalculateOutputShape(int outputDepth, int outputHeight, int outputWidth)
    {
        return [outputHeight, outputWidth, outputDepth];
    }

    /// <summary>
    /// Calculates the output dimension after applying a convolution operation.
    /// </summary>
    /// <param name="inputDimension">The input dimension (height or width).</param>
    /// <param name="kernelSize">The size of the kernel (filter).</param>
    /// <param name="stride">The stride (step size) of the convolution.</param>
    /// <param name="padding">The amount of padding added to the input.</param>
    /// <returns>The calculated output dimension.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output dimension (height or width) after applying a convolution operation
    /// with the specified parameters. The formula used is (inputDimension - kernelSize + 2 * padding) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how big the output will be after applying the layer.
    /// 
    /// The output size depends on:
    /// - How big your input is
    /// - How big your filter is
    /// - How much you move the filter each step (stride)
    /// - How much extra border you add (padding)
    /// 
    /// For example, if you have:
    /// - Input size = 28
    /// - Kernel size = 3
    /// - Stride = 1
    /// - Padding = 1
    /// 
    /// The output size will be (28 - 3 + 2*1)/1 + 1 = 28
    /// (In this case, the padding makes the output the same size as the input)
    /// </para>
    /// </remarks>
    private static int CalculateOutputDimension(int inputDimension, int kernelSize, int stride, int padding)
    {
        return (inputDimension - kernelSize + 2 * padding) / stride + 1;
    }

    /// <summary>
    /// Applies the activation function to a single value.
    /// </summary>
    /// <param name="value">The value to apply the activation function to.</param>
    /// <returns>The result after applying the activation function.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the layer's activation function to a single value. It handles both scalar
    /// and vector activation functions by appropriately wrapping the input as needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies a mathematical function to transform a value.
    /// 
    /// The activation function:
    /// - Adds non-linearity to the network
    /// - Typically keeps useful signals and reduces unhelpful ones
    /// - Helps the network learn complex patterns
    /// 
    /// For example, the popular ReLU activation:
    /// - Keeps positive values unchanged
    /// - Changes all negative values to zero
    /// 
    /// This allows the network to learn more complex relationships than just straight lines.
    /// </para>
    /// </remarks>
    private T ApplyActivation(T value)
    {
        if (UsingVectorActivation)
        {
            return VectorActivation!.Activate(new Vector<T>([value]))[0];
        }
        else
        {
            return ScalarActivation!.Activate(value);
        }
    }

    /// <summary>
    /// Processes the input data through the depthwise separable convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after depthwise separable convolution and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the forward pass of the depthwise separable convolutional layer. It first applies
    /// the depthwise convolution, then the pointwise convolution, adds biases, and finally applies the
    /// activation function. The result is a tensor where each channel represents different features detected
    /// by the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the two-step filtering process to your input data.
    /// 
    /// During the forward pass:
    /// - First, apply depthwise convolution (filter each channel separately)
    /// - Next, apply pointwise convolution (mix filtered channels together)
    /// - Add biases to each output channel
    /// - Apply the activation function to make results non-linear
    /// 
    /// Think of it like a cooking process where you:
    /// 1. Process each ingredient separately (depthwise)
    /// 2. Mix the processed ingredients together (pointwise)
    /// 3. Add seasoning (biases)
    /// 4. Cook everything (activation function)
    /// 
    /// The result shows which patterns were detected in the input data.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int outputHeight = CalculateOutputDimension(input.Shape[1], _kernelSize, _stride, _padding);
        int outputWidth = CalculateOutputDimension(input.Shape[2], _kernelSize, _stride, _padding);

        // Depthwise convolution
        var depthwiseOutput = DepthwiseConvolution(input);
        _lastDepthwiseOutput = depthwiseOutput;

        // Pointwise convolution
        var pointwiseOutput = PointwiseConvolution(depthwiseOutput);

        // Add biases and apply activation
        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _outputDepth]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    for (int c = 0; c < _outputDepth; c++)
                    {
                        T value = NumOps.Add(pointwiseOutput[b, h, w, c], _biases[c]);
                        output[b, h, w, c] = ApplyActivation(value);
                    }
                }
            }
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Applies the depthwise convolution step to the input data.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after depthwise convolution.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the depthwise convolution step, where each input channel is convolved with
    /// its own filter. This differs from standard convolution because it doesn't mix channels - each
    /// output channel depends only on the corresponding input channel.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the first step of the filtering process.
    /// 
    /// Depthwise convolution:
    /// - Works on each input channel independently
    /// - Each channel gets its own dedicated filter
    /// - No mixing between channels happens yet
    /// 
    /// For example, if your input has red, green, and blue channels:
    /// - The red channel is filtered to produce a filtered red channel
    /// - The green channel is filtered to produce a filtered green channel
    /// - The blue channel is filtered to produce a filtered blue channel
    /// 
    /// This focuses on finding spatial patterns within each individual channel.
    /// </para>
    /// </remarks>
    private Tensor<T> DepthwiseConvolution(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputHeight = input.Shape[1];
        int inputWidth = input.Shape[2];
        int outputHeight = CalculateOutputDimension(inputHeight, _kernelSize, _stride, _padding);
        int outputWidth = CalculateOutputDimension(inputWidth, _kernelSize, _stride, _padding);

        var output = new Tensor<T>([batchSize, outputHeight, outputWidth, _inputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _inputDepth; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = NumOps.Zero;
                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh - _padding;
                                int iw = ow * _stride + kw - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, ih, iw, c], _depthwiseKernels[c, 0, kh, kw]));
                                }
                            }
                        }

                        output[b, oh, ow, c] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies the pointwise convolution step to the depthwise convolution output.
    /// </summary>
    /// <param name="input">The tensor output from depthwise convolution.</param>
    /// <returns>The output tensor after pointwise convolution.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the pointwise convolution step, which applies 1×1 convolutions to combine the
    /// channels output by the depthwise convolution. This is where the channel mixing occurs, allowing
    /// the layer to learn relationships between features detected in different channels.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the second step of the filtering process.
    /// 
    /// Pointwise convolution:
    /// - Uses 1×1 filters (just one pixel in size)
    /// - Combines information across all channels
    /// - Creates new output channels based on combinations of input channels
    /// 
    /// Think of it like mixing colors:
    /// - After each color has been filtered separately by the depthwise step
    /// - The pointwise step creates new colors by combining the filtered colors
    /// - Each new output channel is a different "recipe" for combining input channels
    /// 
    /// This step allows the layer to detect patterns that span across multiple channels.
    /// </para>
    /// </remarks>
    private Tensor<T> PointwiseConvolution(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int height = input.Shape[1];
        int width = input.Shape[2];

        var output = new Tensor<T>([batchSize, height, width, _outputDepth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int oc = 0; oc < _outputDepth; oc++)
                    {
                        T sum = NumOps.Zero;
                        for (int ic = 0; ic < _inputDepth; ic++)
                        {
                            sum = NumOps.Add(sum, NumOps.Multiply(input[b, h, w, ic], _pointwiseKernels[oc, ic, 0, 0]));
                        }
                        output[b, h, w, oc] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Calculates gradients for the input, kernels, and biases during backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the backward pass of the depthwise separable convolutional layer during training.
    /// It calculates the gradients for the depthwise kernels, pointwise kernels, biases, and the input.
    /// These gradients indicate how each parameter should be adjusted to reduce the loss.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps the layer learn from its mistakes.
    /// 
    /// During the backward pass:
    /// - The layer receives information about how wrong its output was
    /// - It calculates how to adjust each of its filters to be more accurate
    /// - It prepares the adjustments but doesn't apply them yet
    /// - It passes information back to previous layers so they can learn too
    /// 
    /// The layer has to figure out:
    /// - How to adjust the depthwise filters (first step)
    /// - How to adjust the pointwise filters (second step)
    /// - How to adjust the biases
    /// 
    /// This is where the actual "learning" happens in the neural network.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastDepthwiseOutput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = outputGradient.Shape[0];
        int outputHeight = outputGradient.Shape[1];
        int outputWidth = outputGradient.Shape[2];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];

        // Initialize gradients
        _depthwiseKernelsGradient = new Tensor<T>(_depthwiseKernels.Shape);
        _pointwiseKernelsGradient = new Tensor<T>(_pointwiseKernels.Shape);
        _biasesGradient = new Vector<T>(_biases.Length);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        // Compute gradients for biases and pointwise kernels
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                for (int w = 0; w < outputWidth; w++)
                {
                    for (int oc = 0; oc < _outputDepth; oc++)
                    {
                        T gradValue = ApplyActivationDerivative(outputGradient[b, h, w, oc], _lastOutput[b, h, w, oc]);
                        _biasesGradient[oc] = NumOps.Add(_biasesGradient[oc], gradValue);

                        for (int ic = 0; ic < _inputDepth; ic++)
                        {
                            _pointwiseKernelsGradient[oc, ic, 0, 0] = NumOps.Add(_pointwiseKernelsGradient[oc, ic, 0, 0],
                                NumOps.Multiply(gradValue, _lastDepthwiseOutput[b, h, w, ic]));
                        }
                    }
                }
            }
        }

        // Compute gradients for depthwise kernels and input
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _inputDepth; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T gradValue = NumOps.Zero;
                        for (int oc = 0; oc < _outputDepth; oc++)
                        {
                            gradValue = NumOps.Add(gradValue, NumOps.Multiply(
                                ApplyActivationDerivative(outputGradient[b, oh, ow, oc], _lastOutput[b, oh, ow, oc]),
                                _pointwiseKernels[oc, c, 0, 0]));
                        }

                        for (int kh = 0; kh < _kernelSize; kh++)
                        {
                            for (int kw = 0; kw < _kernelSize; kw++)
                            {
                                int ih = oh * _stride + kh - _padding;
                                int iw = ow * _stride + kw - _padding;

                                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                {
                                    _depthwiseKernelsGradient[c, 0, kh, kw] = NumOps.Add(_depthwiseKernelsGradient[c, 0, kh, kw],
                                        NumOps.Multiply(gradValue, _lastInput[b, ih, iw, c]));
                                    inputGradient[b, ih, iw, c] = NumOps.Add(inputGradient[b, ih, iw, c],
                                        NumOps.Multiply(gradValue, _depthwiseKernels[c, 0, kh, kw]));
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Applies the derivative of the activation function during backpropagation.
    /// </summary>
    /// <param name="gradient">The gradient flowing back from the next layer.</param>
    /// <param name="output">The output value from the forward pass.</param>
    /// <returns>The gradient after applying the activation derivative.</returns>
    /// <exception cref="InvalidOperationException">Thrown when activation functions are not set.</exception>
    /// <remarks>
    /// <para>
    /// This method applies the derivative of the layer's activation function during backpropagation.
    /// It handles both scalar and vector activation functions appropriately.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps determine how sensitive the output is to small changes.
    /// 
    /// During backpropagation:
    /// - The network needs to know how much a small change in the input affects the output
    /// - This is calculated by applying the derivative of the activation function
    /// - The result tells us how to adjust the parameters to improve the network
    /// 
    /// Think of it like figuring out how steep a hill is - the steeper the hill,
    /// the more a small step will change your elevation.
    /// </para>
    /// </remarks>
    protected new T ApplyActivationDerivative(T gradient, T output)
    {
        if (UsingVectorActivation)
        {
            if (VectorActivation == null)
                throw new InvalidOperationException("Vector<double> activation function is not set.");

            // Create a vector with a single element
            var outputVector = new Vector<T>([output]);
        
            // Get the derivative matrix (1x1 in this case)
            var derivativeMatrix = VectorActivation.Derivative(outputVector);
        
            // Multiply the gradient with the single element of the derivative matrix
            return NumOps.Multiply(gradient, derivativeMatrix[0, 0]);
        }
        else
        {
            if (ScalarActivation == null)
                throw new InvalidOperationException("Scalar activation function is not set.");

            // For scalar activation, we directly multiply the gradient with the derivative
            return NumOps.Multiply(gradient, ScalarActivation.Derivative(output));
        }
    }

    /// <summary>
    /// Updates the layer's parameters (kernel weights and biases) using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (depthwise kernels, pointwise kernels, and biases)
    /// based on the gradients calculated during the backward pass. The learning rate controls the
    /// step size of the update.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the lessons learned during training.
    /// 
    /// When updating parameters:
    /// - The learning rate controls how big each adjustment is
    /// - Small learning rate = small, careful changes
    /// - Large learning rate = big, faster changes (but might overshoot)
    /// 
    /// The layer updates:
    /// - The depthwise filters (first step filters)
    /// - The pointwise filters (second step filters)
    /// - The biases
    /// 
    /// This happens after each batch of data, gradually improving the layer's performance.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_depthwiseKernelsGradient == null || _pointwiseKernelsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update depthwise kernels
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        _depthwiseKernels[i, j, k, l] = NumOps.Subtract(_depthwiseKernels[i, j, k, l],
                            NumOps.Multiply(learningRate, _depthwiseKernelsGradient[i, j, k, l]));
                    }
                }
            }
        }

        // Update pointwise kernels
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        _pointwiseKernels[i, j, k, l] = NumOps.Subtract(_pointwiseKernels[i, j, k, l],
                            NumOps.Multiply(learningRate, _pointwiseKernelsGradient[i, j, k, l]));
                    }
                }
            }
        }

        // Update biases
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, _biasesGradient[i]));
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all depthwise kernels, pointwise kernels, and biases.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts all trainable parameters from the layer and returns them as a single vector.
    /// This includes all depthwise kernels, pointwise kernels, and biases, concatenated in that order.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the learned values from the layer.
    /// 
    /// The parameters include:
    /// - All depthwise filter values (first step filters)
    /// - All pointwise filter values (second step filters)
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
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;
        var parameters = new Vector<T>(totalParams);
    
        int index = 0;
    
        // Copy depthwise kernel parameters
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        parameters[index++] = _depthwiseKernels[i, j, k, l];
                    }
                }
            }
        }
    
        // Copy pointwise kernel parameters
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        parameters[index++] = _pointwiseKernels[i, j, k, l];
                    }
                }
            }
        }
    
        // Copy bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            parameters[index++] = _biases[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets all trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector. The vector must
    /// contain values for all depthwise kernels, pointwise kernels, and biases, in that order.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learned values at once.
    /// 
    /// When setting parameters:
    /// - The vector must have exactly the right number of values
    /// - The values are assigned in order: depthwise filters, pointwise filters, then biases
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
        int totalParams = _depthwiseKernels.Length + _pointwiseKernels.Length + _biases.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set depthwise kernel parameters
        for (int i = 0; i < _depthwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _depthwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _depthwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _depthwiseKernels.Shape[3]; l++)
                    {
                        _depthwiseKernels[i, j, k, l] = parameters[index++];
                    }
                }
            }
        }
    
        // Set pointwise kernel parameters
        for (int i = 0; i < _pointwiseKernels.Shape[0]; i++)
        {
            for (int j = 0; j < _pointwiseKernels.Shape[1]; j++)
            {
                for (int k = 0; k < _pointwiseKernels.Shape[2]; k++)
                {
                    for (int l = 0; l < _pointwiseKernels.Shape[3]; l++)
                    {
                        _pointwiseKernels[i, j, k, l] = parameters[index++];
                    }
                }
            }
        }
    
        // Set bias parameters
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears the cached values from the forward and backward passes, including the
    /// input, intermediate outputs, and gradients. This is useful when starting to process a new
    /// batch or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The layer forgets the last input it processed
    /// - It forgets the intermediate results (after depthwise convolution)
    /// - It forgets the final output it produced
    /// - It clears any calculated gradients
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
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastDepthwiseOutput = null;
        _lastOutput = null;
        _depthwiseKernelsGradient = null;
        _pointwiseKernelsGradient = null;
        _biasesGradient = null;
    }
}