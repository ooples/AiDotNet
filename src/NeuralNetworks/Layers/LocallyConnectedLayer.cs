#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Locally Connected layer which applies different filters to different regions of the input, unlike a convolutional layer which shares filters.
/// </summary>
/// <remarks>
/// <para>
/// The Locally Connected layer is similar to a convolutional layer in that it applies filters to local regions 
/// of the input, but differs in that it uses different filter weights for each spatial location. This increases
/// the number of parameters and the expressiveness of the model, but reduces generalization capabilities.
/// It's useful when the patterns in different regions of the input are inherently different, such as in
/// face recognition where different parts of a face have different characteristics.
/// </para>
/// <para><b>For Beginners:</b> This layer is like a specialized convolutional layer where each region gets its own unique filter.
/// 
/// Think of a Locally Connected layer like having specialized detectors for different regions:
/// - In a regular convolutional layer, the same filter slides across the entire input
/// - In a locally connected layer, each position has its own unique filter
/// - This means the layer can learn location-specific features
/// 
/// For example, in face recognition:
/// - A convolutional layer would use the same detector for eyes, whether looking at the top-left or bottom-right
/// - A locally connected layer would use different detectors depending on where it's looking
/// 
/// This specialization increases the model's power but:
/// - Requires more parameters
/// - May not generalize as well to new examples
/// - Is more computationally intensive
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 4, 4, 1", TestConstructorArgs = "4, 4, 1, 2, 3, 1, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class LocallyConnectedLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensors for the locally connected filters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the filter weights, with a separate set of weights for each spatial location.
    /// The shape is [outputHeight, outputWidth, outputChannels, kernelSize, kernelSize, inputChannels],
    /// which captures the fact that there's a unique kernel for each output position.
    /// </para>
    /// <para><b>For Beginners:</b> These are the learnable filter values specific to each location.
    /// 
    /// The weights tensor:
    /// - Contains the filter values for each position
    /// - Is 6-dimensional to capture all the necessary information
    /// - Has different filters for each (x,y) position in the output
    /// 
    /// The 6 dimensions are:
    /// 1. Output height position
    /// 2. Output width position
    /// 3. Output channel
    /// 4. Kernel height position
    /// 5. Kernel width position
    /// 6. Input channel
    /// 
    /// This complex structure allows each position to have its own specialized filter.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;

    /// <summary>
    /// The bias values for each output channel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor contains the bias values that are added to each output channel after the filter
    /// application. These are shared across spatial locations but different for each output channel.
    /// </para>
    /// <para><b>For Beginners:</b> These are additional learnable values added to each output channel.
    ///
    /// The biases:
    /// - Are added to each output after applying the filters
    /// - Help the network learn by providing an adjustable baseline
    /// - Are the same for a given channel across all spatial positions
    ///
    /// They're like a "starting point" that the network can adjust during learning.
    /// </para>
    /// </remarks>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _biases;

    /// <summary>
    /// Stores the input tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores the pre-activation output from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Stores the output tensor from the last forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the gradients for the weights calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _weightGradients;

    /// <summary>
    /// Stores the gradients for the biases calculated during the backward pass.
    /// </summary>
    private Tensor<T>? _biasGradients;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuOutput;
    private int[]? _gpuInputShape4D;
    private bool _gpuAddedBatchDimension;

    #region GPU Weight Storage Fields

    // GPU weight tensors for GPU-resident training
    private Tensor<T>? _gpuWeights;
    private Tensor<T>? _gpuBiases;

    // GPU gradient tensors from BackwardGpu
    private Tensor<T>? _gpuWeightGradient;
    private Tensor<T>? _gpuBiasGradient;

    // Optimizer state tensors for SGD/NAG/LARS (velocity)
    private Tensor<T>? _gpuWeightVelocity;
    private Tensor<T>? _gpuBiasVelocity;

    // Optimizer state tensors for Adam/AdamW/LAMB (M and V)
    private Tensor<T>? _gpuWeightM;
    private Tensor<T>? _gpuWeightV;
    private Tensor<T>? _gpuBiasM;
    private Tensor<T>? _gpuBiasV;

    #endregion

    /// <summary>
    /// The height of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the height dimension of the input tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height (number of rows) of the input data.
    /// 
    /// For example, if processing 28x28 images, this would be 28.
    /// </para>
    /// </remarks>
    private readonly int _inputHeight;

    /// <summary>
    /// The width of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the width dimension of the input tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width (number of columns) of the input data.
    /// 
    /// For example, if processing 28x28 images, this would be 28.
    /// </para>
    /// </remarks>
    private readonly int _inputWidth;

    /// <summary>
    /// The number of channels in the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the number of channels in the input tensor. For RGB images, this would be 3.
    /// For grayscale images, this would be 1. For feature maps from previous layers, this could be any number.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of feature channels in the input data.
    /// 
    /// - For color images: 3 channels (Red, Green, Blue)
    /// - For grayscale images: 1 channel
    /// - For hidden layers: However many features the previous layer produced
    /// </para>
    /// </remarks>
    private readonly int _inputChannels;

    /// <summary>
    /// The height of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the height dimension of the output tensor, calculated as (inputHeight - kernelSize) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This is the height of the output after applying the filters.
    /// 
    /// It depends on:
    /// - The input height
    /// - The kernel size 
    /// - The stride (how many pixels the filter moves each step)
    /// 
    /// It's calculated as: (inputHeight - kernelSize) / stride + 1
    /// </para>
    /// </remarks>
    private readonly int _outputHeight;

    /// <summary>
    /// The width of the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the width dimension of the output tensor, calculated as (inputWidth - kernelSize) / stride + 1.
    /// </para>
    /// <para><b>For Beginners:</b> This is the width of the output after applying the filters.
    /// 
    /// It depends on:
    /// - The input width
    /// - The kernel size 
    /// - The stride (how many pixels the filter moves each step)
    /// 
    /// It's calculated as: (inputWidth - kernelSize) / stride + 1
    /// </para>
    /// </remarks>
    private readonly int _outputWidth;

    /// <summary>
    /// The number of channels in the output tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the number of channels in the output tensor, which is the number of different
    /// filters applied at each spatial location.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of features the layer will produce.
    /// 
    /// Each output channel:
    /// - Represents a different feature or pattern the layer is looking for
    /// - Is created by a separate filter
    /// - Helps the network learn more complex representations
    /// 
    /// The more output channels, the more patterns the layer can detect,
    /// but also the more parameters it needs to learn.
    /// </para>
    /// </remarks>
    private readonly int _outputChannels;

    /// <summary>
    /// The size of the kernel (filter) in both height and width dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents the size of the square kernel (filter) applied to the input. The kernel
    /// has dimensions kernelSize x kernelSize.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of each filter window that processes the input.
    /// 
    /// The kernel size:
    /// - Determines how large an area each filter covers
    /// - Is the same for both height and width (square filter)
    /// - Affects how local or global the detected features are
    /// 
    /// Common values are 3x3, 5x5, or 7x7. Larger kernels can detect more global patterns
    /// but require more parameters.
    /// </para>
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// The stride (step size) of the kernel when moving across the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value represents how many pixels the kernel moves in each step when scanning the input.
    /// A stride of 1 means the kernel moves one pixel at a time, while a stride of 2 means it skips
    /// every other pixel.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many pixels the filter moves each step.
    /// 
    /// The stride:
    /// - Controls how much overlap there is between filter applications
    /// - Affects the size of the output (larger stride = smaller output)
    /// - Can help reduce computation by processing fewer positions
    /// 
    /// A stride of 1 means the filter moves one pixel at a time (maximum overlap).
    /// A stride of 2 means the filter jumps two pixels each time (less overlap, smaller output).
    /// </para>
    /// </remarks>
    private readonly int _stride;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> because this layer has trainable parameters (weights and biases).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The LocallyConnectedLayer always returns true because it contains trainable weights and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has parameters that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// The Locally Connected layer always supports training because it has weights 
    /// and biases that are learned during training.
    /// </para>
    /// </remarks>
    public override int ParameterCount => _weights.Length + _biases.Length;
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="LocallyConnectedLayer{T}"/> class with the specified dimensions, kernel parameters, and element-wise activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="inputChannels">The number of channels in the input tensor.</param>
    /// <param name="outputChannels">The number of channels in the output tensor.</param>
    /// <param name="kernelSize">The size of the kernel (filter) in both height and width dimensions.</param>
    /// <param name="stride">The stride (step size) of the kernel when moving across the input.</param>
    /// <param name="activationFunction">The activation function to apply after the locally connected operation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Locally Connected layer with the specified dimensions, kernel parameters,
    /// and element-wise activation function. It initializes the weights and biases and calculates the output
    /// dimensions based on the input dimensions, kernel size, and stride.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new locally connected layer with standard activation function.
    /// 
    /// When creating this layer, you specify:
    /// - inputHeight, inputWidth: The dimensions of your input data
    /// - inputChannels: How many channels your input data has
    /// - outputChannels: How many different features you want the layer to detect
    /// - kernelSize: The size of each filter window (e.g., 3 for a 3x3 filter)
    /// - stride: How many pixels the filter moves each step
    /// - activationFunction: What function to apply to the output (default is ReLU)
    /// 
    /// For example, to process 28x28 grayscale images with 16 output features, 3x3 filters,
    /// and a stride of 1, you would use: inputHeight=28, inputWidth=28, inputChannels=1,
    /// outputChannels=16, kernelSize=3, stride=1.
    /// </para>
    /// </remarks>
    public LocallyConnectedLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels],
            [
                (inputHeight - kernelSize) / stride + 1,
                (inputWidth - kernelSize) / stride + 1,
                outputChannels
            ],
            activationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Tensor<T>([_outputChannels]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LocallyConnectedLayer{T}"/> class with the specified dimensions, kernel parameters, and vector activation function.
    /// </summary>
    /// <param name="inputHeight">The height of the input tensor.</param>
    /// <param name="inputWidth">The width of the input tensor.</param>
    /// <param name="inputChannels">The number of channels in the input tensor.</param>
    /// <param name="outputChannels">The number of channels in the output tensor.</param>
    /// <param name="kernelSize">The size of the kernel (filter) in both height and width dimensions.</param>
    /// <param name="stride">The stride (step size) of the kernel when moving across the input.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after the locally connected operation. Defaults to ReLU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Locally Connected layer with the specified dimensions, kernel parameters,
    /// and vector activation function. Vector activation functions operate on entire vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new locally connected layer with an advanced vector-based activation.
    /// 
    /// Vector activation functions:
    /// - Process entire groups of numbers together, not just one at a time
    /// - Can capture relationships between different features
    /// - May be more powerful for complex patterns
    /// 
    /// Otherwise, this constructor works just like the standard one, setting up the layer with:
    /// - The specified dimensions and parameters
    /// - Proper calculation of output dimensions
    /// - Initialization of weights and biases
    /// </para>
    /// </remarks>
    public LocallyConnectedLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride,
        IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(
            [inputHeight, inputWidth, inputChannels],
            [
                (inputHeight - kernelSize) / stride + 1,
                (inputWidth - kernelSize) / stride + 1,
                outputChannels
            ],
            vectorActivationFunction ?? new ReLUActivation<T>())
    {
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputHeight = (inputHeight - kernelSize) / stride + 1;
        _outputWidth = (inputWidth - kernelSize) / stride + 1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        // Initialize weights and biases
        _weights = new Tensor<T>([_outputHeight, _outputWidth, _outputChannels, _kernelSize, _kernelSize, _inputChannels]);
        _biases = new Tensor<T>([_outputChannels]);

        InitializeParameters();

        // Register tensors for GPU memory persistence
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes the weights and biases of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weights using Xavier initialization, which scales the random values
    /// based on the number of input and output connections. The biases are initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the starting values for the layer's weights and biases.
    /// 
    /// For weights:
    /// - Values are randomized with Xavier initialization
    /// - This helps prevent the signals from growing too large or too small
    /// - Different weights for each position, output channel, and filter position
    /// 
    /// For biases:
    /// - All values start at zero
    /// - They will adjust during training to fit the data better
    /// 
    /// Good initialization is important because it affects how quickly and how well the network learns.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // Xavier initialization for weights using vectorized operations
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_kernelSize * _kernelSize * _inputChannels + _outputChannels)));
        T half = NumOps.FromDouble(0.5);

        // Generate random values [0, 1) and transform to [-0.5, 0.5] * scale
        int totalElements = _weights.Length;
        var randomTensor = Tensor<T>.CreateRandom(totalElements, 1); // [0, 1]

        // Create tensor filled with 0.5 for subtraction
        var halfTensor = new Tensor<T>([totalElements]);
        halfTensor.Fill(half);

        // Transform to [-0.5, 0.5] * scale using Engine ops
        var centeredTensor = Engine.TensorSubtract(randomTensor.Reshape([totalElements]), halfTensor);
        var scaledTensor = Engine.TensorMultiplyScalar(centeredTensor, scale);

        // Assign the scaled tensor to weights (preserving shape)
        // Note: Array.Copy to _weights.ToArray() creates a temporary copy, not updating _weights
        _weights = scaledTensor.Reshape(_weights._shape);

        // Initialize biases to zero
        _biases.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of the locally connected layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape should be [batchSize, inputHeight, inputWidth, inputChannels].</param>
    /// <returns>The output tensor after applying the locally connected operation and activation. Shape will be [batchSize, outputHeight, outputWidth, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the locally connected layer. It applies different filters
    /// to each spatial location of the input, followed by adding biases and applying the activation function.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the locally connected filters.
    /// 
    /// During the forward pass:
    /// 1. For each position in the output:
    ///    - Apply a unique filter to the corresponding region of the input
    ///    - Sum up the results of element-wise multiplications
    ///    - Add the bias for the output channel
    /// 2. Apply the activation function to add non-linearity
    /// 
    /// This process is similar to a convolution, but instead of re-using the same filter for all
    /// positions, each position has its own specialized filter.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Normalize to 4D NHWC [batch, height, width, channels] for processing
        Tensor<T> processInput;

        if (rank < 4)
        {
            // For lower-rank inputs, add leading dimensions
            // 3D [H, W, C] -> [1, H, W, C]
            // 2D [W, C] -> [1, 1, W, C]
            // 1D [C] -> [1, 1, 1, C]
            var shape4D = new int[4];
            int offset = 4 - rank;
            for (int i = 0; i < offset; i++)
                shape4D[i] = 1;
            for (int i = 0; i < rank; i++)
                shape4D[offset + i] = input.Shape[i];
            processInput = Engine.Reshape(input, shape4D);
        }
        else if (rank == 4)
        {
            // Standard 4D NHWC
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dimensions into batch
            // e.g., 5D [B1, B2, H, W, C] -> [B1*B2, H, W, C]
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            int height = input.Shape[rank - 3];
            int width = input.Shape[rank - 2];
            int channels = input.Shape[rank - 1];
            processInput = Engine.Reshape(input, [flatBatch, height, width, channels]);
        }

        _lastInput = processInput;

        // === GPU-Accelerated LocallyConnectedConv2D ===
        // The layer uses NHWC format but Engine expects NCHW, so we transpose.
        // Every shape op via Engine so the gradient tape records the transformation —
        // direct Tensor<T>.Transpose bypasses the tape and disconnects _weights / _biases
        // from the backward graph.
        var inputNCHW = Engine.TensorPermute(processInput, new[] { 0, 3, 1, 2 });

        // Weights need to be permuted from [oh, ow, oc, kh, kw, ic] to [oh, ow, oc, ic, kh, kw]
        var weightsPermuted = Engine.TensorPermute(_weights, new[] { 0, 1, 2, 5, 3, 4 });

        // Pass bias as 1D tensor [outChannels] to ensure consistent behavior across
        // CPU fallback, GPU, and JIT paths. The engine handles per-channel bias internally.
        int[] strideArr = [_stride, _stride];
        var outputNCHW = Engine.LocallyConnectedConv2D(inputNCHW, weightsPermuted, _biases, strideArr);

        // Transpose output back from NCHW [batch, channels, height, width] to NHWC [batch, height, width, channels]
        var preActivation = Engine.TensorPermute(outputNCHW, new[] { 0, 2, 3, 1 });

        // Apply activation function
        var result = ApplyActivation(preActivation);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastPreActivation = preActivation;
            _lastOutput = result;
        }

        // Restore output shape to match original input rank
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            // Get output spatial dimensions from the 4D output
            int outHeight = result.Shape[1];
            int outWidth = result.Shape[2];
            int outChannels = result.Shape[3];

            if (_originalInputShape.Length < 4)
            {
                // Restore lower-rank shape: remove leading 1s
                // 3D output [H', W', C'], 2D output [W', C'], 1D output [C']
                var outShape = new int[_originalInputShape.Length];
                int offset = 4 - _originalInputShape.Length;
                if (_originalInputShape.Length >= 3) outShape[_originalInputShape.Length - 3] = outHeight;
                if (_originalInputShape.Length >= 2) outShape[_originalInputShape.Length - 2] = outWidth;
                outShape[_originalInputShape.Length - 1] = outChannels;
                return Engine.Reshape(result, outShape);
            }
            else
            {
                // Restore higher-rank shape: expand batch dimension back
                var outShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 3; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 3] = outHeight;
                outShape[_originalInputShape.Length - 2] = outWidth;
                outShape[_originalInputShape.Length - 1] = outChannels;
                return Engine.Reshape(result, outShape);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor [batch, inChannels, inHeight, inWidth] in NCHW format.</param>
    /// <returns>GPU-resident output tensor [batch, outChannels, outHeight, outWidth] in NCHW format.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// All data stays on the GPU throughout the computation, avoiding expensive CPU-GPU transfers.</para>
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

        // Validate input shape - GPU uses NCHW format [batch, channels, height, width]
        if (input.Shape.Length < 3)
        {
            throw new ArgumentException(
                $"LocallyConnected input requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");
        }

        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Reshape input to 4D NCHW [B, C, H, W] for locally connected operation
        Tensor<T> input4D;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
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
        if (actualInputChannels != _inputChannels)
        {
            throw new ArgumentException(
                $"Expected input channels {_inputChannels}, but got {actualInputChannels}.");
        }

        // Weights need to be permuted from [oh, ow, oc, kh, kw, ic] to [oh, ow, oc, ic, kh, kw] for GPU kernel
        var weightsPermuted = _weights.Transpose([0, 1, 2, 5, 3, 4]);

        // Map activation function to FusedActivationType
        var fusedActivation = MapActivationToFused();

        // Execute GPU-fused LocallyConnected Conv2D + Bias + Activation
        var result = gpuEngine.LocallyConnectedConv2DGpu(
            input4D,
            weightsPermuted,
            _biases,
            _stride, _stride,      // strideH, strideW
            fusedActivation);

        // Cache tensors for backward pass during training
        if (IsTrainingMode)
        {
            _gpuInput = input4D;
            _gpuOutput = result;
            _gpuInputShape4D = input4D._shape;
            _gpuAddedBatchDimension = rank == 3;
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
            outputShape[_originalInputShape.Length - 3] = _outputChannels;
            outputShape[_originalInputShape.Length - 2] = result.Shape[2];
            outputShape[_originalInputShape.Length - 1] = result.Shape[3];
            return result.Reshape(outputShape);
        }

        if (rank == 3)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            return result.Reshape([_outputChannels, result.Shape[2], result.Shape[3]]);
        }

        return result;
    }

    /// <summary>
    /// Computes the activation gradient on GPU for locally connected layer backward pass.
    /// </summary>
    private Tensor<T> ComputeActivationGradientGpu(DirectGpuTensorEngine gpuEngine, Tensor<T> gradOutput, Tensor<T> output, FusedActivationType activation)
    {
        // Flatten tensors for element-wise activation backward
        int totalElements = gradOutput.Length;
        var flat2DShape = new[] { totalElements, 1 };
        var flatGrad = gradOutput.Reshape(flat2DShape);
        var flatOutput = output.Reshape(flat2DShape);

        Tensor<T> flatResult = activation switch
        {
            FusedActivationType.ReLU => gpuEngine.ReluBackwardGpu<T>(flatGrad, flatOutput),
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
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when Backward has not been called before UpdateParameters.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients calculated during
    /// the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// - All weights and biases are adjusted to reduce prediction errors
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer "learns" from data over time, gradually improving
    /// its ability to extract useful features from the input.
    /// 
    /// The method will throw an error if you try to run it before performing a backward pass.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightGradients == null || _biasGradients == null)
        {
            throw new InvalidOperationException("UpdateParameters called before Backward. Gradients are null.");
        }

        _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(_weightGradients, learningRate));
        _biases = Engine.TensorSubtract(_biases, Engine.TensorMultiplyScalar(_biasGradients, learningRate));

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights and biases) and combines them into a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving and loading
    /// model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all the unique filter weights (which can be very many!) and biases
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// For locally connected layers, this vector can be very large due to the
    /// unique filters for each spatial location.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Get weight parameters as vector
        var weightVector = new Vector<T>(_weights.ToArray());

        // Get bias parameters as vector
        var biasVector = new Vector<T>(_biases.ToArray());

        // Concatenate weights and biases
        return Vector<T>.Concatenate(weightVector, biasVector);
    }

    /// <summary>
    /// Sets the trainable parameters of the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all the weights and biases of the layer from a single vector of parameters.
    /// The vector must have the correct length to match the total number of parameters in the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The values are distributed to all the weights and biases in the correct order
    /// - Throws an error if the input doesn't match the expected number of parameters
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Setting specific parameter values for testing
    /// 
    /// For locally connected layers, this vector needs to be very large to account for
    /// all the unique filters at each spatial location.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var wGrad = _weightGradients != null
            ? new Vector<T>(_weightGradients.ToArray())
            : new Vector<T>(_weights.Length);
        var bGrad = _biasGradients != null
            ? new Vector<T>(_biasGradients.ToArray())
            : new Vector<T>(_biases.Length);
        return Vector<T>.Concatenate(wGrad, bGrad);
    }

    public override void ClearGradients()
    {
        _weightGradients = null;
        _biasGradients = null;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _weights.Length + _biases.Length;
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Split parameters into weights and biases using Vector.Slice
        int weightsLength = _weights.Length;
        var weightsVector = parameters.Slice(0, weightsLength);
        var biasesVector = parameters.Slice(weightsLength, _biases.Length);

        // Convert vectors to tensors and assign
        _weights = Tensor<T>.FromVector(weightsVector, _weights._shape);
        _biases = Tensor<T>.FromVector(biasesVector, _biases._shape);

        // Notify engine that parameters have changed (for GPU cache invalidation)
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer, clearing cached values from forward and backward passes.
    /// This includes the last input tensor and the weight and bias gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input from the last forward pass is cleared
    /// - All gradient information from the last backward pass is cleared
    /// - The layer is ready for new data without being influenced by previous data
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// It helps ensure that each training or prediction batch is processed independently.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastPreActivation = null;
        _lastOutput = null;
        _weightGradients = null;
        _biasGradients = null;

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuOutput = null;
        _gpuInputShape4D = null;
        _gpuAddedBatchDimension = false;
    }

    #region GPU Parameter Updates

    /// <summary>
    /// Updates parameters using GPU-based optimizer.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend() ?? throw new InvalidOperationException("GPU backend not available");

        if (_gpuWeightGradient == null || _gpuBiasGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure GPU weight tensors exist
        _gpuWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _weights, GpuTensorRole.Weight);
        _gpuBiases ??= GpuTensorHelper.UploadToGpu<T>(backend, _biases, GpuTensorRole.Bias);

        // Ensure optimizer state buffers exist
        EnsureLocallyConnectedOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeights.Buffer, _gpuWeightGradient.Buffer, BuildLocallyConnectedOptimizerState("weights"), _weights.Length);
        config.ApplyUpdate(backend, _gpuBiases.Buffer, _gpuBiasGradient.Buffer, BuildLocallyConnectedOptimizerState("biases"), _biases.Length);

        // Sync back to CPU tensors for compatibility
        _weights = _gpuWeights;
        _biases = _gpuBiases;
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all locally connected parameters.
    /// </summary>
    private void EnsureLocallyConnectedOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = _weights.Length;
        int biasSize = _biases.Length;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Velocity buffers
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // Squared average buffers (reuse velocity fields)
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    /// <summary>
    /// Builds the optimizer state for a specific locally connected parameter.
    /// </summary>
    private GpuOptimizerState BuildLocallyConnectedOptimizerState(string paramName)
    {
        return paramName switch
        {
            "weights" => new GpuOptimizerState { Velocity = _gpuWeightVelocity?.Buffer, M = _gpuWeightM?.Buffer, V = _gpuWeightV?.Buffer, SquaredAvg = _gpuWeightVelocity?.Buffer, AccumulatedGrad = _gpuWeightVelocity?.Buffer },
            "biases" => new GpuOptimizerState { Velocity = _gpuBiasVelocity?.Buffer, M = _gpuBiasM?.Buffer, V = _gpuBiasV?.Buffer, SquaredAvg = _gpuBiasVelocity?.Buffer, AccumulatedGrad = _gpuBiasVelocity?.Buffer },
            _ => new GpuOptimizerState()
        };
    }

    #endregion
}
