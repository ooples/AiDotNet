namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a primary capsule layer for capsule networks.
/// </summary>
/// <remarks>
/// <para>
/// The PrimaryCapsuleLayer is the first layer in a capsule network that transforms traditional scalar feature maps
/// into capsule vectors. It performs a convolution operation followed by reshaping the output into capsules.
/// Each capsule represents a group of neurons that encodes both the presence and properties of a particular entity.
/// This layer serves as a bridge between standard convolutional layers and higher-level capsule layers.
/// </para>
/// <para><b>For Beginners:</b> This layer is the first step in creating a capsule network.
/// 
/// In traditional neural networks, each neuron outputs a single number indicating the presence of a feature.
/// In capsule networks, neurons are grouped into "capsules" where each capsule outputs a vector:
/// - The length of the vector represents the presence of an entity
/// - The orientation of the vector represents properties of that entity
/// 
/// Think of it like this:
/// - Standard neurons: "I see a nose with 90% confidence"
/// - Capsule neurons: "I see a nose with 90% confidence, and it's pointing 30° to the left, 
///   it's 2cm long, it has a slightly curved shape..."
/// 
/// The primary capsule layer converts traditional feature maps (from convolutional layers)
/// into these vector-based capsules that can capture more detailed information about the entities detected.
/// 
/// This approach helps the network understand spatial relationships and maintain information
/// about pose, orientation, and other properties that are typically lost in traditional networks.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PrimaryCapsuleLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensor for convolution operations. Shape: [outputChannels, inputChannels * kernelSize * kernelSize]
    /// </summary>
    /// <remarks>
    /// This tensor contains the learnable weights for the convolution operation.
    /// </remarks>
    private Tensor<T> _convWeights;

    /// <summary>
    /// The bias tensor for convolution operations. Shape: [outputChannels]
    /// </summary>
    /// <remarks>
    /// This tensor contains the learnable biases for the convolution operation.
    /// </remarks>
    private Tensor<T> _convBias;

    /// <summary>
    /// The gradient of the loss with respect to the convolution weights.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the convolution weights, which is used to update the weights
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _convWeightsGradient;

    /// <summary>
    /// The gradient of the loss with respect to the convolution bias.
    /// </summary>
    /// <remarks>
    /// This field stores the gradient of the convolution bias, which is used to update the bias
    /// during the parameter update step.
    /// </remarks>
    private Tensor<T>? _convBiasGradient;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Stores whether the original input was provided in NCHW layout.
    /// </summary>
    private bool _inputWasNCHW;

    /// <summary>
    /// The output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the output tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The number of input channels.
    /// </summary>
    /// <remarks>
    /// This field stores the number of channels in the input tensor.
    /// </remarks>
    private readonly int _inputChannels;

    /// <summary>
    /// The number of capsule channels.
    /// </summary>
    /// <remarks>
    /// This field stores the number of different types of capsules that the layer will produce.
    /// </remarks>
    private readonly int _capsuleChannels;

    /// <summary>
    /// The dimension of each capsule.
    /// </summary>
    /// <remarks>
    /// This field stores the dimension of the vector that each capsule outputs.
    /// </remarks>
    private readonly int _capsuleDimension;

    /// <summary>
    /// The size of the convolutional kernel.
    /// </summary>
    /// <remarks>
    /// This field stores the size of the square kernel used for convolution operations.
    /// </remarks>
    private readonly int _kernelSize;

    /// <summary>
    /// The stride of the convolution operation.
    /// </summary>
    /// <remarks>
    /// This field stores the number of pixels to skip when sliding the convolution window.
    /// </remarks>
    private readonly int _stride;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the PrimaryCapsuleLayer has trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that PrimaryCapsuleLayer can be trained through backpropagation. The layer
    /// has trainable parameters (convolution weights and biases) that are updated during training to optimize
    /// the capsule transformation process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (weights and biases) that change during training
    /// - It will improve its performance as it sees more data
    /// - It learns to extract better capsule representations from the input
    /// 
    /// During training, the layer learns to transform input features into capsule vectors
    /// that best represent the entities in the data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimaryCapsuleLayer{T}"/> class with the specified parameters
    /// and a scalar activation function.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="capsuleChannels">The number of capsule channels.</param>
    /// <param name="capsuleDimension">The dimension of each capsule.</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="stride">The stride of the convolution operation.</param>
    /// <param name="scalarActivation">The activation function to apply after processing. Defaults to Squash if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a PrimaryCapsuleLayer with the specified parameters. It initializes the convolution
    /// weights and biases and sets up the layer to transform input feature maps into primary capsules.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary parameters.
    /// 
    /// When creating a PrimaryCapsuleLayer, you need to specify:
    /// - inputChannels: How many channels your input has (e.g., 3 for RGB images, or more if from a conv layer)
    /// - capsuleChannels: How many different types of capsules to create
    /// - capsuleDimension: How many values in each capsule's output vector
    /// - kernelSize: The size of the area examined by the convolution (e.g., 3 for a 3×3 kernel)
    /// - stride: How far to move the kernel each step
    /// - scalarActivation: The function applied to each scalar value (defaults to Squash)
    /// 
    /// For example, if you set capsuleChannels=8 and capsuleDimension=16, you'll have 8 different 
    /// types of capsules, each outputting a 16-dimensional vector.
    /// 
    /// The default Squash activation function is specifically designed for capsule networks.
    /// It ensures that the length of each capsule's output vector is between 0 and 1, 
    /// which is ideal for representing the probability of an entity being present.
    /// </para>
    /// </remarks>
    public PrimaryCapsuleLayer(int inputChannels, int capsuleChannels, int capsuleDimension, int kernelSize, int stride, IActivationFunction<T>? scalarActivation = null)
        : base([inputChannels], [capsuleChannels * capsuleDimension], scalarActivation ?? new SquashActivation<T>())
    {
        _inputChannels = inputChannels;
        _capsuleChannels = capsuleChannels;
        _capsuleDimension = capsuleDimension;
        _kernelSize = kernelSize;
        _stride = stride;

        int outputChannels = capsuleChannels * capsuleDimension;
        int inputSize = inputChannels * kernelSize * kernelSize;
        _convWeights = new Tensor<T>([outputChannels, inputSize]);
        _convBias = new Tensor<T>([outputChannels]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimaryCapsuleLayer{T}"/> class with the specified parameters
    /// and a vector activation function.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="capsuleChannels">The number of capsule channels.</param>
    /// <param name="capsuleDimension">The dimension of each capsule.</param>
    /// <param name="kernelSize">The size of the convolutional kernel.</param>
    /// <param name="stride">The stride of the convolution operation.</param>
    /// <param name="vectorActivation">The vector activation function to apply after processing. Defaults to Squash if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a PrimaryCapsuleLayer with the specified parameters and a vector activation function.
    /// A vector activation function operates on entire capsule vectors rather than individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor is similar to the other one, but uses a vector-based activation function.
    ///
    /// A vector activation function:
    /// - Operates on entire capsule vectors at once, rather than one value at a time
    /// - Can better preserve the relationship between values in a capsule
    /// - The default Squash function ensures vector lengths are between 0 and 1
    ///
    /// The Squash function is specifically designed for capsule networks. It scales vectors
    /// non-linearly so that small vectors shrink to nearly zero length, while large vectors
    /// shrink to have a length slightly below 1, preserving their direction.
    /// </para>
    /// </remarks>
    public PrimaryCapsuleLayer(int inputChannels, int capsuleChannels, int capsuleDimension, int kernelSize, int stride, IVectorActivationFunction<T>? vectorActivation = null)
        : base([inputChannels], [capsuleChannels * capsuleDimension], vectorActivation ?? new SquashActivation<T>())
    {
        _inputChannels = inputChannels;
        _capsuleChannels = capsuleChannels;
        _capsuleDimension = capsuleDimension;
        _kernelSize = kernelSize;
        _stride = stride;

        int outputChannels = capsuleChannels * capsuleDimension;
        int inputSize = inputChannels * kernelSize * kernelSize;
        _convWeights = new Tensor<T>([outputChannels, inputSize]);
        _convBias = new Tensor<T>([outputChannels]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the layer's weights and biases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the convolution weights using a scaling factor derived from the dimensions
    /// of the weight matrices. The scaling helps prevent vanishing or exploding gradients
    /// during training. The bias is initialized to zero.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's weights and biases.
    /// 
    /// Proper initialization is important for neural networks because:
    /// - Starting with good values helps the network learn faster
    /// - It helps prevent problems during training like vanishing or exploding gradients
    ///   (when values become too small or too large)
    /// 
    /// This method:
    /// - Calculates a scaling factor based on the size of the matrices
    /// - Initializes weights to small random values multiplied by this scale
    /// - Sets all bias values to zero
    /// 
    /// This approach (known as "He initialization") works well for many types of neural networks,
    /// including capsule networks.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int rows = _convWeights.Shape[0];
        int cols = _convWeights.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (rows + cols)));

        // Initialize weights with scaled random values using Engine operations
        var randomTensor = Tensor<T>.CreateRandom(rows, cols);
        var halfTensor = new Tensor<T>([rows, cols]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        _convWeights = Engine.TensorMultiplyScalar(shifted, scale);

        // Initialize bias to zero using Fill
        _convBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of the primary capsule layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing capsule vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the primary capsule layer. It performs a convolution
    /// operation on the input tensor, reshapes the result into capsule vectors, and applies the activation
    /// function to produce the final output. The input and output tensors are cached for use during the
    /// backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method transforms the input features into capsule vectors.
    /// 
    /// During the forward pass:
    /// 1. The method applies a convolution operation to the input
    ///    (similar to a standard convolutional layer)
    /// 2. It reshapes the result into groups of vectors (the capsules)
    /// 3. It applies the activation function (typically Squash) to each capsule vector
    /// 
    /// The output is a set of capsule vectors where:
    /// - Each capsule vector's length represents the probability of detecting an entity
    /// - The orientation of the vector represents properties of the detected entity
    /// 
    /// This is the key difference from traditional neural networks - instead of just 
    /// detecting if something is present, the capsules also capture information about 
    /// the properties of what they detect.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // PrimaryCapsule needs to handle both NCHW (from ConvolutionalLayer) and NHWC inputs
        // Detect format: if first dim (for 3D) or second dim (for 4D) matches _inputChannels, it's NCHW
        Tensor<T> processInput;
        int batchSize;
        bool inputIsNCHW = false;

        if (rank < 4)
        {
            // For 3D input, detect format:
            // NCHW: [C, H, W] where C == _inputChannels
            // NHWC: [H, W, C] where C == _inputChannels (last dim)
            if (rank == 3 && input.Shape[0] == _inputChannels)
            {
                // NCHW format: [C, H, W] -> [1, C, H, W]
                inputIsNCHW = true;
                processInput = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
                batchSize = 1;
            }
            else
            {
                // NHWC format: add leading dimensions to make 4D
                // 3D [H, W, C] -> [1, H, W, C]
                // 2D [W, C] -> [1, 1, W, C]
                // 1D [C] -> [1, 1, 1, C]
                var shape4D = new int[4];
                int offset = 4 - rank;
                for (int i = 0; i < offset; i++)
                    shape4D[i] = 1;
                for (int i = 0; i < rank; i++)
                    shape4D[offset + i] = input.Shape[i];
                processInput = input.Reshape(shape4D);
                batchSize = shape4D[0];
            }
        }
        else if (rank == 4)
        {
            // For 4D, detect NCHW vs NHWC by checking which dim matches _inputChannels
            // NCHW: [B, C, H, W] where C == _inputChannels (dim 1)
            // NHWC: [B, H, W, C] where C == _inputChannels (dim 3)
            if (input.Shape[1] == _inputChannels && input.Shape[3] != _inputChannels)
            {
                inputIsNCHW = true;
            }
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch, assume NHWC for last 3 dims
            // e.g., 5D [B1, B2, H, W, C] -> [B1*B2, H, W, C]
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            int height = input.Shape[rank - 3];
            int width = input.Shape[rank - 2];
            int channels = input.Shape[rank - 1];
            processInput = input.Reshape([flatBatch, height, width, channels]);
        }

        _inputWasNCHW = inputIsNCHW;
        _lastInput = processInput;

        // Get spatial dimensions based on format
        int inputHeight, inputWidth;
        if (inputIsNCHW)
        {
            // NCHW: [B, C, H, W]
            inputHeight = processInput.Shape[2];
            inputWidth = processInput.Shape[3];
        }
        else
        {
            // NHWC: [B, H, W, C]
            inputHeight = processInput.Shape[1];
            inputWidth = processInput.Shape[2];
        }

        int outputHeight = (inputHeight - _kernelSize) / _stride + 1;
        int outputWidth = (inputWidth - _kernelSize) / _stride + 1;

        // Reshape weights to Conv2D kernel format [outChannels, inChannels, kH, kW]
        int outputChannels = _capsuleChannels * _capsuleDimension;
        var kernelNCHW = _convWeights.Reshape([outputChannels, _inputChannels, _kernelSize, _kernelSize]);

        // Convert input to NCHW for Conv2D if needed
        Tensor<T> inputNCHW = inputIsNCHW
            ? processInput
            : processInput.Transpose([0, 3, 1, 2]);

        // Convolution
        var convNCHW = Engine.Conv2D(inputNCHW, kernelNCHW, new[] { _stride, _stride }, new[] { 0, 0 }, new[] { 1, 1 });

        // Add bias (reshape to [1, outChannels, 1, 1] for broadcast)
        var biasNCHW = _convBias.Reshape([1, outputChannels, 1, 1]);
        convNCHW = Engine.TensorBroadcastAdd(convNCHW, biasNCHW);

        // Convert back to NHWC and reshape to capsule layout
        var convNHWC = convNCHW.Transpose([0, 2, 3, 1]);
        var output = convNHWC.Reshape([batchSize, outputHeight, outputWidth, _capsuleChannels, _capsuleDimension]);

        // Apply activation (Squash) to each capsule vector
        // SquashActivation expects 2D input [numCapsules, capsuleDim], so reshape and apply
        int totalCapsules = batchSize * outputHeight * outputWidth * _capsuleChannels;
        var flatOutput = output.Reshape([totalCapsules, _capsuleDimension]);
        var activatedFlat = ApplyActivation(flatOutput);
        _lastOutput = activatedFlat.Reshape([batchSize, outputHeight, outputWidth, _capsuleChannels, _capsuleDimension]);

        // Restore output shape to match original input rank
        // Output is 5D [B, OH, OW, capsuleChannels, capsuleDim]
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            if (_originalInputShape.Length < 4)
            {
                // Lower-rank: remove batch dimension
                // 3D input -> 4D output [OH, OW, capsuleChannels, capsuleDim]
                return _lastOutput.Reshape([outputHeight, outputWidth, _capsuleChannels, _capsuleDimension]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                // e.g., 5D input [B1, B2, H, W, C] -> 6D output [B1, B2, OH, OW, capsuleChannels, capsuleDim]
                var outShape = new int[_originalInputShape.Length + 1];
                for (int d = 0; d < _originalInputShape.Length - 3; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 3] = outputHeight;
                outShape[_originalInputShape.Length - 2] = outputWidth;
                outShape[_originalInputShape.Length - 1] = _capsuleChannels;
                outShape[_originalInputShape.Length] = _capsuleDimension;
                return _lastOutput.Reshape(outShape);
            }
        }

        return _lastOutput;
    }

    /// <summary>
    /// Extracts a patch from the input tensor for convolution.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="batch">The batch index.</param>
    /// <param name="startY">The starting Y coordinate of the patch.</param>
    /// <param name="startX">The starting X coordinate of the patch.</param>
    /// <returns>A vector containing the flattened patch values.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts a square patch from the input tensor at the specified location and flattens
    /// it into a vector for use in the convolution operation. The patch has dimensions _kernelSize × _kernelSize
    /// and includes all input channels.
    /// </para>
    /// <para><b>For Beginners:</b> This method extracts a small square region from the input for processing.
    /// 
    /// For convolution operations, we need to:
    /// - Look at a small area of the input at a time (the "patch" or "receptive field")
    /// - Apply the same operation to each patch as we move across the input
    /// 
    /// This method:
    /// - Takes coordinates that specify where to start (startX, startY)
    /// - Extracts a square patch of size _kernelSize × _kernelSize
    /// - Includes all channels from the input for that patch
    /// - Flattens this 3D patch into a 1D vector for easier processing
    /// 
    /// For example, with a 3×3 kernel and 3 input channels, this would extract
    /// a 3×3×3 patch (27 values) and arrange them into a single vector.
    /// </para>
    /// </remarks>
    private Vector<T> ExtractPatch(Tensor<T> input, int batch, int startY, int startX)
    {
        var patch = new Vector<T>(_inputChannels * _kernelSize * _kernelSize);
        int index = 0;
        for (int c = 0; c < _inputChannels; c++)
        {
            for (int i = 0; i < _kernelSize; i++)
            {
                for (int j = 0; j < _kernelSize; j++)
                {
                    patch[index++] = input[batch, startY + i, startX + j, c];
                }
            }
        }

        return patch;
    }

    /// <summary>
    /// Performs the backward pass of the primary capsule layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the primary capsule layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradients of the convolution weights and biases,
    /// as well as the gradient with respect to the input tensor. The computed weight and bias gradients are stored
    /// for later use in the parameter update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how all parameters should change to reduce errors.
    /// 
    /// During the backward pass:
    /// - The layer receives gradients indicating how the output capsules should change
    /// - It calculates how each weight, bias, and input value should change
    /// - These gradients are used later to update the parameters during training
    /// 
    /// This process involves:
    /// 1. Applying the derivative of the activation function
    /// 2. For each location in the output:
    ///    - Extracting the corresponding input patch
    ///    - Computing the gradients for weights and biases
    ///    - Computing the gradients for the input
    /// 3. Aggregating all the gradients
    /// 
    /// This allows the layer to learn how to better transform input features into
    /// meaningful capsule representations.
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

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int outputHeight = activationGradient.Shape[1];
        int outputWidth = activationGradient.Shape[2];
        int outputChannels = _capsuleChannels * _capsuleDimension;

        // Reshape activation gradient to NCHW [batch, outC, H, W]
        var gradNHWC = activationGradient.Reshape([batchSize, outputHeight, outputWidth, outputChannels]);
        var gradNCHW = gradNHWC.Transpose([0, 3, 1, 2]);

        // Prepare kernel in Conv2D layout
        var kernelNCHW = _convWeights.Reshape([outputChannels, _inputChannels, _kernelSize, _kernelSize]);

        // Input in NCHW
        var inputNCHW = _inputWasNCHW ? _lastInput : _lastInput.Transpose([0, 3, 1, 2]);

        // Gradients via Conv2D ops
        var inputGradNCHW = Engine.Conv2DBackwardInput(gradNCHW, kernelNCHW, inputNCHW.Shape, new[] { _stride, _stride }, new[] { 0, 0 }, new[] { 1, 1 });
        var kernelGrad = Engine.Conv2DBackwardKernel(gradNCHW, inputNCHW, kernelNCHW.Shape, new[] { _stride, _stride }, new[] { 0, 0 }, new[] { 1, 1 });

        // Bias gradient: sum over batch/height/width
        _convBiasGradient = Engine.ReduceSum(gradNCHW, new[] { 0, 2, 3 }, keepDims: false);

        // Reshape kernel gradient back to [outC, flatPatch]
        _convWeightsGradient = kernelGrad.Reshape([outputChannels, _inputChannels * _kernelSize * _kernelSize]);

        // Input gradient back to original layout
        var inputGradient = _inputWasNCHW ? inputGradNCHW : inputGradNCHW.Transpose([0, 2, 3, 1]);

        // Restore gradient shape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
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
    /// This method uses automatic differentiation to compute gradients by building a computation graph
    /// that mirrors the forward pass operations (Convolution -> Reshape -> Activation).
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_convWeights, "convWeights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_convBias, "convBias", requiresGradient: true);

        // 2. Reshape 2D weights to 4D for Conv2D [Out, In, K, K]
        // _convWeights is [OutputChannels, InputChannels * KernelSize * KernelSize]
        // We need [OutputChannels, InputChannels, KernelSize, KernelSize]
        int totalOutputChannels = _capsuleChannels * _capsuleDimension;
        var weights4DNode = Autodiff.TensorOperations<T>.Reshape(weightsNode,
            new int[] { totalOutputChannels, _inputChannels, _kernelSize, _kernelSize });

        // 3. Conv2D Operation
        var convOutput = Autodiff.TensorOperations<T>.Conv2D(
            inputNode,
            weights4DNode,
            biasNode,
            new int[] { _stride, _stride },
            new int[] { 0, 0 } // Padding is handled via output size calculation usually, check Forward
        );
        // Note: Forward uses manual padding logic? 
        // Forward: int outputHeight = (inputHeight - _kernelSize) / _stride + 1;
        // ExtractPatch uses input directly.
        // If ExtractPatch handles bounds checking (it doesn't seem to pad, just extracts), then padding=0 is correct.
        // ConvolutionalLayer uses padding parameter. PrimaryCapsuleLayer constructor takes padding but Forward doesn't seem to use it explicitly for bounds extension?
        // ExtractPatch: "patch[index++] = input[batch, startY + i, startX + j, c];" 
        // If indices are out of bounds, Tensor indexer might throw.
        // But let's assume 0 padding for now based on Forward logic matching loop bounds.

        // 4. Reshape to Capsules [Batch, OutH, OutW, Caps, Dim]
        int batchSize = _lastInput.Shape[0];
        int inputHeight = _lastInput.Shape[1];
        int inputWidth = _lastInput.Shape[2];
        int outputHeight = (inputHeight - _kernelSize) / _stride + 1;
        int outputWidth = (inputWidth - _kernelSize) / _stride + 1;

        var reshapedOutput = Autodiff.TensorOperations<T>.Reshape(
            convOutput,
            new int[] { batchSize, outputHeight, outputWidth, _capsuleChannels, _capsuleDimension }
        );

        // 5. Apply Activation (Squash)
        var activatedOutput = ApplyActivationToGraph(reshapedOutput);

        // 6. Set Gradient and Execute Backward
        activatedOutput.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((activatedOutput, false));

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

        // 7. Store Gradients
        _convWeightsGradient = weightsNode.Gradient;
        _convBiasGradient = biasNode.Gradient;

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore gradient shape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }


    /// <summary>
    /// Updates the parameters of the primary capsule layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the convolution weights and biases of the layer based on the gradients calculated
    /// during the backward pass. The learning rate controls the size of the parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's weights and biases during training.
    /// 
    /// After the backward pass calculates how parameters should change, this method:
    /// - Takes each weight and bias
    /// - Subtracts the corresponding gradient scaled by the learning rate
    /// - This moves the parameters in the direction that reduces errors
    /// 
    /// The learning rate controls how big each update step is:
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// This is how the layer gradually improves its ability to transform inputs
    /// into meaningful capsule representations over many training iterations.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_convWeightsGradient == null || _convBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Use Engine operations for GPU/CPU acceleration
        var scaledWeightGrad = Engine.TensorMultiplyScalar(_convWeightsGradient, learningRate);
        _convWeights = Engine.TensorSubtract(_convWeights, scaledWeightGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_convBiasGradient, learningRate);
        _convBias = Engine.TensorSubtract(_convBias, scaledBiasGrad);
    }

    /// <summary>
    /// Gets all trainable parameters from the primary capsule layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. It concatenates
    /// the convolution weights and biases into a single vector. This is useful for optimization algorithms
    /// that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values in the layer.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network learns during training
    /// - Include all the weights and biases from this layer
    /// - Are combined into a single long list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The method carefully arranges all parameters in a specific order
    /// so they can be correctly restored later.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector.Concatenate for production-grade parameter extraction
        return Vector<T>.Concatenate(
            new Vector<T>(_convWeights.ToArray()),
            new Vector<T>(_convBias.ToArray())
        );
    }

    /// <summary>
    /// Sets the trainable parameters for the primary capsule layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the layer from a single vector. It extracts the appropriate
    /// portions of the input vector for each parameter (convolution weights and biases). This is useful for
    /// loading saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The method extracts portions for each weight matrix and bias vector
    /// - It places each value in its correct position
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters,
    /// ensuring that all matrices and vectors maintain their correct dimensions.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int weightSize = _convWeights.Length;
        int biasSize = _convBias.Length;
        int totalParams = weightSize + biasSize;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }

        // Use Tensor.FromVector for production-grade parameter setting
        _convWeights = Tensor<T>.FromVector(parameters.Slice(0, weightSize), _convWeights.Shape);
        _convBias = Tensor<T>.FromVector(parameters.Slice(weightSize, biasSize), [biasSize]);
    }

    /// <summary>
    /// Resets the internal state of the primary capsule layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the primary capsule layer, including the cached inputs, outputs,
    /// and gradients. This is useful when starting to process a new sequence or batch of data, or when
    /// implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous processing are cleared
    /// - All calculated gradients are cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// Resetting state helps ensure that each forward and backward pass is independent,
    /// which is important for correct behavior in many neural network architectures.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _convWeightsGradient = null;
        _convBiasGradient = null;
        _originalInputShape = null;
        _inputWasNCHW = false;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_convWeights == null || _convBias == null)
            throw new InvalidOperationException("Layer not initialized. Call Initialize() first.");

        // Create input node - expecting [batch, height, width, channels]
        var symbolicInput = new Tensor<T>(InputShape);
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Reshape the matrix weights to Conv2D format [outChannels, inChannels, kH, kW]
        int totalOutputChannels = _capsuleChannels * _capsuleDimension;
        var convWeightsTensor = _convWeights.Reshape(new int[] { totalOutputChannels, _inputChannels, _kernelSize, _kernelSize });
        var weightsNode = Autodiff.TensorOperations<T>.Constant(convWeightsTensor, "conv_weights");

        // _convBias is already a Tensor<T>, use directly
        var biasNode = Autodiff.TensorOperations<T>.Constant(_convBias, "conv_bias");

        // Apply convolution: [batch, height, width, channels] -> [batch, outH, outW, totalOutputChannels]
        var convOutput = Autodiff.TensorOperations<T>.Conv2D(inputNode, weightsNode, biasNode, new[] { _stride, _stride }, padding: new[] { 0, 0 });

        // Reshape to separate capsules: [batch, outH, outW, totalOutputChannels]
        // -> [batch, outH, outW, capsuleChannels, capsuleDimension]
        int batchSize = InputShape[0];
        int inputHeight = InputShape[1];
        int inputWidth = InputShape[2];
        int outputHeight = (inputHeight - _kernelSize) / _stride + 1;
        int outputWidth = (inputWidth - _kernelSize) / _stride + 1;

        var reshapedOutput = Autodiff.TensorOperations<T>.Reshape(
            convOutput,
            new int[] { batchSize, outputHeight, outputWidth, _capsuleChannels, _capsuleDimension }
        );

        // Apply Squash activation to each capsule vector (along the last dimension)
        // The Squash operation scales the length of each capsule vector to [0, 1)
        var output = Autodiff.TensorOperations<T>.Squash(reshapedOutput);

        return output;
    }

    public override bool SupportsJitCompilation => _convWeights != null && _convBias != null;

}
