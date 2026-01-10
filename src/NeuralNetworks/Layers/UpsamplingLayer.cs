using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an upsampling layer that increases the spatial dimensions of input tensors using nearest-neighbor interpolation.
/// </summary>
/// <remarks>
/// <para>
/// An upsampling layer increases the spatial dimensions (height and width) of input tensors by repeating values from
/// the input to create a larger output. This implementation uses nearest-neighbor interpolation, which repeats each
/// value in the input tensor multiple times based on the scale factor to create the upsampled output.
/// </para>
/// <para><b>For Beginners:</b> This layer makes images or feature maps larger by simply repeating pixels.
/// 
/// Think of it like zooming in on a digital image:
/// - When you zoom in on a pixelated image, each original pixel becomes a larger square
/// - This layer does the same thing to feature maps inside the neural network
/// - It's like stretching an image without adding any new information
/// 
/// For example, with a scale factor of 2:
/// - A 4Ã—4 image becomes an 8Ã—8 image
/// - Each pixel in the original image is copied to a 2Ã—2 block in the output
/// - This creates a larger image that preserves the original content but with more pixels
/// 
/// This is useful for tasks like image generation or upscaling, where you need to increase
/// the resolution of features that the network has processed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class UpsamplingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The factor by which to increase spatial dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the scale factor used to increase the spatial dimensions (height and width) of the input.
    /// A value of 2 means the output height and width will be twice the input dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how much larger the output will be compared to the input.
    /// 
    /// For example:
    /// - With a scale factor of 2: A 10Ã—10 image becomes 20Ã—20
    /// - With a scale factor of 3: A 10Ã—10 image becomes 30Ã—30
    /// - With a scale factor of 4: A 10Ã—10 image becomes 40Ã—40
    /// 
    /// The scale factor applies equally to both height and width, so the total number of pixels
    /// increases by the square of the scale factor (e.g., a scale factor of 2 means 4 times more pixels).
    /// </para>
    /// </remarks>
    private readonly int _scaleFactor;

    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass, which is needed during the backward
    /// pass to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what it last processed.
    ///
    /// Storing the input is necessary because:
    /// - During training, the layer needs to remember what input it processed
    /// - This helps calculate the correct gradients during the backward pass
    /// - It's part of the layer's "working memory" for the learning process
    ///
    /// This cached input helps the layer understand how to adjust the network's behavior
    /// to improve its performance on future inputs.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached input shape from GPU forward pass for backward pass.
    /// </summary>
    private int[]? _gpuCachedInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, even though it has no trainable parameters, to allow gradient propagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the upsampling layer can be included in the training process. Although this layer
    /// does not have trainable parameters, it returns true to allow gradient propagation through the layer during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer participates in the learning process.
    /// 
    /// A value of true means:
    /// - The layer is part of the training process
    /// - It can pass gradients backward to previous layers
    /// - It helps the network learn, even though it doesn't have its own parameters to adjust
    /// 
    /// This is like being a messenger that relays feedback to earlier parts of the network,
    /// even though the messenger doesn't change its own behavior.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="UpsamplingLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="scaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an upsampling layer with the specified input shape and scale factor. The output shape
    /// is calculated based on the input shape and scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new upsampling layer.
    /// 
    /// The parameters you provide determine:
    /// - inputShape: The dimensions of the data coming into this layer
    /// - scaleFactor: How much larger the output should be compared to the input
    /// 
    /// For example, if inputShape is [3, 32, 32] (representing 3 channels of a 32Ã—32 image)
    /// and scaleFactor is 2, the output shape will be [3, 64, 64] - the same number of
    /// channels but twice the height and width.
    /// </para>
    /// </remarks>
    public UpsamplingLayer(int[] inputShape, int scaleFactor)
        : base(inputShape, CalculateOutputShape(inputShape, scaleFactor))
    {
        _scaleFactor = scaleFactor;
    }

    /// <summary>
    /// Calculates the output shape based on input shape and scale factor.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="scaleFactor">The factor by which to increase spatial dimensions.</param>
    /// <returns>The calculated output shape.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the upsampling layer by multiplying the height and width dimensions
    /// of the input shape by the scale factor, while keeping the number of channels the same.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of data that will come out of this layer.
    /// 
    /// It works by:
    /// - Taking the input shape
    /// - Keeping the channel dimension (first element) the same
    /// - Multiplying the height and width (second and third elements) by the scale factor
    /// 
    /// For example, if the input shape is [16, 20, 30] (16 channels, 20 height, 30 width)
    /// and the scale factor is 2, the output shape will be [16, 40, 60].
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int scaleFactor)
    {
        // Industry-standard: support tensors of any rank
        // The last two dimensions are always height and width for upsampling
        // Supports: 2D [H, W], 3D [C, H, W], 4D [B, C, H, W], etc.
        if (inputShape.Length < 2)
            throw new ArgumentException("Input shape must have at least 2 dimensions for upsampling.");

        var outputShape = new int[inputShape.Length];

        // Copy all dimensions except the last two
        for (int i = 0; i < inputShape.Length - 2; i++)
        {
            outputShape[i] = inputShape[i];
        }

        // Scale the last two dimensions (height and width)
        int heightIdx = inputShape.Length - 2;
        int widthIdx = inputShape.Length - 1;
        outputShape[heightIdx] = inputShape[heightIdx] * scaleFactor;
        outputShape[widthIdx] = inputShape[widthIdx] * scaleFactor;

        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the upsampling layer.
    /// </summary>
    /// <param name="input">The input tensor to upsample.</param>
    /// <returns>The upsampled output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the upsampling layer using nearest-neighbor interpolation. It repeats
    /// each value in the input tensor according to the scale factor to create a larger output tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a larger version of the input by repeating values.
    /// 
    /// During the forward pass:
    /// 1. The layer receives an input tensor (like a stack of feature maps)
    /// 2. For each value in the input:
    ///    - The value is copied multiple times based on the scale factor
    ///    - These copies form a block in the output tensor
    /// 3. This creates an output that is larger but contains the same information
    /// 
    /// For example, with a scale factor of 2, each pixel becomes a 2Ã—2 block of identical pixels.
    /// This is the simplest form of upsampling, which preserves the original content
    /// but increases the spatial dimensions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        return Engine.Upsample(input, _scaleFactor, _scaleFactor);
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after upsampling.</returns>
    /// <exception cref="ArgumentException">Thrown when no input tensor is provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when GPU backend is unavailable.</exception>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        // Cache input shape for backward pass during training
        if (IsTrainingMode)
        {
            _gpuCachedInputShape = (int[])input.Shape.Clone();
        }

        return gpuEngine.UpsampleGpu(input, _scaleFactor);
    }

    /// <summary>
    /// Performs the backward pass on GPU tensors.
    /// </summary>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when GPU backend is unavailable or forward pass was not called.
    /// </exception>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> gradOutput)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        if (_gpuCachedInputShape == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Get original input dimensions
        int inputHeight = _gpuCachedInputShape[^2];
        int inputWidth = _gpuCachedInputShape[^1];

        return gpuEngine.UpsampleBackwardGpu(gradOutput, inputHeight, inputWidth, _scaleFactor);
    }

    /// <summary>
    /// Performs the backward pass of the upsampling layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to perform a backward pass before a forward pass.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the upsampling layer, which is used during training to propagate
    /// error gradients back through the network. For each position in the input gradient, it sums up the corresponding
    /// gradients from the scale factor Ã— scale factor region in the output gradient.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the layer's input should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The layer receives information about how its output should change (outputGradient)
    /// 2. For each position in the input:
    ///    - It finds all the positions in the output that came from this input position
    ///    - It sums up the gradients from all those output positions
    ///    - This sum becomes the gradient for the input position
    /// 
    /// This is like collecting feedback from all the copies made during the forward pass
    /// and combining it into a single piece of feedback for the original value.
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
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        return Engine.UpsampleBackward(outputGradient, _lastInput.Shape, _scaleFactor, _scaleFactor);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Upsample TensorOperation for automatic gradient computation.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Convert input to computation node
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Apply upsampling operation
        var outputNode = Autodiff.TensorOperations<T>.Upsample(inputNode, _scaleFactor);

        // Perform backward pass with inline topological sort
        outputNode.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
                continue;

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

        // Execute backward pass in reverse topological order
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
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is empty as the upsampling layer does not have any trainable parameters to update.
    /// It is included to conform to the base class interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing because this layer has no learnable values.
    /// 
    /// The upsampling layer:
    /// - Performs a fixed, predefined operation (repeating values)
    /// - Has no weights or biases to adjust during training
    /// - Only passes gradients backward without changing itself
    /// 
    /// This method exists only to satisfy the requirements of the base layer class,
    /// similar to how a purely functional node in a network would need to implement
    /// this method even though it has nothing to update.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector, as the layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector as the upsampling layer does not have any trainable parameters.
    /// It is included to conform to the base class interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because this layer has no learnable values.
    /// 
    /// Since the upsampling layer:
    /// - Has no weights or biases
    /// - Performs a fixed operation that doesn't need to be learned
    /// - Only transforms the input according to predefined rules
    /// 
    /// It returns an empty vector, indicating to the optimization process that
    /// there are no parameters to update for this layer.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // This layer doesn't have any trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Exports this layer's computation as a differentiable computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which input variable nodes should be added.</param>
    /// <returns>The output computation node representing this layer's operation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representation of the upsampling operation using nearest-neighbor
    /// interpolation. The operation repeats each value in the input based on the scale factor.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an optimized version of the upsampling operation.
    ///
    /// For upsampling layers:
    /// - Creates a placeholder for the input tensor
    /// - Applies the upsampling operation (repeat values)
    /// - Returns a computation graph for efficient execution
    ///
    /// This allows for faster inference by pre-compiling the upsampling operation.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create placeholder for input tensor
        // Input shape: [channels, height, width]
        var inputPlaceholder = new Tensor<T>(InputShape);
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "input");
        inputNodes.Add(inputNode);

        // Apply upsampling operation
        var outputNode = TensorOperations<T>.Upsample(inputNode, _scaleFactor);

        // Upsampling layers typically don't use activation, but we return the result
        // No activation to apply for upsampling layers (they use identity by default)
        return outputNode;
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <value>Always returns true as upsampling operations can be efficiently compiled.</value>
    /// <remarks>
    /// <para>
    /// Upsampling layers support JIT compilation since the nearest-neighbor interpolation
    /// is a straightforward operation that can be optimized at compile time.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the upsampling layer by clearing the cached input tensor.
    /// This is useful when starting to process a new, unrelated input.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of what it last processed.
    ///
    /// When resetting the state:
    /// - The layer forgets what input it recently processed
    /// - This helps prepare it for processing new, unrelated inputs
    /// - It's like clearing a workspace before starting a new project
    ///
    /// This is mostly important during training, where the layer needs to
    /// maintain consistency between forward and backward passes.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear the cached input
        _lastInput = null;
        _gpuCachedInputShape = null;
    }
}
