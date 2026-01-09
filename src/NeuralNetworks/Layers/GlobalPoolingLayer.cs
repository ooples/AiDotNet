using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a global pooling layer that reduces spatial dimensions to a single value per channel.
/// </summary>
/// <remarks>
/// <para>
/// A global pooling layer reduces the spatial dimensions (height and width) of the input feature maps
/// to a single value per channel. This is achieved by applying a pooling operation (such as max or average)
/// across the entire spatial extent of each channel. Global pooling is often used at the end of convolutional
/// neural networks to reduce the spatial dimensions before connecting to fully connected layers, providing
/// some translation invariance and reducing the number of parameters.
/// </para>
/// <para><b>For Beginners:</b> A global pooling layer summarizes each feature map into a single value.
/// 
/// Imagine you have a set of 2D feature maps (like heat maps showing where different features appear):
/// - Global pooling looks at each entire feature map
/// - It creates a single number that represents that entire feature map
/// - This dramatically reduces the amount of data while preserving the most important information
/// 
/// For example, with 64 feature maps of size 7×7:
/// - Input: 7×7—64 (3,136 values)
/// - Output: 1×1—64 (64 values, one per feature map)
/// 
/// There are two main types of global pooling:
/// - Global Max Pooling: Takes the maximum value from each feature map
///   (useful for detecting if a feature appears anywhere in the input)
/// - Global Average Pooling: Takes the average of all values in each feature map
///   (useful for determining the overall presence of a feature)
/// 
/// Global pooling is often used as the final layer before classification,
/// replacing large fully connected layers and reducing overfitting.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GlobalPoolingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The type of pooling operation to apply globally (Max or Average).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field determines the type of pooling operation to apply across the entire spatial dimensions
    /// of each feature map. Average pooling takes the mean of all values, while max pooling takes the
    /// maximum value.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how each feature map is summarized into a single value.
    /// 
    /// The two main types of global pooling are:
    /// 
    /// 1. Global Max Pooling (PoolingType.Max):
    ///    - Takes the highest value from each feature map
    ///    - Good for detecting if a specific feature appears anywhere in the input
    ///    - Less sensitive to small variations but can be more sensitive to noise
    ///    - Example: If a feature map detects edges, max pooling tells you "how strongly
    ///      the strongest edge was detected" anywhere in the image
    /// 
    /// 2. Global Average Pooling (PoolingType.Average):
    ///    - Takes the average of all values in each feature map
    ///    - Provides a smoother summary of the feature's presence
    ///    - More stable but might miss important localized features
    ///    - Example: If a feature map detects edges, average pooling tells you
    ///      "what the average edge strength was" across the entire image
    /// 
    /// Selecting the right pooling type depends on your specific task and data.
    /// </para>
    /// </remarks>
    private readonly PoolingType _poolingType;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary for computing
    /// gradients during the backward pass, particularly for max pooling which needs to know which
    /// input values were selected as the maximum.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember what input values it processed
    /// - For max pooling, it needs to know which value was the maximum
    /// - For average pooling, it needs the input dimensions
    /// - This helps when calculating how to distribute gradients during backpropagation
    /// 
    /// This is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// The output tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output produced during the last forward pass. It is used during
    /// backpropagation to compute gradients, particularly when an activation function is applied
    /// after pooling.
    /// </para>
    /// <para><b>For Beginners:</b> This stores what the layer output after its most recent calculation.
    /// 
    /// During training:
    /// - The network needs to remember what pooled values it produced
    /// - This helps calculate how to improve the network
    /// - It's especially important if an activation function was applied after pooling
    /// 
    /// This is also cleared after each training batch to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the indices of the maximum values found during global max pooling.
    /// </summary>
    private int[]? _maxIndices;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because global pooling layers have no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the global pooling layer doesn't have any trainable parameters.
    /// The layer simply performs a pooling operation without any weights or biases that need to be
    /// learned. However, it still participates in backpropagation by passing gradients back
    /// to the previous layer.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn or change during training.
    /// 
    /// A value of false means:
    /// - The layer has no weights or biases to adjust
    /// - It performs a fixed operation (pooling) that doesn't change
    /// - It's a transformation layer, not a learning layer
    /// 
    /// Unlike convolutional or fully connected layers which learn patterns from data,
    /// the global pooling layer just reduces spatial dimensions using a fixed operation.
    /// 
    /// It's like a data processing step rather than a learning step in the network.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="GlobalPoolingLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor (typically [batchSize, height, width, channels]).</param>
    /// <param name="poolingType">The type of pooling operation to apply (Max or Average).</param>
    /// <param name="activationFunction">The activation function to apply after pooling. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new global pooling layer with the specified input shape,
    /// pooling type, and activation function. The output shape is calculated to have the same
    /// batch size and number of channels as the input, but with spatial dimensions reduced to 1×1.
    /// The activation function operates on individual scalar values in the output tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the global pooling layer with your chosen settings.
    /// 
    /// When creating a global pooling layer, you need to specify:
    /// - Input shape: The dimensions of your input feature maps
    /// - Pooling type: Whether to use max or average pooling
    /// - Activation function: Any additional function to apply after pooling (optional)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a global average pooling layer for 28×28 feature maps with 64 channels
    /// var globalAvgPool = new GlobalPoolingLayer<float>(
    ///     new int[] { batchSize, 28, 28, 64 }, 
    ///     PoolingType.Average
    /// );
    /// 
    /// // Create a global max pooling layer with ReLU activation
    /// var globalMaxPool = new GlobalPoolingLayer<float>(
    ///     new int[] { batchSize, 14, 14, 128 }, 
    ///     PoolingType.Max,
    ///     new ReLUActivation<float>()
    /// );
    /// ```
    /// 
    /// The output will always have spatial dimensions of 1×1, preserving the batch size and number of channels.
    /// </para>
    /// </remarks>
    public GlobalPoolingLayer(int[] inputShape, PoolingType poolingType, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape), activationFunction ?? new IdentityActivation<T>())
    {
        _poolingType = poolingType;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GlobalPoolingLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor (typically [batchSize, height, width, channels]).</param>
    /// <param name="poolingType">The type of pooling operation to apply (Max or Average).</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after pooling (required to disambiguate from IActivationFunction overload).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new global pooling layer with the specified input shape,
    /// pooling type, and vector activation function. The output shape is calculated to have the same
    /// batch size and number of channels as the input, but with spatial dimensions reduced to 1×1.
    /// Unlike the other constructor, this one accepts a vector activation function that operates on
    /// entire vectors rather than individual scalar values.
    /// </para>
    /// <para><b>For Beginners:</b> This is an alternative setup that uses a different kind of activation function.
    ///
    /// This constructor is almost identical to the first one, but with one key difference:
    /// - Regular activation: processes each pooled value separately
    /// - Vector activation: processes groups of pooled values together
    ///
    /// Vector activation functions are useful when the relationship between
    /// different channels needs to be considered. For example, softmax might
    /// be applied across all channels to normalize them into a probability distribution.
    ///
    /// For most cases, the standard constructor with regular activation functions is sufficient.
    /// </para>
    /// </remarks>
    public GlobalPoolingLayer(int[] inputShape, PoolingType poolingType, IVectorActivationFunction<T> vectorActivationFunction)
        : base(inputShape, CalculateOutputShape(inputShape), vectorActivationFunction)
    {
        _poolingType = poolingType;
    }

    /// <summary>
    /// Calculates the output shape of the global pooling layer based on the input shape.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor (typically [batchSize, height, width, channels]).</param>
    /// <returns>The calculated output shape with spatial dimensions reduced to 1×1.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape for the global pooling layer. Global pooling reduces
    /// the spatial dimensions (height and width) to 1×1 while preserving the batch size and number of channels.
    /// </para>
    /// <para><b>For Beginners:</b> This determines what shape the output data will have after pooling.
    /// 
    /// Global pooling always:
    /// - Keeps the same batch size (number of examples)
    /// - Keeps the same number of channels (feature maps)
    /// - Reduces height and width to 1
    /// 
    /// For example:
    /// - Input shape: [32, 7, 7, 64] (32 examples, 7×7 spatial dimensions, 64 channels)
    /// - Output shape: [32, 1, 1, 64] (32 examples, 1×1 spatial dimensions, 64 channels)
    /// 
    /// This dramatic reduction in spatial dimensions helps prepare the feature maps for
    /// classification or other tasks that require a fixed-size vector input.
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape)
    {
        // Industry-standard: support tensors of any rank
        return inputShape.Length switch
        {
            // 4D CNN format: [batch, height, width, channels] -> [batch, 1, 1, channels]
            4 => [inputShape[0], 1, 1, inputShape[3]],

            // 3D sequence/transformer format: [batch, seq_len, features] -> [batch, features]
            // This is the critical fix for transformer classification pipelines
            3 => [inputShape[0], inputShape[2]],

            // 2D format: [batch, features] -> [batch, features] (nothing to pool)
            2 => [inputShape[0], inputShape[1]],

            // 1D format: [features] -> [1] (pool all features to single value)
            1 => [1],

            // 5D+ format: reduce all spatial dimensions except batch and last (channels/features)
            _ when inputShape.Length > 4 => CreateHighDimensionalOutputShape(inputShape),

            // 0D (scalar): return as-is
            _ => inputShape
        };
    }

    /// <summary>
    /// Creates output shape for tensors with more than 4 dimensions.
    /// </summary>
    private static int[] CreateHighDimensionalOutputShape(int[] inputShape)
    {
        var result = new int[inputShape.Length];
        result[0] = inputShape[0]; // Keep batch dimension
        for (int i = 1; i < inputShape.Length - 1; i++)
        {
            result[i] = 1; // Reduce all spatial dimensions to 1
        }
        result[^1] = inputShape[^1]; // Keep channel/feature dimension
        return result;
    }

    /// <summary>
    /// Performs the forward pass of the global pooling layer.
    /// </summary>
    /// <param name="input">The input tensor to process. Shape: [batchSize, height, width, channels].</param>
    /// <returns>The output tensor after global pooling. Shape: [batchSize, 1, 1, channels].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the global pooling layer. For each channel in each example,
    /// it applies the specified pooling operation (max or average) across the entire spatial dimensions.
    /// For max pooling, it finds the maximum value in each channel. For average pooling, it computes the
    /// mean of all values in each channel. The result is a tensor with the same batch size and number of
    /// channels, but with spatial dimensions reduced to 1×1.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer processes input data by pooling across entire feature maps.
    /// 
    /// The forward pass works in these steps:
    /// 1. For each example in the batch and each channel:
    ///    - If using average pooling: Calculate the average of all values in the feature map
    ///    - If using max pooling: Find the maximum value in the feature map
    /// 2. Store these pooled values in the output tensor
    /// 3. Apply the activation function (if specified)
    /// 4. Save the input and output for use during backpropagation
    /// 
    /// This global pooling operation efficiently summarizes each feature map
    /// into a single value, drastically reducing the data dimensions while
    /// preserving the most important information about each feature.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Industry-standard: dynamically determine axes based on input rank
        // For any rank, reduce all dimensions except first (batch) and last (features/channels)
        var axes = GetReductionAxes(input.Shape.Length);

        Tensor<T> output;
        if (axes.Length == 0)
        {
            // No dimensions to reduce (1D or 2D input)
            output = input;
            _maxIndices = null;
        }
        else if (_poolingType == PoolingType.Average)
        {
            // Use GPU-accelerated ReduceMean
            // keepDims=true for rank >= 4 to preserve dimension structure for subsequent reshape
            output = Engine.ReduceMean(input, axes, keepDims: input.Shape.Length >= 4);
            _maxIndices = null;
        }
        else // Max pooling
        {
            // Use GPU-accelerated ReduceMax
            // keepDims=true for rank >= 4 to preserve dimension structure for subsequent reshape
            output = Engine.ReduceMax(input, axes, keepDims: input.Shape.Length >= 4, out _maxIndices);
        }

        var result = ApplyActivation(output);

        // Only store for backward pass during training - skip during inference
        if (IsTrainingMode)
        {
            _lastOutput = result;
        }

        return result;
    }

    /// <summary>
    /// Gets the axes to reduce over based on input tensor rank.
    /// </summary>
    private static int[] GetReductionAxes(int rank)
    {
        return rank switch
        {
            // 4D: [batch, height, width, channels] -> reduce height (1) and width (2)
            4 => [1, 2],
            // 3D: [batch, seq_len, features] -> reduce seq_len (1)
            3 => [1],
            // 2D: [batch, features] -> nothing to reduce
            2 => [],
            // 1D: [features] -> reduce all to single value
            1 => [0],
            // 5D+: reduce all middle dimensions
            _ when rank > 4 => Enumerable.Range(1, rank - 2).ToArray(),
            // 0D: nothing to reduce
            _ => []
        };
    }

    /// <summary>
    /// Performs the backward pass of the global pooling layer to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, 1, 1, channels].</param>
    /// <returns>The gradient tensor to be passed to the previous layer. Shape: [batchSize, height, width, channels].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the global pooling layer.
    /// For average pooling, the gradient is distributed equally among all positions in the input that
    /// contributed to the average. For max pooling, the gradient is assigned only to the position that
    /// had the maximum value in the forward pass. This reflects how each position in the input
    /// contributed to the output during the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer passes error information back to previous layers.
    ///
    /// The backward pass works differently depending on the pooling type:
    ///
    /// For average pooling:
    /// - The gradient for each output value is divided equally among all input positions
    /// - Every position in a feature map gets the same small portion of the gradient
    /// - This reflects that each input position contributed equally to the average
    ///
    /// For max pooling:
    /// - The gradient for each output value is assigned only to the input position that had the maximum value
    /// - Only the "winning" position gets the gradient, all others get zero
    /// - This reflects that only the maximum value contributed to the output
    ///
    /// This process ensures that the network learns appropriately based on how
    /// each input position influenced the pooled output.
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
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, 1, 1, channels].</param>
    /// <returns>The gradient tensor to be passed to the previous layer. Shape: [batchSize, height, width, channels].</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Apply activation derivative
        outputGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // Industry-standard: dynamically determine axes based on input rank
        var axes = GetReductionAxes(_lastInput.Shape.Length);

        // If no axes to reduce, just return the gradient as-is
        if (axes.Length == 0)
        {
            return outputGradient;
        }

        if (_poolingType == PoolingType.Average)
        {
            // Use GPU-accelerated ReduceMeanBackward
            return Engine.ReduceMeanBackward(outputGradient, _lastInput.Shape, axes);
        }
        else // Max pooling
        {
            if (_maxIndices == null)
                throw new InvalidOperationException("Max indices not available for backward pass.");

            // Use GPU-accelerated ReduceMaxBackward
            return Engine.ReduceMaxBackward(outputGradient, _maxIndices, _lastInput.Shape);
        }
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient tensor from the next layer. Shape: [batchSize, 1, 1, channels].</param>
    /// <returns>The gradient tensor to be passed to the previous layer. Shape: [batchSize, height, width, channels].</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Currently, global pooling operations
    /// are not yet available in TensorOperations, so this method falls back to the manual implementation.
    /// </para>
    /// <para>
    /// Once global pooling operations are added to TensorOperations, this method will provide:
    /// - Automatic gradient computation through the computation graph
    /// - Verification of manual gradient implementations
    /// - Support for rapid prototyping with custom modifications
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // If vector activation is configured, fall back to manual path
        if (VectorActivation != null)
        {
            return BackwardManual(outputGradient);
        }

        // Convert input to computation node
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Apply global pooling using reduce operations
        // Global pooling reduces over spatial dimensions (all non-batch, non-channel axes), keeping channels.
        // The reduction axes are determined dynamically from the input rank. For a 4D NHWC tensor
        // [batch, height, width, channels], this corresponds to reducing over height (1) and width (2).
        var axes = GetReductionAxes(_lastInput.Shape.Length); // Industry-standard: dynamic axes based on input rank

        // If no axes to reduce (1D or 2D input), just return the gradient as-is
        if (axes.Length == 0)
        {
            return outputGradient;
        }

        Autodiff.ComputationNode<T> outputNode;
        if (_poolingType == PoolingType.Max)
        {
            // keepDims=true for rank >= 4 to preserve dimension structure for subsequent reshape
            outputNode = Autodiff.TensorOperations<T>.ReduceMax(inputNode, axes, keepDims: _lastInput.Shape.Length >= 4);
        }
        else // Average pooling
        {
            // keepDims=true for rank >= 4 to preserve dimension structure for subsequent reshape
            outputNode = Autodiff.TensorOperations<T>.ReduceMean(inputNode, axes, keepDims: _lastInput.Shape.Length >= 4);
        }

        // Reshape the reduction result to match this layer's OutputShape (may add/remove dimensions or just validate shape)
        var squeezed = Autodiff.TensorOperations<T>.Reshape(outputNode, OutputShape);

        // Apply activation if present
        var activated = ApplyScalarActivationAutodiff(squeezed);

        // Perform backward pass with inline topological sort
        activated.Gradient = outputGradient;

        // Production-grade: Inline topological sort for backward pass
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((activated, false));

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
    /// Applies scalar activation function with autodiff support.
    /// </summary>
    /// <param name="input">The input computation node.</param>
    /// <returns>The activated computation node, or the input unchanged if no scalar activation is configured.</returns>
    private Autodiff.ComputationNode<T> ApplyScalarActivationAutodiff(Autodiff.ComputationNode<T> input)
    {
        if (ScalarActivation == null)
            return input;

        // Use generic activation support - works for ALL 39 built-in activations
        return Autodiff.TensorOperations<T>.ApplyActivation(input, ScalarActivation);
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <value><c>true</c> if GPU execution is supported; otherwise, <c>false</c>.</value>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Performs GPU-accelerated forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>The GPU-resident output tensor after global pooling.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine");

        var input = inputs[0];

        IGpuTensor<T> output;
        if (_poolingType == PoolingType.Average)
        {
            output = gpuEngine.GlobalMeanPoolGpu(input);
            _maxIndices = null;
        }
        else // Max pooling
        {
            output = gpuEngine.GlobalMaxPoolGpu(input, out _maxIndices);
        }

        // Apply activation on GPU if non-trivial activation exists
        var fusedType = MapActivationToFused();
        if (fusedType != FusedActivationType.None)
        {
            output = gpuEngine.ActivationGpu(output, fusedType);
        }

        // Cache state for backward pass only during training
        // Skip this expensive download during inference (50% overhead reduction)
        if (IsTrainingMode)
        {
            _lastInput = input.ToTensor();
            _lastOutput = output.ToTensor();
        }

        return output;
    }

    /// <inheritdoc/>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");
        }

        var backend = gpuEngine.GetBackend();
        if (backend is null)
        {
            throw new InvalidOperationException("GPU backend unavailable.");
        }

        // CPU fallback: download gradient, compute via CPU Backward, upload result
        var outputGradCpu = outputGradient.ToTensor();
        var inputGradCpu = Backward(outputGradCpu);

        // Return the input gradient as GPU tensor
        return new GpuTensor<T>(backend, inputGradCpu, GpuTensorRole.Gradient);
    }

    /// <summary>
    /// Updates the parameters of the layer based on the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the global pooling layer has no
    /// trainable parameters to update, so it performs no operation.
    /// </para>
    /// <para><b>For Beginners:</b> This method does nothing because pooling layers have no adjustable weights.
    /// 
    /// Unlike layers like convolutional or fully connected layers:
    /// - Global pooling layers don't have weights or biases to learn
    /// - They perform a fixed operation (finding maximum or average values)
    /// - There's nothing to update during training
    /// 
    /// This method exists only to fulfill the requirements of the base layer class.
    /// The pooling layer influences the network by reducing dimensions and providing
    /// translation invariance, not by learning parameters.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a pooling layer
    }

    /// <summary>
    /// Gets the trainable parameters of the layer.
    /// </summary>
    /// <returns>
    /// An empty vector since global pooling layers have no trainable parameters.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method is a required override from the base class, but the global pooling layer has no
    /// trainable parameters to retrieve, so it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because pooling layers have no learnable values.
    /// 
    /// Unlike layers with weights and biases:
    /// - Global pooling layers don't have any parameters that change during training
    /// - They perform a fixed operation (pooling) that doesn't involve learning
    /// - There are no values to save when storing a trained model
    /// 
    /// This method returns an empty vector, indicating there are no parameters to collect.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // GlobalPoolingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input and output
    /// tensors from the previous forward pass. This is useful when starting to process a new batch of data
    /// or when switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input and output tensors are cleared
    /// - This frees up memory and prepares for new data
    /// 
    /// This is typically called:
    /// - Between training batches
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _maxIndices = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Global pooling can be implemented as regular pooling with pool size = spatial dimensions
        // InputShape for CNN: [channels, height, width]
        if (InputShape.Length >= 3)
        {
            int height = InputShape[1];
            int width = InputShape[2];
            var poolSize = new int[] { height, width };
            var strides = new int[] { 1, 1 };

            if (_poolingType == PoolingType.Max)
            {
                return TensorOperations<T>.MaxPool2D(inputNode, poolSize: poolSize, strides: strides);
            }
            else // Average
            {
                return TensorOperations<T>.AvgPool2D(inputNode, poolSize: poolSize, strides: strides);
            }
        }

        // Fallback for other shapes - return identity for now
        return inputNode;
    }

    public override bool SupportsJitCompilation
    {
        get
        {
            return InputShape != null && InputShape.Length > 0;
        }
    }
}
