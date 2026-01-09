using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that splits the input tensor along a specific dimension into multiple equal parts.
/// </summary>
/// <remarks>
/// <para>
/// A split layer divides the input tensor into multiple equal parts along a specified dimension. This is useful
/// for parallel processing of data or for implementing multi-headed attention mechanisms. The layer ensures that 
/// the input size is divisible by the number of splits to maintain consistency.
/// </para>
/// <para><b>For Beginners:</b> This layer breaks up your input data into smaller, equal-sized chunks.
/// 
/// Think of it like cutting a pizza into equal slices:
/// - Your input data is the whole pizza
/// - The number of splits determines how many slices you want
/// - Each slice has the same size and shape
/// 
/// Benefits include:
/// - Processing different parts of the input in parallel
/// - Allowing different operations on different parts of the input
/// - Creating multi-stream architectures where each stream handles a portion of the data
/// 
/// For example, in natural language processing, you might split word embeddings to create
/// multiple "attention heads" that each focus on different aspects of the text.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SplitLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The number of parts to split the input tensor into.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies how many equal parts the input tensor will be divided into. The input size must be
    /// divisible by this number to ensure all splits have the same size.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many equal pieces the input will be cut into.
    /// 
    /// For example:
    /// - If numSplits is 2, the input is cut in half
    /// - If numSplits is 4, the input is cut into quarters
    /// 
    /// The layer will check that the input can be divided equally by this number
    /// without any remainder to ensure all pieces are exactly the same size.
    /// </para>
    /// </remarks>
    private readonly int _numSplits;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    /// 
    /// During training, the layer needs to remember what input it processed so that it can
    /// properly calculate how to improve. This temporary storage is cleared between batches
    /// or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> as split layers can propagate gradients.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the split layer can participate in the training process by propagating gradients.
    /// Although the layer has no trainable parameters itself, it can pass gradients back to previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that the layer can be used during training.
    /// 
    /// Even though this layer doesn't have any parameters that need to be adjusted:
    /// - It can still pass error information backward to previous layers during training
    /// - It participates in the backpropagation process
    /// 
    /// This allows the layer to be included in networks that learn from data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// SplitLayer uses GPU Reshape operations.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SplitLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="numSplits">The number of parts to split the input tensor into.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a split layer with the specified input shape and number of splits. It verifies
    /// that the input size is divisible by the number of splits to ensure all splits have the same size.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new layer that will divide the input into equal parts.
    /// 
    /// When creating a split layer, you need to specify:
    /// - inputShape: The dimensions of the data going into the layer
    /// - numSplits: How many equal pieces to divide the input into
    /// 
    /// The constructor checks that the input can be divided equally by the number of splits.
    /// For example, if your input has 100 features and you want 4 splits, that works (100 ÷ 4 = 25).
    /// But if your input has 100 features and you want 3 splits, that won't work
    /// because you'd get splits of size 33.33... which isn't a whole number.
    /// </para>
    /// </remarks>
    public SplitLayer(int[] inputShape, int numSplits)
        : base(inputShape, CalculateOutputShape(inputShape, numSplits))
    {
        _numSplits = numSplits;
    }

    /// <summary>
    /// Calculates the output shape of the split layer based on input shape and number of splits.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="numSplits">The number of parts to split the input tensor into.</param>
    /// <returns>The calculated output shape for the split layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the input size is not divisible by the number of splits.</exception>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape of the split layer based on the input shape and the number of splits.
    /// It verifies that the input size is divisible by the number of splits and throws an exception if it's not.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data that will come out of this layer.
    /// 
    /// It also performs an important check:
    /// - It verifies that the input can be divided equally by the number of splits
    /// - If not, it throws an error to prevent problems later
    /// 
    /// The output shape will have one more dimension than the input, with the new dimension
    /// representing the different splits, and another dimension representing the size of each split.
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int numSplits)
    {
        if (inputShape[0] % numSplits != 0)
        {
            throw new ArgumentException("Input size must be divisible by the number of splits");
        }

        return [inputShape[0] / numSplits];
    }

    /// <summary>
    /// Performs the forward pass of the split layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after splitting.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the split layer. It divides the input tensor into multiple
    /// equal-sized parts along the specified dimension and returns a new tensor containing all the splits.
    /// </para>
    /// <para><b>For Beginners:</b> This method does the actual work of splitting the input data.
    /// 
    /// During the forward pass:
    /// 1. The input is saved for later use in training
    /// 2. The method calculates how big each split should be
    /// 3. It creates a new tensor with an additional dimension to hold all the splits
    /// 4. It copies the data from the input into the appropriate positions in the output
    /// 
    /// After splitting, the data will have a new dimension that indicates which split each piece belongs to.
    /// For example, if you split a batch of 10 samples with 100 features into 5 splits, you'll get
    /// an output with shape [10, 5, 20], where 20 is the size of each split.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: add batch dim
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // Standard 2D
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 1]]);
        }

        _lastInput = processInput;
        int inputSize = processInput.Shape[1];
        int splitSize = inputSize / _numSplits;
        var output = Engine.Reshape(processInput, new[] { batchSize, _numSplits, splitSize });

        // Restore output shape to match original input rank
        if (_originalInputShape != null && _originalInputShape.Length != 2)
        {
            if (_originalInputShape.Length == 1)
            {
                // 1D input: output shape becomes [numSplits, splitSize]
                return output.Reshape([_numSplits, splitSize]);
            }
            else
            {
                // Higher-rank: reconstruct leading dims with split dimensions
                var outShape = new int[_originalInputShape.Length + 1];
                for (int d = 0; d < _originalInputShape.Length - 1; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 1] = _numSplits;
                outShape[_originalInputShape.Length] = splitSize;
                return output.Reshape(outShape);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensors.</param>
    /// <returns>A GPU-resident output tensor after splitting.</returns>
    /// <remarks>
    /// <para>
    /// SplitLayer is implemented as a reshape operation that stays entirely GPU-resident.
    /// No data is downloaded to CPU during inference.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;
        int rank = shape.Length;

        // Determine batch size and input size
        int batchSize;
        int inputSize;
        IGpuTensor<T> processInput;

        if (rank == 1)
        {
            batchSize = 1;
            inputSize = shape[0];
            processInput = gpuEngine.ReshapeGpu(input, new[] { 1, inputSize });
        }
        else if (rank == 2)
        {
            batchSize = shape[0];
            inputSize = shape[1];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            batchSize = 1;
            for (int d = 0; d < rank - 1; d++)
                batchSize *= shape[d];
            inputSize = shape[rank - 1];
            processInput = gpuEngine.ReshapeGpu(input, new[] { batchSize, inputSize });
        }

        // Cache for backward pass during training
        if (IsTrainingMode)
        {
            _lastInput = processInput.ToTensor();
            _originalInputShape = shape;
        }

        int splitSize = inputSize / _numSplits;

        // Reshape to [batch, numSplits, splitSize]
        var output = gpuEngine.ReshapeGpu(processInput, new[] { batchSize, _numSplits, splitSize });

        // Restore output shape to match original input rank
        if (rank == 1)
        {
            return gpuEngine.ReshapeGpu(output, new[] { _numSplits, splitSize });
        }
        else if (rank > 2)
        {
            var outShape = new int[rank + 1];
            for (int d = 0; d < rank - 1; d++)
                outShape[d] = shape[d];
            outShape[rank - 1] = _numSplits;
            outShape[rank] = splitSize;
            return gpuEngine.ReshapeGpu(output, outShape);
        }

        return output;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// Since split is a reshape operation, backward is also just a reshape to restore the original shape.
    /// </remarks>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_originalInputShape == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Split is a reshape, so backward is just reshaping gradient to original input shape
        return gpuEngine.ReshapeGpu(outputGradient, _originalInputShape);
    }

    /// <summary>
    /// Performs the backward pass of the split layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the split layer, which is used during training to propagate
    /// error gradients back through the network. It recombines the gradients from all splits into a single
    /// gradient tensor matching the original input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This method reverses the splitting process for training.
    /// 
    /// During the backward pass:
    /// 1. The method throws an error if the forward pass hasn't been called first
    /// 2. It calculates how big each split is
    /// 3. It creates a gradient tensor matching the original input shape
    /// 4. It copies the gradient values from each split back to their original positions
    /// 
    /// This process ensures that error information flows backward through the network properly,
    /// allowing layers before the split to learn from the training process.
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
        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        var inputGradient = Engine.Reshape(outputGradient, new[] { batchSize, inputSize });

        // Restore gradient to original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 2)
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
    /// This method uses automatic differentiation to compute gradients using the Reshape operation.
    /// The split layer is effectively a reshape operation that adds a new dimension by dividing
    /// one dimension into two.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Create computation node
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);

        // Split is effectively a reshape: [batch, inputSize] -> [batch, numSplits, splitSize]
        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        int splitSize = inputSize / _numSplits;
        var outputShape = new int[] { batchSize, _numSplits, splitSize };

        var outputNode = Autodiff.TensorOperations<T>.Reshape(inputNode, outputShape);

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

        // Extract input gradient
        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore gradient to original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 2)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is a no-op for the split layer since it has no trainable parameters to update.
    /// It is implemented to satisfy the interface requirements of LayerBase.
    /// </para>
    /// <para><b>For Beginners:</b> This method doesn't do anything in the split layer.
    /// 
    /// Since the split layer doesn't have any trainable parameters:
    /// - There's nothing to update during training
    /// - This method exists just to fulfill the requirements of being a layer
    /// 
    /// Other layers would use this method to update their weights and biases,
    /// but the split layer simply passes data through without modification.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since this layer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector since the split layer has no trainable parameters.
    /// It is implemented to satisfy the interface requirements of LayerBase.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because the layer has no parameters.
    /// 
    /// Since the split layer doesn't modify the data in any way that requires learning:
    /// - There are no weights or biases to adjust
    /// - This method returns an empty vector (a list with no elements)
    /// 
    /// Other layers would return their weights and biases here, which would be
    /// used for saving the model or applying optimization techniques.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // SplitLayer has no trainable parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the split layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the split layer, clearing the cached input.
    /// This is useful when starting to process a new batch or when implementing stateful networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The stored input from the previous forward pass is cleared
    /// 
    /// This is important for:
    /// - Processing a new batch of unrelated data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like clearing your workspace before starting a new project -
    /// it ensures that old information doesn't interfere with new processing.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
    }

    /// <summary>
    /// Exports the split layer as a computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which the input node will be added.</param>
    /// <returns>The output computation node representing the split operation.</returns>
    /// <remarks>
    /// <para>
    /// The split layer is implemented as a reshape operation that adds a new dimension.
    /// Input shape [batch, inputSize] is reshaped to [batch, numSplits, splitSize].
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Input shape: [batch, inputSize]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "split_input");
        inputNodes.Add(inputNode);

        // Split is implemented as a reshape: [batch, inputSize] → [batch, numSplits, splitSize]
        // This matches the Forward() implementation which creates a tensor with shape [batchSize, _numSplits, splitSize]
        int inputSize = InputShape[0];
        int splitSize = inputSize / _numSplits;
        var outputShape = new int[] { 1, _numSplits, splitSize };

        return TensorOperations<T>.Reshape(inputNode, outputShape);
    }

    public override bool SupportsJitCompilation => true;
}
