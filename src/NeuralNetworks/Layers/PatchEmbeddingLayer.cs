using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a patch embedding layer for Vision Transformer (ViT) architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// The patch embedding layer divides an input image into fixed-size patches and projects them into an embedding space.
/// This is a key component of Vision Transformers, converting 2D spatial information into a sequence of embeddings
/// that can be processed by transformer encoder blocks.
/// </para>
/// <para>
/// <b>For Beginners:</b> This layer breaks an image into small square pieces (patches) and converts each patch
/// into a numerical representation that can be processed by a transformer.
///
/// Think of it like cutting a photo into a grid of smaller squares, then describing each square with numbers.
/// For example, a 224x224 image with 16x16 patches would be cut into 196 patches (14x14 grid), and each
/// patch would be represented by a vector of numbers (the embedding).
///
/// This allows transformers, which were originally designed for text, to process images by treating
/// the patches like "words" in a sentence.
/// </para>
/// </remarks>
public class PatchEmbeddingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of each square patch (both width and height).
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// The dimension of the embedding vector for each patch.
    /// </summary>
    private readonly int _embeddingDim;

    /// <summary>
    /// The height of the input image.
    /// </summary>
    private readonly int _imageHeight;

    /// <summary>
    /// The width of the input image.
    /// </summary>
    private readonly int _imageWidth;

    /// <summary>
    /// The number of color channels in the input image (e.g., 3 for RGB).
    /// </summary>
    private readonly int _channels;

    /// <summary>
    /// The number of patches along the height dimension.
    /// </summary>
    private readonly int _numPatchesHeight;

    /// <summary>
    /// The number of patches along the width dimension.
    /// </summary>
    private readonly int _numPatchesWidth;

    /// <summary>
    /// The total number of patches (height x width).
    /// </summary>
    private readonly int _numPatches;

    /// <summary>
    /// The projection weights that transform flattened patches to embeddings.
    /// </summary>
    private Tensor<T> _projectionWeights;

    /// <summary>
    /// The bias terms added to the projected embeddings.
    /// </summary>
    private Tensor<T> _projectionBias;

    /// <summary>
    /// Cached input from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gradients for projection weights calculated during backward pass.
    /// </summary>
    private Tensor<T>? _projectionWeightsGradient;

    /// <summary>
    /// Gradients for projection bias calculated during backward pass.
    /// </summary>
    private Tensor<T>? _projectionBiasGradient;

    /// <summary>
    /// Cached pre-activation tensor from forward pass for use in activation derivative calculation.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuInput;
    private IGpuTensor<T>? _gpuPatchesFlat;
    private int _gpuBatchSize;
    private bool _gpuHasBatch;

    #region GPU Weight Storage Fields

    // GPU tensors for GPU-resident training
    private GpuTensor<T>? _gpuWeights;
    private GpuTensor<T>? _gpuBias;
    private GpuTensor<T>? _gpuWeightGradient;
    private GpuTensor<T>? _gpuBiasGradient;
    private GpuTensor<T>? _gpuWeightVelocity;
    private GpuTensor<T>? _gpuBiasVelocity;
    private GpuTensor<T>? _gpuWeightM;
    private GpuTensor<T>? _gpuWeightV;
    private GpuTensor<T>? _gpuBiasM;
    private GpuTensor<T>? _gpuBiasV;

    #endregion

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the total number of parameters in this layer.
    /// </summary>
    /// <value>
    /// The total number of trainable parameters (projection weights + projection bias).
    /// </value>
    public override int ParameterCount => _projectionWeights.Shape[0] * _projectionWeights.Shape[1] + _projectionBias.Length;

    /// <summary>
    /// Creates a new patch embedding layer with the specified dimensions.
    /// </summary>
    /// <param name="imageHeight">The height of the input image.</param>
    /// <param name="imageWidth">The width of the input image.</param>
    /// <param name="channels">The number of color channels in the input image.</param>
    /// <param name="patchSize">The size of each square patch.</param>
    /// <param name="embeddingDim">The dimension of the embedding vector for each patch.</param>
    /// <param name="activationFunction">The activation function to apply (defaults to identity if null).</param>
    /// <exception cref="ArgumentException">Thrown when image dimensions are not divisible by patch size.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the patch embedding layer with your image specifications.
    ///
    /// The parameters define:
    /// - imageHeight/imageWidth: The size of your input images
    /// - channels: How many color channels (3 for RGB, 1 for grayscale)
    /// - patchSize: How big each patch should be (commonly 16x16 or 32x32)
    /// - embeddingDim: How many numbers represent each patch (commonly 768 or 1024)
    ///
    /// For example, a 224x224 RGB image with 16x16 patches and 768 embedding dimensions
    /// would create 196 patches (14x14 grid), each represented by 768 numbers.
    /// </para>
    /// </remarks>
    public PatchEmbeddingLayer(
        int imageHeight,
        int imageWidth,
        int channels,
        int patchSize,
        int embeddingDim,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [channels, imageHeight, imageWidth],
            [(imageHeight / patchSize) * (imageWidth / patchSize), embeddingDim],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (imageHeight % patchSize != 0)
        {
            throw new ArgumentException($"Image height {imageHeight} must be divisible by patch size {patchSize}", nameof(imageHeight));
        }

        if (imageWidth % patchSize != 0)
        {
            throw new ArgumentException($"Image width {imageWidth} must be divisible by patch size {patchSize}", nameof(imageWidth));
        }

        _imageHeight = imageHeight;
        _imageWidth = imageWidth;
        _channels = channels;
        _patchSize = patchSize;
        _embeddingDim = embeddingDim;

        _numPatchesHeight = imageHeight / patchSize;
        _numPatchesWidth = imageWidth / patchSize;
        _numPatches = _numPatchesHeight * _numPatchesWidth;

        int patchDim = _channels * _patchSize * _patchSize;
        _projectionWeights = new Tensor<T>([patchDim, _embeddingDim]);
        _projectionBias = new Tensor<T>([_embeddingDim]);

        InitializeParameters();

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_projectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_projectionBias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes the weights and biases of the layer using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        int patchDim = _channels * _patchSize * _patchSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (patchDim + _embeddingDim)));

        for (int i = 0; i < _projectionWeights.Shape[0]; i++)
        {
            for (int j = 0; j < _projectionWeights.Shape[1]; j++)
            {
                _projectionWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }

        for (int i = 0; i < _projectionBias.Shape[0]; i++)
        {
            _projectionBias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Performs the forward pass of the patch embedding layer.
    /// </summary>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <returns>The output tensor with shape [batch, num_patches, embedding_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts the image into patch embeddings.
    ///
    /// The forward pass:
    /// 1. Divides the image into patches (like cutting a photo into a grid)
    /// 2. Flattens each patch into a vector (like unrolling each grid square into a line)
    /// 3. Projects each flattened patch to the embedding dimension (transforming the numbers)
    /// 4. Returns a sequence of patch embeddings
    ///
    /// For example, a 224x224 image becomes 196 embeddings, each with 768 dimensions,
    /// ready to be processed by transformer encoder blocks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // PatchEmbedding expects 4D input [B, C, H, W], normalize to 4D
        Tensor<T> processInput;
        int batchSize;

        if (rank < 4)
        {
            // Add leading dimensions to make 4D
            // 3D [C, H, W] -> [1, C, H, W]
            // 2D [H, W] -> [1, 1, H, W]
            // 1D [W] -> [1, 1, 1, W]
            var shape4D = new int[4];
            int offset = 4 - rank;
            for (int i = 0; i < offset; i++)
                shape4D[i] = 1;
            for (int i = 0; i < rank; i++)
                shape4D[offset + i] = input.Shape[i];
            processInput = input.Reshape(shape4D);
            batchSize = shape4D[0];
        }
        else if (rank == 4)
        {
            // Standard 4D [B, C, H, W]
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            // e.g., 5D [B1, B2, C, H, W] -> [B1*B2, C, H, W]
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            int channels = input.Shape[rank - 3];
            int height = input.Shape[rank - 2];
            int width = input.Shape[rank - 1];
            processInput = input.Reshape([flatBatch, channels, height, width]);
        }

        _lastInput = processInput;
        int patchDim = _channels * _patchSize * _patchSize;

        // Efficient Patchify using Reshape and Transpose
        // Input: [B, C, H, W]
        // 1. Reshape to split H and W into patches: [B, C, Nh, P, Nw, P]
        var reshaped = processInput.Reshape(batchSize, _channels, _numPatchesHeight, _patchSize, _numPatchesWidth, _patchSize);

        // 2. Transpose to group patch dimensions: [B, Nh, Nw, C, P, P]
        var transposed = reshaped.Transpose(new[] { 0, 2, 4, 1, 3, 5 });

        // 3. Flatten patches: [B, Nh*Nw, C*P*P] = [B, N, patchDim]
        var patches = transposed.Reshape(batchSize, _numPatches, patchDim);

        // Projection: patches @ weights + bias
        // Reshape to 2D for TensorMatMul: [B*N, patchDim] @ [patchDim, embedDim] -> [B*N, embedDim]
        var patchesFlat = patches.Reshape(batchSize * _numPatches, patchDim);
        var projectedFlat = Engine.TensorMatMul(patchesFlat, _projectionWeights);
        // Reshape back to 3D: [B, N, embedDim]
        var projected = projectedFlat.Reshape(batchSize, _numPatches, _embeddingDim);

        // Add bias (broadcast)
        var biasBroadcast = _projectionBias.Reshape(1, 1, _embeddingDim);
        var preActivation = Engine.TensorBroadcastAdd(projected, biasBroadcast);

        _lastPreActivation = preActivation;
        var output = ApplyActivation(preActivation);

        // Restore output shape to match original input rank
        // Output goes from 3D [B, N, embedDim] to appropriate rank
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            if (_originalInputShape.Length < 4)
            {
                // Lower-rank input: remove batch dimension if it was added
                // 3D input [C, H, W] -> 2D output [N, embedDim]
                // 2D input [H, W] -> 2D output [N, embedDim]
                // 1D input [W] -> 2D output [N, embedDim]
                return output.Reshape([_numPatches, _embeddingDim]);
            }
            else
            {
                // Higher-rank input: restore leading dimensions
                // e.g., 5D input [B1, B2, C, H, W] -> 4D output [B1, B2, N, embedDim]
                var outShape = new int[_originalInputShape.Length - 1];
                for (int d = 0; d < _originalInputShape.Length - 3; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 3] = _numPatches;
                outShape[_originalInputShape.Length - 2] = _embeddingDim;
                return output.Reshape(outShape);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the patch embedding layer.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient to be passed to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how the layer's parameters should change during training.
    ///
    /// During the backward pass:
    /// - The gradient tells us how much each output contributed to the error
    /// - We use this to figure out how to adjust the projection weights and biases
    /// - We also calculate gradients to pass back to earlier layers
    ///
    /// This allows the entire network to learn through backpropagation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }


    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients by building a computation graph
    /// that mirrors the forward pass operations (Reshape -> Permute -> Reshape -> MatMul -> Add).
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_projectionWeights, "weights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_projectionBias, "bias", requiresGradient: true);

        // 2. Patchify Logic
        int batchSize = _lastInput.Shape[0];
        int patchDim = _channels * _patchSize * _patchSize;

        // Reshape to split H and W into patches: [B, C, Nh, P, Nw, P]
        var reshapedNode = Autodiff.TensorOperations<T>.Reshape(inputNode,
            batchSize, _channels, _numPatchesHeight, _patchSize, _numPatchesWidth, _patchSize);

        // Permute to group patch dimensions: [B, Nh, Nw, C, P, P]
        // Using new Permute operation added to TensorOperations
        var transposedNode = Autodiff.TensorOperations<T>.Permute(reshapedNode, 0, 2, 4, 1, 3, 5);

        // Flatten patches: [B, N, patchDim]
        var patchesNode = Autodiff.TensorOperations<T>.Reshape(transposedNode, batchSize, _numPatches, patchDim);

        // 3. Projection: patches @ weights
        var projectedNode = Autodiff.TensorOperations<T>.MatrixMultiply(patchesNode, weightsNode);

        // 4. Add Bias (broadcast)
        // Reshape bias to [1, 1, EmbedDim] to match [B, N, EmbedDim] for broadcasting on last dim
        // Note: TensorOperations.Add supports broadcasting if implemented, or we can explicit reshape
        // TensorOperations.Add usually does broadcast logic.
        // But to be safe and explicit (and match Forward logic), let's reshape bias.
        var biasReshapedNode = Autodiff.TensorOperations<T>.Reshape(biasNode, 1, 1, _embeddingDim);
        var preActivationNode = Autodiff.TensorOperations<T>.Add(projectedNode, biasReshapedNode);

        // 5. Apply Activation
        var activatedOutput = ApplyActivationToGraph(preActivationNode);

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
        _projectionWeightsGradient = weightsNode.Gradient;
        _projectionBiasGradient = biasNode.Gradient;

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore gradient shape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastPreActivation == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var activationGradient = ApplyActivationDerivative(_lastPreActivation, outputGradient);

        int batchSize = _lastInput.Shape[0];
        int patchDim = _channels * _patchSize * _patchSize;

        // 1. Gradient w.r.t Bias: Sum over batch and patches
        _projectionBiasGradient = Engine.ReduceSum(activationGradient, new[] { 0, 1 });

        // 2. Reconstruct patches from input
        var reshapedInput = _lastInput.Reshape(batchSize, _channels, _numPatchesHeight, _patchSize, _numPatchesWidth, _patchSize);
        var transposedInput = reshapedInput.Transpose(new[] { 0, 2, 4, 1, 3, 5 });
        var patches = transposedInput.Reshape(batchSize, _numPatches, patchDim);

        // 3. Gradient w.r.t Weights: patches^T @ grad
        var patchesT = patches.Transpose(new[] { 0, 2, 1 });
        // [B, P, N] @ [B, N, E] -> [B, P, E]
        var weightGradBatch = Engine.BatchMatMul(patchesT, activationGradient);
        // Sum over batch
        _projectionWeightsGradient = Engine.ReduceSum(weightGradBatch, new[] { 0 });

        // 4. Gradient w.r.t Input (Patches)
        // [B*N, E] @ [E, P] -> [B*N, P]
        var weightsT = Engine.TensorTranspose(_projectionWeights);
        var gradFlat = activationGradient.Reshape(batchSize * _numPatches, _embeddingDim);
        var patchesGradFlat = Engine.TensorMatMul(gradFlat, weightsT);
        var patchesGrad = patchesGradFlat.Reshape(batchSize, _numPatches, patchDim);

        // 5. Un-patchify: Reshape/Transpose back to image [B, C, H, W]
        // [B, N, P] -> [B, Nh, Nw, C, P, P]
        var gradReshaped = patchesGrad.Reshape(batchSize, _numPatchesHeight, _numPatchesWidth, _channels, _patchSize, _patchSize);

        // Transpose to [B, C, Nh, P, Nw, P]
        var gradTransposed = gradReshaped.Transpose(new[] { 0, 3, 1, 4, 2, 5 });

        // Reshape to [B, C, H, W]
        var inputGradient = gradTransposed.Reshape(_lastInput.Shape);

        // Restore gradient shape to match original input shape
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate controlling the size of parameter updates.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adjusts the layer's weights and biases to improve performance.
    ///
    /// The learning rate controls:
    /// - How much to adjust each parameter
    /// - Larger values mean bigger adjustments (faster learning but less stable)
    /// - Smaller values mean smaller adjustments (slower but more stable learning)
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_projectionWeightsGradient == null || _projectionBiasGradient == null)
        {
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        }

        _projectionWeights = Engine.TensorSubtract(_projectionWeights, Engine.TensorMultiplyScalar(_projectionWeightsGradient, learningRate));
        _projectionBias = Engine.TensorSubtract(_projectionBias, Engine.TensorMultiplyScalar(_projectionBiasGradient, learningRate));

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_projectionWeights);
        Engine.InvalidatePersistentTensor(_projectionBias);
    }

    /// <summary>
    /// Gets all parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all weights and biases.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method collects all the layer's learnable values into one list.
    /// This is useful for saving the model or for optimization algorithms.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int totalParams = _projectionWeights.Shape[0] * _projectionWeights.Shape[1] + _projectionBias.Shape[0];
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        for (int i = 0; i < _projectionWeights.Shape[0]; i++)
        {
            for (int j = 0; j < _projectionWeights.Shape[1]; j++)
            {
                parameters[index++] = _projectionWeights[i, j];
            }
        }

        for (int i = 0; i < _projectionBias.Shape[0]; i++)
        {
            parameters[index++] = _projectionBias[i];
        }

        return parameters;
    }

    /// <summary>
    /// Sets all parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all weights and biases to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameter vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method loads saved parameter values back into the layer.
    /// This is used when loading a previously trained model.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _projectionWeights.Shape[0] * _projectionWeights.Shape[1] + _projectionBias.Shape[0];

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int index = 0;

        for (int i = 0; i < _projectionWeights.Shape[0]; i++)
        {
            for (int j = 0; j < _projectionWeights.Shape[1]; j++)
            {
                _projectionWeights[i, j] = parameters[index++];
            }
        }

        for (int i = 0; i < _projectionBias.Shape[0]; i++)
        {
            _projectionBias[i] = parameters[index++];
        }

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_projectionWeights);
        Engine.InvalidatePersistentTensor(_projectionBias);
    }

    /// <summary>
    /// Resets the internal state of the patch embedding layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method clears cached values to prepare for processing new data.
    /// It keeps the learned parameters but clears temporary calculation values.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _projectionWeightsGradient = null;
        _projectionBiasGradient = null;

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuPatchesFlat = null;
        _gpuBatchSize = 0;
        _gpuHasBatch = false;
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass for patch embedding.
    /// </summary>
    /// <param name="inputs">The input GPU tensors. Expects one tensor with shape [batch, channels, height, width].</param>
    /// <returns>The GPU-resident output tensor with shape [batch, num_patches, embedding_dim].</returns>
    /// <exception cref="ArgumentException">Thrown when no inputs provided.</exception>
    /// <exception cref="InvalidOperationException">Thrown when engine is not a DirectGpuTensorEngine.</exception>
    /// <remarks>
    /// <para>
    /// This implementation keeps all operations GPU-resident without CPU roundtrips:
    /// 1. Reshape to split spatial dimensions into patches
    /// 2. Permute to group patch dimensions
    /// 3. Reshape to flatten patches
    /// 4. Linear projection with fused bias addition
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;

        // PatchEmbedding expects 4D input [B, C, H, W]
        bool hasBatch = shape.Length == 4;
        int batchSize;
        IGpuTensor<T> processInput;

        if (shape.Length == 3)
        {
            // 3D [C, H, W] -> [1, C, H, W]
            processInput = gpuEngine.ReshapeGpu(input, [1, shape[0], shape[1], shape[2]]);
            batchSize = 1;
        }
        else if (shape.Length == 4)
        {
            processInput = input;
            batchSize = shape[0];
        }
        else
        {
            throw new ArgumentException(
                $"PatchEmbeddingLayer expects 3D [C,H,W] or 4D [B,C,H,W] input, got {shape.Length}D.", nameof(inputs));
        }

        int patchDim = _channels * _patchSize * _patchSize;

        // GPU-resident patchify using Reshape and Permute
        // 1. Reshape to split H and W into patches: [B, C, Nh, P, Nw, P]
        var reshaped = gpuEngine.ReshapeGpu(processInput,
            [batchSize, _channels, _numPatchesHeight, _patchSize, _numPatchesWidth, _patchSize]);

        // 2. Permute to group patch dimensions: [B, Nh, Nw, C, P, P]
        var permuted = gpuEngine.PermuteGpu(reshaped, [0, 2, 4, 1, 3, 5]);

        // 3. Reshape to flatten patches: [B, N, patchDim]
        var patches = gpuEngine.ReshapeGpu(permuted, [batchSize, _numPatches, patchDim]);

        // 4. Flatten for matrix multiplication: [B*N, patchDim]
        var patchesFlat = gpuEngine.ReshapeGpu(patches, [batchSize * _numPatches, patchDim]);

        // 5. GPU-resident linear projection: [B*N, patchDim] @ [patchDim, embedDim] + bias -> [B*N, embedDim]
        var projectedFlat = gpuEngine.FusedLinearGpu(patchesFlat, _projectionWeights, _projectionBias, FusedActivationType.None);

        // 6. Reshape back to 3D: [B, N, embedDim]
        var output = gpuEngine.ReshapeGpu(projectedFlat, [batchSize, _numPatches, _embeddingDim]);

        // Cache GPU tensors for backward pass during training
        if (IsTrainingMode)
        {
            _gpuInput = processInput != input ? processInput : input;
            _gpuPatchesFlat = patchesFlat;
            _gpuBatchSize = batchSize;
            _gpuHasBatch = hasBatch;
        }
        else
        {
            // Dispose intermediate GPU tensors if not training
            if (processInput != input)
                processInput.Dispose();
            patchesFlat.Dispose();
        }

        // Dispose other intermediate GPU tensors that are no longer needed
        reshaped.Dispose();
        permuted.Dispose();
        patches.Dispose();
        projectedFlat.Dispose();

        // Remove batch dimension if input didn't have it
        if (!hasBatch && output.Shape[0] == 1)
        {
            var result = gpuEngine.ReshapeGpu(output, [_numPatches, _embeddingDim]);
            output.Dispose();
            return result;
        }

        return output;
    }

    /// <summary>
    /// Performs GPU-resident backward pass for patch embedding.
    /// </summary>
    /// <param name="outputGradient">The gradient from subsequent layer [B, N, embedDim].</param>
    /// <returns>The gradient with respect to input [B, C, H, W].</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuPatchesFlat == null || _gpuInput == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // outputGradient shape: [B, N, embedDim] or [N, embedDim] if no batch
        int patchDim = _channels * _patchSize * _patchSize;

        // Reshape output gradient for linear backward: [B, N, embedDim] -> [B*N, embedDim]
        IGpuTensor<T> gradFlat;
        if (outputGradient.Shape.Length == 2)
        {
            // [N, embedDim] -> [1*N, embedDim]
            gradFlat = outputGradient;
        }
        else
        {
            // [B, N, embedDim] -> [B*N, embedDim]
            gradFlat = gpuEngine.ReshapeGpu(outputGradient, [_gpuBatchSize * _numPatches, _embeddingDim]);
        }

        // Step 1: Backprop through linear projection
        // Weight gradient: patches^T @ gradOutput -> [patchDim, B*N] @ [B*N, embedDim] = [patchDim, embedDim]
        var patchesFlatT = gpuEngine.TransposeGpu<T>(_gpuPatchesFlat);
        var weightGradGpu = gpuEngine.MatMulGpuTensors<T>(patchesFlatT, gradFlat);
        _projectionWeightsGradient = weightGradGpu.ToTensor();

        // Bias gradient: sum of gradOutput along batch dimension -> [embedDim]
        var biasGradGpu = gpuEngine.SumAxisGpu<T>(gradFlat, 0);
        _projectionBiasGradient = biasGradGpu.ToTensor();

        // Store GPU gradients for GPU-resident training
        _gpuWeightGradient?.Dispose();
        _gpuWeightGradient = new GpuTensor<T>(backend, weightGradGpu.Buffer, weightGradGpu.Shape, GpuTensorRole.Gradient, ownsBuffer: false);
        _gpuBiasGradient?.Dispose();
        _gpuBiasGradient = new GpuTensor<T>(backend, biasGradGpu.Buffer, biasGradGpu.Shape, GpuTensorRole.Gradient, ownsBuffer: false);

        // Input gradient: gradOutput @ weights^T -> [B*N, embedDim] @ [embedDim, patchDim] = [B*N, patchDim]
        var weightsGpu = gpuEngine.UploadToGpu<T>(_projectionWeights, GpuTensorRole.Weight);
        var weightsT = gpuEngine.TransposeGpu<T>(weightsGpu);
        var patchGrad = gpuEngine.MatMulGpuTensors<T>(gradFlat, weightsT);

        // Step 2: Reshape patch gradient back to image space
        // [B*N, patchDim] -> [B, N, patchDim] -> [B, Nh, Nw, C, P, P]
        var patchGrad3D = gpuEngine.ReshapeGpu(patchGrad, [_gpuBatchSize, _numPatches, patchDim]);
        var patchGradSpatial = gpuEngine.ReshapeGpu(patchGrad3D,
            [_gpuBatchSize, _numPatchesHeight, _numPatchesWidth, _channels, _patchSize, _patchSize]);

        // Step 3: Reverse permute: [B, Nh, Nw, C, P, P] -> [B, C, Nh, P, Nw, P]
        var gradPermuted = gpuEngine.PermuteGpu(patchGradSpatial, [0, 3, 1, 4, 2, 5]);

        // Step 4: Reshape back to image: [B, C, Nh, P, Nw, P] -> [B, C, H, W]
        var inputGrad = gpuEngine.ReshapeGpu(gradPermuted, [_gpuBatchSize, _channels, _imageHeight, _imageWidth]);

        // Dispose intermediate tensors
        patchesFlatT.Dispose();
        weightGradGpu.Dispose();
        biasGradGpu.Dispose();
        weightsGpu.Dispose();
        weightsT.Dispose();
        patchGrad.Dispose();
        patchGrad3D.Dispose();
        patchGradSpatial.Dispose();
        gradPermuted.Dispose();
        if (gradFlat != outputGradient)
            gradFlat.Dispose();

        // Remove batch dimension if input didn't have it
        if (!_gpuHasBatch)
        {
            var result = gpuEngine.ReshapeGpu(inputGrad, [_channels, _imageHeight, _imageWidth]);
            inputGrad.Dispose();
            return result;
        }

        return inputGrad;
    }

    /// <summary>
    /// Updates layer parameters using GPU-resident optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Ensure GPU weights are initialized
        _gpuWeights ??= new GpuTensor<T>(backend, _projectionWeights, GpuTensorRole.Weight);
        _gpuBias ??= new GpuTensor<T>(backend, _projectionBias, GpuTensorRole.Bias);

        // Verify gradients exist
        if (_gpuWeightGradient == null || _gpuBiasGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure optimizer state buffers exist
        EnsurePatchEmbeddingOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        int weightCount = _projectionWeights.Length;
        int biasCount = _projectionBias.Length;

        config.ApplyUpdate(backend, _gpuWeights.Buffer, _gpuWeightGradient.Buffer, BuildPatchEmbeddingOptimizerState("weights"), weightCount);
        config.ApplyUpdate(backend, _gpuBias.Buffer, _gpuBiasGradient.Buffer, BuildPatchEmbeddingOptimizerState("bias"), biasCount);

        // Sync back to CPU tensors for compatibility
        _projectionWeights = _gpuWeights.ToTensor();
        _projectionBias = _gpuBias.ToTensor();

        // Invalidate GPU cache after parameter updates
        Engine.InvalidatePersistentTensor(_projectionWeights);
        Engine.InvalidatePersistentTensor(_projectionBias);
    }

    /// <summary>
    /// Ensures optimizer state buffers are allocated for the given optimizer type.
    /// </summary>
    private void EnsurePatchEmbeddingOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = _projectionWeights.Length;
        int biasSize = _projectionBias.Length;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Momentum-based optimizers need velocity buffers
                _gpuWeightVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // Adam-family optimizers need M (first moment) and V (second moment) buffers
                _gpuWeightM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // RmsProp/Adagrad need squared average/accumulated gradient - reuse velocity buffers
                _gpuWeightVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    /// <summary>
    /// Builds the optimizer state for the specified parameter.
    /// </summary>
    private GpuOptimizerState BuildPatchEmbeddingOptimizerState(string paramName)
    {
        return paramName switch
        {
            "weights" => new GpuOptimizerState
            {
                Velocity = _gpuWeightVelocity?.Buffer,
                M = _gpuWeightM?.Buffer,
                V = _gpuWeightV?.Buffer,
                SquaredAvg = _gpuWeightVelocity?.Buffer,
                AccumulatedGrad = _gpuWeightVelocity?.Buffer
            },
            "bias" => new GpuOptimizerState
            {
                Velocity = _gpuBiasVelocity?.Buffer,
                M = _gpuBiasM?.Buffer,
                V = _gpuBiasV?.Buffer,
                SquaredAvg = _gpuBiasVelocity?.Buffer,
                AccumulatedGrad = _gpuBiasVelocity?.Buffer
            },
            _ => throw new ArgumentException($"Unknown parameter: {paramName}", nameof(paramName))
        };
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_projectionWeights == null || _projectionBias == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Weights and biases are already Tensor<T>
        var weightsNode = TensorOperations<T>.Constant(_projectionWeights, "weights");
        var biasNode = TensorOperations<T>.Constant(_projectionBias, "bias");

        var output = TensorOperations<T>.MatrixMultiply(inputNode, weightsNode);
        return TensorOperations<T>.Add(output, biasNode);
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => _projectionWeights != null && _projectionBias != null;
}
