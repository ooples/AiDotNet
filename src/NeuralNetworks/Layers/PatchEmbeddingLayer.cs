#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Helpers;

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
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "1, 3, 8, 8", TestConstructorArgs = "4, 16")]
public partial class PatchEmbeddingLayer<T> : LayerBase<T>
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
    private int _imageHeight;

    /// <summary>
    /// The width of the input image.
    /// </summary>
    private int _imageWidth;

    /// <summary>
    /// The number of color channels in the input image (e.g., 3 for RGB).
    /// </summary>
    private int _channels;

    /// <summary>
    /// The number of patches along the height dimension.
    /// </summary>
    private int _numPatchesHeight;

    /// <summary>
    /// The number of patches along the width dimension.
    /// </summary>
    private int _numPatchesWidth;

    /// <summary>
    /// The total number of patches (height x width).
    /// </summary>
    private int _numPatches;

    /// <summary>
    /// The projection weights that transform flattened patches to embeddings.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _projectionWeights;

    /// <summary>
    /// The bias terms added to the projected embeddings.
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuPatchesFlat;
    private int _gpuBatchSize;
    private bool _gpuHasBatch;

    #region GPU Weight Storage Fields

    // GPU tensors for GPU-resident training
    private Tensor<T>? _gpuWeights;
    private Tensor<T>? _gpuBias;
    private Tensor<T>? _gpuWeightGradient;
    private Tensor<T>? _gpuBiasGradient;
    private Tensor<T>? _gpuWeightVelocity;
    private Tensor<T>? _gpuBiasVelocity;
    private Tensor<T>? _gpuWeightM;
    private Tensor<T>? _gpuWeightV;
    private Tensor<T>? _gpuBiasM;
    private Tensor<T>? _gpuBiasV;

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
        int patchSize,
        int embeddingDim,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            new[] { -1, -1, -1 },
            new[] { -1, embeddingDim },
            activationFunction ?? new IdentityActivation<T>())
    {
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        _patchSize = patchSize;
        _embeddingDim = embeddingDim;

        // Lazy: image H/W, channels, num patches resolved on first Forward.
        _imageHeight = -1;
        _imageWidth = -1;
        _channels = -1;
        _numPatchesHeight = -1;
        _numPatchesWidth = -1;
        _numPatches = -1;

        _projectionWeights = new Tensor<T>([0, embeddingDim]);
        _projectionBias = new Tensor<T>([embeddingDim]);

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.Eager;
    }

    /// <summary>
    /// Resolves image height/width/channels from input on first forward (PyTorch-style).
    /// Per Dosovitskiy et al. 2021, the input is [B, C, H, W] (or [C, H, W]); we read
    /// channels from input.Shape[^3], H from [^2], W from [^1], assert divisibility by
    /// patchSize, and allocate the projection weights [C*P*P, embeddingDim].
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 3)
            throw new ArgumentException(
                $"PatchEmbeddingLayer requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {rank}.",
                nameof(input));

        _channels = input.Shape[rank - 3];
        _imageHeight = input.Shape[rank - 2];
        _imageWidth = input.Shape[rank - 1];

        if (_imageHeight % _patchSize != 0 || _imageWidth % _patchSize != 0)
            throw new ArgumentException(
                $"Image H/W ({_imageHeight}/{_imageWidth}) must be divisible by patchSize ({_patchSize}).",
                nameof(input));

        _numPatchesHeight = _imageHeight / _patchSize;
        _numPatchesWidth = _imageWidth / _patchSize;
        _numPatches = _numPatchesHeight * _numPatchesWidth;

        int patchDim = _channels * _patchSize * _patchSize;
        _projectionWeights = new Tensor<T>([patchDim, _embeddingDim]);
        _projectionBias = new Tensor<T>([_embeddingDim]);

        InitializeParameters();

        RegisterTrainableParameter(_projectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_projectionBias, PersistentTensorRole.Biases);

        ResolveShapes(
            new[] { _channels, _imageHeight, _imageWidth },
            new[] { _numPatches, _embeddingDim });
    }

    /// <summary>
    /// Initializes the weights and biases of the layer using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        int patchDim = _channels * _patchSize * _patchSize;
        InitializeLayerWeights(_projectionWeights, patchDim, _embeddingDim);
        InitializeLayerBiases(_projectionBias);
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
        EnsureInitializedFromInput(input);

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
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
            // Every shape op via Engine so the gradient tape records it — direct
            // Tensor<T>.Reshape / .Transpose bypass the tape and disconnect the
            // patch-embedding weights from the backward graph on every DiT /
            // ViT / MMDiT forward pass.
            var shape4D = new int[4];
            int offset = 4 - rank;
            for (int i = 0; i < offset; i++)
                shape4D[i] = 1;
            for (int i = 0; i < rank; i++)
                shape4D[offset + i] = input.Shape[i];
            processInput = Engine.Reshape(input, shape4D);
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
            processInput = Engine.Reshape(input, new[] { flatBatch, channels, height, width });
        }

        _lastInput = processInput;
        int patchDim = _channels * _patchSize * _patchSize;

        // Efficient Patchify using Reshape and Transpose (all via Engine).
        // Input: [B, C, H, W]
        // 1. Reshape to split H and W into patches: [B, C, Nh, P, Nw, P]
        var reshaped = Engine.Reshape(processInput,
            new[] { batchSize, _channels, _numPatchesHeight, _patchSize, _numPatchesWidth, _patchSize });

        // 2. Transpose to group patch dimensions: [B, Nh, Nw, C, P, P]
        var transposed = Engine.TensorPermute(reshaped, new[] { 0, 2, 4, 1, 3, 5 });

        // 3. Flatten patches: [B, Nh*Nw, C*P*P] = [B, N, patchDim]
        var patches = Engine.Reshape(transposed, new[] { batchSize, _numPatches, patchDim });

        // Projection: patches @ weights + bias
        // Reshape to 2D for TensorMatMul: [B*N, patchDim] @ [patchDim, embedDim] -> [B*N, embedDim]
        var patchesFlat = Engine.Reshape(patches, new[] { batchSize * _numPatches, patchDim });
        var projectedFlat = Engine.TensorMatMul(patchesFlat, _projectionWeights);
        // Reshape back to 3D: [B, N, embedDim]
        var projected = Engine.Reshape(projectedFlat, new[] { batchSize, _numPatches, _embeddingDim });

        // Add bias (broadcast) — reshape bias fresh each call to keep the tape
        // GradFn chain alive (a cached reshape primed during inference would
        // disconnect _projectionBias from the backward walk).
        var biasBroadcast = Engine.Reshape(_projectionBias, new[] { 1, 1, _embeddingDim });
        var preActivation = Engine.TensorBroadcastAdd(projected, biasBroadcast);

        _lastPreActivation = preActivation;
        var output = ApplyActivation(preActivation);

        // Restore output shape to match original input rank (via Engine).
        if (_originalInputShape != null && _originalInputShape.Length != 4)
        {
            if (_originalInputShape.Length < 4)
            {
                return Engine.Reshape(output, new[] { _numPatches, _embeddingDim });
            }
            else
            {
                var outShape = new int[_originalInputShape.Length - 1];
                for (int d = 0; d < _originalInputShape.Length - 3; d++)
                    outShape[d] = _originalInputShape[d];
                outShape[_originalInputShape.Length - 3] = _numPatches;
                outShape[_originalInputShape.Length - 2] = _embeddingDim;
                return Engine.Reshape(output, outShape);
            }
        }

        return output;
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
    public override Vector<T> GetParameterGradients()
    {
        if (_projectionWeightsGradient == null || _projectionBiasGradient == null)
            return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_projectionWeightsGradient.ToArray()),
            new Vector<T>(_projectionBiasGradient.ToArray()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _projectionWeightsGradient = null; _projectionBiasGradient = null;
    }

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

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["PatchSize"] = _patchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input._shape;

        // PatchEmbedding expects 4D input [B, C, H, W]
        bool hasBatch = shape.Length == 4;
        int batchSize;
        Tensor<T> processInput;

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
        _gpuWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _projectionWeights, GpuTensorRole.Weight);
        _gpuBias ??= GpuTensorHelper.UploadToGpu<T>(backend, _projectionBias, GpuTensorRole.Bias);

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
        _projectionWeights = _gpuWeights;
        _projectionBias = _gpuBias;

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
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // Adam-family optimizers need M (first moment) and V (second moment) buffers
                _gpuWeightM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // RmsProp/Adagrad need squared average/accumulated gradient - reuse velocity buffers
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
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
}
