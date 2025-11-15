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
    private Matrix<T> _projectionWeights;

    /// <summary>
    /// The bias terms added to the projected embeddings.
    /// </summary>
    private Vector<T> _projectionBias;

    /// <summary>
    /// Cached input from the forward pass for use in the backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gradients for projection weights calculated during backward pass.
    /// </summary>
    private Matrix<T>? _projectionWeightsGradient;

    /// <summary>
    /// Gradients for projection bias calculated during backward pass.
    /// </summary>
    private Vector<T>? _projectionBiasGradient;

    /// <summary>
    /// Cached pre-activation tensor from forward pass for use in activation derivative calculation.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Indicates whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

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
        : base([channels, imageHeight, imageWidth], [0, 0], activationFunction ?? new IdentityActivation<T>())
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
        _projectionWeights = new Matrix<T>(patchDim, _embeddingDim);
        _projectionBias = new Vector<T>(_embeddingDim);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the layer using Xavier initialization.
    /// </summary>
    private void InitializeParameters()
    {
        int patchDim = _channels * _patchSize * _patchSize;
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (patchDim + _embeddingDim)));

        for (int i = 0; i < _projectionWeights.Rows; i++)
        {
            for (int j = 0; j < _projectionWeights.Columns; j++)
            {
                _projectionWeights[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }
        }

        for (int i = 0; i < _projectionBias.Length; i++)
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
        _lastInput = input;
        int batchSize = input.Shape[0];
        int patchDim = _channels * _patchSize * _patchSize;

        var patches = new Tensor<T>([batchSize, _numPatches, patchDim]);

        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < _numPatchesHeight; ph++)
            {
                for (int pw = 0; pw < _numPatchesWidth; pw++)
                {
                    int flatIdx = 0;
                    for (int c = 0; c < _channels; c++)
                    {
                        for (int h = 0; h < _patchSize; h++)
                        {
                            for (int w = 0; w < _patchSize; w++)
                            {
                                int inputH = ph * _patchSize + h;
                                int inputW = pw * _patchSize + w;
                                patches[b, patchIdx, flatIdx] = input[b, c, inputH, inputW];
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        var preActivation = new Tensor<T>([batchSize, _numPatches, _embeddingDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _numPatches; p++)
            {
                for (int e = 0; e < _embeddingDim; e++)
                {
                    T sum = _projectionBias[e];
                    for (int d = 0; d < patchDim; d++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(patches[b, p, d], _projectionWeights[d, e]));
                    }
                    preActivation[b, p, e] = sum;
                }
            }
        }

        _lastPreActivation = preActivation;
        return ApplyActivation(preActivation);
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
    /// This method uses automatic differentiation to compute gradients. It's slower than the
    /// manual implementation but can be useful for:
    /// - Verifying gradient correctness
    /// - Rapid prototyping with custom modifications
    /// - Research and experimentation
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // For complex/composite layers, delegate to manual implementation
        // Full autodiff requires implementing all sub-operations
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Gets the topological order of nodes in the computation graph.
    /// </summary>
    private List<Autodiff.ComputationNode<T>> GetTopologicalOrder(Autodiff.ComputationNode<T> root)
    {
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var result = new List<Autodiff.ComputationNode<T>>();

        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((root, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                stack.Push((node, true));

                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
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

        _projectionWeightsGradient = new Matrix<T>(patchDim, _embeddingDim);
        _projectionBiasGradient = new Vector<T>(_embeddingDim);

        var patches = new Tensor<T>([batchSize, _numPatches, patchDim]);
        var patchesGradient = new Tensor<T>([batchSize, _numPatches, patchDim]);

        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < _numPatchesHeight; ph++)
            {
                for (int pw = 0; pw < _numPatchesWidth; pw++)
                {
                    int flatIdx = 0;
                    for (int c = 0; c < _channels; c++)
                    {
                        for (int h = 0; h < _patchSize; h++)
                        {
                            for (int w = 0; w < _patchSize; w++)
                            {
                                int inputH = ph * _patchSize + h;
                                int inputW = pw * _patchSize + w;
                                patches[b, patchIdx, flatIdx] = _lastInput[b, c, inputH, inputW];
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < _numPatches; p++)
            {
                for (int e = 0; e < _embeddingDim; e++)
                {
                    T grad = activationGradient[b, p, e];
                    _projectionBiasGradient[e] = NumOps.Add(_projectionBiasGradient[e], grad);

                    for (int d = 0; d < patchDim; d++)
                    {
                        T weightGrad = NumOps.Multiply(patches[b, p, d], grad);
                        _projectionWeightsGradient[d, e] = NumOps.Add(_projectionWeightsGradient[d, e], weightGrad);

                        T patchGrad = NumOps.Multiply(_projectionWeights[d, e], grad);
                        patchesGradient[b, p, d] = NumOps.Add(patchesGradient[b, p, d], patchGrad);
                    }
                }
            }
        }

        var inputGradient = new Tensor<T>(_lastInput.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            int patchIdx = 0;
            for (int ph = 0; ph < _numPatchesHeight; ph++)
            {
                for (int pw = 0; pw < _numPatchesWidth; pw++)
                {
                    int flatIdx = 0;
                    for (int c = 0; c < _channels; c++)
                    {
                        for (int h = 0; h < _patchSize; h++)
                        {
                            for (int w = 0; w < _patchSize; w++)
                            {
                                int inputH = ph * _patchSize + h;
                                int inputW = pw * _patchSize + w;
                                inputGradient[b, c, inputH, inputW] = patchesGradient[b, patchIdx, flatIdx];
                                flatIdx++;
                            }
                        }
                    }
                    patchIdx++;
                }
            }
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

        for (int i = 0; i < _projectionWeights.Rows; i++)
        {
            for (int j = 0; j < _projectionWeights.Columns; j++)
            {
                _projectionWeights[i, j] = NumOps.Subtract(
                    _projectionWeights[i, j],
                    NumOps.Multiply(learningRate, _projectionWeightsGradient[i, j]));
            }
        }

        for (int i = 0; i < _projectionBias.Length; i++)
        {
            _projectionBias[i] = NumOps.Subtract(
                _projectionBias[i],
                NumOps.Multiply(learningRate, _projectionBiasGradient[i]));
        }
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
        int totalParams = _projectionWeights.Rows * _projectionWeights.Columns + _projectionBias.Length;
        var parameters = new Vector<T>(totalParams);
        int index = 0;

        for (int i = 0; i < _projectionWeights.Rows; i++)
        {
            for (int j = 0; j < _projectionWeights.Columns; j++)
            {
                parameters[index++] = _projectionWeights[i, j];
            }
        }

        for (int i = 0; i < _projectionBias.Length; i++)
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
        int totalParams = _projectionWeights.Rows * _projectionWeights.Columns + _projectionBias.Length;

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}", nameof(parameters));
        }

        int index = 0;

        for (int i = 0; i < _projectionWeights.Rows; i++)
        {
            for (int j = 0; j < _projectionWeights.Columns; j++)
            {
                _projectionWeights[i, j] = parameters[index++];
            }
        }

        for (int i = 0; i < _projectionBias.Length; i++)
        {
            _projectionBias[i] = parameters[index++];
        }
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
    }
}
