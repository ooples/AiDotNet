namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Patch embedding layer for Swin Transformer that converts images to patch sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This layer divides an input image into non-overlapping patches and projects each patch
/// to an embedding vector. This is the first step in processing images with Swin Transformer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this layer as cutting an image into small squares (patches)
/// and converting each square into a list of numbers (embedding) that describes its content.
/// This allows the transformer to process images as sequences, similar to how it processes text.
/// </para>
/// <para>
/// Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
/// </para>
/// </remarks>
public class SwinPatchEmbeddingLayer<T> : LayerBase<T>
{
    private readonly int _patchSize;
    private readonly int _inputChannels;
    private readonly int _embedDim;
    private readonly int _inputHeight;
    private readonly int _inputWidth;

    /// <summary>
    /// The convolutional layer used for patch projection.
    /// Uses kernel size = stride = patch size for non-overlapping patches.
    /// </summary>
    private readonly ConvolutionalLayer<T> _projection;

    /// <summary>
    /// Layer normalization applied after patch embedding.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount => _projection.ParameterCount + _norm.ParameterCount;

    /// <summary>
    /// Gets the number of patches produced by this layer.
    /// </summary>
    public int NumPatches => (_inputHeight / _patchSize) * (_inputWidth / _patchSize);

    /// <summary>
    /// Gets the height of the patch grid.
    /// </summary>
    public int PatchGridHeight => _inputHeight / _patchSize;

    /// <summary>
    /// Gets the width of the patch grid.
    /// </summary>
    public int PatchGridWidth => _inputWidth / _patchSize;

    /// <summary>
    /// Creates a new Swin patch embedding layer.
    /// </summary>
    /// <param name="inputHeight">Height of input images.</param>
    /// <param name="inputWidth">Width of input images.</param>
    /// <param name="inputChannels">Number of input channels (default: 3 for RGB).</param>
    /// <param name="patchSize">Size of each patch (default: 4 from Swin paper).</param>
    /// <param name="embedDim">Dimension of patch embeddings (default: 96 for Swin-Tiny).</param>
    /// <exception cref="ArgumentException">Thrown if input dimensions are not divisible by patch size.</exception>
    public SwinPatchEmbeddingLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels = 3,
        int patchSize = 4,
        int embedDim = 96)
        : base([inputChannels, inputHeight, inputWidth], [embedDim])
    {
        if (inputHeight % patchSize != 0)
            throw new ArgumentException($"Input height ({inputHeight}) must be divisible by patch size ({patchSize}).", nameof(inputHeight));
        if (inputWidth % patchSize != 0)
            throw new ArgumentException($"Input width ({inputWidth}) must be divisible by patch size ({patchSize}).", nameof(inputWidth));

        _patchSize = patchSize;
        _inputChannels = inputChannels;
        _embedDim = embedDim;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;

        // Projection: Conv with kernel=stride=patchSize creates non-overlapping patches
        _projection = new ConvolutionalLayer<T>(
            inputChannels,
            inputHeight,
            inputWidth,
            embedDim,
            kernelSize: patchSize,
            stride: patchSize,
            padding: 0);

        // Layer normalization over embedding dimension
        _norm = new LayerNormalizationLayer<T>(embedDim);
    }

    /// <summary>
    /// Performs the forward pass, converting image to patch sequence.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, channels, height, width].</param>
    /// <returns>Output tensor of shape [batch, numPatches, embedDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Input: [batch, channels, height, width]
        int batch = input.Shape[0];

        // Apply convolution: [batch, embedDim, H/patchSize, W/patchSize]
        var projected = _projection.Forward(input);

        int patchH = projected.Shape[2];
        int patchW = projected.Shape[3];
        int numPatches = patchH * patchW;

        // Reshape to sequence: [batch, numPatches, embedDim]
        var sequence = new Tensor<T>([batch, numPatches, _embedDim]);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < patchH; h++)
            {
                for (int w = 0; w < patchW; w++)
                {
                    int seqIdx = h * patchW + w;
                    for (int c = 0; c < _embedDim; c++)
                    {
                        sequence[b, seqIdx, c] = projected[b, c, h, w];
                    }
                }
            }
        }

        // Apply layer normalization
        var normalized = _norm.Forward(sequence);

        return normalized;
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient for the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        int batch = outputGradient.Shape[0];
        int numPatches = outputGradient.Shape[1];

        // Backward through layer norm
        var normGrad = _norm.Backward(outputGradient);

        // Reshape back to conv output shape: [batch, embedDim, patchH, patchW]
        int patchH = PatchGridHeight;
        int patchW = PatchGridWidth;
        var convGrad = new Tensor<T>([batch, _embedDim, patchH, patchW]);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < patchH; h++)
            {
                for (int w = 0; w < patchW; w++)
                {
                    int seqIdx = h * patchW + w;
                    for (int c = 0; c < _embedDim; c++)
                    {
                        convGrad[b, c, h, w] = normGrad[b, seqIdx, c];
                    }
                }
            }
        }

        // Backward through projection conv
        var inputGrad = _projection.Backward(convGrad);

        return inputGrad;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var projParams = _projection.GetParameters();
        var normParams = _norm.GetParameters();

        var result = new T[projParams.Length + normParams.Length];
        Array.Copy(projParams.Data, 0, result, 0, projParams.Length);
        Array.Copy(normParams.Data, 0, result, projParams.Length, normParams.Length);

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int projCount = _projection.ParameterCount;
        int normCount = _norm.ParameterCount;

        var projParams = new T[projCount];
        var normParams = new T[normCount];

        Array.Copy(parameters.Data, 0, projParams, 0, projCount);
        Array.Copy(parameters.Data, projCount, normParams, 0, normCount);

        _projection.SetParameters(new Vector<T>(projParams));
        _norm.SetParameters(new Vector<T>(normParams));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var projGrads = _projection.GetParameterGradients();
        var normGrads = _norm.GetParameterGradients();

        var result = new T[projGrads.Length + normGrads.Length];
        Array.Copy(projGrads.Data, 0, result, 0, projGrads.Length);
        Array.Copy(normGrads.Data, 0, result, projGrads.Length, normGrads.Length);

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _projection.ResetState();
        _norm.ResetState();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _projection.UpdateParameters(learningRate);
        _norm.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation =>
        _projection != null && _projection.SupportsJitCompilation &&
        _norm != null && _norm.SupportsJitCompilation;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create symbolic input node for image: [batch, channels, height, width]
        var symbolicInput = new Tensor<T>([1, _inputChannels, _inputHeight, _inputWidth]);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "swin_patch_embed_input");
        inputNodes.Add(inputNode);

        // Export projection convolution graph
        var projectedNode = _projection.ExportComputationGraph(inputNodes);

        // The reshape from [batch, embedDim, H/patchSize, W/patchSize] to [batch, numPatches, embedDim]
        // is a structural operation that would be represented as a reshape node
        var reshapeNode = TensorOperations<T>.Reshape(projectedNode, [1, NumPatches, _embedDim]);

        // Export norm graph - create a subgraph for layer normalization
        var normInputNodes = new List<ComputationNode<T>> { reshapeNode };
        var outputNode = _norm.ExportComputationGraph(normInputNodes);

        return outputNode;
    }
}
