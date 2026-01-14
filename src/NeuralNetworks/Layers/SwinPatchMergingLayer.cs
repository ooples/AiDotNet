namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Patch merging layer for Swin Transformer that performs downsampling between stages.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This layer merges 2x2 neighboring patches into a single patch, reducing spatial
/// resolution by half while doubling the channel dimension. This creates the hierarchical
/// structure characteristic of Swin Transformer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like pooling in CNNs, but instead of taking
/// max or average, we concatenate 4 neighboring patches together (2x2 grid) and then
/// use a linear layer to reduce the combined channels. This lets the network process
/// information at multiple scales.
/// </para>
/// <para>
/// Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
/// </para>
/// </remarks>
public class SwinPatchMergingLayer<T> : LayerBase<T>
{
    private readonly int _inputDim;
    private readonly int _outputDim;

    /// <summary>
    /// Linear reduction layer that projects concatenated patches to output dimension.
    /// Input: 4 * inputDim (concatenated 2x2 patches), Output: 2 * inputDim
    /// </summary>
    private readonly DenseLayer<T> _reduction;

    /// <summary>
    /// Layer normalization applied before reduction.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm;

    // Cached values for backward pass
    private int _cachedBatch;
    private int _cachedH;
    private int _cachedW;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount => _reduction.ParameterCount + _norm.ParameterCount;

    /// <summary>
    /// Creates a new Swin patch merging layer.
    /// </summary>
    /// <param name="inputDim">Input channel dimension.</param>
    /// <exception cref="ArgumentException">Thrown if inputDim is not positive.</exception>
    public SwinPatchMergingLayer(int inputDim)
        : base([inputDim], [inputDim * 2])
    {
        if (inputDim <= 0)
            throw new ArgumentException("Input dimension must be positive.", nameof(inputDim));

        _inputDim = inputDim;
        _outputDim = inputDim * 2;

        // Layer normalization over concatenated dimension (4 * inputDim)
        _norm = new LayerNormalizationLayer<T>(inputDim * 4);

        // Linear reduction: 4 * inputDim -> 2 * inputDim
        _reduction = new DenseLayer<T>(inputDim * 4, _outputDim);
    }

    /// <summary>
    /// Performs the forward pass, merging 2x2 patches.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, seqLen, dim] where seqLen = H*W.</param>
    /// <returns>Output tensor of shape [batch, seqLen/4, dim*2].</returns>
    /// <exception cref="InvalidOperationException">Thrown if spatial dimensions are not even.</exception>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dim = input.Shape[2];

        // Infer spatial dimensions
        int h, w;
        FindSpatialDimensions(seqLen, out h, out w);

        // Validate dimensions are even
        if (h % 2 != 0 || w % 2 != 0)
        {
            throw new InvalidOperationException(
                $"Spatial dimensions must be even for patch merging. Got h={h}, w={w}.");
        }

        _cachedBatch = batch;
        _cachedH = h;
        _cachedW = w;

        int newH = h / 2;
        int newW = w / 2;
        int newSeqLen = newH * newW;

        // Reshape to spatial and concatenate 2x2 patches
        var merged = new Tensor<T>([batch, newSeqLen, dim * 4]);

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < newH; i++)
            {
                for (int j = 0; j < newW; j++)
                {
                    int newIdx = i * newW + j;

                    // Get 4 patch indices from original grid
                    int idx0 = (2 * i) * w + (2 * j);         // Top-left
                    int idx1 = (2 * i) * w + (2 * j + 1);     // Top-right
                    int idx2 = (2 * i + 1) * w + (2 * j);     // Bottom-left
                    int idx3 = (2 * i + 1) * w + (2 * j + 1); // Bottom-right

                    // Concatenate channels from 4 patches
                    for (int d = 0; d < dim; d++)
                    {
                        merged[b, newIdx, d] = input[b, idx0, d];
                        merged[b, newIdx, dim + d] = input[b, idx1, d];
                        merged[b, newIdx, 2 * dim + d] = input[b, idx2, d];
                        merged[b, newIdx, 3 * dim + d] = input[b, idx3, d];
                    }
                }
            }
        }

        // Apply layer normalization
        var normalized = _norm.Forward(merged);

        // Apply linear reduction: [batch, newSeqLen, 4*dim] -> [batch, newSeqLen, 2*dim]
        var output = new Tensor<T>([batch, newSeqLen, _outputDim]);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < newSeqLen; s++)
            {
                var tokenIn = new Tensor<T>([1, dim * 4]);
                for (int d = 0; d < dim * 4; d++)
                {
                    tokenIn[0, d] = normalized[b, s, d];
                }

                var tokenOut = _reduction.Forward(tokenIn);

                for (int d = 0; d < _outputDim; d++)
                {
                    output[b, s, d] = tokenOut[0, d];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer of shape [batch, newSeqLen, 2*dim].</param>
    /// <returns>Gradient for the input of shape [batch, seqLen, dim].</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        int batch = outputGradient.Shape[0];
        int newSeqLen = outputGradient.Shape[1];
        int newH = _cachedH / 2;
        int newW = _cachedW / 2;

        // Backward through reduction layer
        var reductionGrad = new Tensor<T>([batch, newSeqLen, _inputDim * 4]);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < newSeqLen; s++)
            {
                var tokenGrad = new Tensor<T>([1, _outputDim]);
                for (int d = 0; d < _outputDim; d++)
                {
                    tokenGrad[0, d] = outputGradient[b, s, d];
                }

                var tokenInputGrad = _reduction.Backward(tokenGrad);

                for (int d = 0; d < _inputDim * 4; d++)
                {
                    reductionGrad[b, s, d] = tokenInputGrad[0, d];
                }
            }
        }

        // Backward through layer normalization
        var normGrad = _norm.Backward(reductionGrad);

        // Reverse the patch merging: distribute gradients back to original positions
        int seqLen = _cachedH * _cachedW;
        var inputGrad = new Tensor<T>([batch, seqLen, _inputDim]);

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < newH; i++)
            {
                for (int j = 0; j < newW; j++)
                {
                    int newIdx = i * newW + j;

                    // Get 4 patch indices from original grid
                    int idx0 = (2 * i) * _cachedW + (2 * j);
                    int idx1 = (2 * i) * _cachedW + (2 * j + 1);
                    int idx2 = (2 * i + 1) * _cachedW + (2 * j);
                    int idx3 = (2 * i + 1) * _cachedW + (2 * j + 1);

                    // Distribute gradients back
                    for (int d = 0; d < _inputDim; d++)
                    {
                        inputGrad[b, idx0, d] = normGrad[b, newIdx, d];
                        inputGrad[b, idx1, d] = normGrad[b, newIdx, _inputDim + d];
                        inputGrad[b, idx2, d] = normGrad[b, newIdx, 2 * _inputDim + d];
                        inputGrad[b, idx3, d] = normGrad[b, newIdx, 3 * _inputDim + d];
                    }
                }
            }
        }

        return inputGrad;
    }

    private static void FindSpatialDimensions(int seqLen, out int h, out int w)
    {
        // Find valid factorization where both h and w are even
        int sqrtSeq = (int)Math.Sqrt(seqLen);

        h = 0;
        w = 0;

        // Search for factors close to square
        for (int candidate = sqrtSeq; candidate >= 1; candidate--)
        {
            if (seqLen % candidate == 0)
            {
                int other = seqLen / candidate;
                // Both dimensions must be even for 2×2 patch merging
                if (candidate % 2 == 0 && other % 2 == 0)
                {
                    h = other;
                    w = candidate;
                    return;
                }
            }
        }

        // If no valid even factorization found, throw exception
        throw new InvalidOperationException(
            $"Cannot find valid spatial dimensions from sequence length {seqLen}. " +
            "Sequence length must be factorizable into two even integers for patch merging.");
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var normParams = _norm.GetParameters();
        var reductionParams = _reduction.GetParameters();

        var result = new T[normParams.Length + reductionParams.Length];
        normParams.AsSpan().CopyTo(result.AsSpan(0, normParams.Length));
        reductionParams.AsSpan().CopyTo(result.AsSpan(normParams.Length, reductionParams.Length));

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int normCount = _norm.ParameterCount;
        int reductionCount = _reduction.ParameterCount;

        var normParams = new T[normCount];
        var reductionParams = new T[reductionCount];

        parameters.AsSpan().Slice(0, normCount).CopyTo(normParams);
        parameters.AsSpan().Slice(normCount, reductionCount).CopyTo(reductionParams);

        _norm.SetParameters(new Vector<T>(normParams));
        _reduction.SetParameters(new Vector<T>(reductionParams));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var normGrads = _norm.GetParameterGradients();
        var reductionGrads = _reduction.GetParameterGradients();

        var result = new T[normGrads.Length + reductionGrads.Length];
        normGrads.AsSpan().CopyTo(result.AsSpan(0, normGrads.Length));
        reductionGrads.AsSpan().CopyTo(result.AsSpan(normGrads.Length, reductionGrads.Length));

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm.ResetState();
        _reduction.ResetState();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm.UpdateParameters(learningRate);
        _reduction.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation =>
        _norm != null && _norm.SupportsJitCompilation &&
        _reduction != null && _reduction.SupportsJitCompilation;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create symbolic input node: [batch, seqLen, dim]
        // seqLen = H*W, after merging becomes (H/2)*(W/2) = seqLen/4
        var symbolicInput = new Tensor<T>([1, 4, _inputDim]); // Minimum 2x2 spatial
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "swin_merge_input");
        inputNodes.Add(inputNode);

        // The patch merging operation (concatenate 2x2 patches) would be represented
        // as a reshape/concat operation in the computation graph
        // For JIT, we represent this as: reshape -> concat -> norm -> reduction
        var mergedShape = new Tensor<T>([1, 1, _inputDim * 4]); // 4 patches concatenated
        var mergedNode = TensorOperations<T>.Reshape(inputNode, [1, 1, _inputDim * 4]);

        // Export norm graph
        var normNodes = new List<ComputationNode<T>> { mergedNode };
        var normedNode = _norm.ExportComputationGraph(normNodes);

        // Export reduction graph
        var reductionNodes = new List<ComputationNode<T>> { normedNode };
        var outputNode = _reduction.ExportComputationGraph(reductionNodes);

        return outputNode;
    }
}
