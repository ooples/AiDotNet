using AiDotNet.ActivationFunctions;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Swin Transformer block layer with windowed multi-head self-attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This layer implements the core Swin Transformer block with:
/// - Window-based multi-head self-attention (W-MSA or SW-MSA)
/// - Shifted window partitioning for cross-window connections
/// - Two-layer MLP with GELU activation
/// - Pre-norm architecture with residual connections
/// - Learnable relative position bias
/// </para>
/// <para>
/// <b>For Beginners:</b> Unlike standard transformers that compute attention across all tokens
/// (which is expensive for images), Swin Transformer divides the image into windows and
/// computes attention only within each window. To allow information flow between windows,
/// alternate layers use "shifted" windows that overlap the original window boundaries.
/// </para>
/// <para>
/// Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
/// </para>
/// </remarks>
public class SwinTransformerBlockLayer<T> : LayerBase<T>
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _windowSize;
    private readonly int _shiftSize;
    private readonly double _scale;
    private readonly int _mlpRatio;

    // Pre-norm layer normalizations
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly LayerNormalizationLayer<T> _norm2;

    // Window attention projections
    private readonly DenseLayer<T> _qkvProj;
    private readonly DenseLayer<T> _outProj;

    // Relative position bias table: (2*windowSize-1)^2 entries for each head
    private Tensor<T> _relativePositionBiasTable;
    private readonly int[,] _relativePositionIndex;

    // MLP layers
    private readonly DenseLayer<T> _mlpFc1;
    private readonly DenseLayer<T> _mlpFc2;

    // Cached values for backward pass
    private Tensor<T>? _cachedInput;
    private Tensor<T>? _cachedNorm1Output;
    private Tensor<T>? _cachedAttnOutput;
    private Tensor<T>? _cachedResidual1;
    private Tensor<T>? _cachedNorm2Output;
    private int _cachedH;
    private int _cachedW;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount =>
        _norm1.ParameterCount +
        _norm2.ParameterCount +
        _qkvProj.ParameterCount +
        _outProj.ParameterCount +
        _relativePositionBiasTable.Length +
        _mlpFc1.ParameterCount +
        _mlpFc2.ParameterCount;

    /// <summary>
    /// Gets whether this block uses shifted windows.
    /// </summary>
    public bool UsesShiftedWindows => _shiftSize > 0;

    /// <summary>
    /// Creates a new Swin Transformer block layer.
    /// </summary>
    /// <param name="dim">Number of input/output channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="windowSize">Window size for attention (default: 7 from paper, 10 for Donut).</param>
    /// <param name="shiftSize">Shift size for SW-MSA (0 for W-MSA, windowSize/2 for SW-MSA).</param>
    /// <param name="mlpRatio">Ratio of MLP hidden dim to embedding dim (default: 4).</param>
    /// <exception cref="ArgumentException">Thrown if dim is not divisible by numHeads.</exception>
    public SwinTransformerBlockLayer(
        int dim,
        int numHeads,
        int windowSize = 7,
        int shiftSize = 0,
        int mlpRatio = 4)
        : base([dim], [dim])
    {
        if (dim % numHeads != 0)
            throw new ArgumentException($"Dimension ({dim}) must be divisible by number of heads ({numHeads}).", nameof(dim));

        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _windowSize = windowSize;
        _shiftSize = shiftSize;
        _mlpRatio = mlpRatio;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Layer normalizations (pre-norm architecture)
        _norm1 = new LayerNormalizationLayer<T>(dim);
        _norm2 = new LayerNormalizationLayer<T>(dim);

        // QKV projection (3 * dim for Q, K, V combined)
        _qkvProj = new DenseLayer<T>(dim, dim * 3);
        _outProj = new DenseLayer<T>(dim, dim);

        // Relative position bias table
        int biasTableSize = (2 * windowSize - 1) * (2 * windowSize - 1);
        _relativePositionBiasTable = new Tensor<T>([biasTableSize, numHeads]);
        InitializeRelativePositionBias();

        // Compute relative position index for windows
        _relativePositionIndex = ComputeRelativePositionIndex(windowSize);

        // MLP (2-layer with GELU)
        int mlpHiddenDim = dim * mlpRatio;
        _mlpFc1 = new DenseLayer<T>(dim, mlpHiddenDim, (IActivationFunction<T>)new GELUActivation<T>());
        _mlpFc2 = new DenseLayer<T>(mlpHiddenDim, dim);
    }

    private void InitializeRelativePositionBias()
    {
        // Initialize with truncated normal distribution (std=0.02 from paper)
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            double value = random.NextGaussian(0, 0.02);
            // Truncate to [-0.04, 0.04]
            value = Math.Max(-0.04, Math.Min(0.04, value));
            _relativePositionBiasTable[i] = NumOps.FromDouble(value);
        }
    }

    private static int[,] ComputeRelativePositionIndex(int windowSize)
    {
        int windowArea = windowSize * windowSize;
        var relativePositionIndex = new int[windowArea, windowArea];

        for (int i = 0; i < windowArea; i++)
        {
            int h1 = i / windowSize;
            int w1 = i % windowSize;

            for (int j = 0; j < windowArea; j++)
            {
                int h2 = j / windowSize;
                int w2 = j % windowSize;

                // Relative position
                int relH = h1 - h2 + windowSize - 1;
                int relW = w1 - w2 + windowSize - 1;

                // Index into bias table
                relativePositionIndex[i, j] = relH * (2 * windowSize - 1) + relW;
            }
        }

        return relativePositionIndex;
    }

    /// <summary>
    /// Performs the forward pass through the Swin Transformer block.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, seqLen, dim].</param>
    /// <returns>Output tensor of shape [batch, seqLen, dim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, seq_len, dim]
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Cache for backward
        _cachedInput = input;

        // Infer spatial dimensions (assume square or find valid factorization)
        int h = (int)Math.Sqrt(seqLen);
        int w = seqLen / h;
        if (h * w != seqLen)
        {
            // Find valid factorization
            for (int candidate = h; candidate >= 1; candidate--)
            {
                if (seqLen % candidate == 0)
                {
                    h = seqLen / candidate;
                    w = candidate;
                    break;
                }
            }
        }
        _cachedH = h;
        _cachedW = w;

        // Pre-norm + Window attention + residual
        var normed1 = _norm1.Forward(input);
        _cachedNorm1Output = normed1;

        var attnOut = WindowAttention(normed1, h, w, batch);
        _cachedAttnOutput = attnOut;

        var residual1 = AddTensors(input, attnOut);
        _cachedResidual1 = residual1;

        // Pre-norm + MLP + residual
        var normed2 = _norm2.Forward(residual1);
        _cachedNorm2Output = normed2;

        var mlpOut = ApplyMLP(normed2);
        var output = AddTensors(residual1, mlpOut);

        return output;
    }

    private Tensor<T> WindowAttention(Tensor<T> x, int h, int w, int batch)
    {
        int seqLen = x.Shape[1];

        // Reshape to spatial: [B, H, W, C]
        var spatial = ReshapeToSpatial(x, batch, h, w, _dim);

        // Apply cyclic shift if needed
        if (_shiftSize > 0)
        {
            spatial = CyclicShift(spatial, -_shiftSize);
        }

        // Partition into windows: [numWindows*B, windowSize*windowSize, C]
        var (windows, numWindowsH, numWindowsW) = WindowPartition(spatial);

        // Apply attention within each window
        var attnOut = WindowedSelfAttention(windows);

        // Merge windows back: [B, H, W, C]
        var merged = WindowReverse(attnOut, numWindowsH, numWindowsW, batch, h, w);

        // Reverse cyclic shift if applied
        if (_shiftSize > 0)
        {
            merged = CyclicShift(merged, _shiftSize);
        }

        // Reshape back to sequence: [B, H*W, C]
        return ReshapeToSequence(merged);
    }

    private Tensor<T> ReshapeToSpatial(Tensor<T> x, int batch, int h, int w, int c)
    {
        var spatial = new Tensor<T>([batch, h, w, c]);
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    int seqIdx = i * w + j;
                    for (int d = 0; d < c; d++)
                    {
                        spatial[b, i, j, d] = x[b, seqIdx, d];
                    }
                }
            }
        }
        return spatial;
    }

    private Tensor<T> ReshapeToSequence(Tensor<T> spatial)
    {
        int batch = spatial.Shape[0];
        int h = spatial.Shape[1];
        int w = spatial.Shape[2];
        int c = spatial.Shape[3];

        var seq = new Tensor<T>([batch, h * w, c]);
        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    int seqIdx = i * w + j;
                    for (int d = 0; d < c; d++)
                    {
                        seq[b, seqIdx, d] = spatial[b, i, j, d];
                    }
                }
            }
        }
        return seq;
    }

    private Tensor<T> CyclicShift(Tensor<T> x, int shift)
    {
        int batch = x.Shape[0];
        int h = x.Shape[1];
        int w = x.Shape[2];
        int c = x.Shape[3];

        var shifted = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // Compute source indices with cyclic wrapping
                    int srcI = (i - shift % h + h) % h;
                    int srcJ = (j - shift % w + w) % w;

                    for (int d = 0; d < c; d++)
                    {
                        shifted[b, i, j, d] = x[b, srcI, srcJ, d];
                    }
                }
            }
        }

        return shifted;
    }

    private (Tensor<T> windows, int numWindowsH, int numWindowsW) WindowPartition(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int h = x.Shape[1];
        int w = x.Shape[2];
        int c = x.Shape[3];

        // Pad if necessary to make dimensions divisible by window size
        int padH = (_windowSize - h % _windowSize) % _windowSize;
        int padW = (_windowSize - w % _windowSize) % _windowSize;
        int paddedH = h + padH;
        int paddedW = w + padW;

        Tensor<T> padded;
        if (padH > 0 || padW > 0)
        {
            padded = new Tensor<T>([batch, paddedH, paddedW, c]);
            for (int b = 0; b < batch; b++)
            {
                for (int i = 0; i < paddedH; i++)
                {
                    for (int j = 0; j < paddedW; j++)
                    {
                        for (int d = 0; d < c; d++)
                        {
                            if (i < h && j < w)
                                padded[b, i, j, d] = x[b, i, j, d];
                            else
                                padded[b, i, j, d] = NumOps.Zero;
                        }
                    }
                }
            }
        }
        else
        {
            padded = x;
        }

        int numWindowsH = paddedH / _windowSize;
        int numWindowsW = paddedW / _windowSize;
        int numWindows = numWindowsH * numWindowsW;
        int windowArea = _windowSize * _windowSize;

        var windows = new Tensor<T>([batch * numWindows, windowArea, c]);

        for (int b = 0; b < batch; b++)
        {
            for (int wh = 0; wh < numWindowsH; wh++)
            {
                for (int ww = 0; ww < numWindowsW; ww++)
                {
                    int windowIdx = b * numWindows + wh * numWindowsW + ww;
                    int startH = wh * _windowSize;
                    int startW = ww * _windowSize;

                    for (int i = 0; i < _windowSize; i++)
                    {
                        for (int j = 0; j < _windowSize; j++)
                        {
                            int tokenIdx = i * _windowSize + j;
                            for (int d = 0; d < c; d++)
                            {
                                windows[windowIdx, tokenIdx, d] = padded[b, startH + i, startW + j, d];
                            }
                        }
                    }
                }
            }
        }

        return (windows, numWindowsH, numWindowsW);
    }

    private Tensor<T> WindowReverse(Tensor<T> windows, int numWindowsH, int numWindowsW, int batch, int h, int w)
    {
        int numWindows = numWindowsH * numWindowsW;
        int c = windows.Shape[2];

        var spatial = new Tensor<T>([batch, h, w, c]);

        for (int b = 0; b < batch; b++)
        {
            for (int wh = 0; wh < numWindowsH; wh++)
            {
                for (int ww = 0; ww < numWindowsW; ww++)
                {
                    int windowIdx = b * numWindows + wh * numWindowsW + ww;
                    int startH = wh * _windowSize;
                    int startW = ww * _windowSize;

                    for (int i = 0; i < _windowSize; i++)
                    {
                        for (int j = 0; j < _windowSize; j++)
                        {
                            int outH = startH + i;
                            int outW = startW + j;

                            // Only copy if within original bounds
                            if (outH < h && outW < w)
                            {
                                int tokenIdx = i * _windowSize + j;
                                for (int d = 0; d < c; d++)
                                {
                                    spatial[b, outH, outW, d] = windows[windowIdx, tokenIdx, d];
                                }
                            }
                        }
                    }
                }
            }
        }

        return spatial;
    }

    private Tensor<T> WindowedSelfAttention(Tensor<T> windows)
    {
        int numWindows = windows.Shape[0];
        int windowArea = windows.Shape[1];
        int c = windows.Shape[2];

        // Project to Q, K, V for all windows
        var qkv = new Tensor<T>([numWindows, windowArea, 3 * c]);
        for (int win = 0; win < numWindows; win++)
        {
            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>([1, c]);
                for (int d = 0; d < c; d++)
                {
                    tokenIn[0, d] = windows[win, t, d];
                }
                var tokenQkv = _qkvProj.Forward(tokenIn);
                for (int d = 0; d < 3 * c; d++)
                {
                    qkv[win, t, d] = tokenQkv[0, d];
                }
            }
        }

        // Compute attention per window
        var output = new Tensor<T>([numWindows, windowArea, c]);

        for (int win = 0; win < numWindows; win++)
        {
            // Compute attention scores for this window
            var attnScores = new double[_numHeads, windowArea, windowArea];

            for (int head = 0; head < _numHeads; head++)
            {
                int headOffset = head * _headDim;

                for (int i = 0; i < windowArea; i++)
                {
                    for (int j = 0; j < windowArea; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            double q = NumOps.ToDouble(qkv[win, i, headOffset + d]);
                            double k = NumOps.ToDouble(qkv[win, j, c + headOffset + d]);
                            score += q * k;
                        }
                        score *= _scale;

                        // Add relative position bias
                        int biasIdx = _relativePositionIndex[i, j];
                        score += NumOps.ToDouble(_relativePositionBiasTable[biasIdx, head]);

                        attnScores[head, i, j] = score;
                    }
                }
            }

            // Softmax per head per query
            var attnProbs = new double[_numHeads, windowArea, windowArea];
            for (int head = 0; head < _numHeads; head++)
            {
                for (int i = 0; i < windowArea; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < windowArea; j++)
                    {
                        if (attnScores[head, i, j] > maxScore)
                            maxScore = attnScores[head, i, j];
                    }

                    double sumExp = 0;
                    for (int j = 0; j < windowArea; j++)
                    {
                        attnProbs[head, i, j] = Math.Exp(attnScores[head, i, j] - maxScore);
                        sumExp += attnProbs[head, i, j];
                    }

                    for (int j = 0; j < windowArea; j++)
                    {
                        attnProbs[head, i, j] /= sumExp;
                    }
                }
            }

            // Apply attention to values and concatenate heads
            var attnOut = new double[windowArea, c];
            for (int head = 0; head < _numHeads; head++)
            {
                int headOffset = head * _headDim;
                int vOffset = 2 * c + headOffset;

                for (int i = 0; i < windowArea; i++)
                {
                    for (int d = 0; d < _headDim; d++)
                    {
                        double val = 0;
                        for (int j = 0; j < windowArea; j++)
                        {
                            val += attnProbs[head, i, j] * NumOps.ToDouble(qkv[win, j, vOffset + d]);
                        }
                        attnOut[i, headOffset + d] = val;
                    }
                }
            }

            // Output projection
            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>([1, c]);
                for (int d = 0; d < c; d++)
                {
                    tokenIn[0, d] = NumOps.FromDouble(attnOut[t, d]);
                }
                var tokenOut = _outProj.Forward(tokenIn);
                for (int d = 0; d < c; d++)
                {
                    output[win, t, d] = tokenOut[0, d];
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyMLP(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var tokenIn = new Tensor<T>([1, _dim]);
                for (int d = 0; d < _dim; d++)
                {
                    tokenIn[0, d] = x[b, s, d];
                }

                // FC1 + GELU (activation built into layer)
                var hidden = _mlpFc1.Forward(tokenIn);

                // FC2
                var tokenOut = _mlpFc2.Forward(hidden);
                for (int d = 0; d < _dim; d++)
                {
                    result[b, s, d] = tokenOut[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer.</param>
    /// <returns>Gradient for the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Simplified backward pass - compute gradients through the network
        // Full implementation would require caching intermediate values

        // Backward through second residual
        var mlpGrad = BackwardMLP(outputGradient);
        var norm2Grad = _norm2.Backward(mlpGrad);
        var residual1Grad = AddTensors(outputGradient, norm2Grad);

        // Backward through first residual
        var attnGrad = BackwardWindowAttention(residual1Grad);
        var norm1Grad = _norm1.Backward(attnGrad);
        var inputGrad = AddTensors(residual1Grad, norm1Grad);

        return inputGrad;
    }

    private Tensor<T> BackwardMLP(Tensor<T> gradient)
    {
        int batch = gradient.Shape[0];
        int seqLen = gradient.Shape[1];

        var result = new Tensor<T>(gradient.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var tokenGrad = new Tensor<T>([1, _dim]);
                for (int d = 0; d < _dim; d++)
                {
                    tokenGrad[0, d] = gradient[b, s, d];
                }

                var fc2Grad = _mlpFc2.Backward(tokenGrad);
                var fc1Grad = _mlpFc1.Backward(fc2Grad);

                for (int d = 0; d < _dim; d++)
                {
                    result[b, s, d] = fc1Grad[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> BackwardWindowAttention(Tensor<T> gradient)
    {
        // Simplified backward - in full implementation would need to track
        // all intermediate values and compute gradients through attention
        int batch = gradient.Shape[0];
        int seqLen = gradient.Shape[1];

        var result = new Tensor<T>(gradient.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var tokenGrad = new Tensor<T>([1, _dim]);
                for (int d = 0; d < _dim; d++)
                {
                    tokenGrad[0, d] = gradient[b, s, d];
                }

                var outProjGrad = _outProj.Backward(tokenGrad);
                var qkvGrad = _qkvProj.Backward(outProjGrad);

                for (int d = 0; d < _dim; d++)
                {
                    result[b, s, d] = qkvGrad[0, d];
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        allParams.AddRange(_norm1.GetParameters().Data);
        allParams.AddRange(_norm2.GetParameters().Data);
        allParams.AddRange(_qkvProj.GetParameters().Data);
        allParams.AddRange(_outProj.GetParameters().Data);

        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            allParams.Add(_relativePositionBiasTable[i]);
        }

        allParams.AddRange(_mlpFc1.GetParameters().Data);
        allParams.AddRange(_mlpFc2.GetParameters().Data);

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Norm1
        int norm1Count = _norm1.ParameterCount;
        _norm1.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(norm1Count).ToArray()));
        offset += norm1Count;

        // Norm2
        int norm2Count = _norm2.ParameterCount;
        _norm2.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(norm2Count).ToArray()));
        offset += norm2Count;

        // QKV projection
        int qkvCount = _qkvProj.ParameterCount;
        _qkvProj.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(qkvCount).ToArray()));
        offset += qkvCount;

        // Output projection
        int outCount = _outProj.ParameterCount;
        _outProj.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(outCount).ToArray()));
        offset += outCount;

        // Relative position bias
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            _relativePositionBiasTable[i] = parameters[offset + i];
        }
        offset += _relativePositionBiasTable.Length;

        // MLP FC1
        int fc1Count = _mlpFc1.ParameterCount;
        _mlpFc1.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(fc1Count).ToArray()));
        offset += fc1Count;

        // MLP FC2
        int fc2Count = _mlpFc2.ParameterCount;
        _mlpFc2.SetParameters(new Vector<T>(parameters.Data.Skip(offset).Take(fc2Count).ToArray()));
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        allGrads.AddRange(_norm1.GetParameterGradients().Data);
        allGrads.AddRange(_norm2.GetParameterGradients().Data);
        allGrads.AddRange(_qkvProj.GetParameterGradients().Data);
        allGrads.AddRange(_outProj.GetParameterGradients().Data);

        // Bias table gradients (would need to be computed in backward)
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            allGrads.Add(NumOps.Zero);
        }

        allGrads.AddRange(_mlpFc1.GetParameterGradients().Data);
        allGrads.AddRange(_mlpFc2.GetParameterGradients().Data);

        return new Vector<T>([.. allGrads]);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _norm1.ResetState();
        _norm2.ResetState();
        _qkvProj.ResetState();
        _outProj.ResetState();
        _mlpFc1.ResetState();
        _mlpFc2.ResetState();

        _cachedInput = null;
        _cachedNorm1Output = null;
        _cachedAttnOutput = null;
        _cachedResidual1 = null;
        _cachedNorm2Output = null;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _qkvProj.UpdateParameters(learningRate);
        _outProj.UpdateParameters(learningRate);
        _mlpFc1.UpdateParameters(learningRate);
        _mlpFc2.UpdateParameters(learningRate);

        // Update relative position bias table (simplified gradient descent)
        // In a full implementation, this would use stored gradients
    }

    /// <inheritdoc/>
    public override bool SupportsJitCompilation =>
        _norm1 != null && _norm1.SupportsJitCompilation &&
        _norm2 != null && _norm2.SupportsJitCompilation &&
        _qkvProj != null && _qkvProj.SupportsJitCompilation &&
        _outProj != null && _outProj.SupportsJitCompilation &&
        _mlpFc1 != null && _mlpFc1.SupportsJitCompilation &&
        _mlpFc2 != null && _mlpFc2.SupportsJitCompilation;

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        // Create symbolic input node: [batch, seqLen, dim]
        var symbolicInput = new Tensor<T>([1, _windowSize * _windowSize, _dim]);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "swin_block_input");
        inputNodes.Add(inputNode);

        // Export norm1 graph
        var norm1Nodes = new List<ComputationNode<T>> { inputNode };
        var normed1 = _norm1.ExportComputationGraph(norm1Nodes);

        // Export QKV projection
        var qkvNodes = new List<ComputationNode<T>> { normed1 };
        var qkvOut = _qkvProj.ExportComputationGraph(qkvNodes);

        // Window attention is complex - for JIT we represent it as a composite operation
        // In practice, this would be optimized as a fused attention kernel
        var outProjNodes = new List<ComputationNode<T>> { qkvOut };
        var attnOut = _outProj.ExportComputationGraph(outProjNodes);

        // First residual: input + attn
        var residual1 = TensorOperations<T>.Add(inputNode, attnOut);

        // Export norm2 graph
        var norm2Nodes = new List<ComputationNode<T>> { residual1 };
        var normed2 = _norm2.ExportComputationGraph(norm2Nodes);

        // Export MLP: fc1 -> gelu -> fc2
        var fc1Nodes = new List<ComputationNode<T>> { normed2 };
        var mlpHidden = _mlpFc1.ExportComputationGraph(fc1Nodes);

        var fc2Nodes = new List<ComputationNode<T>> { mlpHidden };
        var mlpOut = _mlpFc2.ExportComputationGraph(fc2Nodes);

        // Second residual: residual1 + mlp
        var output = TensorOperations<T>.Add(residual1, mlpOut);

        return output;
    }
}
