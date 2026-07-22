using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.Transformer)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "1, 16, 16", TestConstructorArgs = "16, 2, 4")]
public partial class SwinTransformerBlockLayer<T> : LayerBase<T>
{
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _windowSize;
    private readonly int _shiftSize;
    private readonly double _scale;
    private readonly int _mlpRatio;
    // Stochastic depth (DropPath) rate per Swin (Liu et al. 2021): each residual branch is dropped
    // per-sample with this probability during training and scaled by 1/(1-rate) so the expected
    // activation is preserved; at inference it is the identity. 0 (default) keeps the block unchanged.
    private readonly double _dropPathRate;
    // Forward-call counter so RandomSeed-seeded drop masks are bit-identical across reruns at the
    // same step (mirrors DropoutLayer's determinism contract).
    private long _dropPathForwardCounter;

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
    private Tensor<T>? _cachedQkv; // [numWindows, windowArea, 3*dim]
    private int _cachedNumWindows;
    private int _cachedWindowArea;
    private int _cachedH;
    private int _cachedW;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount =>
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
        int mlpRatio = 4,
        double dropPathRate = 0.0)
        : base([dim], [dim])
    {
        if (dim % numHeads != 0)
            throw new ArgumentException($"Dimension ({dim}) must be divisible by number of heads ({numHeads}).", nameof(dim));
        if (dropPathRate < 0.0 || dropPathRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(dropPathRate), "Drop-path rate must be in [0, 1).");

        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _windowSize = windowSize;
        _shiftSize = shiftSize;
        _mlpRatio = mlpRatio;
        _dropPathRate = dropPathRate;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Layer normalizations (pre-norm architecture)
        _norm1 = new LayerNormalizationLayer<T>();
        _norm2 = new LayerNormalizationLayer<T>();

        // QKV projection (3 * dim for Q, K, V combined)
        _qkvProj = new DenseLayer<T>(dim * 3);
        _outProj = new DenseLayer<T>(dim);

        // Relative position bias table
        int biasTableSize = (2 * windowSize - 1) * (2 * windowSize - 1);
        _relativePositionBiasTable = new Tensor<T>([biasTableSize, numHeads]);
        InitializeRelativePositionBias();
        RegisterTrainableParameter(_relativePositionBiasTable, PersistentTensorRole.Weights);

        // Compute relative position index for windows
        _relativePositionIndex = ComputeRelativePositionIndex(windowSize);

        // MLP (2-layer with GELU)
        int mlpHiddenDim = dim * mlpRatio;
        _mlpFc1 = new DenseLayer<T>(mlpHiddenDim, (IActivationFunction<T>)new GELUActivation<T>());
        _mlpFc2 = new DenseLayer<T>(dim);

        RegisterSubLayer(_norm1);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_qkvProj);
        RegisterSubLayer(_outProj);
        RegisterSubLayer(_mlpFc1);
        RegisterSubLayer(_mlpFc2);
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

        // Stochastic depth on the attention residual branch (Swin, Liu et al. 2021).
        var residual1 = AddTensors(input, DropPath(attnOut, batch));
        _cachedResidual1 = residual1;

        // Pre-norm + MLP + residual
        var normed2 = _norm2.Forward(residual1);
        _cachedNorm2Output = normed2;

        var mlpOut = ApplyMLP(normed2);
        // Stochastic depth on the MLP residual branch (independent draw from the attention branch).
        var output = AddTensors(residual1, DropPath(mlpOut, batch));

        return output;
    }

    private Tensor<T> WindowAttention(Tensor<T> x, int h, int w, int batch)
    {
        int seqLen = x.Shape[1];

        int effectiveShift = h <= _windowSize || w <= _windowSize
            ? 0
            : Math.Min(_shiftSize, _windowSize - 1);

        // Work in sequence form throughout. The gather-based rearrangements
        // below preserve the autodiff edge while expressing the same spatial
        // window layout as [B,H,W,C].
        var shifted = x;

        // Apply cyclic shift if needed
        if (effectiveShift > 0)
        {
            shifted = CyclicShift(shifted, h, w, -effectiveShift);
        }

        // Partition into windows: [numWindows*B, windowSize*windowSize, C]
        var (windows, numWindowsH, numWindowsW) = WindowPartition(shifted, h, w);

        // Apply attention within each window
        var attnOut = WindowedSelfAttention(
            windows, batch, h, w, numWindowsH, numWindowsW, effectiveShift);

        // Merge windows back: [B, H*W, C]
        var merged = WindowReverse(attnOut, numWindowsH, numWindowsW, batch, h, w);

        // Reverse cyclic shift if applied
        if (effectiveShift > 0)
        {
            merged = CyclicShift(merged, h, w, effectiveShift);
        }

        return merged;
    }

    private static Tensor<int> CreateIndices(int[] values)
    {
        var indices = new Tensor<int>([values.Length]);
        for (int i = 0; i < values.Length; i++) indices[i] = values[i];
        return indices;
    }

    private Tensor<T> CyclicShift(Tensor<T> x, int h, int w, int shift)
    {
        var order = new int[h * w];
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                int srcI = (i - shift % h + h) % h;
                int srcJ = (j - shift % w + w) % w;
                order[i * w + j] = srcI * w + srcJ;
            }
        }
        return Engine.TensorGather(x, CreateIndices(order), axis: 1);
    }

    private (Tensor<T> windows, int numWindowsH, int numWindowsW) WindowPartition(
        Tensor<T> x, int h, int w)
    {
        int batch = x.Shape[0];
        int c = x.Shape[2];

        // Pad if necessary to make dimensions divisible by window size
        int padH = (_windowSize - h % _windowSize) % _windowSize;
        int padW = (_windowSize - w % _windowSize) % _windowSize;
        int paddedH = h + padH;
        int paddedW = w + padW;

        Tensor<T> source;
        if (padH > 0 || padW > 0)
        {
            // Append one zero token and point every padded position at it.
            // The constant pad has no gradient; valid gathers remain connected.
            source = Engine.TensorConcatenate(
                [x, new Tensor<T>([batch, 1, c])], axis: 1);
        }
        else
        {
            source = x;
        }

        int numWindowsH = paddedH / _windowSize;
        int numWindowsW = paddedW / _windowSize;
        int numWindows = numWindowsH * numWindowsW;
        int windowArea = _windowSize * _windowSize;

        var order = new int[paddedH * paddedW];
        int destination = 0;
        int padIndex = h * w;
        for (int wh = 0; wh < numWindowsH; wh++)
        {
            for (int ww = 0; ww < numWindowsW; ww++)
            {
                for (int i = 0; i < _windowSize; i++)
                {
                    for (int j = 0; j < _windowSize; j++)
                    {
                        int row = wh * _windowSize + i;
                        int col = ww * _windowSize + j;
                        order[destination++] = row < h && col < w
                            ? row * w + col
                            : padIndex;
                    }
                }
            }
        }

        var gathered = Engine.TensorGather(source, CreateIndices(order), axis: 1);
        var windows = Engine.Reshape(gathered, [batch * numWindows, windowArea, c]);
        return (windows, numWindowsH, numWindowsW);
    }

    private Tensor<T> WindowReverse(Tensor<T> windows, int numWindowsH, int numWindowsW, int batch, int h, int w)
    {
        int numWindows = numWindowsH * numWindowsW;
        int c = windows.Shape[2];

        var windowSequence = Engine.Reshape(
            windows, [batch, numWindows * _windowSize * _windowSize, c]);
        var order = new int[h * w];
        for (int row = 0; row < h; row++)
        {
            for (int col = 0; col < w; col++)
            {
                int wh = row / _windowSize;
                int ww = col / _windowSize;
                int localRow = row % _windowSize;
                int localCol = col % _windowSize;
                int window = wh * numWindowsW + ww;
                order[row * w + col] = window * _windowSize * _windowSize
                    + localRow * _windowSize + localCol;
            }
        }
        return Engine.TensorGather(windowSequence, CreateIndices(order), axis: 1);
    }

    private Tensor<T> WindowedSelfAttention(
        Tensor<T> windows,
        int batch,
        int h,
        int w,
        int numWindowsH,
        int numWindowsW,
        int effectiveShift)
    {
        int numWindows = windows.Shape[0];
        int windowArea = windows.Shape[1];
        int c = windows.Shape[2];

        // Project to Q, K, V for all windows (batched for correct backward)
        var flatWindows = Engine.Reshape(windows, [numWindows * windowArea, c]);
        var flatQkv = _qkvProj.Forward(flatWindows);
        var qkv = Engine.Reshape(flatQkv, [numWindows, windowArea, 3 * c]);
        _cachedQkv = qkv;
        _cachedNumWindows = numWindows;
        _cachedWindowArea = windowArea;

        var qFlat = Engine.TensorSlice(qkv, [0, 0, 0], [numWindows, windowArea, c]);
        var kFlat = Engine.TensorSlice(qkv, [0, 0, c], [numWindows, windowArea, c]);
        var vFlat = Engine.TensorSlice(qkv, [0, 0, 2 * c], [numWindows, windowArea, c]);
        var q = Engine.TensorPermute(
            Engine.Reshape(qFlat, [numWindows, windowArea, _numHeads, _headDim]),
            [0, 2, 1, 3]);
        var k = Engine.TensorPermute(
            Engine.Reshape(kFlat, [numWindows, windowArea, _numHeads, _headDim]),
            [0, 2, 3, 1]);
        var v = Engine.TensorPermute(
            Engine.Reshape(vFlat, [numWindows, windowArea, _numHeads, _headDim]),
            [0, 2, 1, 3]);

        var scores = Engine.TensorMultiplyScalar(
            Engine.BatchMatMul(q, k), NumOps.FromDouble(_scale));

        // Gather the paper's learned relative-position bias table into
        // [1, heads, windowArea, windowArea] and broadcast across windows.
        var biasIndices = new int[windowArea * windowArea];
        for (int i = 0; i < windowArea; i++)
            for (int j = 0; j < windowArea; j++)
                biasIndices[i * windowArea + j] = _relativePositionIndex[i, j];
        var relativeBias = Engine.TensorGather(
            _relativePositionBiasTable, CreateIndices(biasIndices), axis: 0);
        relativeBias = Engine.Reshape(relativeBias, [windowArea, windowArea, _numHeads]);
        relativeBias = Engine.TensorPermute(relativeBias, [2, 0, 1]);
        relativeBias = Engine.Reshape(relativeBias, [1, _numHeads, windowArea, windowArea]);
        scores = Engine.TensorBroadcastAdd(scores, relativeBias);

        var mask = CreateAttentionMask(
            batch, h, w, numWindowsH, numWindowsW, effectiveShift);
        if (mask is not null)
            scores = Engine.TensorBroadcastAdd(scores, mask);

        var probabilities = Engine.Softmax(scores, axis: -1);
        var context = Engine.BatchMatMul(probabilities, v);
        context = Engine.TensorPermute(context, [0, 2, 1, 3]);
        var output = Engine.Reshape(context, [numWindows * windowArea, c]);

        // Batch output projection across ALL windows and tokens for correct backward
        var flatProjOut = _outProj.Forward(output);
        return Engine.Reshape(flatProjOut, [numWindows, windowArea, c]);
    }

    private Tensor<T>? CreateAttentionMask(
        int batch,
        int h,
        int w,
        int numWindowsH,
        int numWindowsW,
        int shift)
    {
        int paddedH = numWindowsH * _windowSize;
        int paddedW = numWindowsW * _windowSize;
        bool hasPadding = paddedH != h || paddedW != w;
        if (shift == 0 && !hasPadding) return null;

        int windowArea = _windowSize * _windowSize;
        int windowsPerBatch = numWindowsH * numWindowsW;
        var mask = new Tensor<T>([batch * windowsPerBatch, 1, windowArea, windowArea]);
        T blocked = NumOps.FromDouble(-100.0);

        static int Region(int coordinate, int length, int windowSize, int shiftSize)
        {
            if (shiftSize == 0) return 0;
            if (coordinate < length - windowSize) return 0;
            if (coordinate < length - shiftSize) return 1;
            return 2;
        }

        for (int b = 0; b < batch; b++)
        {
            for (int wh = 0; wh < numWindowsH; wh++)
            {
                for (int ww = 0; ww < numWindowsW; ww++)
                {
                    int window = b * windowsPerBatch + wh * numWindowsW + ww;
                    for (int query = 0; query < windowArea; query++)
                    {
                        int qi = wh * _windowSize + query / _windowSize;
                        int qj = ww * _windowSize + query % _windowSize;
                        int queryRegion = Region(qi, paddedH, _windowSize, shift) * 3
                            + Region(qj, paddedW, _windowSize, shift);
                        for (int key = 0; key < windowArea; key++)
                        {
                            int ki = wh * _windowSize + key / _windowSize;
                            int kj = ww * _windowSize + key % _windowSize;
                            int keyRegion = Region(ki, paddedH, _windowSize, shift) * 3
                                + Region(kj, paddedW, _windowSize, shift);
                            if (queryRegion != keyRegion || ki >= h || kj >= w)
                                mask[window, 0, query, key] = blocked;
                        }
                    }
                }
            }
        }

        return mask;
    }

    private Tensor<T> ApplyMLP(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];

        // Batch all tokens for correct Forward/Backward (single _lastInput)
        var flat = x.Reshape([batch * seqLen, _dim]);
        var hidden = _mlpFc1.Forward(flat);
        var flatOut = _mlpFc2.Forward(hidden);
        return flatOut.Reshape([batch, seqLen, _dim]);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    /// <summary>
    /// Stochastic depth (drop-path), as used by the Swin Transformer (Liu et al. 2021, following
    /// Huang et al. 2016). During training each sample's entire residual <paramref name="branch"/> is
    /// dropped with probability <c>_dropPathRate</c> and the survivors are scaled by <c>1/(1-rate)</c>
    /// so the expected contribution is preserved; at inference (or when the rate is 0) it is the
    /// identity. The per-sample keep/drop mask is applied with the tape-recorded
    /// <see cref="IEngine.TensorMultiply{T}"/> so the gradient flows back through it automatically
    /// (grad·mask), exactly like Dropout. The mask is constant w.r.t. the parameters, so the attention
    /// and MLP sub-layers see their gradients scaled by the same factor that scaled their forward output.
    /// </summary>
    private Tensor<T> DropPath(Tensor<T> branch, int batch)
    {
        if (_dropPathRate <= 0.0 || !IsTrainingMode)
            return branch;

        double keepProb = 1.0 - _dropPathRate;
        T scale = NumOps.FromDouble(1.0 / keepProb);

        // Per-call seed derived from RandomSeed + a forward counter so identically-seeded models
        // produce bit-identical drop masks at the same step (matches DropoutLayer's determinism
        // contract). With no seed, fall back to a cryptographically secure RNG (production default).
        // long (not ulong): .NET Framework 4.7.1 has no Interlocked.Increment(ref ulong) overload.
        long counter = System.Threading.Interlocked.Increment(ref _dropPathForwardCounter);
        var rng = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(unchecked((int)((uint)RandomSeed.Value * 2654435761u ^ (uint)counter)))
            : RandomHelper.CreateSecureRandom();

        int seqLen = branch.Shape[1];
        int dim = branch.Shape[2];
        var mask = new Tensor<T>([batch, seqLen, dim]);
        for (int b = 0; b < batch; b++)
        {
            // One keep/drop decision per sample — the whole branch survives or vanishes together.
            T v = rng.NextDouble() < _dropPathRate ? NumOps.Zero : scale;
            for (int s = 0; s < seqLen; s++)
                for (int d = 0; d < dim; d++)
                    mask[b, s, d] = v;
        }
        return Engine.TensorMultiply(branch, mask);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        allParams.AddRange(_norm1.GetParameters().ToArray());
        allParams.AddRange(_norm2.GetParameters().ToArray());
        allParams.AddRange(_qkvProj.GetParameters().ToArray());
        allParams.AddRange(_outProj.GetParameters().ToArray());

        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            allParams.Add(_relativePositionBiasTable[i]);
        }

        allParams.AddRange(_mlpFc1.GetParameters().ToArray());
        allParams.AddRange(_mlpFc2.GetParameters().ToArray());

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Lazy ctor: sublayers may be in placeholder shape state. Resolve from
        // known constants — every Dense/Norm operates on the per-token feature
        // dim (dim) except _mlpFc2 which contracts back from dim*mlpRatio.
        if (!_norm1.IsShapeResolved) _norm1.ResolveFromShape(new[] { _dim });
        if (!_norm2.IsShapeResolved) _norm2.ResolveFromShape(new[] { _dim });
        if (!_qkvProj.IsShapeResolved) _qkvProj.ResolveFromShape(new[] { _dim });
        if (!_outProj.IsShapeResolved) _outProj.ResolveFromShape(new[] { _dim });
        if (!_mlpFc1.IsShapeResolved) _mlpFc1.ResolveFromShape(new[] { _dim });
        if (!_mlpFc2.IsShapeResolved) _mlpFc2.ResolveFromShape(new[] { _dim * _mlpRatio });

        int offset = 0;

        // Norm1
        int norm1Count = checked((int)_norm1.ParameterCount);
        _norm1.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, norm1Count).ToArray()));
        offset += norm1Count;

        // Norm2
        int norm2Count = checked((int)_norm2.ParameterCount);
        _norm2.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, norm2Count).ToArray()));
        offset += norm2Count;

        // QKV projection
        int qkvCount = checked((int)_qkvProj.ParameterCount);
        _qkvProj.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, qkvCount).ToArray()));
        offset += qkvCount;

        // Output projection
        int outCount = checked((int)_outProj.ParameterCount);
        _outProj.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, outCount).ToArray()));
        offset += outCount;

        // Relative position bias
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            _relativePositionBiasTable[i] = parameters[offset + i];
        }
        offset += _relativePositionBiasTable.Length;

        // MLP FC1
        int fc1Count = checked((int)_mlpFc1.ParameterCount);
        _mlpFc1.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, fc1Count).ToArray()));
        offset += fc1Count;

        // MLP FC2
        int fc2Count = checked((int)_mlpFc2.ParameterCount);
        _mlpFc2.SetParameters(new Vector<T>(parameters.AsSpan().Slice(offset, fc2Count).ToArray()));
    }

    /// <summary>
    /// Emits the constructor-level settings that cannot be inferred from the layer's
    /// input/output shapes alone, so the deserializer can rebuild this block with the
    /// exact same configuration (and therefore the exact same per-sublayer parameter
    /// counts). Without these keys the reflection-driven deserialization fallback
    /// reconstructs the block with default head/window/MLP settings, producing a
    /// different ParameterCount than was serialized and throwing in SetParameters.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        // dim is recoverable from the input shape, but persist it too so the
        // metadata-match scorer locks onto the intended constructor unambiguously.
        metadata["Dim"] = _dim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["WindowSize"] = _windowSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ShiftSize"] = _shiftSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["MlpRatio"] = _mlpRatio.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        allGrads.AddRange(_norm1.GetParameterGradients().ToArray());
        allGrads.AddRange(_norm2.GetParameterGradients().ToArray());
        allGrads.AddRange(_qkvProj.GetParameterGradients().ToArray());
        allGrads.AddRange(_outProj.GetParameterGradients().ToArray());

        // Bias table gradients (would need to be computed in backward)
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            allGrads.Add(NumOps.Zero);
        }

        allGrads.AddRange(_mlpFc1.GetParameterGradients().ToArray());
        allGrads.AddRange(_mlpFc2.GetParameterGradients().ToArray());

        return new Vector<T>([.. allGrads]);
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _norm1.ClearGradients(); _norm2.ClearGradients();
        _qkvProj.ClearGradients(); _outProj.ClearGradients();
        _mlpFc1.ClearGradients(); _mlpFc2.ClearGradients();
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

}
