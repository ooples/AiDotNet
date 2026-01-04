using System.IO;
using AiDotNet.Extensions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Swin Transformer backbone for hierarchical vision transformer feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Swin Transformer is a hierarchical vision transformer
/// that uses shifted windows for efficient attention computation. Unlike ViT which
/// processes the entire image at once, Swin processes local windows and shifts them
/// between layers for cross-window connections.</para>
///
/// <para>Key features:
/// - Hierarchical structure with patch merging (like CNN stages)
/// - Window-based multi-head self-attention for efficiency
/// - Shifted window partitioning for cross-window connections
/// </para>
///
/// <para>Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021</para>
/// </remarks>
public class SwinTransformer<T> : BackboneBase<T>
{
    private readonly PatchEmbeddingBlock<T> _patchEmbed;
    private readonly List<SwinStage<T>> _stages;
    private readonly SwinVariant _variant;
    private readonly int _embedDim;
    private readonly int _windowSize;

    /// <inheritdoc/>
    public override string Name => $"Swin-{_variant}";

    /// <inheritdoc/>
    public override int[] OutputChannels { get; }

    /// <inheritdoc/>
    public override int[] Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Creates a new Swin Transformer backbone.
    /// </summary>
    /// <param name="variant">Swin variant (Tiny, Small, Base, Large).</param>
    /// <param name="windowSize">Window size for attention (default 7).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public SwinTransformer(SwinVariant variant = SwinVariant.SwinTiny, int windowSize = 7, int inChannels = 3)
    {
        _variant = variant;
        _windowSize = windowSize;
        _stages = new List<SwinStage<T>>();

        // Get configuration for variant
        var (embedDim, depths, numHeads) = GetVariantConfig(variant);
        _embedDim = embedDim;

        // Calculate output channels (doubles each stage after patch merging)
        OutputChannels = new int[4];
        for (int i = 0; i < 4; i++)
        {
            OutputChannels[i] = embedDim * (1 << i);
        }

        // Patch embedding: convert image to sequence of patches
        _patchEmbed = new PatchEmbeddingBlock<T>(
            patchSize: 4,
            inChannels: inChannels,
            embedDim: embedDim
        );

        // Build stages
        int currentDim = embedDim;
        for (int i = 0; i < 4; i++)
        {
            bool downsample = i > 0;

            var stage = new SwinStage<T>(
                dim: currentDim,
                depth: depths[i],
                numHeads: numHeads[i],
                windowSize: windowSize,
                downsample: downsample
            );

            _stages.Add(stage);

            if (downsample)
            {
                currentDim *= 2;
            }
        }
    }

    /// <summary>
    /// Gets the configuration for each Swin variant.
    /// </summary>
    private static (int embedDim, int[] depths, int[] numHeads) GetVariantConfig(SwinVariant variant) => variant switch
    {
        SwinVariant.SwinTiny => (96, new[] { 2, 2, 6, 2 }, new[] { 3, 6, 12, 24 }),
        SwinVariant.SwinSmall => (96, new[] { 2, 2, 18, 2 }, new[] { 3, 6, 12, 24 }),
        SwinVariant.SwinBase => (128, new[] { 2, 2, 18, 2 }, new[] { 4, 8, 16, 32 }),
        SwinVariant.SwinLarge => (192, new[] { 2, 2, 18, 2 }, new[] { 6, 12, 24, 48 }),
        _ => (96, new[] { 2, 2, 6, 2 }, new[] { 3, 6, 12, 24 })
    };

    /// <inheritdoc/>
    public override List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();

        // Patch embedding
        var x = _patchEmbed.Forward(input);

        // Stages with feature extraction
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);

            // Reshape to [batch, channels, height, width] for feature map output
            var featureMap = ReshapeToFeatureMap(x, input.Shape[2], input.Shape[3], i);
            features.Add(featureMap);
        }

        return features;
    }

    /// <summary>
    /// Reshapes sequence tensor to feature map format.
    /// </summary>
    private Tensor<T> ReshapeToFeatureMap(Tensor<T> x, int inputHeight, int inputWidth, int stageIdx)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];

        // Calculate output spatial dimensions
        int stride = Strides[stageIdx];

        // Validate input dimensions are divisible by stride
        if (inputHeight % stride != 0)
        {
            throw new ArgumentException(
                $"Input height ({inputHeight}) must be divisible by stride ({stride}) at stage {stageIdx}.",
                nameof(inputHeight));
        }
        if (inputWidth % stride != 0)
        {
            throw new ArgumentException(
                $"Input width ({inputWidth}) must be divisible by stride ({stride}) at stage {stageIdx}.",
                nameof(inputWidth));
        }

        int height = inputHeight / stride;
        int width = inputWidth / stride;

        // Validate sequence length matches expected spatial dimensions
        int expectedSeqLen = height * width;
        if (seqLen != expectedSeqLen)
        {
            throw new InvalidOperationException(
                $"Sequence length mismatch at stage {stageIdx}: expected {expectedSeqLen} (height={height}, width={width}), got {seqLen}. " +
                "Ensure input dimensions are divisible by patch size and all stride factors.");
        }

        var featureMap = new Tensor<T>(new[] { batch, dim, height, width });

        for (int n = 0; n < batch; n++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    for (int c = 0; c < dim; c++)
                    {
                        featureMap[n, c, h, w] = x[n, seqIdx, c];
                    }
                }
            }
        }

        return featureMap;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = _patchEmbed.GetParameterCount();
        for (int i = 0; i < _stages.Count; i++)
        {
            count += _stages[i].GetParameterCount();
        }
        return count;
    }

    /// <inheritdoc/>
    public override void WriteParameters(BinaryWriter writer)
    {
        // Write configuration
        writer.Write((int)_variant);
        writer.Write(_embedDim);
        writer.Write(_windowSize);
        writer.Write(_stages.Count);

        // Write patch embedding
        _patchEmbed.WriteParameters(writer);

        // Write stages
        foreach (var stage in _stages)
        {
            stage.WriteParameters(writer);
        }
    }

    /// <inheritdoc/>
    public override void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration
        var variant = (SwinVariant)reader.ReadInt32();
        int embedDim = reader.ReadInt32();
        int windowSize = reader.ReadInt32();
        int stageCount = reader.ReadInt32();

        if (variant != _variant)
        {
            throw new InvalidOperationException($"SwinTransformer variant mismatch: expected {_variant}, got {variant}.");
        }

        if (embedDim != _embedDim)
        {
            throw new InvalidOperationException($"SwinTransformer embed dim mismatch: expected {_embedDim}, got {embedDim}.");
        }

        if (windowSize != _windowSize)
        {
            throw new InvalidOperationException($"SwinTransformer window size mismatch: expected {_windowSize}, got {windowSize}.");
        }

        if (stageCount != _stages.Count)
        {
            throw new InvalidOperationException($"SwinTransformer stage count mismatch: expected {_stages.Count}, got {stageCount}.");
        }

        // Read patch embedding
        _patchEmbed.ReadParameters(reader);

        // Read stages
        foreach (var stage in _stages)
        {
            stage.ReadParameters(reader);
        }
    }
}

/// <summary>
/// Swin Transformer variant enumeration.
/// </summary>
public enum SwinVariant
{
    /// <summary>Swin-Tiny: ~28M parameters.</summary>
    SwinTiny,
    /// <summary>Swin-Small: ~50M parameters.</summary>
    SwinSmall,
    /// <summary>Swin-Base: ~88M parameters.</summary>
    SwinBase,
    /// <summary>Swin-Large: ~197M parameters.</summary>
    SwinLarge
}

/// <summary>
/// Patch embedding layer that converts image to sequence of patches.
/// </summary>
internal class PatchEmbeddingBlock<T>
{
    private readonly Conv2D<T> _proj;
    private readonly int _patchSize;

    public PatchEmbeddingBlock(int patchSize, int inChannels, int embedDim)
    {
        _patchSize = patchSize;

        // Use conv to project patches
        _proj = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: embedDim,
            kernelSize: patchSize,
            stride: patchSize,
            padding: 0,
            useBias: true
        );
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, channels, height, width]
        var x = _proj.Forward(input);

        // Reshape to sequence: [batch, num_patches, embed_dim]
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];
        int numPatches = height * width;

        var sequence = new Tensor<T>(new[] { batch, numPatches, channels });

        for (int n = 0; n < batch; n++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    for (int c = 0; c < channels; c++)
                    {
                        sequence[n, seqIdx, c] = x[n, c, h, w];
                    }
                }
            }
        }

        return sequence;
    }

    public long GetParameterCount()
    {
        return _proj.GetParameterCount();
    }

    public void WriteParameters(BinaryWriter writer)
    {
        _proj.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        _proj.ReadParameters(reader);
    }
}

/// <summary>
/// A stage in Swin Transformer containing multiple transformer blocks.
/// </summary>
internal class SwinStage<T>
{
    private readonly List<SwinTransformerBlock<T>> _blocks;
    private readonly PatchMergingBlock<T>? _patchMerge;
    private readonly int _dim;
    private readonly int _depth;

    public SwinStage(int dim, int depth, int numHeads, int windowSize, bool downsample)
    {
        _dim = dim;
        _depth = depth;
        _blocks = new List<SwinTransformerBlock<T>>();

        // Create transformer blocks (alternating window and shifted window attention)
        for (int i = 0; i < depth; i++)
        {
            bool shiftWindows = i % 2 == 1;
            _blocks.Add(new SwinTransformerBlock<T>(
                dim: dim,
                numHeads: numHeads,
                windowSize: windowSize,
                shiftSize: shiftWindows ? windowSize / 2 : 0
            ));
        }

        // Patch merging for downsampling
        if (downsample)
        {
            _patchMerge = new PatchMergingBlock<T>(dim);
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;

        // Apply transformer blocks
        foreach (var block in _blocks)
        {
            x = block.Forward(x);
        }

        // Downsample via patch merging
        if (_patchMerge is not null)
        {
            x = _patchMerge.Forward(x);
        }

        return x;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var block in _blocks)
        {
            count += block.GetParameterCount();
        }
        if (_patchMerge is not null)
        {
            count += _patchMerge.GetParameterCount();
        }
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_blocks.Count);
        writer.Write(_patchMerge is not null);

        foreach (var block in _blocks)
        {
            block.WriteParameters(writer);
        }

        if (_patchMerge is not null)
        {
            _patchMerge.WriteParameters(writer);
        }
    }

    public void ReadParameters(BinaryReader reader)
    {
        int blockCount = reader.ReadInt32();
        bool hasPatchMerge = reader.ReadBoolean();

        if (blockCount != _blocks.Count)
        {
            throw new InvalidOperationException($"SwinStage block count mismatch: expected {_blocks.Count}, got {blockCount}.");
        }

        if (hasPatchMerge != (_patchMerge is not null))
        {
            throw new InvalidOperationException("SwinStage patch merge configuration mismatch.");
        }

        foreach (var block in _blocks)
        {
            block.ReadParameters(reader);
        }

        if (_patchMerge is not null)
        {
            _patchMerge.ReadParameters(reader);
        }
    }
}

/// <summary>
/// Single Swin Transformer block with windowed multi-head self-attention.
/// </summary>
/// <remarks>
/// <para>Implements the full Swin Transformer block with:
/// - Pre-norm architecture (LayerNorm before attention and MLP)
/// - Window-based multi-head self-attention with learnable relative position bias
/// - Shifted window partitioning for cross-window connections
/// - Two-layer MLP with GELU activation
/// </para>
/// </remarks>
internal class SwinTransformerBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _windowSize;
    private readonly int _shiftSize;
    private readonly double _scale;

    // Layer normalization with learnable parameters
    private readonly SwinLayerNorm<T> _norm1;
    private readonly SwinLayerNorm<T> _norm2;

    // Window attention projections
    private readonly Dense<T> _qkvProj;
    private readonly Dense<T> _outProj;

    // Relative position bias table: (2*windowSize-1) * (2*windowSize-1) entries for each head
    private readonly Tensor<T> _relativePositionBiasTable;
    private readonly int[,] _relativePositionIndex;

    // MLP layers
    private readonly Dense<T> _mlpFc1;
    private readonly Dense<T> _mlpFc2;

    public SwinTransformerBlock(int dim, int numHeads, int windowSize, int shiftSize)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _windowSize = windowSize;
        _shiftSize = shiftSize;
        _scale = 1.0 / Math.Sqrt(_headDim);

        // Layer normalization
        _norm1 = new SwinLayerNorm<T>(dim);
        _norm2 = new SwinLayerNorm<T>(dim);

        // QKV projection (3 * dim for Q, K, V)
        _qkvProj = new Dense<T>(dim, dim * 3);
        _outProj = new Dense<T>(dim, dim);

        // Relative position bias table
        int biasTableSize = (2 * windowSize - 1) * (2 * windowSize - 1);
        _relativePositionBiasTable = new Tensor<T>(new[] { biasTableSize, numHeads });
        InitializeRelativePositionBias();

        // Compute relative position index for windows
        _relativePositionIndex = ComputeRelativePositionIndex(windowSize);

        // MLP (2-layer with GELU)
        int mlpHiddenDim = dim * 4;
        _mlpFc1 = new Dense<T>(dim, mlpHiddenDim);
        _mlpFc2 = new Dense<T>(mlpHiddenDim, dim);
    }

    private void InitializeRelativePositionBias()
    {
        // Initialize with truncated normal distribution (std=0.02)
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            double value = random.NextGaussian(0, 0.02);
            // Truncate to [-0.04, 0.04]
            value = Math.Max(-0.04, Math.Min(0.04, value));
            _relativePositionBiasTable[i] = _numOps.FromDouble(value);
        }
    }

    private int[,] ComputeRelativePositionIndex(int windowSize)
    {
        // Create coordinate grid
        var coordsH = new int[windowSize];
        var coordsW = new int[windowSize];
        for (int i = 0; i < windowSize; i++)
        {
            coordsH[i] = i;
            coordsW[i] = i;
        }

        // Compute pairwise relative positions
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

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, seq_len, dim]
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];

        // Infer spatial dimensions (assume square or near-square)
        int h = (int)Math.Sqrt(seqLen);
        int w = seqLen / h;
        if (h * w != seqLen)
        {
            // Try to find valid factorization
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

        // Pre-norm + Window attention + residual
        var normed1 = _norm1.Forward(input);
        var attnOut = WindowAttention(normed1, h, w);
        var x = AddTensors(input, attnOut);

        // Pre-norm + MLP + residual
        var normed2 = _norm2.Forward(x);
        var mlpOut = ApplyMLP(normed2);
        x = AddTensors(x, mlpOut);

        return x;
    }

    private Tensor<T> WindowAttention(Tensor<T> x, int h, int w)
    {
        int batch = x.Shape[0];
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
        int numWindows = numWindowsH * numWindowsW;

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
        var spatial = new Tensor<T>(new[] { batch, h, w, c });
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

        var seq = new Tensor<T>(new[] { batch, h * w, c });
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
            padded = new Tensor<T>(new[] { batch, paddedH, paddedW, c });
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
                                padded[b, i, j, d] = _numOps.FromDouble(0.0);
                        }
                    }
                }
            }
        }
        else
        {
            padded = x;
            paddedH = h;
            paddedW = w;
        }

        int numWindowsH = paddedH / _windowSize;
        int numWindowsW = paddedW / _windowSize;
        int numWindows = numWindowsH * numWindowsW;
        int windowArea = _windowSize * _windowSize;

        var windows = new Tensor<T>(new[] { batch * numWindows, windowArea, c });

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
        int paddedH = numWindowsH * _windowSize;
        int paddedW = numWindowsW * _windowSize;

        var spatial = new Tensor<T>(new[] { batch, h, w, c });

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

        // Project to Q, K, V
        var qkv = new Tensor<T>(new[] { numWindows, windowArea, 3 * c });
        for (int w = 0; w < numWindows; w++)
        {
            var windowIn = new Tensor<T>(new[] { windowArea, c });
            for (int t = 0; t < windowArea; t++)
            {
                for (int d = 0; d < c; d++)
                {
                    windowIn[t, d] = windows[w, t, d];
                }
            }

            // Apply QKV projection per token
            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>(new[] { 1, c });
                for (int d = 0; d < c; d++)
                {
                    tokenIn[0, d] = windowIn[t, d];
                }
                var tokenQkv = _qkvProj.Forward(tokenIn);
                for (int d = 0; d < 3 * c; d++)
                {
                    qkv[w, t, d] = tokenQkv[0, d];
                }
            }
        }

        // Compute attention per window
        var output = new Tensor<T>(new[] { numWindows, windowArea, c });

        for (int w = 0; w < numWindows; w++)
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
                            double q = _numOps.ToDouble(qkv[w, i, headOffset + d]);
                            double k = _numOps.ToDouble(qkv[w, j, c + headOffset + d]);
                            score += q * k;
                        }
                        score *= _scale;

                        // Add relative position bias
                        int biasIdx = _relativePositionIndex[i, j];
                        score += _numOps.ToDouble(_relativePositionBiasTable[biasIdx, head]);

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
                            val += attnProbs[head, i, j] * _numOps.ToDouble(qkv[w, j, vOffset + d]);
                        }
                        attnOut[i, headOffset + d] = val;
                    }
                }
            }

            // Output projection
            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>(new[] { 1, c });
                for (int d = 0; d < c; d++)
                {
                    tokenIn[0, d] = _numOps.FromDouble(attnOut[t, d]);
                }
                var tokenOut = _outProj.Forward(tokenIn);
                for (int d = 0; d < c; d++)
                {
                    output[w, t, d] = tokenOut[0, d];
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
                var tokenIn = new Tensor<T>(new[] { 1, _dim });
                for (int d = 0; d < _dim; d++)
                {
                    tokenIn[0, d] = x[b, s, d];
                }

                // FC1 + GELU
                var hidden = _mlpFc1.Forward(tokenIn);
                for (int d = 0; d < hidden.Shape[1]; d++)
                {
                    double val = _numOps.ToDouble(hidden[0, d]);
                    double gelu = 0.5 * val * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (val + 0.044715 * val * val * val)));
                    hidden[0, d] = _numOps.FromDouble(gelu);
                }

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
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }

    public long GetParameterCount()
    {
        // LayerNorm: 2 * dim each (gamma + beta)
        long normParams = 2 * 2 * _dim;

        // QKV projection: dim * 3*dim + 3*dim bias
        long qkvParams = _dim * 3 * _dim + 3 * _dim;

        // Output projection: dim * dim + dim bias
        long outProjParams = _dim * _dim + _dim;

        // Relative position bias table
        int biasTableSize = (2 * _windowSize - 1) * (2 * _windowSize - 1);
        long biasParams = biasTableSize * _numHeads;

        // MLP: fc1 (dim * 4*dim + 4*dim) + fc2 (4*dim * dim + dim)
        long mlpParams = _dim * 4 * _dim + 4 * _dim + 4 * _dim * _dim + _dim;

        return normParams + qkvParams + outProjParams + biasParams + mlpParams;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        // Write layer norms
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);

        // Write projections
        _qkvProj.WriteParameters(writer);
        _outProj.WriteParameters(writer);

        // Write relative position bias table
        writer.Write(_relativePositionBiasTable.Shape[0]);
        writer.Write(_relativePositionBiasTable.Shape[1]);
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_relativePositionBiasTable[i]));
        }

        // Write MLP
        _mlpFc1.WriteParameters(writer);
        _mlpFc2.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        // Read layer norms
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);

        // Read projections
        _qkvProj.ReadParameters(reader);
        _outProj.ReadParameters(reader);

        // Read relative position bias table
        int biasTableDim0 = reader.ReadInt32();
        int biasTableDim1 = reader.ReadInt32();
        if (biasTableDim0 != _relativePositionBiasTable.Shape[0] || biasTableDim1 != _relativePositionBiasTable.Shape[1])
        {
            throw new InvalidOperationException("SwinTransformerBlock relative position bias table shape mismatch.");
        }
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            _relativePositionBiasTable[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read MLP
        _mlpFc1.ReadParameters(reader);
        _mlpFc2.ReadParameters(reader);
    }
}

/// <summary>
/// Layer normalization with learnable affine parameters for Swin Transformer.
/// </summary>
internal class SwinLayerNorm<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _dim;
    private readonly Tensor<T> _gamma;
    private readonly Tensor<T> _beta;
    private readonly double _eps;

    public SwinLayerNorm(int dim, double eps = 1e-6)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _eps = eps;

        _gamma = new Tensor<T>(new[] { dim });
        _beta = new Tensor<T>(new[] { dim });

        // Initialize gamma to 1 and beta to 0
        for (int i = 0; i < dim; i++)
        {
            _gamma[i] = _numOps.FromDouble(1.0);
            _beta[i] = _numOps.FromDouble(0.0);
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Compute mean
                double mean = 0;
                for (int d = 0; d < dim; d++)
                {
                    mean += _numOps.ToDouble(x[b, s, d]);
                }
                mean /= dim;

                // Compute variance
                double variance = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = _numOps.ToDouble(x[b, s, d]) - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                // Normalize and apply affine transformation
                double std = Math.Sqrt(variance + _eps);
                for (int d = 0; d < dim; d++)
                {
                    double normalized = (_numOps.ToDouble(x[b, s, d]) - mean) / std;
                    double gamma = _numOps.ToDouble(_gamma[d]);
                    double beta = _numOps.ToDouble(_beta[d]);
                    result[b, s, d] = _numOps.FromDouble(gamma * normalized + beta);
                }
            }
        }

        return result;
    }

    public long GetParameterCount() => 2 * _dim;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_dim);
        for (int i = 0; i < _dim; i++)
        {
            writer.Write(_numOps.ToDouble(_gamma[i]));
        }
        for (int i = 0; i < _dim; i++)
        {
            writer.Write(_numOps.ToDouble(_beta[i]));
        }
    }

    public void ReadParameters(BinaryReader reader)
    {
        int dim = reader.ReadInt32();
        if (dim != _dim)
        {
            throw new InvalidOperationException($"SwinLayerNorm dim mismatch: expected {_dim}, got {dim}.");
        }
        for (int i = 0; i < _dim; i++)
        {
            _gamma[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        for (int i = 0; i < _dim; i++)
        {
            _beta[i] = _numOps.FromDouble(reader.ReadDouble());
        }
    }
}

/// <summary>
/// Patch merging layer for downsampling in Swin Transformer.
/// </summary>
internal class PatchMergingBlock<T>
{
    private readonly Dense<T> _reduction;
    private readonly int _dim;

    public PatchMergingBlock(int dim)
    {
        _dim = dim;
        // Merge 2x2 patches and reduce dimension
        _reduction = new Dense<T>(dim * 4, dim * 2);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        return Forward(input, null, null);
    }

    public Tensor<T> Forward(Tensor<T> input, int? inputHeight, int? inputWidth)
    {
        // input: [batch, seq_len, dim]
        // Need to reshape to 2D spatial and merge 2x2 patches

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dim = input.Shape[2];

        // Calculate spatial dimensions - use provided values or infer from aspect ratio
        int h, w;
        if (inputHeight.HasValue && inputWidth.HasValue)
        {
            h = inputHeight.Value;
            w = inputWidth.Value;

            // Validate that dimensions match sequence length
            if (h * w != seqLen)
            {
                throw new ArgumentException(
                    $"Provided dimensions ({h} x {w} = {h * w}) do not match sequence length {seqLen}.");
            }

            // Validate dimensions are even for patch merging
            if (h % 2 != 0 || w % 2 != 0)
            {
                throw new ArgumentException(
                    $"Both dimensions must be even for 2x2 patch merging. Got h={h}, w={w}.");
            }
        }
        else
        {
            // Find valid factorization h×w = seqLen where both h and w are even (required for 2×2 patch merging)
            // Start from sqrt and search for closest factors
            int sqrtSeq = (int)Math.Sqrt(seqLen);
            h = 0;
            w = 0;

            // Search for factors, preferring those close to square
            for (int candidate = sqrtSeq; candidate >= 1; candidate--)
            {
                if (seqLen % candidate == 0)
                {
                    int other = seqLen / candidate;
                    // Both dimensions must be even for 2×2 patch merging
                    if (candidate % 2 == 0 && other % 2 == 0)
                    {
                        h = other;  // height >= width (for typical image aspect ratios)
                        w = candidate;
                        break;
                    }
                }
            }

            // If no valid even factorization found, throw exception
            if (h == 0 || w == 0)
            {
                throw new ArgumentException(
                    $"Cannot infer spatial dimensions from sequence length {seqLen}. " +
                    "Sequence length must be factorizable into two even integers for patch merging. " +
                    "Please provide explicit inputHeight and inputWidth parameters.");
            }
        }

        int newH = h / 2;
        int newW = w / 2;
        int newSeqLen = newH * newW;

        // Concatenate 2x2 patches
        var merged = new Tensor<T>(new[] { batch, newSeqLen, dim * 4 });

        for (int n = 0; n < batch; n++)
        {
            for (int i = 0; i < newH; i++)
            {
                for (int j = 0; j < newW; j++)
                {
                    int newIdx = i * newW + j;

                    // Get 4 patch indices
                    int idx0 = (2 * i) * w + (2 * j);
                    int idx1 = (2 * i) * w + (2 * j + 1);
                    int idx2 = (2 * i + 1) * w + (2 * j);
                    int idx3 = (2 * i + 1) * w + (2 * j + 1);

                    for (int d = 0; d < dim; d++)
                    {
                        merged[n, newIdx, d] = input[n, idx0, d];
                        merged[n, newIdx, dim + d] = input[n, idx1, d];
                        merged[n, newIdx, 2 * dim + d] = input[n, idx2, d];
                        merged[n, newIdx, 3 * dim + d] = input[n, idx3, d];
                    }
                }
            }
        }

        // Reduce dimension
        var output = _reduction.Forward(merged);
        return output;
    }

    public long GetParameterCount()
    {
        return _dim * 4 * _dim * 2 + _dim * 2; // Dense layer params
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_dim);
        _reduction.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int dim = reader.ReadInt32();
        if (dim != _dim)
        {
            throw new InvalidOperationException($"PatchMergingBlock dim mismatch: expected {_dim}, got {dim}.");
        }
        _reduction.ReadParameters(reader);
    }
}
