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
        int height = inputHeight / stride;
        int width = inputWidth / stride;

        var featureMap = new Tensor<T>(new[] { batch, dim, height, width });

        for (int n = 0; n < batch; n++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    if (seqIdx < seqLen)
                    {
                        for (int c = 0; c < dim; c++)
                        {
                            featureMap[n, c, h, w] = x[n, seqIdx, c];
                        }
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
}

/// <summary>
/// Single Swin Transformer block with windowed attention.
/// </summary>
internal class SwinTransformerBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _windowSize;
    private readonly int _shiftSize;
    private readonly MultiHeadSelfAttention<T> _attention;
    private readonly Dense<T> _mlpFc1;
    private readonly Dense<T> _mlpFc2;

    public SwinTransformerBlock(int dim, int numHeads, int windowSize, int shiftSize)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _numHeads = numHeads;
        _windowSize = windowSize;
        _shiftSize = shiftSize;

        // Window attention
        _attention = new MultiHeadSelfAttention<T>(dim, numHeads);

        // MLP (2-layer feed-forward with GELU)
        int mlpHiddenDim = dim * 4;
        _mlpFc1 = new Dense<T>(dim, mlpHiddenDim);
        _mlpFc2 = new Dense<T>(mlpHiddenDim, dim);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, seq_len, dim]

        // Self-attention with residual
        var attnOut = _attention.Forward(input);
        var x = AddTensors(input, attnOut);

        // MLP with residual
        var mlpInput = x;
        var mlpHidden = _mlpFc1.Forward(x);
        mlpHidden = ApplyGELU(mlpHidden);
        var mlpOut = _mlpFc2.Forward(mlpHidden);

        x = AddTensors(mlpInput, mlpOut);

        return x;
    }

    public long GetParameterCount()
    {
        return _attention.GetParameterCount() +
               _dim * _dim * 4 + _dim * 4 + // MLP fc1
               _dim * 4 * _dim + _dim; // MLP fc2
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

    private Tensor<T> ApplyGELU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))
            double gelu = 0.5 * val * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (val + 0.044715 * val * val * val)));
            result[i] = _numOps.FromDouble(gelu);
        }
        return result;
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
        // input: [batch, seq_len, dim]
        // Need to reshape to 2D spatial and merge 2x2 patches

        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dim = input.Shape[2];

        // Assume square spatial dimensions
        int h = (int)Math.Sqrt(seqLen);
        int w = h;
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
}
