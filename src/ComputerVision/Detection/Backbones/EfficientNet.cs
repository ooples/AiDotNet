using System.IO;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// EfficientNet backbone for efficient feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EfficientNet is a family of models that were designed
/// using neural architecture search to find the optimal balance between width, depth,
/// and resolution. It achieves state-of-the-art accuracy with significantly fewer
/// parameters than other architectures.</para>
///
/// <para>Key features:
/// - MBConv (Mobile Inverted Bottleneck) blocks
/// - Squeeze-and-Excitation for channel attention
/// - Compound scaling for width, depth, and resolution
/// </para>
///
/// <para>Reference: Tan et al., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019</para>
/// </remarks>
public class EfficientNet<T> : BackboneBase<T>
{
    private readonly Conv2D<T> _stem;
    private readonly List<MBConvBlock<T>> _blocks;
    private readonly EfficientNetVariant _variant;
    private readonly int _stemChannels;

    /// <inheritdoc/>
    public override string Name => $"EfficientNet-{_variant}";

    /// <inheritdoc/>
    public override int[] OutputChannels { get; }

    /// <inheritdoc/>
    public override int[] Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Indices of blocks that produce output features (P2, P3, P4, P5).
    /// </summary>
    private readonly int[] _featureIndices;

    /// <summary>
    /// Creates a new EfficientNet backbone.
    /// </summary>
    /// <param name="variant">EfficientNet variant (B0-B7).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public EfficientNet(EfficientNetVariant variant = EfficientNetVariant.B0, int inChannels = 3)
    {
        _variant = variant;
        _blocks = new List<MBConvBlock<T>>();

        // Get scaling factors for variant
        var (widthMult, depthMult) = GetScalingFactors(variant);

        // Base block configurations: (expand_ratio, channels, num_blocks, stride, kernel_size)
        var blockConfigs = new[]
        {
            (1, 16, 1, 1, 3),   // Stage 1
            (6, 24, 2, 2, 3),   // Stage 2 - P2
            (6, 40, 2, 2, 5),   // Stage 3 - P3
            (6, 80, 3, 2, 3),   // Stage 4
            (6, 112, 3, 1, 5),  // Stage 5
            (6, 192, 4, 2, 5),  // Stage 6 - P4
            (6, 320, 1, 1, 3),  // Stage 7 - P5
        };

        // Calculate scaled channels
        _stemChannels = ScaleChannels(32, widthMult);
        OutputChannels = new[]
        {
            ScaleChannels(24, widthMult),
            ScaleChannels(40, widthMult),
            ScaleChannels(192, widthMult),
            ScaleChannels(320, widthMult)
        };

        // Stem convolution
        _stem = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: _stemChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            useBias: false
        );

        // Build MBConv blocks
        int currentChannels = _stemChannels;
        var featureIndicesList = new List<int>();
        int blockIdx = 0;

        for (int stageIdx = 0; stageIdx < blockConfigs.Length; stageIdx++)
        {
            var (expandRatio, outChannels, numBlocks, stride, kernelSize) = blockConfigs[stageIdx];
            int scaledChannels = ScaleChannels(outChannels, widthMult);
            int scaledBlocks = ScaleDepth(numBlocks, depthMult);

            for (int i = 0; i < scaledBlocks; i++)
            {
                int blockStride = i == 0 ? stride : 1;
                int blockInChannels = i == 0 ? currentChannels : scaledChannels;

                _blocks.Add(new MBConvBlock<T>(
                    inChannels: blockInChannels,
                    outChannels: scaledChannels,
                    kernelSize: kernelSize,
                    stride: blockStride,
                    expandRatio: expandRatio,
                    useSE: true
                ));

                blockIdx++;
            }

            currentChannels = scaledChannels;

            // Mark feature output stages (indices 1, 2, 5, 6 -> P2, P3, P4, P5)
            if (stageIdx == 1 || stageIdx == 2 || stageIdx == 5 || stageIdx == 6)
            {
                featureIndicesList.Add(blockIdx - 1);
            }
        }

        _featureIndices = featureIndicesList.ToArray();
    }

    /// <summary>
    /// Gets the scaling factors for each EfficientNet variant.
    /// </summary>
    private static (double width, double depth) GetScalingFactors(EfficientNetVariant variant)
    {
        return variant switch
        {
            EfficientNetVariant.B0 => (1.0, 1.0),
            EfficientNetVariant.B1 => (1.0, 1.1),
            EfficientNetVariant.B2 => (1.1, 1.2),
            EfficientNetVariant.B3 => (1.2, 1.4),
            EfficientNetVariant.B4 => (1.4, 1.8),
            EfficientNetVariant.B5 => (1.6, 2.2),
            EfficientNetVariant.B6 => (1.8, 2.6),
            EfficientNetVariant.B7 => (2.0, 3.1),
            _ => (1.0, 1.0)
        };
    }

    /// <summary>
    /// Scales channel count and rounds to nearest multiple of 8.
    /// </summary>
    private static int ScaleChannels(int channels, double multiplier)
    {
        int scaled = (int)(channels * multiplier);
        // Round to nearest multiple of 8
        return Math.Max(8, ((scaled + 4) / 8) * 8);
    }

    /// <summary>
    /// Scales depth and rounds up.
    /// </summary>
    private static int ScaleDepth(int depth, double multiplier)
    {
        return (int)Math.Ceiling(depth * multiplier);
    }

    /// <inheritdoc/>
    public override List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();

        // Stem
        var x = _stem.Forward(input);
        x = ApplySwish(x);

        // Blocks
        for (int i = 0; i < _blocks.Count; i++)
        {
            x = _blocks[i].Forward(x);

            // Collect features at marked indices
            if (_featureIndices.Contains(i))
            {
                features.Add(x);
            }
        }

        return features;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = _stem.GetParameterCount();
        foreach (var block in _blocks)
        {
            count += block.GetParameterCount();
        }
        return count;
    }

    /// <inheritdoc/>
    public override void WriteParameters(BinaryWriter writer)
    {
        // Write configuration
        writer.Write((int)_variant);
        writer.Write(_blocks.Count);
        writer.Write(_featureIndices.Length);
        foreach (int idx in _featureIndices)
        {
            writer.Write(idx);
        }

        // Write stem
        _stem.WriteParameters(writer);

        // Write blocks
        foreach (var block in _blocks)
        {
            block.WriteParameters(writer);
        }
    }

    /// <inheritdoc/>
    public override void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration
        var variant = (EfficientNetVariant)reader.ReadInt32();
        int blockCount = reader.ReadInt32();
        int featureIndexCount = reader.ReadInt32();
        var featureIndices = new int[featureIndexCount];
        for (int i = 0; i < featureIndexCount; i++)
        {
            featureIndices[i] = reader.ReadInt32();
        }

        if (variant != _variant)
        {
            throw new InvalidOperationException($"EfficientNet variant mismatch: expected {_variant}, got {variant}.");
        }

        if (blockCount != _blocks.Count)
        {
            throw new InvalidOperationException($"EfficientNet block count mismatch: expected {_blocks.Count}, got {blockCount}.");
        }

        // Validate feature indices match
        if (featureIndexCount != _featureIndices.Length)
        {
            throw new InvalidOperationException(
                $"EfficientNet feature index count mismatch: expected {_featureIndices.Length}, got {featureIndexCount}.");
        }

        for (int i = 0; i < featureIndexCount; i++)
        {
            if (featureIndices[i] != _featureIndices[i])
            {
                throw new InvalidOperationException(
                    $"EfficientNet feature index mismatch at position {i}: expected {_featureIndices[i]}, got {featureIndices[i]}.");
            }
        }

        // Read stem
        _stem.ReadParameters(reader);

        // Read blocks
        foreach (var block in _blocks)
        {
            block.ReadParameters(reader);
        }
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double swish = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = NumOps.FromDouble(swish);
        }
        return result;
    }
}

/// <summary>
/// EfficientNet variant enumeration.
/// </summary>
public enum EfficientNetVariant
{
    /// <summary>EfficientNet-B0: baseline model.</summary>
    B0,
    /// <summary>EfficientNet-B1: 1.1x depth scaling.</summary>
    B1,
    /// <summary>EfficientNet-B2: 1.1x width, 1.2x depth.</summary>
    B2,
    /// <summary>EfficientNet-B3: 1.2x width, 1.4x depth.</summary>
    B3,
    /// <summary>EfficientNet-B4: 1.4x width, 1.8x depth.</summary>
    B4,
    /// <summary>EfficientNet-B5: 1.6x width, 2.2x depth.</summary>
    B5,
    /// <summary>EfficientNet-B6: 1.8x width, 2.6x depth.</summary>
    B6,
    /// <summary>EfficientNet-B7: 2.0x width, 3.1x depth.</summary>
    B7
}

/// <summary>
/// Mobile Inverted Bottleneck Convolution block (MBConv).
/// </summary>
internal class MBConvBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T>? _expand;
    private readonly Conv2D<T> _depthwise;
    private readonly SqueezeExcitation<T>? _se;
    private readonly Conv2D<T> _project;
    private readonly bool _useResidual;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _expandRatio;

    public MBConvBlock(int inChannels, int outChannels, int kernelSize, int stride, int expandRatio, bool useSE)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _inChannels = inChannels;
        _outChannels = outChannels;
        _expandRatio = expandRatio;
        _useResidual = inChannels == outChannels && stride == 1;

        int hiddenDim = inChannels * expandRatio;

        // Expansion phase (if expand ratio > 1)
        if (expandRatio != 1)
        {
            _expand = new Conv2D<T>(
                inChannels: inChannels,
                outChannels: hiddenDim,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                useBias: false
            );
        }

        // Depthwise convolution
        int padding = (kernelSize - 1) / 2;
        _depthwise = new Conv2D<T>(
            inChannels: hiddenDim,
            outChannels: hiddenDim,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            useBias: false
        );

        // Squeeze-and-Excitation
        if (useSE)
        {
            int seChannels = Math.Max(1, inChannels / 4);
            _se = new SqueezeExcitation<T>(hiddenDim, seChannels);
        }

        // Projection phase
        _project = new Conv2D<T>(
            inChannels: hiddenDim,
            outChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            useBias: false
        );
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;

        // Expansion
        if (_expand is not null)
        {
            x = _expand.Forward(x);
            x = ApplySwish(x);
        }

        // Depthwise conv
        x = _depthwise.Forward(x);
        x = ApplySwish(x);

        // Squeeze-and-Excitation
        if (_se is not null)
        {
            x = _se.Forward(x);
        }

        // Projection
        x = _project.Forward(x);

        // Residual connection
        if (_useResidual)
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = _numOps.Add(x[i], input[i]);
            }
        }

        return x;
    }

    public long GetParameterCount()
    {
        long count = 0;
        int hiddenDim = _inChannels * _expandRatio;

        // Expansion conv (1x1)
        if (_expandRatio != 1)
        {
            count += _inChannels * hiddenDim;
        }

        // Depthwise conv (approx 3x3 per channel)
        count += hiddenDim * 9;

        // SE layers
        if (_se is not null)
        {
            int seChannels = Math.Max(1, _inChannels / 4);
            count += hiddenDim * seChannels + seChannels * hiddenDim;
        }

        // Projection conv (1x1)
        count += hiddenDim * _outChannels;

        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        // Write configuration
        writer.Write(_expandRatio != 1);
        writer.Write(_se is not null);

        // Write expand conv
        if (_expand is not null)
        {
            _expand.WriteParameters(writer);
        }

        // Write depthwise conv
        _depthwise.WriteParameters(writer);

        // Write SE
        if (_se is not null)
        {
            _se.WriteParameters(writer);
        }

        // Write project conv
        _project.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        bool hasExpand = reader.ReadBoolean();
        bool hasSE = reader.ReadBoolean();

        if (hasExpand != (_expand is not null))
        {
            throw new InvalidOperationException("MBConvBlock expand configuration mismatch.");
        }
        if (hasSE != (_se is not null))
        {
            throw new InvalidOperationException("MBConvBlock SE configuration mismatch.");
        }

        if (_expand is not null)
        {
            _expand.ReadParameters(reader);
        }

        _depthwise.ReadParameters(reader);

        if (_se is not null)
        {
            _se.ReadParameters(reader);
        }

        _project.ReadParameters(reader);
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double swish = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(swish);
        }
        return result;
    }
}

/// <summary>
/// Squeeze-and-Excitation block for channel attention.
/// </summary>
internal class SqueezeExcitation<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dense<T> _fc1;
    private readonly Dense<T> _fc2;
    private readonly int _channels;

    public SqueezeExcitation(int channels, int reducedChannels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _channels = channels;
        _fc1 = new Dense<T>(channels, reducedChannels);
        _fc2 = new Dense<T>(reducedChannels, channels);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // input: [batch, channels, height, width]
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        // Global average pooling
        var squeezed = new Tensor<T>(new[] { batch, channels, 1 });
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sum = 0;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum += _numOps.ToDouble(input[n, c, h, w]);
                    }
                }
                squeezed[n, c, 0] = _numOps.FromDouble(sum / (height * width));
            }
        }

        // Excitation: FC -> Swish -> FC -> Sigmoid
        var excited = _fc1.Forward(squeezed);
        excited = ApplySwish(excited);
        excited = _fc2.Forward(excited);
        excited = ApplySigmoid(excited);

        // Scale input channels
        var output = new Tensor<T>(input.Shape);
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                T scale = excited[n, c, 0];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[n, c, h, w] = _numOps.Multiply(input[n, c, h, w], scale);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double swish = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(swish);
        }
        return result;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-val));
            result[i] = _numOps.FromDouble(sigmoid);
        }
        return result;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        _fc1.WriteParameters(writer);
        _fc2.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        _fc1.ReadParameters(reader);
        _fc2.ReadParameters(reader);
    }
}
