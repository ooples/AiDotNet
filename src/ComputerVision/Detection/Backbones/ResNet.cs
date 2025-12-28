using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// ResNet backbone network for feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> ResNet (Residual Network) is a foundational architecture
/// that introduced skip connections to enable training of very deep networks. It's widely
/// used as a backbone for detection models like Faster R-CNN.</para>
///
/// <para>Key features:
/// - Residual blocks with skip connections prevent gradient vanishing
/// - Multiple variants: ResNet-18, 34, 50, 101, 152
/// - Bottleneck blocks (3 convolutions) for deeper networks
/// </para>
///
/// <para>Reference: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016</para>
/// </remarks>
public class ResNet<T> : BackboneBase<T>
{
    private readonly Conv2D<T> _conv1;
    private readonly List<ResNetStage<T>> _stages;
    private readonly ResNetVariant _variant;

    /// <inheritdoc/>
    public override string Name => $"ResNet-{GetLayerCount(_variant)}";

    /// <inheritdoc/>
    public override int[] OutputChannels { get; }

    /// <inheritdoc/>
    public override int[] Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Creates a new ResNet backbone.
    /// </summary>
    /// <param name="variant">ResNet variant (18, 34, 50, 101, or 152).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public ResNet(ResNetVariant variant = ResNetVariant.ResNet50, int inChannels = 3)
    {
        _variant = variant;
        _stages = new List<ResNetStage<T>>();

        // Base channels and expansion factor
        bool useBottleneck = variant >= ResNetVariant.ResNet50;
        int expansion = useBottleneck ? 4 : 1;
        int[] baseChannels = { 64, 128, 256, 512 };
        OutputChannels = baseChannels.Select(c => c * expansion).ToArray();

        // Initial conv layer (7x7, stride 2)
        _conv1 = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: 64,
            kernelSize: 7,
            stride: 2,
            padding: 3,
            useBias: false
        );

        // Get block counts for this variant
        int[] blockCounts = GetBlockCounts(variant);

        // Build stages
        int currentChannels = 64;
        for (int i = 0; i < 4; i++)
        {
            int outChannels = baseChannels[i];
            int stride = i == 0 ? 1 : 2; // First stage doesn't downsample (pool already did)

            var stage = new ResNetStage<T>(
                inChannels: currentChannels,
                outChannels: outChannels,
                numBlocks: blockCounts[i],
                stride: stride,
                useBottleneck: useBottleneck
            );

            _stages.Add(stage);
            currentChannels = outChannels * expansion;
        }
    }

    private static int GetLayerCount(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => 18,
        ResNetVariant.ResNet34 => 34,
        ResNetVariant.ResNet50 => 50,
        ResNetVariant.ResNet101 => 101,
        ResNetVariant.ResNet152 => 152,
        _ => 50
    };

    /// <summary>
    /// Gets the block counts for each ResNet variant.
    /// </summary>
    private static int[] GetBlockCounts(ResNetVariant variant) => variant switch
    {
        ResNetVariant.ResNet18 => new[] { 2, 2, 2, 2 },
        ResNetVariant.ResNet34 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet50 => new[] { 3, 4, 6, 3 },
        ResNetVariant.ResNet101 => new[] { 3, 4, 23, 3 },
        ResNetVariant.ResNet152 => new[] { 3, 8, 36, 3 },
        _ => new[] { 3, 4, 6, 3 }
    };

    /// <inheritdoc/>
    public override List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();

        // Stem: conv + bn + relu + maxpool
        var x = _conv1.Forward(input);
        x = ApplyReLU(x);
        x = MaxPool2D(x, kernelSize: 3, stride: 2, padding: 1);

        // Stages with feature extraction
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            features.Add(x); // C2, C3, C4, C5
        }

        return features;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = _conv1.GetParameterCount();
        for (int i = 0; i < _stages.Count; i++)
        {
            count += _stages[i].GetParameterCount();
        }
        return count;
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }

    private Tensor<T> MaxPool2D(Tensor<T> x, int kernelSize, int stride, int padding)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
        int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

        var output = new Tensor<T>(new[] { batch, channels, outHeight, outWidth });

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        double maxVal = double.NegativeInfinity;
                        for (int kh = 0; kh < kernelSize; kh++)
                        {
                            for (int kw = 0; kw < kernelSize; kw++)
                            {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    double val = NumOps.ToDouble(x[n, c, ih, iw]);
                                    maxVal = Math.Max(maxVal, val);
                                }
                            }
                        }
                        output[n, c, oh, ow] = NumOps.FromDouble(maxVal == double.NegativeInfinity ? 0 : maxVal);
                    }
                }
            }
        }

        return output;
    }
}

/// <summary>
/// ResNet variant enumeration.
/// </summary>
public enum ResNetVariant
{
    /// <summary>ResNet-18: 18 layers, basic blocks.</summary>
    ResNet18,
    /// <summary>ResNet-34: 34 layers, basic blocks.</summary>
    ResNet34,
    /// <summary>ResNet-50: 50 layers, bottleneck blocks.</summary>
    ResNet50,
    /// <summary>ResNet-101: 101 layers, bottleneck blocks.</summary>
    ResNet101,
    /// <summary>ResNet-152: 152 layers, bottleneck blocks.</summary>
    ResNet152
}

/// <summary>
/// A stage in ResNet containing multiple residual blocks.
/// </summary>
internal class ResNetStage<T>
{
    private readonly List<ResidualBlock<T>> _blocks;

    public ResNetStage(int inChannels, int outChannels, int numBlocks, int stride, bool useBottleneck)
    {
        _blocks = new List<ResidualBlock<T>>();
        int expansion = useBottleneck ? 4 : 1;

        // First block may downsample and change channels
        _blocks.Add(new ResidualBlock<T>(
            inChannels: inChannels,
            outChannels: outChannels,
            stride: stride,
            useBottleneck: useBottleneck,
            downsample: inChannels != outChannels * expansion || stride != 1
        ));

        // Remaining blocks maintain channel count
        for (int i = 1; i < numBlocks; i++)
        {
            _blocks.Add(new ResidualBlock<T>(
                inChannels: outChannels * expansion,
                outChannels: outChannels,
                stride: 1,
                useBottleneck: useBottleneck,
                downsample: false
            ));
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;
        foreach (var block in _blocks)
        {
            x = block.Forward(x);
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
        return count;
    }
}

/// <summary>
/// Individual residual block in ResNet.
/// </summary>
internal class ResidualBlock<T>
{
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly Conv2D<T>? _conv3; // For bottleneck
    private readonly Conv2D<T>? _downsample;
    private readonly bool _useBottleneck;
    private readonly INumericOperations<T> _numOps;
    private readonly int _outChannels;

    public ResidualBlock(int inChannels, int outChannels, int stride, bool useBottleneck, bool downsample)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _useBottleneck = useBottleneck;
        _outChannels = outChannels;

        int expansion = useBottleneck ? 4 : 1;

        if (useBottleneck)
        {
            // Bottleneck: 1x1 -> 3x3 -> 1x1
            _conv1 = new Conv2D<T>(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                useBias: false
            );

            _conv2 = new Conv2D<T>(
                inChannels: outChannels,
                outChannels: outChannels,
                kernelSize: 3,
                stride: stride,
                padding: 1,
                useBias: false
            );

            _conv3 = new Conv2D<T>(
                inChannels: outChannels,
                outChannels: outChannels * expansion,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                useBias: false
            );
        }
        else
        {
            // Basic block: 3x3 -> 3x3
            _conv1 = new Conv2D<T>(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: 3,
                stride: stride,
                padding: 1,
                useBias: false
            );

            _conv2 = new Conv2D<T>(
                inChannels: outChannels,
                outChannels: outChannels * expansion,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                useBias: false
            );
        }

        if (downsample)
        {
            _downsample = new Conv2D<T>(
                inChannels: inChannels,
                outChannels: outChannels * expansion,
                kernelSize: 1,
                stride: stride,
                padding: 0,
                useBias: false
            );
        }
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> identity = input;

        if (_downsample is not null)
        {
            identity = _downsample.Forward(input);
        }

        var x = _conv1.Forward(input);
        x = ApplyReLU(x);

        x = _conv2.Forward(x);
        x = ApplyReLU(x);

        if (_useBottleneck && _conv3 is not null)
        {
            x = _conv3.Forward(x);
        }

        // Add residual connection
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = _numOps.Add(x[i], identity[i]);
        }

        x = ApplyReLU(x);
        return x;
    }

    public long GetParameterCount()
    {
        long count = _conv1.GetParameterCount() + _conv2.GetParameterCount();
        if (_conv3 is not null)
        {
            count += _conv3.GetParameterCount();
        }
        if (_downsample is not null)
        {
            count += _downsample.GetParameterCount();
        }
        return count;
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            result[i] = _numOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }
}
