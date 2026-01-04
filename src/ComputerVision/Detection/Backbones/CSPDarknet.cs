using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// CSP-Darknet backbone network used in YOLO family models (v5, v7, v8).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CSP-Darknet is a specialized feature extraction network
/// designed for real-time object detection. It uses Cross-Stage Partial connections
/// to reduce computation while maintaining accuracy.</para>
///
/// <para>Key features:
/// - Cross-Stage Partial (CSP) blocks to reduce redundant gradient information
/// - Dark blocks with residual connections for gradient flow
/// - Multi-scale feature extraction at different depths
/// </para>
///
/// <para>Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection"</para>
/// </remarks>
public class CSPDarknet<T> : BackboneBase<T>
{
    private readonly List<CSPBlock<T>> _stages;
    private readonly Conv2D<T> _stem;
    private readonly int _depth;
    private readonly double _widthMultiplier;

    /// <inheritdoc/>
    public override string Name => $"CSPDarknet-{_widthMultiplier:0.0}x";

    /// <inheritdoc/>
    /// <remarks>
    /// Returns channel counts for P3, P4, P5 feature levels (stages 1, 2, 3).
    /// Stage 0 (P2) is not extracted as it's typically not used in detection heads.
    /// </remarks>
    public override int[] OutputChannels { get; }

    /// <summary>
    /// Internal channel configuration for all 4 stages (used during construction).
    /// </summary>
    private readonly int[] _stageChannels;

    /// <inheritdoc/>
    /// <remarks>
    /// Returns strides for P3, P4, P5 feature levels: 8, 16, 32.
    /// </remarks>
    public override int[] Strides => new[] { 8, 16, 32 };

    /// <summary>
    /// Creates a new CSP-Darknet backbone.
    /// </summary>
    /// <param name="depth">Depth multiplier for number of blocks (default 1.0 = medium).</param>
    /// <param name="widthMultiplier">Width multiplier for channel counts (default 1.0 = medium).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public CSPDarknet(double depth = 1.0, double widthMultiplier = 1.0, int inChannels = 3)
    {
        _depth = Math.Max(1, (int)Math.Round(depth));
        _widthMultiplier = widthMultiplier;
        _stages = new List<CSPBlock<T>>();

        // Calculate channel sizes based on width multiplier
        // _stageChannels contains all 4 stage channels for building the network
        int[] baseChannels = { 64, 128, 256, 512 };
        _stageChannels = baseChannels.Select(c => (int)(c * widthMultiplier)).ToArray();

        // OutputChannels only reflects stages 1, 2, 3 (P3, P4, P5) which are actually extracted
        OutputChannels = new[] { _stageChannels[1], _stageChannels[2], _stageChannels[3] };

        // Stem: Initial convolution (stride 2)
        _stem = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: _stageChannels[0] / 2,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            useBias: false
        );

        // Build stages
        int currentChannels = _stageChannels[0] / 2;
        for (int i = 0; i < 4; i++)
        {
            int outChannels = _stageChannels[i];
            int numBlocks = GetBlockCount(i, _depth);

            var stage = new CSPBlock<T>(
                inChannels: currentChannels,
                outChannels: outChannels,
                numBlocks: numBlocks,
                stride: 2
            );

            _stages.Add(stage);
            currentChannels = outChannels;
        }
    }

    /// <summary>
    /// Gets the number of blocks for each stage based on depth.
    /// </summary>
    private int GetBlockCount(int stage, int depth)
    {
        // Base block counts for CSP-Darknet53
        int[] baseCounts = { 1, 2, 8, 8 };
        return Math.Max(1, (int)Math.Round(baseCounts[stage] * depth * 0.33));
    }

    /// <inheritdoc/>
    public override List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();

        // Stem
        var x = _stem.Forward(input);
        x = ApplySiLU(x);

        // Stages with feature extraction
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);

            // Collect features from stages 1, 2, 3 (P3, P4, P5 in FPN)
            if (i >= 1)
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
        for (int i = 0; i < _stages.Count; i++)
        {
            count += _stages[i].GetParameterCount();
        }
        return count;
    }

    /// <inheritdoc/>
    public override void WriteParameters(System.IO.BinaryWriter writer)
    {
        // Write stem parameters
        _stem.WriteParameters(writer);

        // Write number of stages
        writer.Write(_stages.Count);

        // Write each stage's parameters
        foreach (var stage in _stages)
        {
            stage.WriteParameters(writer);
        }
    }

    /// <inheritdoc/>
    public override void ReadParameters(System.IO.BinaryReader reader)
    {
        // Read stem parameters
        _stem.ReadParameters(reader);

        // Read number of stages
        int stageCount = reader.ReadInt32();
        if (stageCount != _stages.Count)
        {
            throw new InvalidOperationException($"Expected {_stages.Count} stages but found {stageCount}.");
        }

        // Read each stage's parameters
        foreach (var stage in _stages)
        {
            stage.ReadParameters(reader);
        }
    }

    /// <summary>
    /// Applies SiLU (Swish) activation.
    /// </summary>
    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = NumOps.FromDouble(silu);
        }
        return result;
    }
}

/// <summary>
/// Cross-Stage Partial block used in CSP-Darknet.
/// </summary>
internal class CSPBlock<T>
{
    private readonly Conv2D<T> _downsample;
    private readonly Conv2D<T> _cv1;
    private readonly Conv2D<T> _cv2;
    private readonly Conv2D<T> _cv3;
    private readonly List<BottleneckBlock<T>> _bottlenecks;
    private readonly INumericOperations<T> _numOps;
    private readonly int _outChannels;

    public CSPBlock(int inChannels, int outChannels, int numBlocks, int stride = 1)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _outChannels = outChannels;

        int hiddenChannels = outChannels / 2;

        // Downsample if stride > 1
        _downsample = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            useBias: false
        );

        // Split path 1
        _cv1 = new Conv2D<T>(
            inChannels: outChannels,
            outChannels: hiddenChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            useBias: false
        );

        // Split path 2 (goes through bottlenecks)
        _cv2 = new Conv2D<T>(
            inChannels: outChannels,
            outChannels: hiddenChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            useBias: false
        );

        // Bottleneck blocks
        _bottlenecks = new List<BottleneckBlock<T>>();
        for (int i = 0; i < numBlocks; i++)
        {
            _bottlenecks.Add(new BottleneckBlock<T>(hiddenChannels, hiddenChannels));
        }

        // Concatenation and final conv
        _cv3 = new Conv2D<T>(
            inChannels: hiddenChannels * 2,
            outChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            useBias: false
        );
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Downsample
        var x = _downsample.Forward(input);
        x = ApplySiLU(x);

        // Split
        var y1 = _cv1.Forward(x);
        y1 = ApplySiLU(y1);

        var y2 = _cv2.Forward(x);
        y2 = ApplySiLU(y2);

        // Bottlenecks on y2
        foreach (var bottleneck in _bottlenecks)
        {
            y2 = bottleneck.Forward(y2);
        }

        // Concatenate along channel dimension
        var concat = ConcatenateChannels(y1, y2);

        // Final conv
        var output = _cv3.Forward(concat);
        output = ApplySiLU(output);

        return output;
    }

    public long GetParameterCount()
    {
        long count = 0;
        count += _downsample.GetParameterCount();
        count += _cv1.GetParameterCount();
        count += _cv2.GetParameterCount();
        count += _cv3.GetParameterCount();
        foreach (var b in _bottlenecks)
        {
            count += b.GetParameterCount();
        }
        return count;
    }

    public void WriteParameters(System.IO.BinaryWriter writer)
    {
        _downsample.WriteParameters(writer);
        _cv1.WriteParameters(writer);
        _cv2.WriteParameters(writer);
        _cv3.WriteParameters(writer);
        writer.Write(_bottlenecks.Count);
        foreach (var b in _bottlenecks)
        {
            b.WriteParameters(writer);
        }
    }

    public void ReadParameters(System.IO.BinaryReader reader)
    {
        _downsample.ReadParameters(reader);
        _cv1.ReadParameters(reader);
        _cv2.ReadParameters(reader);
        _cv3.ReadParameters(reader);
        int bottleneckCount = reader.ReadInt32();
        if (bottleneckCount != _bottlenecks.Count)
        {
            throw new InvalidOperationException($"Expected {_bottlenecks.Count} bottlenecks but found {bottleneckCount}.");
        }
        foreach (var b in _bottlenecks)
        {
            b.ReadParameters(reader);
        }
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(silu);
        }
        return result;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int batch = a.Shape[0];
        int channelsA = a.Shape[1];
        int channelsB = b.Shape[1];
        int height = a.Shape[2];
        int width = a.Shape[3];

        var result = new Tensor<T>(new[] { batch, channelsA + channelsB, height, width });

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channelsA; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[n, c, h, w] = a[n, c, h, w];
                    }
                }
            }
            for (int c = 0; c < channelsB; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[n, channelsA + c, h, w] = b[n, c, h, w];
                    }
                }
            }
        }

        return result;
    }
}

/// <summary>
/// Bottleneck block with residual connection.
/// </summary>
internal class BottleneckBlock<T>
{
    private readonly Conv2D<T> _cv1;
    private readonly Conv2D<T> _cv2;
    private readonly bool _add;
    private readonly INumericOperations<T> _numOps;

    public BottleneckBlock(int inChannels, int outChannels, bool add = true)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _add = add && inChannels == outChannels;

        int hiddenChannels = outChannels;

        _cv1 = new Conv2D<T>(
            inChannels: inChannels,
            outChannels: hiddenChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useBias: false
        );

        _cv2 = new Conv2D<T>(
            inChannels: hiddenChannels,
            outChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useBias: false
        );
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var y = _cv1.Forward(input);
        y = ApplySiLU(y);
        y = _cv2.Forward(y);
        y = ApplySiLU(y);

        if (_add)
        {
            // Residual connection
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = _numOps.Add(y[i], input[i]);
            }
        }

        return y;
    }

    public long GetParameterCount()
    {
        return _cv1.GetParameterCount() + _cv2.GetParameterCount();
    }

    public void WriteParameters(System.IO.BinaryWriter writer)
    {
        _cv1.WriteParameters(writer);
        _cv2.WriteParameters(writer);
    }

    public void ReadParameters(System.IO.BinaryReader reader)
    {
        _cv1.ReadParameters(reader);
        _cv2.ReadParameters(reader);
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(silu);
        }
        return result;
    }
}
