using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// EfficientNet backbone for efficient feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Reference: Tan et al., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    "https://arxiv.org/abs/1905.11946",
    Year = 2019,
    Authors = "Mingxing Tan, Quoc V. Le")]
public class EfficientNet<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly ConvolutionalLayer<T> _stem;
    private readonly List<MBConvBlock<T>> _blocks;
    private readonly EfficientNetVariant _variant;
    private readonly int _stemChannels;
    private readonly int _inChannels;
    private readonly int[] _featureIndices;

    public bool IsFrozen { get; private set; }
    public string Name => $"EfficientNet-{_variant}";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Creates a new EfficientNet backbone.
    /// </summary>
    /// <param name="variant">EfficientNet variant (B0-B7).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public EfficientNet(EfficientNetVariant variant = EfficientNetVariant.B0, int inChannels = 3)
        : base(NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ImageClassification,
                channels: inChannels,
                outputSize: 1),
              new MeanSquaredErrorLoss<T>())
    {
        _variant = variant;
        _inChannels = inChannels;
        _blocks = new List<MBConvBlock<T>>();

        var (widthMult, depthMult) = GetScalingFactors(variant);

        var blockConfigs = new[]
        {
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        };

        _stemChannels = ScaleChannels(32, widthMult);
        OutputChannels = new[]
        {
            ScaleChannels(24, widthMult),
            ScaleChannels(40, widthMult),
            ScaleChannels(192, widthMult),
            ScaleChannels(320, widthMult)
        };

        _stem = new ConvolutionalLayer<T>(_stemChannels, kernelSize: 3, stride: 2, padding: 1);

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

                _blocks.Add(new MBConvBlock<T>(blockInChannels, scaledChannels, kernelSize, blockStride, expandRatio, useSE: true));
                blockIdx++;
            }
            currentChannels = scaledChannels;

            if (stageIdx == 1 || stageIdx == 2 || stageIdx == 5 || stageIdx == 6)
                featureIndicesList.Add(blockIdx - 1);
        }

        _featureIndices = featureIndicesList.ToArray();
    }

    private static (double width, double depth) GetScalingFactors(EfficientNetVariant variant) => variant switch
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

    private static int ScaleChannels(int channels, double multiplier)
    {
        int scaled = (int)(channels * multiplier);
        return Math.Max(8, ((scaled + 4) / 8) * 8);
    }

    private static int ScaleDepth(int depth, double multiplier) => (int)Math.Ceiling(depth * multiplier);

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        var features = new List<Tensor<T>>();
        var x = _stem.Forward(input);
        x = ApplySwish(x);
        for (int i = 0; i < _blocks.Count; i++)
        {
            x = _blocks[i].Forward(x);
            if (_featureIndices.Contains(i)) features.Add(x);
        }
        return features;
    }

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Sum across stem + every MBConv block. Inherited
    /// <c>NeuralNetworkBase&lt;T&gt;.GetParameterCount()</c> delegates to this
    /// virtual property, satisfying the <see cref="IDetectionBackbone{T}"/> contract.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long count = _stem.ParameterCount;
            foreach (var block in _blocks) count += block.GetParameterCount();
            return count;
        }
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write((int)_variant);
        writer.Write(_blocks.Count);
        writer.Write(_featureIndices.Length);
        foreach (int idx in _featureIndices) writer.Write(idx);
        BackboneSerialization.WriteLayerParameters(writer, _stem);
        foreach (var block in _blocks) block.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        var variant = (EfficientNetVariant)reader.ReadInt32();
        int blockCount = reader.ReadInt32();
        int featureIndexCount = reader.ReadInt32();
        var featureIndices = new int[featureIndexCount];
        for (int i = 0; i < featureIndexCount; i++) featureIndices[i] = reader.ReadInt32();

        if (variant != _variant)
            throw new InvalidOperationException($"EfficientNet variant mismatch: expected {_variant}, got {variant}.");
        if (blockCount != _blocks.Count)
            throw new InvalidOperationException($"EfficientNet block count mismatch: expected {_blocks.Count}, got {blockCount}.");
        if (featureIndexCount != _featureIndices.Length)
            throw new InvalidOperationException($"EfficientNet feature index count mismatch: expected {_featureIndices.Length}, got {featureIndexCount}.");
        for (int i = 0; i < featureIndexCount; i++)
            if (featureIndices[i] != _featureIndices[i])
                throw new InvalidOperationException($"EfficientNet feature index mismatch at position {i}: expected {_featureIndices[i]}, got {featureIndices[i]}.");

        BackboneSerialization.ReadLayerParameters(reader, _stem);
        foreach (var block in _blocks) block.ReadParameters(reader);
    }

    public virtual void Freeze() => IsFrozen = true;
    public virtual void Unfreeze() => IsFrozen = false;
    public (int Height, int Width) GetExpectedInputSize() => (640, 640);

    public override Tensor<T> Predict(Tensor<T> input)
    {
        var features = ExtractFeatures(input);
        if (features.Count == 0)
            throw new InvalidOperationException(
                $"{GetType().Name}.ExtractFeatures returned no feature maps.");
        return features[features.Count - 1];
    }

    protected override void InitializeLayers() { }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer) => WriteParameters(writer);
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) => ReadParameters(reader);

    /// <inheritdoc />
    /// <remarks>
    /// Constructs a fresh EfficientNet with the same variant and input-channel configuration.
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new EfficientNet<T>(_variant, _inChannels);

    public override ModelMetadata<T> GetModelMetadata() => new ModelMetadata<T>
    {
        Name = Name,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["BackboneName"] = Name,
            ["OutputChannels"] = OutputChannels,
            ["Strides"] = Strides
        }
    };

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException(
            $"{GetType().Name}: detection backbones train as part of a parent detector.");

    public override Vector<T> GetParameters() =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not expose a flat parameter vector. Use WriteParameters/ReadParameters.");

    public override void SetParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter vector. Use ReadParameters.");

    public override void UpdateParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: backbones do not accept a flat parameter update vector.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters) =>
        throw new NotSupportedException(
            $"{GetType().Name}: WithParameters(Vector<T>) is unsupported on backbones.");

    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => (EfficientNet<T>)MemberwiseClone();

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < x.Length; i++)
        {
            double val = ops.ToDouble(x[i]);
            result[i] = ops.FromDouble(val * (1.0 / (1.0 + Math.Exp(-val))));
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
    private readonly ConvolutionalLayer<T>? _expand;
    private readonly ConvolutionalLayer<T> _depthwise;
    private readonly SqueezeExcitation<T>? _se;
    private readonly ConvolutionalLayer<T> _project;
    private readonly bool _useResidual;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _expandRatio;

    public MBConvBlock(int inChannels, int outChannels, int kernelSize, int stride, int expandRatio, bool useSE)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _expandRatio = expandRatio;
        _useResidual = inChannels == outChannels && stride == 1;

        int hiddenDim = inChannels * expandRatio;

        if (expandRatio != 1)
            _expand = new ConvolutionalLayer<T>(hiddenDim, kernelSize: 1, stride: 1, padding: 0);

        int padding = (kernelSize - 1) / 2;
        _depthwise = new ConvolutionalLayer<T>(hiddenDim, kernelSize: kernelSize, stride: stride, padding: padding);

        if (useSE)
        {
            int seChannels = Math.Max(1, inChannels / 4);
            _se = new SqueezeExcitation<T>(hiddenDim, seChannels);
        }

        _project = new ConvolutionalLayer<T>(outChannels, kernelSize: 1, stride: 1, padding: 0);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;

        if (_expand is not null)
        {
            x = _expand.Forward(x);
            x = ApplySwish(x);
        }

        x = _depthwise.Forward(x);
        x = ApplySwish(x);

        if (_se is not null) x = _se.Forward(x);

        x = _project.Forward(x);

        if (_useResidual) x = BackboneOps<T>.AddResidual(x, input);

        return x;
    }

    public long GetParameterCount()
    {
        long count = (_expand?.ParameterCount ?? 0) + _depthwise.ParameterCount + _project.ParameterCount;
        if (_se is not null) count += _se.GetParameterCount();
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_expand is not null);
        writer.Write(_se is not null);
        if (_expand is not null) BackboneSerialization.WriteLayerParameters(writer, _expand);
        BackboneSerialization.WriteLayerParameters(writer, _depthwise);
        if (_se is not null) _se.WriteParameters(writer);
        BackboneSerialization.WriteLayerParameters(writer, _project);
    }

    public void ReadParameters(BinaryReader reader)
    {
        bool hasExpand = reader.ReadBoolean();
        bool hasSE = reader.ReadBoolean();
        if (hasExpand != (_expand is not null))
            throw new InvalidOperationException("MBConvBlock expand configuration mismatch.");
        if (hasSE != (_se is not null))
            throw new InvalidOperationException("MBConvBlock SE configuration mismatch.");
        if (_expand is not null) BackboneSerialization.ReadLayerParameters(reader, _expand);
        BackboneSerialization.ReadLayerParameters(reader, _depthwise);
        if (_se is not null) _se.ReadParameters(reader);
        BackboneSerialization.ReadLayerParameters(reader, _project);
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < x.Length; i++)
        {
            double val = ops.ToDouble(x[i]);
            result[i] = ops.FromDouble(val * (1.0 / (1.0 + Math.Exp(-val))));
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
    private readonly DenseLayer<T> _fc1;
    private readonly DenseLayer<T> _fc2;

    public SqueezeExcitation(int channels, int reducedChannels)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _fc1 = new DenseLayer<T>(reducedChannels, (Interfaces.IActivationFunction<T>?)null);
        _fc2 = new DenseLayer<T>(channels, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        // Global average pool → [batch, channels]
        var squeezed = new Tensor<T>(new[] { batch, channels });
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sum = 0;
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        sum += _numOps.ToDouble(input[n, c, h, w]);
                squeezed[n, c] = _numOps.FromDouble(sum / (height * width));
            }
        }

        var excited = _fc1.Forward(squeezed);
        excited = ApplySwish(excited);
        excited = _fc2.Forward(excited);
        excited = ApplySigmoid(excited);

        var output = new Tensor<T>(input._shape);
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                T scale = excited[n, c];
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[n, c, h, w] = _numOps.Multiply(input[n, c, h, w], scale);
            }
        }

        return output;
    }

    public long GetParameterCount() => _fc1.ParameterCount + _fc2.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        BackboneSerialization.WriteLayerParameters(writer, _fc1);
        BackboneSerialization.WriteLayerParameters(writer, _fc2);
    }

    public void ReadParameters(BinaryReader reader)
    {
        BackboneSerialization.ReadLayerParameters(reader, _fc1);
        BackboneSerialization.ReadLayerParameters(reader, _fc2);
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            result[i] = _numOps.FromDouble(val * (1.0 / (1.0 + Math.Exp(-val))));
        }
        return result;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> x)
    {
        var result = new Tensor<T>(x._shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            result[i] = _numOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }
}
