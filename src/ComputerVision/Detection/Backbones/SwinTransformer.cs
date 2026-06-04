using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
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
/// Swin Transformer backbone for hierarchical vision transformer feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
    "https://arxiv.org/abs/2103.14030",
    Year = 2021,
    Authors = "Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo")]
public class SwinTransformer<T> : NeuralNetworkBase<T>, IDetectionBackbone<T>
{
    private readonly PatchEmbeddingBlock<T> _patchEmbed;
    private readonly List<SwinStage<T>> _stages;
    private readonly SwinVariant _variant;
    private readonly int _embedDim;
    private readonly int _windowSize;
    private readonly int _inChannels;

    public bool IsFrozen { get; private set; }
    public string Name => $"Swin-{_variant}";
    public IReadOnlyList<int> OutputChannels { get; }
    public IReadOnlyList<int> Strides => new[] { 4, 8, 16, 32 };

    /// <summary>
    /// Creates a new Swin Transformer backbone.
    /// </summary>
    /// <param name="variant">Swin variant (Tiny, Small, Base, Large).</param>
    /// <param name="windowSize">Window size for attention (default 7).</param>
    /// <param name="inChannels">Number of input channels (default 3 for RGB).</param>
    public SwinTransformer(SwinVariant variant = SwinVariant.SwinTiny, int windowSize = 7, int inChannels = 3)
        : base(NeuralNetworkArchitecture<T>.CreateDynamicSpatial(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.ImageClassification,
                channels: inChannels,
                outputSize: 1),
              new MeanSquaredErrorLoss<T>())
    {
        _variant = variant;
        _windowSize = windowSize;
        _inChannels = inChannels;
        _stages = new List<SwinStage<T>>();

        var (embedDim, depths, numHeads) = GetVariantConfig(variant);
        _embedDim = embedDim;

        var outputChannels = new int[4];
        for (int i = 0; i < 4; i++) outputChannels[i] = embedDim * (1 << i);
        OutputChannels = outputChannels;

        _patchEmbed = new PatchEmbeddingBlock<T>(patchSize: 4, embedDim: embedDim);

        int currentDim = embedDim;
        for (int i = 0; i < 4; i++)
        {
            bool downsample = i > 0;
            var stage = new SwinStage<T>(currentDim, depths[i], numHeads[i], windowSize, downsample);
            _stages.Add(stage);
            if (downsample) currentDim *= 2;
        }
    }

    private static (int embedDim, int[] depths, int[] numHeads) GetVariantConfig(SwinVariant variant) => variant switch
    {
        SwinVariant.SwinTiny => (96, new[] { 2, 2, 6, 2 }, new[] { 3, 6, 12, 24 }),
        SwinVariant.SwinSmall => (96, new[] { 2, 2, 18, 2 }, new[] { 3, 6, 12, 24 }),
        SwinVariant.SwinBase => (128, new[] { 2, 2, 18, 2 }, new[] { 4, 8, 16, 32 }),
        SwinVariant.SwinLarge => (192, new[] { 2, 2, 18, 2 }, new[] { 6, 12, 24, 48 }),
        _ => (96, new[] { 2, 2, 6, 2 }, new[] { 3, 6, 12, 24 })
    };

    public List<Tensor<T>> ExtractFeatures(Tensor<T> input)
    {
        // Accept both batched [N, C, H, W] and unbatched [C, H, W] image input. The patch
        // embedding (a Conv2D) and the input.Shape[2]/[3] reads below require 4D NCHW; an
        // unbatched [C, H, W] tensor (e.g. a single image, which is what the model-family test
        // harness feeds) otherwise indexes past the end of the shape inside
        // PatchEmbeddingBlock.Forward. Promote rank-3 to a batch of 1 so both ranks work.
        input = EnsureBatchedNchw(input);

        var features = new List<Tensor<T>>();
        var x = _patchEmbed.Forward(input);
        for (int i = 0; i < _stages.Count; i++)
        {
            x = _stages[i].Forward(x);
            var featureMap = ReshapeToFeatureMap(x, input.Shape[2], input.Shape[3], i);
            features.Add(featureMap);
        }
        return features;
    }

    public IReadOnlyList<Tensor<T>> GetFeatureMaps(Tensor<T> input) => ExtractFeatures(input);

    /// <summary>
    /// Normalizes an image tensor to batched NCHW. A rank-4 <c>[N, C, H, W]</c> tensor is
    /// returned unchanged; a rank-3 <c>[C, H, W]</c> tensor is promoted to <c>[1, C, H, W]</c>
    /// so single-image callers (and the model-family test harness) work without a batch axis.
    /// Any other rank is rejected with a clear message rather than failing deep in the conv.
    /// </summary>
    private static Tensor<T> EnsureBatchedNchw(Tensor<T> input)
    {
        if (input.Shape.Length == 4) return input;
        if (input.Shape.Length == 3)
            return input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        throw new ArgumentException(
            $"SwinTransformer expects a [C, H, W] or [N, C, H, W] image tensor, but got a rank-" +
            $"{input.Shape.Length} tensor with shape [{string.Join(", ", input.Shape.ToArray())}].",
            nameof(input));
    }

    private Tensor<T> ReshapeToFeatureMap(Tensor<T> x, int inputHeight, int inputWidth, int stageIdx)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int dim = x.Shape[2];

        // Derive the feature-map grid from the ACTUAL sequence length rather than
        // inputHeight/stride. Patch merging pads odd grids up to even (PyTorch-faithful), so after
        // a stage the grid no longer equals inputHeight / Strides[stageIdx]; the sequence itself
        // carries the true H×W. inputHeight/inputWidth are no longer required to be divisible by
        // the stride, which is what let non-multiple-of-32 inputs (e.g. 112×112) work. The patch
        // grid is square for square inputs (the Swin norm), so the most-square factorization
        // recovers H,W exactly.
        var (height, width) = SwinFactorizeMostSquare(seqLen);
        if (height == 0)
            throw new InvalidOperationException(
                $"Cannot infer feature-map dimensions from sequence length {seqLen} at stage {stageIdx}.");

        var featureMap = new Tensor<T>(new[] { batch, dim, height, width });
        for (int n = 0; n < batch; n++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    for (int c = 0; c < dim; c++)
                        featureMap[n, c, h, w] = x[n, seqIdx, c];
                }
        return featureMap;
    }

    // Most-square factor pair (H >= W) of seqLen, or (0, 0) if seqLen <= 0.
    private static (int H, int W) SwinFactorizeMostSquare(int seqLen)
    {
        for (int candidate = (int)Math.Sqrt(seqLen); candidate >= 1; candidate--)
            if (seqLen % candidate == 0) return (seqLen / candidate, candidate);
        return (0, 0);
    }

    /// <summary>
    /// Sum across patch embedding + every Swin stage. Inherited
    /// <c>NeuralNetworkBase&lt;T&gt;.GetParameterCount()</c> delegates to this
    /// virtual property, satisfying the <see cref="IDetectionBackbone{T}"/> contract.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long count = _patchEmbed.GetParameterCount();
            for (int i = 0; i < _stages.Count; i++) count += _stages[i].GetParameterCount();
            return count;
        }
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write((int)_variant);
        writer.Write(_embedDim);
        writer.Write(_windowSize);
        writer.Write(_stages.Count);
        _patchEmbed.WriteParameters(writer);
        foreach (var stage in _stages) stage.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        var variant = (SwinVariant)reader.ReadInt32();
        int embedDim = reader.ReadInt32();
        int windowSize = reader.ReadInt32();
        int stageCount = reader.ReadInt32();

        if (variant != _variant)
            throw new InvalidOperationException($"SwinTransformer variant mismatch: expected {_variant}, got {variant}.");
        if (embedDim != _embedDim)
            throw new InvalidOperationException($"SwinTransformer embed dim mismatch: expected {_embedDim}, got {embedDim}.");
        if (windowSize != _windowSize)
            throw new InvalidOperationException($"SwinTransformer window size mismatch: expected {_windowSize}, got {windowSize}.");
        if (stageCount != _stages.Count)
            throw new InvalidOperationException($"SwinTransformer stage count mismatch: expected {_stages.Count}, got {stageCount}.");

        _patchEmbed.ReadParameters(reader);
        foreach (var stage in _stages) stage.ReadParameters(reader);
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
    /// Constructs a fresh Swin Transformer with the same variant, window size, and
    /// input-channel configuration.
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new SwinTransformer<T>(_variant, _windowSize, _inChannels);

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

    /// <inheritdoc />
    /// <remarks>
    /// Round-trips the parameter binary stream through a fresh
    /// <see cref="CreateNewInstance"/> so internal patch-embedding /
    /// transformer / patch-merging blocks and their tensor buffers are
    /// independent copies — see ResNet.DeepCopy.
    /// </remarks>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = (SwinTransformer<T>)CreateNewInstance();
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            WriteParameters(writer);
        }
        ms.Position = 0;
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            copy.ReadParameters(reader);
        }
        return copy;
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
    private readonly ConvolutionalLayer<T> _proj;
    private readonly int _patchSize;

    public PatchEmbeddingBlock(int patchSize, int embedDim)
    {
        _patchSize = patchSize;
        _proj = new ConvolutionalLayer<T>(outputDepth: embedDim, kernelSize: patchSize, stride: patchSize, padding: 0);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = _proj.Forward(input);
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];
        int numPatches = height * width;

        var sequence = new Tensor<T>(new[] { batch, numPatches, channels });
        for (int n = 0; n < batch; n++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    for (int c = 0; c < channels; c++)
                        sequence[n, seqIdx, c] = x[n, c, h, w];
                }
        return sequence;
    }

    public long GetParameterCount() => _proj.ParameterCount;

    public void WriteParameters(BinaryWriter writer) =>
        BackboneSerialization.WriteLayerParameters(writer, _proj);

    public void ReadParameters(BinaryReader reader) =>
        BackboneSerialization.ReadLayerParameters(reader, _proj);
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
        for (int i = 0; i < depth; i++)
        {
            bool shiftWindows = i % 2 == 1;
            _blocks.Add(new SwinTransformerBlock<T>(dim, numHeads, windowSize,
                shiftSize: shiftWindows ? windowSize / 2 : 0));
        }
        if (downsample) _patchMerge = new PatchMergingBlock<T>(dim);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;
        foreach (var block in _blocks) x = block.Forward(x);
        if (_patchMerge is not null) x = _patchMerge.Forward(x);
        return x;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var block in _blocks) count += block.GetParameterCount();
        if (_patchMerge is not null) count += _patchMerge.GetParameterCount();
        return count;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_blocks.Count);
        writer.Write(_patchMerge is not null);
        foreach (var block in _blocks) block.WriteParameters(writer);
        if (_patchMerge is not null) _patchMerge.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int blockCount = reader.ReadInt32();
        bool hasPatchMerge = reader.ReadBoolean();
        if (blockCount != _blocks.Count)
            throw new InvalidOperationException($"SwinStage block count mismatch: expected {_blocks.Count}, got {blockCount}.");
        if (hasPatchMerge != (_patchMerge is not null))
            throw new InvalidOperationException("SwinStage patch merge configuration mismatch.");
        foreach (var block in _blocks) block.ReadParameters(reader);
        if (_patchMerge is not null) _patchMerge.ReadParameters(reader);
    }
}

/// <summary>
/// Single Swin Transformer block with windowed multi-head self-attention.
/// </summary>
internal class SwinTransformerBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _windowSize;
    private readonly int _shiftSize;
    private readonly double _scale;

    private readonly SwinLayerNorm<T> _norm1;
    private readonly SwinLayerNorm<T> _norm2;

    private readonly DenseLayer<T> _qkvProj;
    private readonly DenseLayer<T> _outProj;

    private readonly Tensor<T> _relativePositionBiasTable;
    private readonly int[,] _relativePositionIndex;

    private readonly DenseLayer<T> _mlpFc1;
    private readonly DenseLayer<T> _mlpFc2;

    public SwinTransformerBlock(int dim, int numHeads, int windowSize, int shiftSize)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _numHeads = numHeads;
        _headDim = dim / numHeads;
        _windowSize = windowSize;
        _shiftSize = shiftSize;
        _scale = 1.0 / Math.Sqrt(_headDim);

        _norm1 = new SwinLayerNorm<T>(dim);
        _norm2 = new SwinLayerNorm<T>(dim);

        _qkvProj = new DenseLayer<T>(dim * 3, (Interfaces.IActivationFunction<T>?)null);
        _outProj = new DenseLayer<T>(dim, (Interfaces.IActivationFunction<T>?)null);

        int biasTableSize = (2 * windowSize - 1) * (2 * windowSize - 1);
        _relativePositionBiasTable = new Tensor<T>(new[] { biasTableSize, numHeads });
        InitializeRelativePositionBias();

        _relativePositionIndex = ComputeRelativePositionIndex(windowSize);

        int mlpHiddenDim = dim * 4;
        _mlpFc1 = new DenseLayer<T>(mlpHiddenDim, (Interfaces.IActivationFunction<T>?)null);
        _mlpFc2 = new DenseLayer<T>(dim, (Interfaces.IActivationFunction<T>?)null);
    }

    private void InitializeRelativePositionBias()
    {
        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
        {
            double value = random.NextGaussian(0, 0.02);
            value = Math.Max(-0.04, Math.Min(0.04, value));
            _relativePositionBiasTable[i] = _numOps.FromDouble(value);
        }
    }

    private int[,] ComputeRelativePositionIndex(int windowSize)
    {
        int windowArea = windowSize * windowSize;
        var idx = new int[windowArea, windowArea];
        for (int i = 0; i < windowArea; i++)
        {
            int h1 = i / windowSize;
            int w1 = i % windowSize;
            for (int j = 0; j < windowArea; j++)
            {
                int h2 = j / windowSize;
                int w2 = j % windowSize;
                int relH = h1 - h2 + windowSize - 1;
                int relW = w1 - w2 + windowSize - 1;
                idx[i, j] = relH * (2 * windowSize - 1) + relW;
            }
        }
        return idx;
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        int seqLen = input.Shape[1];
        int h = (int)Math.Sqrt(seqLen);
        int w = seqLen / h;
        if (h * w != seqLen)
        {
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

        var normed1 = _norm1.Forward(input);
        var attnOut = WindowAttention(normed1, h, w);
        var x = AddTensors(input, attnOut);

        var normed2 = _norm2.Forward(x);
        var mlpOut = ApplyMLP(normed2);
        x = AddTensors(x, mlpOut);

        return x;
    }

    private Tensor<T> WindowAttention(Tensor<T> x, int h, int w)
    {
        int batch = x.Shape[0];
        var spatial = ReshapeToSpatial(x, batch, h, w, _dim);
        if (_shiftSize > 0) spatial = CyclicShift(spatial, -_shiftSize);

        var (windows, numWindowsH, numWindowsW) = WindowPartition(spatial);
        var attnOut = WindowedSelfAttention(windows);
        var merged = WindowReverse(attnOut, numWindowsH, numWindowsW, batch, h, w);

        if (_shiftSize > 0) merged = CyclicShift(merged, _shiftSize);
        return ReshapeToSequence(merged);
    }

    private Tensor<T> ReshapeToSpatial(Tensor<T> x, int batch, int h, int w, int c)
    {
        var spatial = new Tensor<T>(new[] { batch, h, w, c });
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int seqIdx = i * w + j;
                    for (int d = 0; d < c; d++) spatial[b, i, j, d] = x[b, seqIdx, d];
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
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int seqIdx = i * w + j;
                    for (int d = 0; d < c; d++) seq[b, seqIdx, d] = spatial[b, i, j, d];
                }
        return seq;
    }

    private Tensor<T> CyclicShift(Tensor<T> x, int shift)
    {
        int batch = x.Shape[0];
        int h = x.Shape[1];
        int w = x.Shape[2];
        int c = x.Shape[3];
        var shifted = new Tensor<T>(x._shape);
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < h; i++)
                for (int j = 0; j < w; j++)
                {
                    int srcI = (i - shift % h + h) % h;
                    int srcJ = (j - shift % w + w) % w;
                    for (int d = 0; d < c; d++) shifted[b, i, j, d] = x[b, srcI, srcJ, d];
                }
        return shifted;
    }

    private (Tensor<T> windows, int numWindowsH, int numWindowsW) WindowPartition(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int h = x.Shape[1];
        int w = x.Shape[2];
        int c = x.Shape[3];

        int padH = (_windowSize - h % _windowSize) % _windowSize;
        int padW = (_windowSize - w % _windowSize) % _windowSize;
        int paddedH = h + padH;
        int paddedW = w + padW;

        Tensor<T> padded;
        if (padH > 0 || padW > 0)
        {
            padded = new Tensor<T>(new[] { batch, paddedH, paddedW, c });
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < paddedH; i++)
                    for (int j = 0; j < paddedW; j++)
                        for (int d = 0; d < c; d++)
                            padded[b, i, j, d] = (i < h && j < w) ? x[b, i, j, d] : _numOps.FromDouble(0.0);
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
            for (int wh = 0; wh < numWindowsH; wh++)
                for (int ww = 0; ww < numWindowsW; ww++)
                {
                    int windowIdx = b * numWindows + wh * numWindowsW + ww;
                    int startH = wh * _windowSize;
                    int startW = ww * _windowSize;
                    for (int i = 0; i < _windowSize; i++)
                        for (int j = 0; j < _windowSize; j++)
                        {
                            int tokenIdx = i * _windowSize + j;
                            for (int d = 0; d < c; d++)
                                windows[windowIdx, tokenIdx, d] = padded[b, startH + i, startW + j, d];
                        }
                }
        return (windows, numWindowsH, numWindowsW);
    }

    private Tensor<T> WindowReverse(Tensor<T> windows, int numWindowsH, int numWindowsW, int batch, int h, int w)
    {
        int numWindows = numWindowsH * numWindowsW;
        int c = windows.Shape[2];

        var spatial = new Tensor<T>(new[] { batch, h, w, c });
        for (int b = 0; b < batch; b++)
            for (int wh = 0; wh < numWindowsH; wh++)
                for (int ww = 0; ww < numWindowsW; ww++)
                {
                    int windowIdx = b * numWindows + wh * numWindowsW + ww;
                    int startH = wh * _windowSize;
                    int startW = ww * _windowSize;
                    for (int i = 0; i < _windowSize; i++)
                        for (int j = 0; j < _windowSize; j++)
                        {
                            int outH = startH + i;
                            int outW = startW + j;
                            if (outH < h && outW < w)
                            {
                                int tokenIdx = i * _windowSize + j;
                                for (int d = 0; d < c; d++)
                                    spatial[b, outH, outW, d] = windows[windowIdx, tokenIdx, d];
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

        var qkv = new Tensor<T>(new[] { numWindows, windowArea, 3 * c });
        for (int wIdx = 0; wIdx < numWindows; wIdx++)
        {
            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>(new[] { 1, c });
                for (int d = 0; d < c; d++) tokenIn[0, d] = windows[wIdx, t, d];
                var tokenQkv = _qkvProj.Forward(tokenIn);
                for (int d = 0; d < 3 * c; d++) qkv[wIdx, t, d] = tokenQkv[0, d];
            }
        }

        var output = new Tensor<T>(new[] { numWindows, windowArea, c });
        for (int wIdx = 0; wIdx < numWindows; wIdx++)
        {
            var attnScores = new double[_numHeads, windowArea, windowArea];
            for (int head = 0; head < _numHeads; head++)
            {
                int headOffset = head * _headDim;
                for (int i = 0; i < windowArea; i++)
                    for (int j = 0; j < windowArea; j++)
                    {
                        double score = 0;
                        for (int d = 0; d < _headDim; d++)
                        {
                            double q = _numOps.ToDouble(qkv[wIdx, i, headOffset + d]);
                            double k = _numOps.ToDouble(qkv[wIdx, j, c + headOffset + d]);
                            score += q * k;
                        }
                        score *= _scale;
                        int biasIdx = _relativePositionIndex[i, j];
                        score += _numOps.ToDouble(_relativePositionBiasTable[biasIdx, head]);
                        attnScores[head, i, j] = score;
                    }
            }

            var attnProbs = new double[_numHeads, windowArea, windowArea];
            for (int head = 0; head < _numHeads; head++)
            {
                for (int i = 0; i < windowArea; i++)
                {
                    double maxScore = double.NegativeInfinity;
                    for (int j = 0; j < windowArea; j++)
                        if (attnScores[head, i, j] > maxScore) maxScore = attnScores[head, i, j];

                    double sumExp = 0;
                    for (int j = 0; j < windowArea; j++)
                    {
                        attnProbs[head, i, j] = Math.Exp(attnScores[head, i, j] - maxScore);
                        sumExp += attnProbs[head, i, j];
                    }
                    for (int j = 0; j < windowArea; j++) attnProbs[head, i, j] /= sumExp;
                }
            }

            var attnOut = new double[windowArea, c];
            for (int head = 0; head < _numHeads; head++)
            {
                int headOffset = head * _headDim;
                int vOffset = 2 * c + headOffset;
                for (int i = 0; i < windowArea; i++)
                    for (int d = 0; d < _headDim; d++)
                    {
                        double val = 0;
                        for (int j = 0; j < windowArea; j++)
                            val += attnProbs[head, i, j] * _numOps.ToDouble(qkv[wIdx, j, vOffset + d]);
                        attnOut[i, headOffset + d] = val;
                    }
            }

            for (int t = 0; t < windowArea; t++)
            {
                var tokenIn = new Tensor<T>(new[] { 1, c });
                for (int d = 0; d < c; d++) tokenIn[0, d] = _numOps.FromDouble(attnOut[t, d]);
                var tokenOut = _outProj.Forward(tokenIn);
                for (int d = 0; d < c; d++) output[wIdx, t, d] = tokenOut[0, d];
            }
        }
        return output;
    }

    private Tensor<T> ApplyMLP(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        var result = new Tensor<T>(x._shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var tokenIn = new Tensor<T>(new[] { 1, _dim });
                for (int d = 0; d < _dim; d++) tokenIn[0, d] = x[b, s, d];

                var hidden = _mlpFc1.Forward(tokenIn);
                for (int d = 0; d < hidden.Shape[1]; d++)
                {
                    double val = _numOps.ToDouble(hidden[0, d]);
                    double gelu = 0.5 * val * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (val + 0.044715 * val * val * val)));
                    hidden[0, d] = _numOps.FromDouble(gelu);
                }

                var tokenOut = _mlpFc2.Forward(hidden);
                for (int d = 0; d < _dim; d++) result[b, s, d] = tokenOut[0, d];
            }
        }
        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        AiDotNetEngine.Current.TensorAdd(a, b);

    public long GetParameterCount()
    {
        long normParams = 2 * 2 * _dim;
        long qkvParams = _qkvProj.ParameterCount;
        long outProjParams = _outProj.ParameterCount;
        int biasTableSize = (2 * _windowSize - 1) * (2 * _windowSize - 1);
        long biasParams = biasTableSize * _numHeads;
        long mlpParams = _mlpFc1.ParameterCount + _mlpFc2.ParameterCount;
        return normParams + qkvParams + outProjParams + biasParams + mlpParams;
    }

    public void WriteParameters(BinaryWriter writer)
    {
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
        BackboneSerialization.WriteLayerParameters(writer, _qkvProj);
        BackboneSerialization.WriteLayerParameters(writer, _outProj);

        writer.Write(_relativePositionBiasTable.Shape[0]);
        writer.Write(_relativePositionBiasTable.Shape[1]);
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
            writer.Write(_numOps.ToDouble(_relativePositionBiasTable[i]));

        BackboneSerialization.WriteLayerParameters(writer, _mlpFc1);
        BackboneSerialization.WriteLayerParameters(writer, _mlpFc2);
    }

    public void ReadParameters(BinaryReader reader)
    {
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
        BackboneSerialization.ReadLayerParameters(reader, _qkvProj);
        BackboneSerialization.ReadLayerParameters(reader, _outProj);

        int biasTableDim0 = reader.ReadInt32();
        int biasTableDim1 = reader.ReadInt32();
        if (biasTableDim0 != _relativePositionBiasTable.Shape[0] || biasTableDim1 != _relativePositionBiasTable.Shape[1])
            throw new InvalidOperationException("SwinTransformerBlock relative position bias table shape mismatch.");
        for (int i = 0; i < _relativePositionBiasTable.Length; i++)
            _relativePositionBiasTable[i] = _numOps.FromDouble(reader.ReadDouble());

        BackboneSerialization.ReadLayerParameters(reader, _mlpFc1);
        BackboneSerialization.ReadLayerParameters(reader, _mlpFc2);
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
        _numOps = MathHelper.GetNumericOperations<T>();
        _dim = dim;
        _eps = eps;
        _gamma = new Tensor<T>(new[] { dim });
        _beta = new Tensor<T>(new[] { dim });
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
        var result = new Tensor<T>(x._shape);

        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                double mean = 0;
                for (int d = 0; d < dim; d++) mean += _numOps.ToDouble(x[b, s, d]);
                mean /= dim;

                double variance = 0;
                for (int d = 0; d < dim; d++)
                {
                    double diff = _numOps.ToDouble(x[b, s, d]) - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                double std = Math.Sqrt(variance + _eps);
                for (int d = 0; d < dim; d++)
                {
                    double normalized = (_numOps.ToDouble(x[b, s, d]) - mean) / std;
                    double gamma = _numOps.ToDouble(_gamma[d]);
                    double beta = _numOps.ToDouble(_beta[d]);
                    result[b, s, d] = _numOps.FromDouble(gamma * normalized + beta);
                }
            }
        return result;
    }

    public long GetParameterCount() => 2 * _dim;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_dim);
        for (int i = 0; i < _dim; i++) writer.Write(_numOps.ToDouble(_gamma[i]));
        for (int i = 0; i < _dim; i++) writer.Write(_numOps.ToDouble(_beta[i]));
    }

    public void ReadParameters(BinaryReader reader)
    {
        int dim = reader.ReadInt32();
        if (dim != _dim)
            throw new InvalidOperationException($"SwinLayerNorm dim mismatch: expected {_dim}, got {dim}.");
        for (int i = 0; i < _dim; i++) _gamma[i] = _numOps.FromDouble(reader.ReadDouble());
        for (int i = 0; i < _dim; i++) _beta[i] = _numOps.FromDouble(reader.ReadDouble());
    }
}

/// <summary>
/// Patch merging layer for downsampling in Swin Transformer.
/// </summary>
internal class PatchMergingBlock<T>
{
    private readonly DenseLayer<T> _reduction;
    private readonly int _dim;

    public PatchMergingBlock(int dim)
    {
        _dim = dim;
        _reduction = new DenseLayer<T>(dim * 2, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => Forward(input, null, null);

    public Tensor<T> Forward(Tensor<T> input, int? inputHeight, int? inputWidth)
    {
        int batch = input.Shape[0];
        int seqLen = input.Shape[1];
        int dim = input.Shape[2];

        int h, w;
        if (inputHeight.HasValue && inputWidth.HasValue)
        {
            h = inputHeight.Value;
            w = inputWidth.Value;
            if (h * w != seqLen)
                throw new ArgumentException($"Provided dimensions ({h} x {w} = {h * w}) do not match sequence length {seqLen}.");
        }
        else
        {
            // Prefer an exact even × even factorization (the common case for real images whose
            // patch grid stays even through the stages). If none exists — e.g. an odd-sided grid
            // such as 7 × 7 = 49 from a small 28 × 28 input — fall back to the MOST-SQUARE
            // factorization (allowing odd sides) and pad the odd dimension(s) to even below,
            // mirroring PyTorch's SwinTransformer which F.pads odd H/W before patch merging
            // instead of rejecting the input.
            (h, w) = FactorizeEvenEven(seqLen);
            if (h == 0) (h, w) = FactorizeMostSquare(seqLen);
            if (h == 0)
                throw new ArgumentException(
                    $"Cannot infer spatial dimensions from sequence length {seqLen} for patch merging.");
        }

        // Pad odd H/W up to the next even size (zeros), so the 2×2 merge always has full quads.
        int hPad = h + (h & 1);
        int wPad = w + (w & 1);
        Tensor<T> src = input;
        if (hPad != h || wPad != w)
        {
            var padded = new Tensor<T>(new[] { batch, hPad * wPad, dim });
            for (int n = 0; n < batch; n++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        int srcIdx = i * w + j;
                        int dstIdx = i * wPad + j;
                        for (int d = 0; d < dim; d++)
                            padded[n, dstIdx, d] = input[n, srcIdx, d];
                    }
            src = padded;
        }

        int newH = hPad / 2;
        int newW = wPad / 2;
        int newSeqLen = newH * newW;
        var merged = new Tensor<T>(new[] { batch, newSeqLen, dim * 4 });

        for (int n = 0; n < batch; n++)
        {
            for (int i = 0; i < newH; i++)
                for (int j = 0; j < newW; j++)
                {
                    int newIdx = i * newW + j;
                    int idx0 = (2 * i) * wPad + (2 * j);
                    int idx1 = (2 * i) * wPad + (2 * j + 1);
                    int idx2 = (2 * i + 1) * wPad + (2 * j);
                    int idx3 = (2 * i + 1) * wPad + (2 * j + 1);
                    for (int d = 0; d < dim; d++)
                    {
                        merged[n, newIdx, d] = src[n, idx0, d];
                        merged[n, newIdx, dim + d] = src[n, idx1, d];
                        merged[n, newIdx, 2 * dim + d] = src[n, idx2, d];
                        merged[n, newIdx, 3 * dim + d] = src[n, idx3, d];
                    }
                }
        }

        return _reduction.Forward(merged);
    }

    // Returns the even × even factor pair of seqLen closest to square, or (0, 0) if none.
    private static (int H, int W) FactorizeEvenEven(int seqLen)
    {
        for (int candidate = (int)Math.Sqrt(seqLen); candidate >= 1; candidate--)
            if (seqLen % candidate == 0)
            {
                int other = seqLen / candidate;
                if ((candidate & 1) == 0 && (other & 1) == 0) return (other, candidate);
            }
        return (0, 0);
    }

    // Returns the factor pair of seqLen closest to square (sides may be odd), or (0, 0).
    private static (int H, int W) FactorizeMostSquare(int seqLen)
    {
        for (int candidate = (int)Math.Sqrt(seqLen); candidate >= 1; candidate--)
            if (seqLen % candidate == 0) return (seqLen / candidate, candidate);
        return (0, 0);
    }

    public long GetParameterCount() => _reduction.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_dim);
        BackboneSerialization.WriteLayerParameters(writer, _reduction);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int dim = reader.ReadInt32();
        if (dim != _dim)
            throw new InvalidOperationException($"PatchMergingBlock dim mismatch: expected {_dim}, got {dim}.");
        BackboneSerialization.ReadLayerParameters(reader, _reduction);
    }
}
