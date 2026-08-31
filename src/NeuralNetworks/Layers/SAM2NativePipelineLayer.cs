using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Native, trainable SAM 2 image/video segmentation pipeline.</summary>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3,
    TestInputShape = "3, 32, 32",
    TestConstructorArgs = "3, 32, 32, 16, new int[] { 1, 1, 1, 1 }, 1, new int[] { 8, 4, 2, 1 }, new int[] { 2 }, 16, 8, 4, 1, 2, 64, 10000.0")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SAM2NativePipelineLayer<T> : LayerBase<T>, IShapeContract
{
    private const int MaskCandidateCount = 4;
    private const int OutputTokenCount = 2 + MaskCandidateCount;

    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _hieraEmbeddingDimension;
    private readonly int[] _hieraStageDepths;
    private readonly int _hieraInitialHeads;
    private readonly int[] _hieraWindowSizes;
    private readonly int[] _hieraGlobalAttentionBlockIndexes;
    private readonly int _modelDimension;
    private readonly int _memoryDimension;
    private readonly int _decoderHeads;
    private readonly int _memoryAttentionLayers;
    private readonly int _decoderDepth;
    private readonly int _decoderMlpDimension;
    private readonly double _ropeTheta;
    private readonly double _memoryMaskScale;
    private readonly double _memoryMaskBias;
    private readonly int _featureHeight;
    private readonly int _featureWidth;
    private readonly int _maskEmbeddingDimension;

    private readonly SAM2HieraImageEncoderLayer<T> _imageEncoder;

    private readonly ConvolutionalLayer<T>[] _promptMaskDownsampler;
    private readonly ConvolutionalLayer<T>[] _memoryMaskDownsampler;
    private readonly ConvolutionalLayer<T> _memoryPixelProjection;
    private readonly ConvolutionalLayer<T>[] _memoryDepthwise;
    private readonly ConvolutionalLayer<T>[] _memoryExpand;
    private readonly ConvolutionalLayer<T>[] _memoryContract;
    private readonly ConvolutionalLayer<T> _memoryOutputProjection;

    private readonly SAM2RoPEAttentionLayer<T>[] _memorySelfAttention;
    private readonly SAM2RoPEAttentionLayer<T>[] _memoryCrossAttention;
    private readonly LayerNormalizationLayer<T>[] _memoryNorm1;
    private readonly LayerNormalizationLayer<T>[] _memoryNorm2;
    private readonly LayerNormalizationLayer<T>[] _memoryNorm3;
    private readonly DenseLayer<T>[] _memoryFfn1;
    private readonly DenseLayer<T>[] _memoryFfn2;
    private readonly LayerNormalizationLayer<T> _memoryFinalNorm;

    private readonly CrossAttentionLayer<T>[] _decoderSelfAttention;
    private readonly CrossAttentionLayer<T>[] _decoderTokenToImage;
    private readonly CrossAttentionLayer<T>[] _decoderImageToToken;
    private readonly LayerNormalizationLayer<T>[] _decoderNorm1;
    private readonly LayerNormalizationLayer<T>[] _decoderNorm2;
    private readonly LayerNormalizationLayer<T>[] _decoderNorm3;
    private readonly LayerNormalizationLayer<T>[] _decoderNorm4;
    private readonly DenseLayer<T>[] _decoderFfn1;
    private readonly DenseLayer<T>[] _decoderFfn2;
    private readonly CrossAttentionLayer<T> _decoderFinalAttention;
    private readonly LayerNormalizationLayer<T> _decoderFinalNorm;
    private readonly ConvolutionalLayer<T> _upscaleStride8;
    private readonly ConvolutionalLayer<T> _upscaleStride4;
    private readonly ConvolutionalLayer<T> _highResStride8Projection;
    private readonly ConvolutionalLayer<T> _highResStride4Projection;
    private readonly DenseLayer<T>[][] _maskHypernetworks;
    private readonly DenseLayer<T>[] _iouHead;
    private readonly DenseLayer<T>[] _objectHead;
    private readonly DenseLayer<T>[] _objectPointerProjection;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _outputTokens;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _pointTypeEmbeddings;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _noMaskEmbedding;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _noMemoryEmbedding;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _noObjectPointer;

    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _memoryLayerScale;

    [Scratch] private Tensor<T>? _lastObjectPointer;

    [Scratch] private Tensor<T>? _lastMasks;
    [Scratch] private Tensor<T>? _lastMaskLogits;
    [Scratch] private Tensor<T>? _lastIouScores;
    [Scratch] private Tensor<T>? _lastIouLogits;
    [Scratch] private Tensor<T>? _lastObjectPresenceScores;
    [Scratch] private Tensor<T>? _lastObjectPresenceLogits;

    /// <summary>Gets the masks produced by the latest decoder forward.</summary>
    internal Tensor<T> LastMasks => _lastMasks
        ?? throw new InvalidOperationException("Decode a mask before reading mask candidates.");

    /// <summary>Gets the quality scores produced by the latest decoder forward.</summary>
    internal Tensor<T> LastIouScores => _lastIouScores
        ?? throw new InvalidOperationException("Decode a mask before reading IoU scores.");

    /// <summary>Gets the presence scores produced by the latest decoder forward.</summary>
    internal Tensor<T> LastObjectPresenceScores => _lastObjectPresenceScores
        ?? throw new InvalidOperationException("Decode a mask before reading object-presence scores.");

    /// <summary>Gets the raw mask logits produced by the latest decoder forward.</summary>
    internal Tensor<T> LastMaskLogits => _lastMaskLogits
        ?? throw new InvalidOperationException("Decode a mask before reading mask logits.");

    /// <summary>Gets the raw IoU predictions produced by the latest decoder forward.</summary>
    internal Tensor<T> LastIouLogits => _lastIouLogits
        ?? throw new InvalidOperationException("Decode a mask before reading IoU logits.");

    /// <summary>Gets the raw object-presence logits produced by the latest decoder forward.</summary>
    internal Tensor<T> LastObjectPresenceLogits => _lastObjectPresenceLogits
        ?? throw new InvalidOperationException("Decode a mask before reading object-presence logits.");

    /// <summary>Gets the IoU prediction-head layers for fidelity verification.</summary>
    internal IReadOnlyList<DenseLayer<T>> IouHeadLayers => _iouHead;

    /// <summary>Gets the object-presence prediction-head layers for fidelity verification.</summary>
    internal IReadOnlyList<DenseLayer<T>> ObjectPresenceHeadLayers => _objectHead;

    /// <summary>Gets the paper's four mask candidates.</summary>
    internal int CandidateCount => MaskCandidateCount;

    /// <summary>Gets the memory transformer depth.</summary>
    internal int MemoryAttentionLayerCount => _memoryAttentionLayers;

    /// <summary>Gets the two-way mask-decoder depth.</summary>
    internal int DecoderDepth => _decoderDepth;

    /// <summary>Gets the spatial-memory width.</summary>
    internal int MemoryDimension => _memoryDimension;

    /// <summary>Gets the mask-decoder embedding width.</summary>
    internal int ModelDimension => _modelDimension;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank == 3)
        {
            return
            [
                new(TensorAxis.Channels, AxisRelation.Fixed(1)),
                new(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 1, 4)),
                new(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 1, 4))
            ];
        }
        if (inputRank == 4)
        {
            return
            [
                new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new(TensorAxis.Channels, AxisRelation.Fixed(1)),
                new(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 1, 4)),
                new(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 1, 4))
            ];
        }
        return null;
    }

    /// <summary>Gets the object pointer produced by the latest decoder forward.</summary>
    internal Tensor<T> LastObjectPointer => _lastObjectPointer
        ?? throw new InvalidOperationException("Decode a mask before reading its object pointer.");

    /// <summary>Gets the Hiera encoder used by this pipeline.</summary>
    internal SAM2HieraImageEncoderLayer<T> ImageEncoder => _imageEncoder;

    /// <summary>Creates the paper topology, with explicit capacity controls.</summary>
    public SAM2NativePipelineLayer(
        [LayerState] int inputChannels,
        [LayerState] int inputHeight,
        [LayerState] int inputWidth,
        [LayerState] int hieraEmbeddingDimension,
        [LayerState] int[] hieraStageDepths,
        [LayerState] int hieraInitialHeads,
        [LayerState] int[] hieraWindowSizes,
        [LayerState] int[] hieraGlobalAttentionBlockIndexes,
        [LayerState] int modelDimension = 256,
        [LayerState] int memoryDimension = 64,
        [LayerState] int decoderHeads = 8,
        [LayerState] int memoryAttentionLayers = 4,
        [LayerState] int decoderDepth = 2,
        [LayerState] int decoderMlpDimension = 2048,
        [LayerState] double ropeTheta = 10000.0,
        [LayerState] double memoryMaskScale = 20.0,
        [LayerState] double memoryMaskBias = -10.0)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [1, Math.Max(1, inputHeight / 4), Math.Max(1, inputWidth / 4)])
    {
        if (modelDimension <= 0) throw new ArgumentOutOfRangeException(nameof(modelDimension));
        if (memoryDimension <= 0 || modelDimension % memoryDimension != 0)
            throw new ArgumentOutOfRangeException(nameof(memoryDimension),
                "Memory dimension must be positive and divide the model dimension for object-pointer tokens.");
        if (decoderHeads <= 0 || modelDimension % decoderHeads != 0)
            throw new ArgumentOutOfRangeException(nameof(decoderHeads));
        if (memoryAttentionLayers <= 0) throw new ArgumentOutOfRangeException(nameof(memoryAttentionLayers));
        if (decoderDepth <= 0) throw new ArgumentOutOfRangeException(nameof(decoderDepth));
        if (decoderMlpDimension <= 0) throw new ArgumentOutOfRangeException(nameof(decoderMlpDimension));
        if (double.IsNaN(memoryMaskScale) || double.IsInfinity(memoryMaskScale) || memoryMaskScale <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(memoryMaskScale));
        if (double.IsNaN(memoryMaskBias) || double.IsInfinity(memoryMaskBias)) throw new ArgumentOutOfRangeException(nameof(memoryMaskBias));

        _inputChannels = inputChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _hieraEmbeddingDimension = hieraEmbeddingDimension;
        _hieraStageDepths = hieraStageDepths.ToArray();
        _hieraInitialHeads = hieraInitialHeads;
        _hieraWindowSizes = hieraWindowSizes.ToArray();
        _hieraGlobalAttentionBlockIndexes = hieraGlobalAttentionBlockIndexes.ToArray();
        _modelDimension = modelDimension;
        _memoryDimension = memoryDimension;
        _decoderHeads = decoderHeads;
        _memoryAttentionLayers = memoryAttentionLayers;
        _decoderDepth = decoderDepth;
        _decoderMlpDimension = decoderMlpDimension;
        _ropeTheta = ropeTheta;
        _memoryMaskScale = memoryMaskScale;
        _memoryMaskBias = memoryMaskBias;
        _featureHeight = Math.Max(1, inputHeight / 16);
        _featureWidth = Math.Max(1, inputWidth / 16);

        var identity = new IdentityActivation<T>();
        var gelu = new GELUActivation<T>();
        var relu = new ReLUActivation<T>();

        _imageEncoder = new SAM2HieraImageEncoderLayer<T>(
            inputChannels, inputHeight, inputWidth, hieraEmbeddingDimension,
            hieraStageDepths, hieraInitialHeads, hieraWindowSizes,
            hieraGlobalAttentionBlockIndexes, modelDimension);
        RegisterSubLayer(_imageEncoder);

        _outputTokens = new Tensor<T>([OutputTokenCount, modelDimension]);
        _pointTypeEmbeddings = new Tensor<T>([4, modelDimension]);
        _noMaskEmbedding = new Tensor<T>([1, modelDimension, 1, 1]);
        _noMemoryEmbedding = new Tensor<T>([1, modelDimension, 1, 1]);
        _noObjectPointer = new Tensor<T>([1, 1, modelDimension]);
        _memoryLayerScale = Tensor<T>.CreateDefault([1, modelDimension, 1, 1], NumOps.FromDouble(1e-6));
        InitializeLayerWeights(_outputTokens, OutputTokenCount, modelDimension);
        InitializeLayerWeights(_pointTypeEmbeddings, 4, modelDimension);
        InitializeLayerWeights(_noMaskEmbedding, modelDimension, modelDimension);
        InitializeLayerWeights(_noMemoryEmbedding, modelDimension, modelDimension);
        InitializeLayerWeights(_noObjectPointer, 1, modelDimension);
        RegisterTrainableParameter(_outputTokens, PersistentTensorRole.Embeddings);
        RegisterTrainableParameter(_pointTypeEmbeddings, PersistentTensorRole.Embeddings);
        RegisterTrainableParameter(_noMaskEmbedding, PersistentTensorRole.Embeddings);
        RegisterTrainableParameter(_noMemoryEmbedding, PersistentTensorRole.Embeddings);
        RegisterTrainableParameter(_noObjectPointer, PersistentTensorRole.Embeddings);
        RegisterTrainableParameter(_memoryLayerScale, PersistentTensorRole.NormalizationParams);

        _promptMaskDownsampler =
        [
            CreateConv(1, Math.Max(4, modelDimension / 4), 2, 2, 0,
                Math.Max(1, inputHeight / 4), Math.Max(1, inputWidth / 4), gelu),
            CreateConv(Math.Max(4, modelDimension / 4), modelDimension, 2, 2, 0,
                Math.Max(1, inputHeight / 8), Math.Max(1, inputWidth / 8), identity)
        ];
        RegisterAll(_promptMaskDownsampler);

        _memoryMaskDownsampler = new ConvolutionalLayer<T>[4];
        int maskChannels = 1;
        int maskHeight = inputHeight;
        int maskWidth = inputWidth;
        for (int i = 0; i < 4; i++)
        {
            int nextChannels = Math.Min(modelDimension, 1 << (2 * (i + 1)));
            _memoryMaskDownsampler[i] = CreateConv(
                maskChannels, nextChannels, 3, 2, 1, maskHeight, maskWidth, gelu);
            RegisterSubLayer(_memoryMaskDownsampler[i]);
            maskChannels = nextChannels;
            maskHeight = ConvOutput(maskHeight, 3, 2, 1);
            maskWidth = ConvOutput(maskWidth, 3, 2, 1);
        }
        if (maskChannels != modelDimension)
        {
            throw new ArgumentException(
                "Model dimension must be at most 256 and reachable by SAM 2's four mask-downsampling stages.",
                nameof(modelDimension));
        }

        _memoryPixelProjection = CreateConv(
            modelDimension, modelDimension, 1, 1, 0, _featureHeight, _featureWidth, identity);
        RegisterSubLayer(_memoryPixelProjection);
        _memoryDepthwise = new ConvolutionalLayer<T>[2];
        _memoryExpand = new ConvolutionalLayer<T>[2];
        _memoryContract = new ConvolutionalLayer<T>[2];
        for (int i = 0; i < 2; i++)
        {
            _memoryDepthwise[i] = CreateConv(
                modelDimension, modelDimension, 7, 1, 3, _featureHeight, _featureWidth,
                identity, groups: modelDimension);
            _memoryExpand[i] = CreateConv(
                modelDimension, modelDimension * 4, 1, 1, 0,
                _featureHeight, _featureWidth, gelu);
            _memoryContract[i] = CreateConv(
                modelDimension * 4, modelDimension, 1, 1, 0,
                _featureHeight, _featureWidth, identity);
            RegisterSubLayer(_memoryDepthwise[i]);
            RegisterSubLayer(_memoryExpand[i]);
            RegisterSubLayer(_memoryContract[i]);
        }
        _memoryOutputProjection = CreateConv(
            modelDimension, memoryDimension, 1, 1, 0, _featureHeight, _featureWidth, identity);
        RegisterSubLayer(_memoryOutputProjection);

        _memorySelfAttention = new SAM2RoPEAttentionLayer<T>[memoryAttentionLayers];
        _memoryCrossAttention = new SAM2RoPEAttentionLayer<T>[memoryAttentionLayers];
        _memoryNorm1 = new LayerNormalizationLayer<T>[memoryAttentionLayers];
        _memoryNorm2 = new LayerNormalizationLayer<T>[memoryAttentionLayers];
        _memoryNorm3 = new LayerNormalizationLayer<T>[memoryAttentionLayers];
        _memoryFfn1 = new DenseLayer<T>[memoryAttentionLayers];
        _memoryFfn2 = new DenseLayer<T>[memoryAttentionLayers];
        for (int i = 0; i < memoryAttentionLayers; i++)
        {
            _memorySelfAttention[i] = new SAM2RoPEAttentionLayer<T>(
                modelDimension, modelDimension, 1, _featureHeight, _featureWidth, ropeTheta);
            _memoryCrossAttention[i] = new SAM2RoPEAttentionLayer<T>(
                modelDimension, memoryDimension, 1, _featureHeight, _featureWidth, ropeTheta);
            _memoryNorm1[i] = new LayerNormalizationLayer<T>(modelDimension);
            _memoryNorm2[i] = new LayerNormalizationLayer<T>(modelDimension);
            _memoryNorm3[i] = new LayerNormalizationLayer<T>(modelDimension);
            _memoryFfn1[i] = CreateDense(modelDimension, modelDimension * 8, relu);
            _memoryFfn2[i] = CreateDense(modelDimension * 8, modelDimension, identity);
            RegisterSubLayer(_memorySelfAttention[i]);
            RegisterSubLayer(_memoryCrossAttention[i]);
            RegisterSubLayer(_memoryNorm1[i]);
            RegisterSubLayer(_memoryNorm2[i]);
            RegisterSubLayer(_memoryNorm3[i]);
            RegisterSubLayer(_memoryFfn1[i]);
            RegisterSubLayer(_memoryFfn2[i]);
        }
        _memoryFinalNorm = new LayerNormalizationLayer<T>(modelDimension);
        RegisterSubLayer(_memoryFinalNorm);

        _decoderSelfAttention = new CrossAttentionLayer<T>[decoderDepth];
        _decoderTokenToImage = new CrossAttentionLayer<T>[decoderDepth];
        _decoderImageToToken = new CrossAttentionLayer<T>[decoderDepth];
        _decoderNorm1 = CreateNorms(decoderDepth, modelDimension);
        _decoderNorm2 = CreateNorms(decoderDepth, modelDimension);
        _decoderNorm3 = CreateNorms(decoderDepth, modelDimension);
        _decoderNorm4 = CreateNorms(decoderDepth, modelDimension);
        _decoderFfn1 = new DenseLayer<T>[decoderDepth];
        _decoderFfn2 = new DenseLayer<T>[decoderDepth];
        for (int i = 0; i < decoderDepth; i++)
        {
            int maxSequence = Math.Max(_featureHeight * _featureWidth, 16);
            _decoderSelfAttention[i] = new CrossAttentionLayer<T>(
                modelDimension, modelDimension, decoderHeads, maxSequence);
            _decoderTokenToImage[i] = new CrossAttentionLayer<T>(
                modelDimension, modelDimension, decoderHeads, maxSequence);
            _decoderImageToToken[i] = new CrossAttentionLayer<T>(
                modelDimension, modelDimension, decoderHeads, maxSequence);
            _decoderFfn1[i] = CreateDense(modelDimension, decoderMlpDimension, relu);
            _decoderFfn2[i] = CreateDense(decoderMlpDimension, modelDimension, identity);
            RegisterSubLayer(_decoderSelfAttention[i]);
            RegisterSubLayer(_decoderTokenToImage[i]);
            RegisterSubLayer(_decoderImageToToken[i]);
            RegisterSubLayer(_decoderFfn1[i]);
            RegisterSubLayer(_decoderFfn2[i]);
        }
        RegisterAll(_decoderNorm1);
        RegisterAll(_decoderNorm2);
        RegisterAll(_decoderNorm3);
        RegisterAll(_decoderNorm4);
        _decoderFinalAttention = new CrossAttentionLayer<T>(
            modelDimension, modelDimension, decoderHeads,
            Math.Max(_featureHeight * _featureWidth, 16));
        _decoderFinalNorm = new LayerNormalizationLayer<T>(modelDimension);
        RegisterSubLayer(_decoderFinalAttention);
        RegisterSubLayer(_decoderFinalNorm);

        int stride8Dimension = Math.Max(4, modelDimension / 4);
        _maskEmbeddingDimension = Math.Max(2, modelDimension / 8);
        _upscaleStride8 = CreateConv(
            modelDimension, stride8Dimension, 3, 1, 1,
            Math.Max(1, inputHeight / 8), Math.Max(1, inputWidth / 8), gelu);
        _highResStride8Projection = CreateConv(
            modelDimension, stride8Dimension, 1, 1, 0,
            Math.Max(1, inputHeight / 8), Math.Max(1, inputWidth / 8), identity);
        _upscaleStride4 = CreateConv(
            stride8Dimension, _maskEmbeddingDimension, 3, 1, 1,
            Math.Max(1, inputHeight / 4), Math.Max(1, inputWidth / 4), gelu);
        _highResStride4Projection = CreateConv(
            modelDimension, _maskEmbeddingDimension, 1, 1, 0,
            Math.Max(1, inputHeight / 4), Math.Max(1, inputWidth / 4), identity);
        RegisterSubLayer(_upscaleStride8);
        RegisterSubLayer(_highResStride8Projection);
        RegisterSubLayer(_upscaleStride4);
        RegisterSubLayer(_highResStride4Projection);

        _maskHypernetworks = new DenseLayer<T>[MaskCandidateCount][];
        for (int mask = 0; mask < MaskCandidateCount; mask++)
        {
            _maskHypernetworks[mask] =
            [
                CreateDense(modelDimension, modelDimension, relu),
                CreateDense(modelDimension, modelDimension, relu),
                CreateDense(modelDimension, _maskEmbeddingDimension, identity)
            ];
            RegisterAll(_maskHypernetworks[mask]);
        }
        _iouHead = CreateMlp(modelDimension, modelDimension, MaskCandidateCount, identity);
        _objectHead = CreateMlp(modelDimension, modelDimension, 1, identity);
        _objectPointerProjection = CreateMlp(modelDimension, modelDimension, modelDimension, identity);
        RegisterAll(_iouHead);
        RegisterAll(_objectHead);
        RegisterAll(_objectPointerProjection);
        ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Encodes an image and preserves Hiera high-resolution skips for decoding.</summary>
    internal Tensor<T> EncodeImage(Tensor<T> image) => _imageEncoder.Forward(image);

    /// <summary>Adds the learned no-memory token used on the first video frame.</summary>
    internal Tensor<T> AddNoMemoryEmbedding(Tensor<T> current)
    {
        var noMemory = Engine.TensorTile(
            _noMemoryEmbedding,
            [current.Shape[0], 1, current.Shape[2], current.Shape[3]]);
        return Engine.TensorAdd(current, noMemory);
    }

    /// <summary>Encodes foreground/background point prompts with random-Fourier-style sine/cosine positions.</summary>
    internal Tensor<T> EncodePoints(float[,] points, int[] labels)
    {
        if (points.GetLength(0) != labels.Length)
            throw new ArgumentException("Every point must have one label.", nameof(labels));
        var positional = BuildCoordinateEncoding(points, cornerOffset: 0);
        var labelIndices = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++) labelIndices[i] = labels[i] > 0 ? 1 : 0;
        var types = Engine.TensorGather(
            _pointTypeEmbeddings, new Tensor<int>(labelIndices, [labelIndices.Length]), axis: 0);
        types = Engine.Reshape(types, [1, labels.Length, _modelDimension]);
        return Engine.TensorAdd(positional, types);
    }

    /// <summary>Encodes a box as the two distinct corner tokens used by SAM's prompt encoder.</summary>
    internal Tensor<T> EncodeBox(float[] box)
    {
        if (box.Length != 4) throw new ArgumentException("A box is [x1,y1,x2,y2].", nameof(box));
        var corners = new float[,] { { box[0], box[1] }, { box[2], box[3] } };
        var positional = BuildCoordinateEncoding(corners, cornerOffset: 0);
        var types = Engine.TensorGather(
            _pointTypeEmbeddings, new Tensor<int>([2, 3], [2]), axis: 0);
        return Engine.TensorAdd(positional, Engine.Reshape(types, [1, 2, _modelDimension]));
    }

    /// <summary>Encodes a dense mask prompt at the main stride-16 image-embedding resolution.</summary>
    internal Tensor<T> EncodeMaskPrompt(Tensor<T> mask)
    {
        if (mask.Rank == 2) mask = Engine.Reshape(mask, [1, 1, mask.Shape[0], mask.Shape[1]]);
        else if (mask.Rank == 3) mask = Engine.Reshape(mask, [mask.Shape[0], 1, mask.Shape[1], mask.Shape[2]]);
        if (mask.Rank != 4 || mask.Shape[1] != 1)
            throw new ArgumentException("Mask prompt must be [H,W], [B,H,W], or [B,1,H,W].", nameof(mask));
        if (mask.Shape[2] != Math.Max(1, _inputHeight / 4)
            || mask.Shape[3] != Math.Max(1, _inputWidth / 4))
        {
            mask = Engine.Interpolate(
                mask, [Math.Max(1, _inputHeight / 4), Math.Max(1, _inputWidth / 4)],
                InterpolateMode.Bilinear, alignCorners: false);
        }
        foreach (var layer in _promptMaskDownsampler) mask = layer.Forward(mask);
        return mask;
    }

    /// <summary>Applies the four-layer SAM 2 memory transformer to the current image embedding.</summary>
    internal Tensor<T> ApplyMemoryAttention(
        Tensor<T> current,
        IReadOnlyList<Tensor<T>> memories,
        IReadOnlyList<Tensor<T>> objectPointers)
    {
        if (memories.Count == 0)
            return AddNoMemoryEmbedding(current);

        var output = MapToTokens(current, _modelDimension);
        output = Engine.TensorAdd(
            output,
            Engine.TensorMultiplyScalar(BuildSpatialEncoding(
                output.Shape[0], output.Shape[1], _modelDimension), NumOps.FromDouble(0.1)));

        var contexts = memories.Select(memory => MapToTokens(memory, _memoryDimension)).ToList();
        int pointerTokenCount = 0;
        foreach (var pointer in objectPointers)
        {
            int batch = pointer.Shape[0];
            int tokens = _modelDimension / _memoryDimension;
            contexts.Add(Engine.Reshape(pointer, [batch, tokens, _memoryDimension]));
            pointerTokenCount += tokens;
        }
        var context = contexts.Count == 1
            ? contexts[0]
            : Engine.TensorConcatenate(contexts.ToArray(), axis: 1);

        for (int i = 0; i < _memoryAttentionLayers; i++)
        {
            var normalized = _memoryNorm1[i].Forward(output);
            output = Engine.TensorAdd(output, _memorySelfAttention[i].Forward(normalized));
            normalized = _memoryNorm2[i].Forward(output);
            var cross = _memoryCrossAttention[i].ForwardWithUnrotatedContextTail(
                normalized, context, pointerTokenCount);
            output = Engine.TensorAdd(output, cross);
            normalized = _memoryNorm3[i].Forward(output);
            var ffn = _memoryFfn2[i].Forward(_memoryFfn1[i].Forward(normalized));
            output = Engine.TensorAdd(output, ffn);
        }
        output = _memoryFinalNorm.Forward(output);
        return TokensToMap(output, _modelDimension, _featureHeight, _featureWidth);
    }

    /// <summary>Runs SAM's two-way decoder and all mask, quality, presence, and pointer heads.</summary>
    internal SAM2PipelineDecodeResult<T> Decode(
        Tensor<T> imageFeatures,
        Tensor<T>? sparsePrompt,
        Tensor<T>? densePrompt)
    {
        int batch = imageFeatures.Shape[0];
        if (densePrompt is not null)
        {
            if (densePrompt.Shape[0] == 1 && batch > 1)
                densePrompt = Engine.TensorTile(densePrompt, [batch, 1, 1, 1]);
            imageFeatures = Engine.TensorAdd(imageFeatures, densePrompt);
        }
        else
        {
            var noMask = Engine.TensorTile(
                _noMaskEmbedding,
                [batch, 1, imageFeatures.Shape[2], imageFeatures.Shape[3]]);
            imageFeatures = Engine.TensorAdd(imageFeatures, noMask);
        }

        var outputTokens = Engine.TensorTile(
            Engine.Reshape(_outputTokens, [1, OutputTokenCount, _modelDimension]),
            [batch, 1, 1]);
        var queries = outputTokens;
        if (sparsePrompt is not null)
        {
            if (sparsePrompt.Shape[0] == 1 && batch > 1)
                sparsePrompt = Engine.TensorTile(sparsePrompt, [batch, 1, 1]);
            queries = Engine.TensorConcatenate([outputTokens, sparsePrompt], axis: 1);
        }
        var imageTokens = MapToTokens(imageFeatures, _modelDimension);

        for (int i = 0; i < _decoderDepth; i++)
        {
            var self = _decoderSelfAttention[i].Forward(queries, queries);
            queries = _decoderNorm1[i].Forward(i == 0 ? self : Engine.TensorAdd(queries, self));
            var toImage = _decoderTokenToImage[i].Forward(queries, imageTokens);
            queries = _decoderNorm2[i].Forward(Engine.TensorAdd(queries, toImage));
            var ffn = _decoderFfn2[i].Forward(_decoderFfn1[i].Forward(queries));
            queries = _decoderNorm3[i].Forward(Engine.TensorAdd(queries, ffn));
            var toTokens = _decoderImageToToken[i].Forward(imageTokens, queries);
            imageTokens = _decoderNorm4[i].Forward(Engine.TensorAdd(imageTokens, toTokens));
        }
        queries = _decoderFinalNorm.Forward(
            Engine.TensorAdd(queries, _decoderFinalAttention.Forward(queries, imageTokens)));

        var objectToken = Engine.TensorNarrow(queries, 1, 0, 1);
        var iouToken = Engine.TensorNarrow(queries, 1, 1, 1);
        var maskTokens = Engine.TensorNarrow(queries, 1, 2, MaskCandidateCount);
        var updatedImage = TokensToMap(imageTokens, _modelDimension, _featureHeight, _featureWidth);
        updatedImage = Engine.Interpolate(
            updatedImage.Contiguous(), [Math.Max(1, _inputHeight / 8), Math.Max(1, _inputWidth / 8)],
            InterpolateMode.Bilinear, alignCorners: false);
        updatedImage = Engine.TensorAdd(
            _upscaleStride8.Forward(updatedImage),
            _highResStride8Projection.Forward(_imageEncoder.LastStride8));
        updatedImage = Engine.Interpolate(
            updatedImage, [Math.Max(1, _inputHeight / 4), Math.Max(1, _inputWidth / 4)],
            InterpolateMode.Bilinear, alignCorners: false);
        updatedImage = Engine.TensorAdd(
            _upscaleStride4.Forward(updatedImage),
            _highResStride4Projection.Forward(_imageEncoder.LastStride4));

        var hyperOutputs = new Tensor<T>[MaskCandidateCount];
        for (int mask = 0; mask < MaskCandidateCount; mask++)
        {
            var token = Engine.TensorNarrow(maskTokens, 1, mask, 1);
            foreach (var layer in _maskHypernetworks[mask]) token = layer.Forward(token);
            hyperOutputs[mask] = token;
        }
        var hyper = Engine.TensorConcatenate(hyperOutputs, axis: 1);
        var spatial = Engine.Reshape(
            updatedImage,
            [batch, _maskEmbeddingDimension, updatedImage.Shape[2] * updatedImage.Shape[3]]);
        var maskLogits = Engine.TensorBatchMatMul(hyper, spatial);
        maskLogits = Engine.Reshape(
            maskLogits, [batch, MaskCandidateCount, updatedImage.Shape[2], updatedImage.Shape[3]]);
        var masks = Engine.Sigmoid(maskLogits);

        var iouLogits = Engine.Reshape(
            RunMlp(iouToken, _iouHead), [batch, MaskCandidateCount]);
        var iouScores = Engine.Sigmoid(iouLogits);
        var objectPresenceLogits = Engine.Reshape(
            RunMlp(objectToken, _objectHead), [batch, 1]);
        var objectPresence = Engine.Sigmoid(objectPresenceLogits);
        // The default object pointer comes from the single-mask token (token 0).
        // Multimask IoU ranking must not change tracking identity.
        var bestToken = Engine.TensorGather(
            maskTokens, new Tensor<int>([0], [1]), axis: 1);
        _lastMasks = masks;
        _lastMaskLogits = maskLogits;
        _lastIouScores = iouScores;
        _lastIouLogits = iouLogits;
        _lastObjectPresenceScores = objectPresence;
        _lastObjectPresenceLogits = objectPresenceLogits;
        var rawObjectPointer = RunMlp(bestToken, _objectPointerProjection);
        var pointerPresence = Engine.Reshape(objectPresence, [batch, 1, 1]);
        var noObjectPointer = Engine.TensorTile(_noObjectPointer, [batch, 1, 1]);
        var pointerAbsence = Engine.TensorNegate(
            Engine.TensorSubtractScalar(pointerPresence, NumOps.One));
        _lastObjectPointer = Engine.TensorAdd(
            Engine.TensorMultiply(rawObjectPointer, pointerPresence),
            Engine.TensorMultiply(noObjectPointer, pointerAbsence));
        return new SAM2PipelineDecodeResult<T>(masks, iouScores, objectPresence, _lastObjectPointer);
    }

    /// <summary>Encodes a predicted mask and its pixels into the 64-channel SAM 2 memory representation.</summary>
    internal Tensor<T> EncodeMemory(Tensor<T> imageFeatures, Tensor<T> selectedMask)
    {
        var fullMask = Engine.Interpolate(
            selectedMask, [_inputHeight, _inputWidth], InterpolateMode.Bilinear, alignCorners: false);
        fullMask = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(fullMask, NumOps.FromDouble(_memoryMaskScale)),
            NumOps.FromDouble(_memoryMaskBias));
        foreach (var layer in _memoryMaskDownsampler) fullMask = layer.Forward(fullMask);

        var fused = Engine.TensorAdd(_memoryPixelProjection.Forward(imageFeatures), fullMask);
        for (int i = 0; i < 2; i++)
        {
            var branch = _memoryDepthwise[i].Forward(fused);
            branch = _memoryContract[i].Forward(_memoryExpand[i].Forward(branch));
            branch = Engine.TensorMultiply(branch, _memoryLayerScale);
            fused = Engine.TensorAdd(fused, branch);
        }
        return _memoryOutputProjection.Forward(fused);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        if (unbatched) input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        var image = AddNoMemoryEmbedding(EncodeImage(input));
        var decoded = Decode(image, null, null);
        // No multimask_output request exists on Layer.Forward, so match Meta's
        // single-mask path and return token 0 rather than IoU-ranking all four tokens.
        var selected = Engine.TensorGather(
            decoded.Masks, new Tensor<int>([0], [1]), axis: 1);
        return unbatched
            ? Engine.Reshape(selected, [1, selected.Shape[2], selected.Shape[3]])
            : selected;
    }

    private Tensor<T> BuildCoordinateEncoding(float[,] points, int cornerOffset)
    {
        int count = points.GetLength(0);
        var result = new Tensor<T>([1, count, _modelDimension]);
        int pairs = Math.Max(1, _modelDimension / 4);
        for (int point = 0; point < count; point++)
        {
            double x = (points[point, 0] + 0.5 + cornerOffset) / _inputWidth;
            double y = (points[point, 1] + 0.5 + cornerOffset) / _inputHeight;
            for (int d = 0; d < _modelDimension; d++)
            {
                int pair = d / 2;
                bool xAxis = pair < (_modelDimension / 4);
                int axisPair = xAxis ? pair : pair - (_modelDimension / 4);
                double frequency = Math.Pow(10000.0, -((double)axisPair / pairs));
                double angle = 2.0 * Math.PI * (xAxis ? x : y) * frequency;
                result[0, point, d] = NumOps.FromDouble((d & 1) == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }
        return result;
    }

    private Tensor<T> BuildSpatialEncoding(int batch, int length, int dimension)
    {
        var result = new Tensor<T>([batch, length, dimension]);
        int pairs = Math.Max(1, dimension / 4);
        for (int b = 0; b < batch; b++)
        for (int token = 0; token < length; token++)
        for (int d = 0; d < dimension; d++)
        {
            int pair = d / 2;
            bool xAxis = pair < dimension / 4;
            int axisPair = xAxis ? pair : pair - dimension / 4;
            double frequency = Math.Pow(10000.0, -((double)axisPair / pairs));
            int position = xAxis ? token % _featureWidth : token / _featureWidth;
            double angle = position * frequency;
            result[b, token, d] = NumOps.FromDouble((d & 1) == 0 ? Math.Sin(angle) : Math.Cos(angle));
        }
        return result;
    }

    private Tensor<T> SelectBestMask(Tensor<T> masks, Tensor<T> scores)
    {
        var selected = new Tensor<T>[masks.Shape[0]];
        for (int batch = 0; batch < masks.Shape[0]; batch++)
        {
            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int mask = 0; mask < MaskCandidateCount; mask++)
            {
                double score = Convert.ToDouble(scores[batch, mask]);
                if (score > bestScore) { bestScore = score; best = mask; }
            }
            var oneBatch = Engine.TensorNarrow(masks, 0, batch, 1);
            selected[batch] = Engine.TensorNarrow(oneBatch, 1, best, 1);
        }
        return selected.Length == 1 ? selected[0] : Engine.TensorConcatenate(selected, axis: 0);
    }

    private Tensor<T> SelectBestToken(Tensor<T> tokens, Tensor<T> scores)
    {
        var selected = new Tensor<T>[tokens.Shape[0]];
        for (int batch = 0; batch < tokens.Shape[0]; batch++)
        {
            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int mask = 0; mask < MaskCandidateCount; mask++)
            {
                double score = Convert.ToDouble(scores[batch, mask]);
                if (score > bestScore) { bestScore = score; best = mask; }
            }
            var oneBatch = Engine.TensorNarrow(tokens, 0, batch, 1);
            selected[batch] = Engine.TensorNarrow(oneBatch, 1, best, 1);
        }
        return selected.Length == 1 ? selected[0] : Engine.TensorConcatenate(selected, axis: 0);
    }

    private Tensor<T> MapToTokens(Tensor<T> map, int channels)
    {
        var nhwc = Engine.TensorPermute(map, [0, 2, 3, 1]);
        return Engine.Reshape(nhwc, [map.Shape[0], map.Shape[2] * map.Shape[3], channels]);
    }

    private Tensor<T> TokensToMap(Tensor<T> tokens, int channels, int height, int width)
    {
        var nhwc = Engine.Reshape(tokens, [tokens.Shape[0], height, width, channels]);
        return Engine.TensorPermute(nhwc, [0, 3, 1, 2]);
    }

    private Tensor<T> RunMlp(Tensor<T> input, IEnumerable<DenseLayer<T>> layers)
    {
        foreach (var layer in layers) input = layer.Forward(input);
        return input;
    }

    private DenseLayer<T>[] CreateMlp(
        int inputDimension, int hiddenDimension, int outputDimension, IActivationFunction<T> finalActivation)
        =>
        [
            CreateDense(inputDimension, hiddenDimension, new ReLUActivation<T>()),
            CreateDense(hiddenDimension, hiddenDimension, new ReLUActivation<T>()),
            CreateDense(hiddenDimension, outputDimension, finalActivation)
        ];

    private DenseLayer<T> CreateDense(
        int inputDimension, int outputDimension, IActivationFunction<T> activation)
    {
        var layer = new DenseLayer<T>(outputDimension, activation);
        _ = layer.Forward(new Tensor<T>([1, 1, inputDimension]));
        layer.ResetState();
        return layer;
    }

    private LayerNormalizationLayer<T>[] CreateNorms(int count, int dimension)
    {
        var result = new LayerNormalizationLayer<T>[count];
        for (int i = 0; i < count; i++) result[i] = new LayerNormalizationLayer<T>(dimension);
        return result;
    }

    private void RegisterAll(IEnumerable<ILayer<T>> layers)
    {
        foreach (var layer in layers) RegisterSubLayer(layer);
    }

    private static int ConvOutput(int size, int kernel, int stride, int padding)
        => Math.Max(1, ((size + 2 * padding - kernel) / stride) + 1);

    private static ConvolutionalLayer<T> CreateConv(
        int inputChannels, int outputChannels, int kernel, int stride, int padding,
        int inputHeight, int inputWidth, IActivationFunction<T> activation, int groups = 1)
    {
        var layer = new ConvolutionalLayer<T>(
            outputChannels, kernel, stride, padding, activation,
            groups: groups, biasMode: BiasMode.Always);
        layer.ResolveFromShape([inputChannels, inputHeight, inputWidth]);
        return layer;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastObjectPointer = null;
        _lastMasks = null;
        _lastMaskLogits = null;
        _lastIouScores = null;
        _lastIouLogits = null;
        _lastObjectPresenceScores = null;
        _lastObjectPresenceLogits = null;
        foreach (var child in GetSubLayers()) child?.ResetState();
    }
}

/// <summary>Outputs of SAM 2's parallel prediction heads.</summary>
internal readonly record struct SAM2PipelineDecodeResult<T>(
    Tensor<T> Masks,
    Tensor<T> IouScores,
    Tensor<T> ObjectPresenceScores,
    Tensor<T> ObjectPointer);
