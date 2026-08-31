using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Hiera image backbone and FPN neck used by native SAM 2.
/// </summary>
/// <remarks>
/// The four stages use windowed transformer blocks, double width/head count at each transition,
/// and expose stride-4/8 features while the FPN fuses stride-16/32 features. These are the same
/// structural contracts used by Meta's SAM 2 Hiera image encoder; widths and depths are constructor
/// state so bounded tests can scale capacity without replacing the architecture.
/// </remarks>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3,
    TestInputShape = "3, 32, 32",
    TestConstructorArgs = "3, 32, 32, 16, new int[] { 1, 1, 1, 1 }, 1, new int[] { 8, 4, 2, 1 }, new int[] { 2 }, 16, 0.0")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SAM2HieraImageEncoderLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputChannels;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _embeddingDimension;
    private readonly int[] _stageDepths;
    private readonly int _initialHeadCount;
    private readonly int[] _windowSizes;
    private readonly int[] _globalAttentionBlockIndexes;
    private readonly int _outputDimension;
    private readonly double _dropoutRate;
    private readonly int[] _stageHeights = new int[4];
    private readonly int[] _stageWidths = new int[4];
    private readonly int[] _stageDimensions = new int[4];

    private readonly ConvolutionalLayer<T> _patchEmbedding;
    private readonly ConvolutionalLayer<T>[] _stageDownsamples;
    private readonly TransformerEncoderBlock<T>[][] _stageBlocks;
    private readonly ConvolutionalLayer<T>[] _fpnProjections;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private Tensor<T> _positionEmbedding;

    [Scratch]
    private Tensor<T>? _lastStride4;

    [Scratch]
    private Tensor<T>? _lastStride8;

    /// <summary>Gets the FPN-projected stride-4 feature map from the latest forward pass.</summary>
    internal Tensor<T> LastStride4 => _lastStride4
        ?? throw new InvalidOperationException("Run the image encoder before reading high-resolution features.");

    /// <summary>Gets the FPN-projected stride-8 feature map from the latest forward pass.</summary>
    internal Tensor<T> LastStride8 => _lastStride8
        ?? throw new InvalidOperationException("Run the image encoder before reading high-resolution features.");

    /// <summary>Gets a copy of the configured Hiera stage depths.</summary>
    public int[] StageDepths => _stageDepths.ToArray();

    /// <summary>Gets the FPN output width.</summary>
    public int OutputDimension => _outputDimension;

    /// <summary>Gets the zero-based blocks configured for global attention.</summary>
    public int[] GlobalAttentionBlockIndexes => _globalAttentionBlockIndexes.ToArray();

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank == 3)
        {
            return
            [
                new(TensorAxis.Channels, AxisRelation.Fixed(_outputDimension)),
                new(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 1, 16)),
                new(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 1, 16))
            ];
        }
        if (inputRank == 4)
        {
            return
            [
                new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new(TensorAxis.Channels, AxisRelation.Fixed(_outputDimension)),
                new(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 1, 16)),
                new(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 1, 16))
            ];
        }
        return null;
    }

    /// <summary>Creates a Hiera encoder and FPN neck.</summary>
    public SAM2HieraImageEncoderLayer(
        [LayerState] int inputChannels,
        [LayerState] int inputHeight,
        [LayerState] int inputWidth,
        [LayerState] int embeddingDimension,
        [LayerState] int[] stageDepths,
        [LayerState] int initialHeadCount,
        [LayerState] int[] windowSizes,
        [LayerState] int[] globalAttentionBlockIndexes,
        [LayerState] int outputDimension = 256,
        [LayerState] double dropoutRate = 0.0)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [outputDimension, Math.Max(1, inputHeight / 16), Math.Max(1, inputWidth / 16)])
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (inputHeight < 16 || inputWidth < 16 || inputHeight % 16 != 0 || inputWidth % 16 != 0)
            throw new ArgumentOutOfRangeException(
                nameof(inputHeight), "SAM 2 requires dimensions of at least 16 and divisible by 16.");
        if (embeddingDimension <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDimension));
        if (stageDepths is null || stageDepths.Length != 4 || stageDepths.Any(x => x <= 0))
            throw new ArgumentException("Hiera requires four positive stage depths.", nameof(stageDepths));
        if (windowSizes is null || windowSizes.Length != 4 || windowSizes.Any(x => x <= 0))
            throw new ArgumentException("Hiera requires four positive window sizes.", nameof(windowSizes));
        int totalBlocks = stageDepths.Sum();
        if (globalAttentionBlockIndexes is null
            || globalAttentionBlockIndexes.Any(x => x < 0 || x >= totalBlocks)
            || globalAttentionBlockIndexes.Distinct().Count() != globalAttentionBlockIndexes.Length)
        {
            throw new ArgumentException(
                "Global-attention indexes must be unique valid zero-based Hiera block indexes.",
                nameof(globalAttentionBlockIndexes));
        }
        if (initialHeadCount <= 0) throw new ArgumentOutOfRangeException(nameof(initialHeadCount));
        if (outputDimension <= 0) throw new ArgumentOutOfRangeException(nameof(outputDimension));
        if (dropoutRate < 0.0 || dropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate));

        _inputChannels = inputChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _embeddingDimension = embeddingDimension;
        _stageDepths = stageDepths.ToArray();
        _initialHeadCount = initialHeadCount;
        _windowSizes = windowSizes.ToArray();
        _globalAttentionBlockIndexes = globalAttentionBlockIndexes.ToArray();
        _outputDimension = outputDimension;
        _dropoutRate = dropoutRate;

        var identity = new IdentityActivation<T>();
        _stageHeights[0] = ConvOutput(inputHeight, 7, 4, 3);
        _stageWidths[0] = ConvOutput(inputWidth, 7, 4, 3);
        _stageDimensions[0] = embeddingDimension;
        for (int stage = 1; stage < 4; stage++)
        {
            _stageHeights[stage] = ConvOutput(_stageHeights[stage - 1], 3, 2, 1);
            _stageWidths[stage] = ConvOutput(_stageWidths[stage - 1], 3, 2, 1);
            _stageDimensions[stage] = checked(_stageDimensions[stage - 1] * 2);
        }

        _patchEmbedding = CreateConv(
            inputChannels, embeddingDimension, 7, 4, 3, inputHeight, inputWidth, identity);
        RegisterSubLayer(_patchEmbedding);

        _positionEmbedding = new Tensor<T>(
            [1, embeddingDimension, _stageHeights[0], _stageWidths[0]]);
        InitializeLayerWeights(
            _positionEmbedding, embeddingDimension, _stageHeights[0] * _stageWidths[0]);
        RegisterTrainableParameter(_positionEmbedding, PersistentTensorRole.Embeddings);

        _stageDownsamples = new ConvolutionalLayer<T>[3];
        for (int stage = 1; stage < 4; stage++)
        {
            _stageDownsamples[stage - 1] = CreateConv(
                _stageDimensions[stage - 1], _stageDimensions[stage], 3, 2, 1,
                _stageHeights[stage - 1], _stageWidths[stage - 1], identity);
            RegisterSubLayer(_stageDownsamples[stage - 1]);
        }

        _stageBlocks = new TransformerEncoderBlock<T>[4][];
        for (int stage = 0; stage < 4; stage++)
        {
            int heads = checked(initialHeadCount << stage);
            int dimension = _stageDimensions[stage];
            if (dimension % heads != 0)
                throw new ArgumentException(
                    $"Stage {stage} dimension {dimension} is not divisible by {heads} heads.");

            _stageBlocks[stage] = new TransformerEncoderBlock<T>[_stageDepths[stage]];
            for (int block = 0; block < _stageDepths[stage]; block++)
            {
                var transformer = new TransformerEncoderBlock<T>(
                    dimension, heads, dimension * 4, dropoutRate, new GELUActivation<T>());
                transformer.SetTrainingMode(false);
                _ = transformer.Forward(new Tensor<T>([1, 1, dimension]));
                transformer.ResetState();
                _stageBlocks[stage][block] = transformer;
                RegisterSubLayer(transformer);
            }
        }

        _fpnProjections = new ConvolutionalLayer<T>[4];
        for (int stage = 0; stage < 4; stage++)
        {
            _fpnProjections[stage] = CreateConv(
                _stageDimensions[stage], outputDimension, 1, 1, 0,
                _stageHeights[stage], _stageWidths[stage], identity);
            RegisterSubLayer(_fpnProjections[stage]);
        }
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 3;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        if (input.Rank != 4 || input.Shape[1] != _inputChannels
            || input.Shape[2] != _inputHeight || input.Shape[3] != _inputWidth)
        {
            throw new ArgumentException(
                $"Expected [B,{_inputChannels},{_inputHeight},{_inputWidth}], got [{string.Join(",", input.Shape)}].",
                nameof(input));
        }

        var stages = new Tensor<T>[4];
        var current = _patchEmbedding.Forward(input);
        current = Engine.TensorAdd(current, _positionEmbedding);

        int globalBlock = 0;
        for (int stage = 0; stage < 4; stage++)
        {
            if (stage > 0)
                current = _stageDownsamples[stage - 1].Forward(current);

            foreach (var block in _stageBlocks[stage])
            {
                int window = _globalAttentionBlockIndexes.Contains(globalBlock)
                    ? 0
                    : _windowSizes[stage];
                current = RunWindowedBlock(
                    current, block, window,
                    _stageHeights[stage], _stageWidths[stage], _stageDimensions[stage]);
                globalBlock++;
            }
            stages[stage] = current;
        }

        // The mask decoder consumes stride-4 and stride-8 skips directly. The main image embedding
        // is the top-down FPN fusion of Hiera's stride-16 and stride-32 stages.
        _lastStride4 = _fpnProjections[0].Forward(stages[0]);
        _lastStride8 = _fpnProjections[1].Forward(stages[1]);
        var stride16 = _fpnProjections[2].Forward(stages[2]);
        var stride32 = _fpnProjections[3].Forward(stages[3]);
        var upsampled32 = NearestUpsample(stride32, _stageHeights[2], _stageWidths[2]);
        var output = Engine.TensorAdd(stride16, upsampled32);

        return unbatched
            ? Engine.Reshape(output, [_outputDimension, _stageHeights[2], _stageWidths[2]])
            : output;
    }

    private Tensor<T> RunWindowedBlock(
        Tensor<T> map,
        TransformerEncoderBlock<T> block,
        int requestedWindow,
        int height,
        int width,
        int channels)
    {
        int batch = map.Shape[0];
        int window = requestedWindow <= 0
            ? 0
            : Math.Min(requestedWindow, Math.Min(height, width));
        if (window <= 1 || height % window != 0 || width % window != 0)
        {
            var allTokens = ToTokens(map, batch, height, width, channels);
            return ToMap(block.Forward(allTokens), batch, height, width, channels);
        }

        var nhwc = Engine.TensorPermute(map, [0, 2, 3, 1]);
        var partitioned = Engine.Reshape(
            nhwc, [batch, height / window, window, width / window, window, channels]);
        partitioned = Engine.TensorPermute(partitioned, [0, 1, 3, 2, 4, 5]);
        var windows = Engine.Reshape(
            partitioned,
            [batch * (height / window) * (width / window), window * window, channels]);
        windows = block.Forward(windows);

        var restored = Engine.Reshape(
            windows, [batch, height / window, width / window, window, window, channels]);
        restored = Engine.TensorPermute(restored, [0, 1, 3, 2, 4, 5]);
        restored = Engine.Reshape(restored, [batch, height, width, channels]);
        return Engine.TensorPermute(restored, [0, 3, 1, 2]);
    }

    private Tensor<T> ToTokens(Tensor<T> map, int batch, int height, int width, int channels)
    {
        var nhwc = Engine.TensorPermute(map, [0, 2, 3, 1]);
        return Engine.Reshape(nhwc, [batch, height * width, channels]);
    }

    private Tensor<T> ToMap(Tensor<T> tokens, int batch, int height, int width, int channels)
    {
        var nhwc = Engine.Reshape(tokens, [batch, height, width, channels]);
        return Engine.TensorPermute(nhwc, [0, 3, 1, 2]);
    }

    private Tensor<T> NearestUpsample(Tensor<T> input, int outputHeight, int outputWidth)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inputHeight = input.Shape[2];
        int inputWidth = input.Shape[3];
        var indices = new int[batch * channels * outputHeight * outputWidth];
        int offset = 0;
        for (int b = 0; b < batch; b++)
        for (int c = 0; c < channels; c++)
        for (int y = 0; y < outputHeight; y++)
        for (int x = 0; x < outputWidth; x++)
        {
            int sourceY = Math.Min(inputHeight - 1, y * inputHeight / outputHeight);
            int sourceX = Math.Min(inputWidth - 1, x * inputWidth / outputWidth);
            indices[offset++] = ((b * channels + c) * inputHeight + sourceY) * inputWidth + sourceX;
        }
        var flat = Engine.Reshape(input, [input.Length]);
        var gathered = Engine.TensorGather(flat, new Tensor<int>(indices, [indices.Length]), axis: 0);
        return Engine.Reshape(gathered, [batch, channels, outputHeight, outputWidth]);
    }

    private static int ConvOutput(int size, int kernel, int stride, int padding)
        => Math.Max(1, ((size + 2 * padding - kernel) / stride) + 1);

    private static ConvolutionalLayer<T> CreateConv(
        int inputChannels,
        int outputChannels,
        int kernel,
        int stride,
        int padding,
        int inputHeight,
        int inputWidth,
        IActivationFunction<T> activation)
    {
        var layer = new ConvolutionalLayer<T>(
            outputChannels, kernel, stride, padding, activation, biasMode: BiasMode.Always);
        layer.ResolveFromShape([inputChannels, inputHeight, inputWidth]);
        return layer;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastStride4 = null;
        _lastStride8 = null;
        _patchEmbedding.ResetState();
        foreach (var layer in _stageDownsamples) layer.ResetState();
        foreach (var stage in _stageBlocks)
        foreach (var layer in stage) layer.ResetState();
        foreach (var layer in _fpnProjections) layer.ResetState();
    }
}
