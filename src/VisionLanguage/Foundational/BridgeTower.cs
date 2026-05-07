using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// BridgeTower: cross-modal alignment through bridge layers connecting vision and text encoder layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BridgeTower (Xu et al., AAAI 2023) introduces bridge layers that connect vision and text encoder
/// layers at multiple levels, enabling fine-grained cross-modal alignment. Each bridge layer consists
/// of cross-attention between corresponding encoder layers, creating bidirectional information flow
/// throughout the encoding process.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning" (Xu et al., AAAI 2023)</item></list></para>
/// <para><b>For Beginners:</b> BridgeTower is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new BridgeTower&lt;double&gt;(architecture, new BridgeTowerOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning", "https://arxiv.org/abs/2206.08657", Year = 2023, Authors = "Xu et al.")]
public class BridgeTower<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly BridgeTowerOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Foundational fusion split: vision in Layers, text + bridge fusion as auxiliaries.
    private readonly List<ILayer<T>> _bridgeFusionLayers = new List<ILayer<T>>();

    public BridgeTower(NeuralNetworkArchitecture<T> architecture, string modelPath, BridgeTowerOptions? options = null) : base(architecture)
    {
        _options = options ?? new BridgeTowerOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public BridgeTower(NeuralNetworkArchitecture<T> architecture, BridgeTowerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new BridgeTowerOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.FusionDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.FusionDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int FusionEmbeddingDim => _options.FusionDim;
    public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers) c = l.Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Vision encoder with bridge cross-attention layers (Layers).
        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Text encoder + bridge fusion (auxiliary stream). The auxiliary
        // stream contains both the text encoder layers and the bridge fusion
        // layers concatenated; we feed the text tokens through the text
        // portion, then concatenate with vision and run through the fusion.
        var textTokens = TokenizeText(text);
        var current = textTokens;
        foreach (var l in _bridgeFusionLayers) current = l.Forward(current);

        var fused = visionOut.ConcatenateTensors(current);
        return fused;
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textTokens = TokenizeText(text);
        Tensor<T> textEmb;
        if (IsOnnxMode && OnnxModel is not null)
        {
            textEmb = L2Normalize(OnnxModel.Run(textTokens));
        }
        else
        {
            var c = textTokens;
            foreach (var l in _bridgeFusionLayers) c = l.Forward(c);
            textEmb = L2Normalize(c);
        }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture is DualStreamArchitecture<T> dual)
        {
            Layers.AddRange(dual.VisionLayers);
            _bridgeFusionLayers.AddRange(dual.TextLayers);
            RegisterAuxiliaryEncoderStream(_bridgeFusionLayers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        // Vision encoder: LN + N vision blocks, with bridge cross-attention at evenly-spaced intervals
        int bridgeInterval = _options.NumVisionLayers > 0 ? Math.Max(1, _options.NumVisionLayers / Math.Max(1, _options.NumBridgeLayers)) : 1;
        int numVisionBridges = _options.NumVisionLayers > 0 ? (_options.NumVisionLayers / bridgeInterval) : 0;
        int bridgeCrossAttnLayers = _options.DropoutRate > 0 ? 3 : 2; // MHA + LN + optional Dropout
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize
            + numVisionBridges * bridgeCrossAttnLayers
            + (_options.VisionDim != _options.FusionDim ? 1 : 0);

        var allLayers = LayerHelper<T>.CreateDefaultBridgeFusionLayers(
            _options.VisionDim, _options.TextDim, _options.FusionDim,
            _options.NumVisionLayers, _options.NumTextLayers, _options.NumBridgeLayers,
            _options.NumHeads, _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else _bridgeFusionLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_bridgeFusionLayers);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        // Pass _optimizer (the AdamW the constructor created with the
        // model's paper-faithful hyperparameters) into TrainWithTape so
        // the train path actually uses it. Without this the call
        // dropped through to GetOrCreateBaseOptimizer's default Adam at
        // lr=1e-3, ignoring whatever the caller configured via the
        // optimizer ctor parameter.
        try { TrainWithTape(PreprocessImage(input), expected, _optimizer); }
        finally { SetTrainingMode(false); }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BridgeTower-Native" : "BridgeTower-ONNX",
            Description = "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning (Xu et al., AAAI 2023)",
            FeatureCount = _options.FusionDim,
            Complexity = _options.NumVisionLayers + _options.NumTextLayers + _options.NumBridgeLayers
        };
        m.AdditionalInfo["Architecture"] = "BridgeTower";
        m.AdditionalInfo["FusionType"] = _options.FusionType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.TextDim);
        writer.Write(_options.FusionDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumTextLayers);
        writer.Write(_options.NumBridgeLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.TextDim = reader.ReadInt32();
        _options.FusionDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumTextLayers = reader.ReadInt32();
        _options.NumBridgeLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new BridgeTower<T>(Architecture, mp, _options);
        return new BridgeTower<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BridgeTower<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) { OnnxModel?.Dispose(); }
        base.Dispose(disposing);
    }
}
