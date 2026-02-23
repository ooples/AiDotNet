using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
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
public class BridgeTower<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly BridgeTowerOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _textLayerEnd;

    public BridgeTower(NeuralNetworkArchitecture<T> architecture, string modelPath, BridgeTowerOptions? options = null) : base(architecture) { _options = options ?? new BridgeTowerOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public BridgeTower(NeuralNetworkArchitecture<T> architecture, BridgeTowerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new BridgeTowerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.FusionDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int FusionEmbeddingDim => _options.FusionDim; public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        // Vision encoder with bridge layers
        var c = p;
        for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Vision encoder with bridge cross-attention layers
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++) visionOut = Layers[i].Forward(visionOut);

        // Text encoder with bridge cross-attention layers
        var textTokens = TokenizeText(text);
        var textOut = textTokens;
        for (int i = _visionLayerEnd; i < _textLayerEnd; i++) textOut = Layers[i].Forward(textOut);

        // Concatenate vision and text for cross-modal fusion layers
        var fused = visionOut.ConcatenateTensors(textOut);
        for (int i = _textLayerEnd; i < Layers.Count; i++) fused = Layers[i].Forward(fused);
        return fused;
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textTokens = TokenizeText(text);
        Tensor<T> textEmb;
        if (IsOnnxMode && OnnxModel is not null) { textEmb = L2Normalize(OnnxModel.Run(textTokens)); }
        else
        {
            var c = textTokens;
            for (int i = _visionLayerEnd; i < _textLayerEnd; i++) c = Layers[i].Forward(c);
            textEmb = L2Normalize(c);
        }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _visionLayerEnd = Layers.Count / 3;
            _textLayerEnd = Layers.Count * 2 / 3;
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultBridgeFusionLayers(_options.VisionDim, _options.TextDim, _options.FusionDim, _options.NumVisionLayers, _options.NumTextLayers, _options.NumBridgeLayers, _options.NumHeads, _options.DropoutRate));
            ComputeBridgeBoundaries();
        }
    }

    private void ComputeBridgeBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        // Vision encoder: LN + N vision blocks, with bridge cross-attention at evenly-spaced intervals
        int bridgeInterval = _options.NumVisionLayers > 0 ? Math.Max(1, _options.NumVisionLayers / Math.Max(1, _options.NumBridgeLayers)) : 1;
        int numVisionBridges = _options.NumVisionLayers > 0 ? (_options.NumVisionLayers / bridgeInterval) : 0;
        int bridgeCrossAttnLayers = _options.DropoutRate > 0 ? 3 : 2; // MHA + LN + optional Dropout
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb + numVisionBridges * bridgeCrossAttnLayers + (_options.VisionDim != _options.FusionDim ? 1 : 0);
        // Text encoder: LN + N text blocks, with bridge cross-attention at evenly-spaced intervals
        int textBridgeInterval = _options.NumTextLayers > 0 ? Math.Max(1, _options.NumTextLayers / Math.Max(1, _options.NumBridgeLayers)) : 1;
        int numTextBridges = _options.NumTextLayers > 0 ? (_options.NumTextLayers / textBridgeInterval) : 0;
        _textLayerEnd = _visionLayerEnd + 1 + _options.NumTextLayers * lpb + numTextBridges * bridgeCrossAttnLayers + (_options.TextDim != _options.FusionDim ? 1 : 0);
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
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "BridgeTower-Native" : "BridgeTower-ONNX", Description = "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning (Xu et al., AAAI 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FusionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers + _options.NumBridgeLayers }; m.AdditionalInfo["Architecture"] = "BridgeTower"; m.AdditionalInfo["FusionType"] = _options.FusionType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.TextDim); writer.Write(_options.FusionDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumTextLayers); writer.Write(_options.NumBridgeLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.TextDim = reader.ReadInt32(); _options.FusionDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumTextLayers = reader.ReadInt32(); _options.NumBridgeLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); if (_useNativeMode) ComputeBridgeBoundaries(); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new BridgeTower<T>(Architecture, mp, _options); return new BridgeTower<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BridgeTower<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxModel?.Dispose(); } base.Dispose(disposing); }
}
