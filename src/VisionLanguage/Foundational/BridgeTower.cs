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
/// </remarks>
public class BridgeTower<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly BridgeTowerOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;

    public BridgeTower(NeuralNetworkArchitecture<T> architecture, string modelPath, BridgeTowerOptions? options = null) : base(architecture) { _options = options ?? new BridgeTowerOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public BridgeTower(NeuralNetworkArchitecture<T> architecture, BridgeTowerOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new BridgeTowerOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.FusionDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int FusionEmbeddingDim => _options.FusionDim; public int MaxSequenceLength => _options.MaxSequenceLength;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); int end = Layers.Count / 3; var c = p; for (int i = 0; i < end; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    public Tensor<T> FuseImageText(Tensor<T> image, string text) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p); var c = p; foreach (var l in Layers) c = l.Forward(c); return c; }
    public T ComputeMatchingScore(Tensor<T> image, string text) { var fused = FuseImageText(image, text); double sum = 0; for (int i = 0; i < fused.Length; i++) sum += NumOps.ToDouble(fused[i]); return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-sum / fused.Length))); }

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultBridgeFusionLayers(_options.VisionDim, _options.TextDim, _options.FusionDim, _options.NumVisionLayers, _options.NumTextLayers, _options.NumBridgeLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "BridgeTower-Native" : "BridgeTower-ONNX", Description = "BridgeTower: Building Bridges Between Encoders in Vision-Language Representation Learning (Xu et al., AAAI 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FusionDim, Complexity = _options.NumVisionLayers + _options.NumTextLayers + _options.NumBridgeLayers }; m.AdditionalInfo["Architecture"] = "BridgeTower"; m.AdditionalInfo["FusionType"] = _options.FusionType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.TextDim); writer.Write(_options.FusionDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumTextLayers); writer.Write(_options.NumBridgeLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.TextDim = reader.ReadInt32(); _options.FusionDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumTextLayers = reader.ReadInt32(); _options.NumBridgeLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new BridgeTower<T>(Architecture, mp, _options); return new BridgeTower<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BridgeTower<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
