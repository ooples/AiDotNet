using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Florence-2 unified vision foundation model for captioning, detection, grounding, and OCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Florence-2 (Xiao et al., Microsoft 2024) is a lightweight sequence-to-sequence vision model
/// (0.23B-0.77B) handling multiple tasks through a unified prompt-based approach. It uses DaViT
/// (Dual Attention ViT) as the vision encoder and a multi-task decoder for captioning, detection,
/// grounding, OCR, and segmentation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" (Xiao et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> Florence2 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class Florence2<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly Florence2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed; private int _encoderEnd;

    public Florence2(NeuralNetworkArchitecture<T> architecture, string modelPath, Florence2Options? options = null) : base(architecture) { _options = options ?? new Florence2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public Florence2(NeuralNetworkArchitecture<T> architecture, Florence2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new Florence2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; InitializeLayers(); }

    public int EmbeddingDimension => _options.EmbeddingDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p); var c = p; for (int i = 0; i < _encoderEnd; i++) c = Layers[i].Forward(c); return c; }

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultFlorence2Layers(_options.EmbeddingDim, _options.DecoderEmbeddingDim, _options.NumLayers, _options.NumDecoderLayers, _options.NumHeads, _options.NumDecoderHeads, _options.DropoutRate)); int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderEnd = 1 + _options.NumLayers * lpb + (_options.EmbeddingDim != _options.DecoderEmbeddingDim ? 1 : 0); } }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Florence-2-Native" : "Florence-2-ONNX", Description = "Florence-2: Unified Vision Foundation Model (Xiao et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.EmbeddingDim, Complexity = _options.NumLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "Florence-2"; m.AdditionalInfo["ModelSize"] = _options.ModelSize.ToString(); m.AdditionalInfo["UseDaViT"] = _options.UseDaViT.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.EmbeddingDim); writer.Write(_options.NumLayers); writer.Write(_options.NumHeads); writer.Write(_options.DecoderEmbeddingDim); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumDecoderHeads); writer.Write((int)_options.ModelSize); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.EmbeddingDim = reader.ReadInt32(); _options.NumLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.DecoderEmbeddingDim = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumDecoderHeads = reader.ReadInt32(); _options.ModelSize = (Florence2ModelSize)reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Florence2<T>(Architecture, mp, _options); return new Florence2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Florence2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxModel?.Dispose(); } base.Dispose(disposing); }
}
