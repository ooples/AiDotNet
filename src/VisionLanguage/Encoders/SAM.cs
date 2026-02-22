using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Segment Anything Model (SAM) vision encoder for promptable image segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SAM (Kirillov et al., 2023) consists of a ViT image encoder producing image embeddings, a prompt
/// encoder that handles points/boxes/masks, and a lightweight mask decoder. The image encoder uses
/// windowed attention with occasional global attention blocks for efficiency at high resolution.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Segment Anything" (Kirillov et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> SAM is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class SAM<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly SAMOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed;

    public SAM(NeuralNetworkArchitecture<T> architecture, string modelPath, SAMOptions? options = null) : base(architecture) { _options = options ?? new SAMOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public SAM(NeuralNetworkArchitecture<T> architecture, SAMOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new SAMOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; InitializeLayers(); }

    public int EmbeddingDimension => _options.EmbeddingDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p); var c = p; foreach (var l in Layers) c = l.Forward(c); return c; }

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultViTLayers(_options.EmbeddingDim, _options.NumLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "SAM-Native" : "SAM-ONNX", Description = "Segment Anything Model (Kirillov et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.EmbeddingDim, Complexity = _options.NumLayers }; m.AdditionalInfo["Architecture"] = "SAM"; m.AdditionalInfo["MaskDecoderDim"] = _options.MaskDecoderDim.ToString(); m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.EmbeddingDim); writer.Write(_options.NumLayers); writer.Write(_options.NumHeads); writer.Write(_options.MaskDecoderDim); writer.Write(_options.WindowSize); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.EmbeddingDim = reader.ReadInt32(); _options.NumLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.MaskDecoderDim = reader.ReadInt32(); _options.WindowSize = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new SAM<T>(Architecture, mp, _options); return new SAM<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SAM<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
