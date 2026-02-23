using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// RADIOv2.5 agglomerative vision foundation model distilling DINOv2, SAM, SigLIP, and CLIP.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RADIOv2.5 (Ranzinger et al., NVIDIA 2025) distills multiple teacher vision models into a single
/// student via multi-teacher distillation. The student produces features compatible with all teachers,
/// serving as a universal drop-in replacement for DINOv2, SAM, SigLIP, and CLIP vision encoders.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "AM-RADIO: Agglomerative Vision Foundation Model" (Ranzinger et al., 2025)</item></list></para>
/// <para><b>For Beginners:</b> RADIOv25 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class RADIOv25<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly RADIOv25Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode; private bool _disposed;

    public RADIOv25(NeuralNetworkArchitecture<T> architecture, string modelPath, RADIOv25Options? options = null) : base(architecture) { _options = options ?? new RADIOv25Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public RADIOv25(NeuralNetworkArchitecture<T> architecture, RADIOv25Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new RADIOv25Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.EmbeddingDim; InitializeLayers(); }

    public int EmbeddingDimension => _options.EmbeddingDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; foreach (var l in Layers) c = l.Forward(c); return L2Normalize(c); }

    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultViTLayers(_options.EmbeddingDim, _options.NumLayers, _options.NumHeads, _options.DropoutRate)); }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "RADIOv2.5-Native" : "RADIOv2.5-ONNX", Description = "AM-RADIO: Agglomerative Vision Foundation Model (Ranzinger et al., 2025)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.EmbeddingDim, Complexity = _options.NumLayers }; m.AdditionalInfo["Architecture"] = "RADIOv2.5"; m.AdditionalInfo["TeacherModels"] = string.Join(", ", _options.TeacherModels); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.EmbeddingDim); writer.Write(_options.NumLayers); writer.Write(_options.NumHeads); writer.Write(_options.AdapterDim); writer.Write(_options.NumSummaryTokens); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.EmbeddingDim = reader.ReadInt32(); _options.NumLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.AdapterDim = reader.ReadInt32(); _options.NumSummaryTokens = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new RADIOv25<T>(Architecture, mp, _options); return new RADIOv25<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RADIOv25<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
