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
/// UNITER (Universal Image-TExt Representation) with conditional masking pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UNITER (Chen et al., ECCV 2020) uses a single-stream transformer with conditional masking where
/// either image regions or text tokens are masked during pre-training, forcing the model to learn
/// cross-modal alignment. Four pre-training tasks: MLM, MRM (with KL divergence), ITM, and WRA.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "UNITER: UNiversal Image-TExt Representation Learning" (Chen et al., ECCV 2020)</item></list></para>
/// <para><b>For Beginners:</b> UNITER is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class UNITER<T> : VisionLanguageModelBase<T>, IVisionLanguageFusionModel<T>
{
    private readonly UNITEROptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _projectionLayerEnd;

    public UNITER(NeuralNetworkArchitecture<T> architecture, string modelPath, UNITEROptions? options = null) : base(architecture) { _options = options ?? new UNITEROptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public UNITER(NeuralNetworkArchitecture<T> architecture, UNITEROptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new UNITEROptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.FusionDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.FusionDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int FusionEmbeddingDim => _options.FusionDim; public int MaxSequenceLength => _options.MaxSequenceLength;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        // Single-stream: run through all layers
        var c = p;
        for (int i = 0; i < Layers.Count; i++) c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> FuseImageText(Tensor<T> image, string text)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Project image features to fusion dim
        var imageProj = p;
        for (int i = 0; i < _projectionLayerEnd; i++) imageProj = Layers[i].Forward(imageProj);

        // UNITER uses conditional masking: single-stream processes concatenated image+text jointly.
        var textTokens = TokenizeText(text);
        var combined = imageProj.ConcatenateTensors(textTokens);
        var c = combined;
        for (int i = _projectionLayerEnd; i < Layers.Count; i++) c = Layers[i].Forward(c);
        return c;
    }

    public T ComputeMatchingScore(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textTokens = TokenizeText(text);
        Tensor<T> textEmb;
        if (IsOnnxMode && OnnxModel is not null) { textEmb = L2Normalize(OnnxModel.Run(textTokens)); }
        else { var c = textTokens; for (int i = 0; i < Layers.Count; i++) c = Layers[i].Forward(c); textEmb = L2Normalize(c); }
        return CosineSimilarity(imageEmb, textEmb);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _projectionLayerEnd = 0; }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSingleStreamFusionLayers(_options.VisionDim, _options.TextDim, _options.FusionDim, _options.NumFusionLayers, _options.NumHeads, _options.DropoutRate));
            _projectionLayerEnd = (_options.VisionDim != _options.FusionDim ? 2 : 0) + (_options.TextDim != _options.FusionDim ? 2 : 0);
        }
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "UNITER-Native" : "UNITER-ONNX", Description = "UNITER: UNiversal Image-TExt Representation Learning (Chen et al., ECCV 2020)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.FusionDim, Complexity = _options.NumFusionLayers }; m.AdditionalInfo["Architecture"] = "UNITER"; m.AdditionalInfo["FusionType"] = _options.FusionType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.TextDim); writer.Write(_options.FusionDim); writer.Write(_options.NumFusionLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.TextDim = reader.ReadInt32(); _options.FusionDim = reader.ReadInt32(); _options.NumFusionLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); if (_useNativeMode) _projectionLayerEnd = (_options.VisionDim != _options.FusionDim ? 2 : 0) + (_options.TextDim != _options.FusionDim ? 2 : 0); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new UNITER<T>(Architecture, mp, _options); return new UNITER<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UNITER<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; if (disposing) { OnnxModel?.Dispose(); } base.Dispose(disposing); }
}
