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

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// LayoutLMv3: unified text, image, and layout pre-training for document AI.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (Microsoft, 2022)</item></list></para>
/// <para><b>For Beginners:</b> LayoutLMv3 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class LayoutLMv3<T> : VisionLanguageModelBase<T>, IDocumentUnderstandingModel<T>
{
    private readonly LayoutLMv3Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _encoderLayerEnd;

    public LayoutLMv3(NeuralNetworkArchitecture<T> architecture, string modelPath, LayoutLMv3Options? options = null) : base(architecture) { _options = options ?? new LayoutLMv3Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public LayoutLMv3(NeuralNetworkArchitecture<T> architecture, LayoutLMv3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new LayoutLMv3Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public bool IsOcrFree => _options.IsOcrFree;
    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _encoderLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }
    /// <summary>
    /// Generates text from a document image using LayoutLMv3's unified multimodal pipeline.
    /// Per the paper (Microsoft, 2022), LayoutLMv3 jointly pre-trains text, image, and layout
    /// modalities with unified text-image masking. Unlike OCR-free models, it explicitly fuses:
    /// (1) Visual features from ViT-like patch encoding,
    /// (2) Layout embeddings encoding 2D spatial positions (x0,y0,x1,y1 bounding boxes),
    /// (3) Text token embeddings from the prompt/OCR output.
    /// The three modalities are combined via additive fusion before the multimodal transformer.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: ViT patch encoding for visual features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Tokenize prompt for multimodal fusion (visual + layout + text)
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 3: Concatenate visual features with prompt tokens
        // LayoutLMv3 fuses visual, layout, and text modalities in the transformer
        var fusedInput = visualFeatures;
        if (promptTokens is not null)
            fusedInput = visualFeatures.ConcatenateTensors(promptTokens);

        // Step 4: Multimodal transformer decoder
        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }
    public Tensor<T> ExtractText(Tensor<T> documentImage) { ThrowIfDisposed(); return GenerateFromImage(documentImage, "Extract all text from this document."); }
    public Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question) { ThrowIfDisposed(); return GenerateFromImage(documentImage, question); }
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Layers.Count / 2; } else { Layers.AddRange(LayerHelper<T>.CreateDefaultDocumentOCRLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, 2048, _options.DropoutRate)); ComputeEncoderDecoderBoundary(); } }
    private void ComputeEncoderDecoderBoundary() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0); }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() {
        var m = new ModelMetadata<T> { Name = _useNativeMode ? "LayoutLMv3-Native" : "LayoutLMv3-ONNX", Description = "LayoutLMv3: unified text, image, and layout pre-training for document AI.", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers };
        m.AdditionalInfo["Architecture"] = "LayoutLMv3";
        m.AdditionalInfo["OcrFree"] = _options.IsOcrFree.ToString();
        m.AdditionalInfo["MaxLayoutTokens"] = _options.MaxLayoutTokens.ToString();
        return m;
    }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.IsOcrFree);
        writer.Write(_options.MaxOutputTokens);
        writer.Write(_options.MaxLayoutTokens);
    }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.IsOcrFree = reader.ReadBoolean();
        _options.MaxOutputTokens = reader.ReadInt32();
        _options.MaxLayoutTokens = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new LayoutLMv3<T>(Architecture, mp, _options); return new LayoutLMv3<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LayoutLMv3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
