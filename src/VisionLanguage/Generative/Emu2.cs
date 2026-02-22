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

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// Emu2: scaled 37B unified understanding and generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Emu2 (Sun et al., 2024) scales the Emu architecture to 37B parameters by using a larger
/// EVA-CLIP-E encoder and a more powerful LLaMA-based decoder. It demonstrates strong in-context
/// learning for multimodal tasks and can generate both text and images from interleaved inputs.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Generative Multimodal Models are In-Context Learners" (Sun et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> Emu2 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class Emu2<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly Emu2Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _decoderLayerEnd;

    public Emu2(NeuralNetworkArchitecture<T> architecture, string modelPath, Emu2Options? options = null) : base(architecture) { _options = options ?? new Emu2Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public Emu2(NeuralNetworkArchitecture<T> architecture, Emu2Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new Emu2Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates using Emu2's scaled 37B unified architecture.
    /// Emu2 (Sun et al., 2024) extends Emu to 37B with:
    /// (1) EVA-CLIP-E encoder (larger ViT) for richer visual representations,
    /// (2) Visual features projected and interleaved with text in shared space,
    /// (3) 37B LLaMA-based decoder with strong in-context learning capability,
    /// (4) Can process interleaved image-text sequences as few-shot demonstrations,
    /// (5) Visual regression head enables both text and image generation.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: EVA-CLIP-E vision encoder + projection
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Tokenize prompt for in-context multimodal learning
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 3: Concatenate visual features with prompt tokens for in-context multimodal learning
        var decoderInput = visionOut;
        if (promptTokens is not null)
            decoderInput = visionOut.ConcatenateTensors(promptTokens);

        // Step 4: LLaMA-based multimodal decoder
        var decoderOut = decoderInput;
        for (int i = _visionLayerEnd; i < _decoderLayerEnd; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        // Step 5: Visual regression head
        var output = decoderOut;
        for (int i = _decoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _decoderLayerEnd = Layers.Count * 2 / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultUnifiedGenerationLayers(_options.VisionDim, _options.DecoderDim, _options.RegressionDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumRegressionLayers, _options.NumHeads, _options.DropoutRate)); ComputeUnifiedBoundaries(); }
    }

    private void ComputeUnifiedBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
        int decoderLpb = _options.DropoutRate > 0 ? 6 : 5;
        _decoderLayerEnd = _visionLayerEnd + _options.NumDecoderLayers * decoderLpb;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Emu2-Native" : "Emu2-ONNX", Description = "Emu2: Generative Multimodal Models are In-Context Learners (Sun et al., 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers + _options.NumRegressionLayers }; m.AdditionalInfo["Architecture"] = "Emu2"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.RegressionDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumRegressionLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.RegressionDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumRegressionLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new Emu2<T>(Architecture, mp, _options); return new Emu2<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Emu2<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
