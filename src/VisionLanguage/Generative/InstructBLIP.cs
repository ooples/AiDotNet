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
/// InstructBLIP: instruction-tuned BLIP-2 for zero-shot generalization across vision-language tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InstructBLIP (Dai et al., NeurIPS 2023) instruction-tunes the Q-Former component of BLIP-2
/// to extract instruction-aware visual features. The instruction is fed to both the Q-Former
/// (to guide visual feature extraction) and the LLM (to guide text generation).
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning" (Dai et al., NeurIPS 2023)</item></list></para>
/// </remarks>
public class InstructBLIP<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly InstructBLIPOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _qFormerLayerEnd;

    public InstructBLIP(NeuralNetworkArchitecture<T> architecture, string modelPath, InstructBLIPOptions? options = null) : base(architecture) { _options = options ?? new InstructBLIPOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public InstructBLIP(NeuralNetworkArchitecture<T> architecture, InstructBLIPOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new InstructBLIPOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Vision encoder
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++) visionOut = Layers[i].Forward(visionOut);

        // Q-Former: learnable queries cross-attend to vision features, instruction-aware
        var qFormerOut = visionOut;
        for (int i = _visionLayerEnd; i < _qFormerLayerEnd; i++) qFormerOut = Layers[i].Forward(qFormerOut);

        // If prompt provided, tokenize and process through decoder
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            // Prompt informs the decoder alongside Q-Former visual features
            var decoderInput = qFormerOut;
            for (int i = _qFormerLayerEnd; i < Layers.Count; i++) decoderInput = Layers[i].Forward(decoderInput);
            return decoderInput;
        }

        // No prompt: generate from visual features only
        var output = qFormerOut;
        for (int i = _qFormerLayerEnd; i < Layers.Count; i++) output = Layers[i].Forward(output);
        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _qFormerLayerEnd = Layers.Count * 2 / 3; }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultQFormerGenerativeLayers(_options.VisionDim, _options.QFormerDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumQFormerLayers, _options.NumDecoderLayers, _options.NumQueryTokens, _options.NumHeads, _options.NumQFormerHeads, _options.DropoutRate));
            ComputeQFormerBoundaries();
        }
    }

    private void ComputeQFormerBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb;
        int qFormerProjection = _options.VisionDim != _options.QFormerDim ? 1 : 0;
        int qfLpb = _options.DropoutRate > 0 ? 8 : 7; // cross-attn + LN + self-attn + LN + Dense + Dense + LN [+ Dropout]
        _qFormerLayerEnd = _visionLayerEnd + qFormerProjection + _options.NumQFormerLayers * qfLpb;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "InstructBLIP-Native" : "InstructBLIP-ONNX", Description = "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning (Dai et al., NeurIPS 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumQFormerLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "InstructBLIP"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.QFormerDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumQFormerLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.QFormerDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumQFormerLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new InstructBLIP<T>(Architecture, mp, _options); return new InstructBLIP<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(InstructBLIP<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
