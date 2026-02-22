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
/// BLIP-3 (xGen-MM): scaled vision-language model with interleaved data and any-to-any generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP-3/xGen-MM (Salesforce, 2024) scales the BLIP-2 architecture with interleaved image-text
/// data (OBELICS), larger Q-Former capacity, and any-to-any generation capabilities.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "xGen-MM (BLIP-3): A Family of Open Large Multimodal Models" (Salesforce, 2024)</item></list></para>
/// <para><b>For Beginners:</b> BLIP3 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class BLIP3<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly BLIP3Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _qFormerLayerEnd;

    public BLIP3(NeuralNetworkArchitecture<T> architecture, string modelPath, BLIP3Options? options = null) : base(architecture) { _options = options ?? new BLIP3Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public BLIP3(NeuralNetworkArchitecture<T> architecture, BLIP3Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new BLIP3Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using BLIP-3/xGen-MM's scaled Q-Former architecture.
    /// BLIP-3 (Salesforce, 2024) extends BLIP-2 with:
    /// (1) Scaled Q-Former with increased capacity for richer visual queries,
    /// (2) Interleaved image-text training on OBELICS data for in-context learning,
    /// (3) Any-to-any generation: visual tokens from Q-Former are projected and
    ///     interleaved with text tokens for flexible multimodal generation,
    /// (4) LLM decoder (Phi-3) generates text conditioned on interleaved visual+text.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: Vision encoder
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Scaled Q-Former cross-attention
        var qFormerOut = visionOut;
        for (int i = _visionLayerEnd; i < _qFormerLayerEnd; i++)
            qFormerOut = Layers[i].Forward(qFormerOut);

        // Step 3: Tokenize prompt for interleaved sequence
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 4: Concatenate Q-Former output with prompt tokens for LLM decoder
        var decoderInput = qFormerOut;
        if (promptTokens is not null)
            decoderInput = qFormerOut.ConcatenateTensors(promptTokens);

        // Step 5: LLM decoder generates output
        var output = decoderInput;
        for (int i = _qFormerLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _qFormerLayerEnd = Layers.Count * 2 / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultQFormerGenerativeLayers(_options.VisionDim, _options.QFormerDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumQFormerLayers, _options.NumDecoderLayers, _options.NumQueryTokens, _options.NumHeads, _options.NumQFormerHeads, _options.DropoutRate)); ComputeQFormerBoundaries(); }
    }

    private void ComputeQFormerBoundaries() { int lpb = _options.DropoutRate > 0 ? 6 : 5; _visionLayerEnd = 1 + _options.NumVisionLayers * lpb; int qfProj = _options.VisionDim != _options.QFormerDim ? 1 : 0; int qfLpb = _options.DropoutRate > 0 ? 8 : 7; _qFormerLayerEnd = _visionLayerEnd + qfProj + _options.NumQFormerLayers * qfLpb; }
    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "BLIP3-Native" : "BLIP3-ONNX", Description = "BLIP-3/xGen-MM: A Family of Open Large Multimodal Models (Salesforce, 2024)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumQFormerLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "BLIP3"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.QFormerDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumQFormerLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.QFormerDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumQFormerLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new BLIP3<T>(Architecture, mp, _options); return new BLIP3<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BLIP3<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
