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
/// KOSMOS-1: multimodal large language model with visual tokens embedded in causal LM.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// KOSMOS-1 (Huang et al., 2023) is a multimodal large language model that embeds visual tokens
/// directly into a causal language model. Image features from a CLIP ViT are linearly projected
/// into the same embedding space as text tokens, then the combined sequence is processed by a
/// causal transformer decoder for unified multimodal understanding and generation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Language Is Not All You Need: Aligning Perception with Language Models" (Huang et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> KOSMOS1 is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class KOSMOS1<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly KOSMOS1Options _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd;

    public KOSMOS1(NeuralNetworkArchitecture<T> architecture, string modelPath, KOSMOS1Options? options = null) : base(architecture) { _options = options ?? new KOSMOS1Options(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public KOSMOS1(NeuralNetworkArchitecture<T> architecture, KOSMOS1Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new KOSMOS1Options(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using KOSMOS-1's unified multimodal causal LM architecture.
    /// KOSMOS-1 (Huang et al., 2023) uses:
    /// (1) CLIP ViT image encoder extracts visual feature tokens,
    /// (2) Linear projection maps visual tokens into the same embedding space as text,
    /// (3) Unified sequence: &lt;s&gt; &lt;image&gt; vis_1 ... vis_N &lt;/image&gt; text_1 ... text_M,
    ///     where special tokens delimit the image region in the sequence,
    /// (4) Causal transformer decoder processes the entire mixed-modality sequence
    ///     with standard causal attention (no separate cross-attention),
    /// (5) Trained on interleaved web data for multimodal in-context learning.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: CLIP ViT vision encoder + linear projection
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);

        // Step 2: Tokenize prompt
        Tensor<T>? promptTokens = null;
        if (prompt is not null)
            promptTokens = TokenizeText(prompt);

        // Step 3: Build unified multimodal sequence [visual_tokens | text_tokens]
        var decoderInput = visionOut;
        if (promptTokens is not null)
            decoderInput = visionOut.ConcatenateTensors(promptTokens);

        // Step 4: Causal transformer decoder
        var output = decoderInput;
        for (int i = _visionLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultCausalMultimodalLayers(_options.VisionDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate)); ComputeCausalBoundary(); }
    }

    private void ComputeCausalBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        // Vision encoder layers + projection to decoder dim
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "KOSMOS-1-Native" : "KOSMOS-1-ONNX", Description = "KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models (Huang et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "KOSMOS-1"; m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString(); return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new KOSMOS1<T>(Architecture, mp, _options); return new KOSMOS1<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KOSMOS1<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
