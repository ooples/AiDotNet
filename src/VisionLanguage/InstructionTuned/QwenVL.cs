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

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// Qwen-VL: visual window attention, multi-resolution, bounding box output via cross-attention resampler.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Qwen-VL (Bai et al., 2023) uses a ViT vision encoder with visual window attention and a
/// cross-attention resampler to compress visual features before feeding into the Qwen language
/// model. It supports multi-resolution input and can output bounding box coordinates for
/// visual grounding tasks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond" (Bai et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> QwenVL is a vision-language model. Default values follow the original paper settings.</para>
/// </remarks>
public class QwenVL<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly QwenVLOptions _options; public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer; private bool _useNativeMode; private bool _disposed;
    private int _visionLayerEnd; private int _resamplerLayerEnd;

    public QwenVL(NeuralNetworkArchitecture<T> architecture, string modelPath, QwenVLOptions? options = null) : base(architecture) { _options = options ?? new QwenVLOptions(); _useNativeMode = false; base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }
    public QwenVL(NeuralNetworkArchitecture<T> architecture, QwenVLOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new QwenVLOptions(); _useNativeMode = true; _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.ImageSize = _options.ImageSize; base.ImageChannels = 3; base.EmbeddingDim = _options.DecoderDim; _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize); InitializeLayers(); }

    public int EmbeddingDimension => _options.DecoderDim; int IVisualEncoder<T>.ImageSize => _options.ImageSize; int IVisualEncoder<T>.ImageChannels => 3; public int MaxGenerationLength => _options.MaxGenerationLength; public int DecoderEmbeddingDim => _options.DecoderDim; public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image) { ThrowIfDisposed(); var p = PreprocessImage(image); if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p)); var c = p; for (int i = 0; i < _visionLayerEnd; i++) c = Layers[i].Forward(c); return L2Normalize(c); }

    /// <summary>
    /// Generates text using Qwen-VL's cross-attention resampler architecture.
    /// Qwen-VL (Bai et al., 2023) uses:
    /// (1) ViT with visual window attention: processes multi-resolution images with
    ///     windowed self-attention for efficient high-resolution encoding,
    /// (2) Cross-attention resampler: 6 layers of cross-attention with 256 learned
    ///     queries attending to all visual tokens, compressing to fixed-length output,
    /// (3) Bounding box output: supports visual grounding by outputting coordinates
    ///     as special tokens interleaved with text,
    /// (4) Qwen decoder backbone.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        int numQueries = _options.MaxVisualTokens; // 256 learned queries

        // Step 1: ViT with visual window attention
        var visionOut = p;
        for (int i = 0; i < _visionLayerEnd; i++)
            visionOut = Layers[i].Forward(visionOut);
        int visLen = visionOut.Length;

        // Step 2: Cross-attention resampler (learned queries attend to visual features)
        var resamplerOut = visionOut;
        for (int i = _visionLayerEnd; i < _resamplerLayerEnd; i++)
            resamplerOut = Layers[i].Forward(resamplerOut);

        // Fuse features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = resamplerOut.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = resamplerOut;
        }

        var output = fusedInput;
        for (int i = _resamplerLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    public Tensor<T> Chat(Tensor<T> image, IEnumerable<(string Role, string Content)> conversationHistory, string userMessage)
    {
        ThrowIfDisposed();
        var sb = new System.Text.StringBuilder();
        sb.Append(_options.SystemPrompt);
        foreach (var (role, content) in conversationHistory) sb.Append($"\n{role}: {content}");
        sb.Append($"\nUser: {userMessage}\nAssistant:");
        return GenerateFromImage(image, sb.ToString());
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); _visionLayerEnd = Layers.Count / 3; _resamplerLayerEnd = Layers.Count * 2 / 3; }
        else { Layers.AddRange(LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(_options.VisionDim, _options.ResamplerDim, _options.DecoderDim, _options.NumVisionLayers, _options.NumResamplerLayers, _options.NumDecoderLayers, _options.MaxVisualTokens, _options.NumHeads, _options.NumResamplerHeads, _options.DropoutRate)); ComputeResamplerBoundaries(); }
    }

    private void ComputeResamplerBoundaries()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _visionLayerEnd = 1 + _options.NumVisionLayers * lpb;
        int resamplerProj = _options.VisionDim != _options.ResamplerDim ? 1 : 0;
        int rLpb = _options.DropoutRate > 0 ? 8 : 7;
        _resamplerLayerEnd = _visionLayerEnd + resamplerProj + _options.NumResamplerLayers * rLpb;
    }

    private Tensor<T> TokenizeText(string text) { if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized."); var encoding = _tokenizer.Encode(text); int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength); var tokens = new Tensor<T>([seqLen]); for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]); return tokens; }

    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode."); SetTrainingMode(true); var o = Predict(input); var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector()); var gt = Tensor<T>.FromVector(g); for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt); _optimizer?.UpdateParameters(Layers); SetTrainingMode(false); }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;
    public override ModelMetadata<T> GetModelMetadata() { var m = new ModelMetadata<T> { Name = _useNativeMode ? "Qwen-VL-Native" : "Qwen-VL-ONNX", Description = "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (Bai et al., 2023)", ModelType = ModelType.NeuralNetwork, FeatureCount = _options.DecoderDim, Complexity = _options.NumVisionLayers + _options.NumResamplerLayers + _options.NumDecoderLayers }; m.AdditionalInfo["Architecture"] = "Qwen-VL"; m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString(); m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName; return m; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.ImageSize); writer.Write(_options.VisionDim); writer.Write(_options.ResamplerDim); writer.Write(_options.DecoderDim); writer.Write(_options.NumVisionLayers); writer.Write(_options.NumResamplerLayers); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.NumResamplerHeads); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.ImageSize = reader.ReadInt32(); _options.VisionDim = reader.ReadInt32(); _options.ResamplerDim = reader.ReadInt32(); _options.DecoderDim = reader.ReadInt32(); _options.NumVisionLayers = reader.ReadInt32(); _options.NumResamplerLayers = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); _options.NumResamplerHeads = reader.ReadInt32(); if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new QwenVL<T>(Architecture, mp, _options); return new QwenVL<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(QwenVL<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
