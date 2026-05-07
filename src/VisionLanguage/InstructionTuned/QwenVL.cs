using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
/// <para><b>For Beginners:</b> Qwen-VL from Alibaba is a versatile model that goes beyond
/// just describing images — it can locate objects by outputting bounding box coordinates.
/// It uses visual window attention (processing image regions locally for efficiency) and a
/// cross-attention resampler that compresses the large number of visual tokens into a fixed
/// smaller set before feeding them to the language model. It supports images at multiple
/// resolutions and can handle tasks like visual question answering, text reading (OCR),
/// and visual grounding (finding where objects are in an image). Default values follow the
/// original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new QwenVL&lt;double&gt;(architecture, new QwenVLOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond", "https://arxiv.org/abs/2308.12966", Year = 2023, Authors = "Bai et al.")]
public class QwenVL<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly QwenVLOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _resamplerLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public QwenVL(NeuralNetworkArchitecture<T> architecture, string modelPath, QwenVLOptions? options = null) : base(architecture)
    {
        _options = options ?? new QwenVLOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public QwenVL(NeuralNetworkArchitecture<T> architecture, QwenVLOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new QwenVLOptions();
        SyncImageSizeWithArchitecture();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    private void SyncImageSizeWithArchitecture()
    {
        int h = Architecture.InputHeight;
        int w = Architecture.InputWidth;
        if (h > 0 && w > 0 && h == w) _options.ImageSize = h;
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public string LanguageModelName => _options.LanguageModelName;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(p));
        var c = p;
        foreach (var l in Layers) c = l.Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates text using Qwen-VL's window-attention + cross-attention resampler architecture.
    /// Qwen-VL (Bai et al., 2023) features:
    /// (1) ViT vision encoder with visual window attention for efficiency,
    /// (2) Cross-attention resampler with learnable query tokens for compression,
    /// (3) Qwen language model decoder with bounding box output capability,
    /// (4) Multi-resolution input support and OCR-friendly tokenization.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        var resamplerOut = visionOut;
        foreach (var l in _resamplerLayers) resamplerOut = l.Forward(resamplerOut);

        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        var decoderInput = resamplerOut;
        if (promptTokens is not null) decoderInput = resamplerOut.ConcatenateTensors(promptTokens);

        var output = decoderInput;
        foreach (var l in _decoderLayers) output = l.Forward(output);

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
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // QwenVL has multiple separable trainable streams (vision in
            // Layers, resampler + decoder as auxiliary streams). A flat caller-
            // supplied Architecture.Layers list cannot be unambiguously
            // split because each stream's layer count is encoded in the
            // model's Options class — this branch would silently leave
            // the auxiliary streams empty and GenerateFromImage would
            // degenerate to a vision-only forward. Reject so the caller
            // either uses the default factory or constructs the streams
            // explicitly post-construction.
            throw new System.NotSupportedException(
                "Custom Architecture.Layers is not supported for QwenVL: the model has multiple " +
                "separable trainable streams (vision, resampler, decoder) and a flat layer list cannot " +
                "be split unambiguously. Use the default factory (no Architecture.Layers) and " +
                "override streams post-construction if needed.");
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int rBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize;
        int rProj = _options.VisionDim != _options.ResamplerDim ? 1 : 0;
        int resamplerLayerEnd = visionLayerEnd + rProj + _options.NumResamplerLayers * rBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(
            _options.VisionDim, _options.ResamplerDim, _options.DecoderDim,
            _options.NumVisionLayers, _options.NumResamplerLayers, _options.NumDecoderLayers,
            _options.MaxVisualTokens, _options.NumHeads, _options.NumResamplerHeads,
            _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else if (idx < resamplerLayerEnd) _resamplerLayers.Add(layer);
            else _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_resamplerLayers);
        RegisterAuxiliaryEncoderStream(_decoderLayers);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    /// <summary>
    /// Vision-encoder-only forward: runs the patch-embedding +
    /// transformer stack in <see cref="NeuralNetworkBase{T}.Layers"/> and
    /// returns the resulting [B, S, VisionEmbeddingDim] embeddings.
    /// Mirrors the embedding-extraction surface of the rest of the
    /// VL family (BiomedCLIP / DFNCLIP / EVACLIP — see #52). For the full
    /// vision → resampler → decoder generation pipeline use
    /// <see cref="GenerateFromImage(Tensor{T}, string)"/>; this method is
    /// the encoder-only fast path callers reach for when they want
    /// pre-fusion image features (zero-shot retrieval, similarity, etc.).
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        // Normalize ONNX inputs the same way the native path does — both
        // EncodeImage / GenerateFromImage and the native Predict call
        // PreprocessImage. Without this, the ONNX fast path would diverge
        // silently from native.
        var c = PreprocessImage(input);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(c);
        SetTrainingMode(false);
        foreach (var l in Layers) c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try { TrainWithTape(PreprocessImage(input), expected); }
        finally { SetTrainingMode(false); }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
        // Sync the auxiliary streams (resampler / abstractor / decoder /
        // visual decoder, depending on model) — see OpenFlamingo.UpdateParameters
        // for full rationale (dual-stream split, GetExtraTrainableLayers
        // widens the flat parameter vector to include them, so a writeback
        // that only walks Layers leaves auxiliary streams on stale weights
        // and the model state silently de-syncs across streams).
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers())
        {
            if (l is null) continue;
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => EnumerateAuxiliaryStreamTrainableLayers();

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Qwen-VL-Native" : "Qwen-VL-ONNX",
            Description = "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (Bai et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumResamplerLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Architecture"] = "Qwen-VL";
        m.AdditionalInfo["InstructionType"] = _options.InstructionArchitectureType.ToString();
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.ResamplerDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumResamplerLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumResamplerHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.ResamplerDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumResamplerLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumResamplerHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new QwenVL<T>(Architecture, mp, _options);
        return new QwenVL<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(QwenVL<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
