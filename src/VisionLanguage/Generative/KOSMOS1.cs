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
/// <para><b>For Beginners:</b> KOSMOS-1 from Microsoft embeds image features directly into
/// a causal language model as if they were text tokens. Image patches from a CLIP ViT are
/// linearly projected into the same embedding space as text, then the combined image-text
/// sequence is processed by a standard causal transformer for unified multimodal understanding
/// and generation. Default values follow the original paper settings.</para>
/// <para><b>Architecture layout:</b> Vision encoder + projection live in
/// <see cref="NeuralNetworkBase{T}.Layers"/>; the causal transformer decoder lives in a private
/// auxiliary stream. <see cref="Predict"/> returns the vision-only embedding;
/// <see cref="GenerateFromImage"/> walks both streams to generate the multimodal output.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new KOSMOS1&lt;double&gt;(architecture, new KOSMOS1Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Language Is Not All You Need: Aligning Perception with Language Models", "https://arxiv.org/abs/2302.14045", Year = 2023, Authors = "Huang et al.")]
public class KOSMOS1<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly KOSMOS1Options _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Decoder layers live outside Layers as an auxiliary stream so the
    // inherited Predict / TrainWithTape walk only the vision encoder.
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public KOSMOS1(NeuralNetworkArchitecture<T> architecture, string modelPath, KOSMOS1Options? options = null) : base(architecture)
    {
        _options = options ?? new KOSMOS1Options();
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

    public KOSMOS1(NeuralNetworkArchitecture<T> architecture, KOSMOS1Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new KOSMOS1Options();
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
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Step 2: Tokenize prompt
        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        // Step 3: Build unified multimodal sequence [visual_tokens | text_tokens]
        var decoderInput = visionOut;
        if (promptTokens is not null) decoderInput = visionOut.ConcatenateTensors(promptTokens);

        // Step 4: Causal transformer decoder
        var output = decoderInput;
        foreach (var l in _decoderLayers) output = l.Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // KOSMOS1 has two trainable streams (vision in Layers,
            // causal decoder in _decoderLayers). A flat caller-supplied
            // Architecture.Layers list can't be unambiguously split — the
            // decoder layer count is encoded in KOSMOS1Options
            // (NumDecoderLayers), and this branch would silently leave
            // _decoderLayers empty so GenerateFromImage returns vision
            // projection without ever running the autoregressive
            // decoder. Reject so the caller uses the default factory
            // path (or overrides streams post-construction).
            throw new System.NotSupportedException(
                "Custom Architecture.Layers is not supported for KOSMOS1: the model has two " +
                "separable trainable streams (vision, causal decoder) and a flat layer list " +
                "cannot be split unambiguously. Use the default factory (no Architecture.Layers) " +
                "and override streams post-construction if needed.");
        }

        // CreateDefaultCausalMultimodalLayers emits:
        //   [pre-norm + N×vision-block + (optional projection), M×decoder-block]
        // Block size = 5 (or 6 with dropout).
        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize
            + (_options.VisionDim != _options.DecoderDim ? 1 : 0);

        var allLayers = LayerHelper<T>.CreateDefaultCausalMultimodalLayers(
            _options.VisionDim, _options.DecoderDim,
            _options.NumVisionLayers, _options.NumDecoderLayers,
            _options.NumHeads, _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else _decoderLayers.Add(layer);
            idx++;
        }

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

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = PreprocessImage(input);
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
            Name = _useNativeMode ? "KOSMOS-1-Native" : "KOSMOS-1-ONNX",
            Description = "KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models (Huang et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Architecture"] = "KOSMOS-1";
        m.AdditionalInfo["GenerativeType"] = _options.ArchitectureType.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new KOSMOS1<T>(Architecture, mp, _options);
        return new KOSMOS1<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KOSMOS1<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            // OnnxModel is allocated by the ONNX-mode constructor and
            // wraps a native ONNX Runtime session — without disposing it
            // here, repeated create/dispose cycles leak unmanaged session
            // memory.
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
