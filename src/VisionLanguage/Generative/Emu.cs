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
/// Emu: unified VQA, captioning, and image generation via EVA-CLIP + LLM + regression head.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Emu (Sun et al., 2023) unifies visual understanding and generation in a single model. It uses
/// an EVA-CLIP vision encoder to extract visual features, a Causal Transformer (LLaMA-based) as
/// the multimodal LLM, and a visual regression head that maps LLM hidden states back to the
/// visual embedding space for image generation via a diffusion decoder.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Generative Pretraining in Multimodality" (Sun et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> Emu unifies visual understanding (answering questions about images,
/// generating captions) and image generation in a single model. It uses an EVA-CLIP vision encoder
/// with a LLaMA-based language model, and a visual regression head that can map LLM outputs back
/// to visual embeddings for image generation via diffusion. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new Emu&lt;double&gt;(architecture, new EmuOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generative Pretraining in Multimodality", "https://arxiv.org/abs/2307.05222", Year = 2023, Authors = "Sun et al.")]
public class Emu<T> : VisionLanguageModelBase<T>, IGenerativeVisionLanguageModel<T>
{
    private readonly EmuOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _regressionLayers = new List<ILayer<T>>();

    public Emu(NeuralNetworkArchitecture<T> architecture, string modelPath, EmuOptions? options = null) : base(architecture)
    {
        _options = options ?? new EmuOptions();
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

    public Emu(NeuralNetworkArchitecture<T> architecture, EmuOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new EmuOptions();
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
    /// Generates using Emu's unified understanding + generation architecture.
    /// Emu (Sun et al., 2023) unifies VQA, captioning, and image generation:
    /// (1) EVA-CLIP vision encoder extracts visual features and projects to LLM dim,
    /// (2) Visual tokens interleaved with text tokens in shared embedding space,
    /// (3) Causal Transformer (LLaMA-based) processes the multimodal sequence,
    /// (4) Visual regression head maps LLM hidden states back to visual embedding
    ///     space, enabling image generation via downstream diffusion decoder,
    /// (5) Generative pretraining: predict next visual/text element autoregressively.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        // Step 1: EVA-CLIP vision encoder + projection
        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        // Step 2: Tokenize prompt for interleaved multimodal sequence
        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        // Step 3: Concatenate visual features with prompt tokens for multimodal sequence
        var decoderInput = visionOut;
        if (promptTokens is not null) decoderInput = visionOut.ConcatenateTensors(promptTokens);

        // Step 4: Causal Transformer decoder
        var decoderOut = decoderInput;
        foreach (var l in _decoderLayers) decoderOut = l.Forward(decoderOut);

        // Step 5: Visual regression head (maps to visual embedding space)
        var output = decoderOut;
        foreach (var l in _regressionLayers) output = l.Forward(output);

        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int decoderBlockSize = _options.DropoutRate > 0 ? 6 : 5;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize
            + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
        int decoderLayerEnd = visionLayerEnd + _options.NumDecoderLayers * decoderBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultUnifiedGenerationLayers(
            _options.VisionDim, _options.DecoderDim, _options.RegressionDim,
            _options.NumVisionLayers, _options.NumDecoderLayers, _options.NumRegressionLayers,
            _options.NumHeads, _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else if (idx < decoderLayerEnd) _decoderLayers.Add(layer);
            else _regressionLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_decoderLayers);
        RegisterAuxiliaryEncoderStream(_regressionLayers);
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
            Name = _useNativeMode ? "Emu-Native" : "Emu-ONNX",
            Description = "Emu: Generative Pretraining in Multimodality (Sun et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers + _options.NumRegressionLayers
        };
        m.AdditionalInfo["Architecture"] = "Emu";
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
        writer.Write(_options.RegressionDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumRegressionLayers);
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
        _options.RegressionDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumRegressionLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new Emu<T>(Architecture, mp, _options);
        return new Emu<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Emu<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
