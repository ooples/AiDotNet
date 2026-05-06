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
/// mPLUG-Owl: modular VLM with visual abstractor for vision-language alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// mPLUG-Owl (Alibaba, 2023) uses a modular architecture with a visual abstractor module that
/// learns to compress and align visual features from the ViT encoder before feeding them into
/// the LLaMA language model. The abstractor uses learnable queries with cross-attention.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality" (Ye et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> mPLUG-Owl from Alibaba uses a modular design where different
/// components can be trained independently. Its key innovation is the "visual abstractor" — a
/// module that sits between the vision encoder and language model, using learnable queries with
/// cross-attention to compress and align visual features into a format the language model can
/// understand. Think of it as a translator that converts raw image features into something
/// the text model can work with effectively. The modular design means you can upgrade
/// individual components without retraining the whole model. Default values follow the
/// original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
/// var trainModel = new MPLUGOwl&lt;double&gt;(architecture, new MPLUGOwlOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality", "https://arxiv.org/abs/2304.14178", Year = 2023, Authors = "Ye et al.")]
public class MPLUGOwl<T> : VisionLanguageModelBase<T>, IInstructionTunedVLM<T>
{
    private readonly MPLUGOwlOptions _options;
    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<ILayer<T>> _abstractorLayers = new List<ILayer<T>>();
    private readonly List<ILayer<T>> _decoderLayers = new List<ILayer<T>>();

    public MPLUGOwl(NeuralNetworkArchitecture<T> architecture, string modelPath, MPLUGOwlOptions? options = null) : base(architecture)
    {
        _options = options ?? new MPLUGOwlOptions();
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

    public MPLUGOwl(NeuralNetworkArchitecture<T> architecture, MPLUGOwlOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new MPLUGOwlOptions();
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
    /// Generates text using mPLUG-Owl's modular architecture.
    /// mPLUG-Owl (Ye et al., 2023) uses three modular components:
    /// (1) Frozen ViT vision encoder for raw visual feature extraction,
    /// (2) Visual abstractor with learnable cross-attention queries that
    ///     compresses and aligns visual features into LLM-compatible tokens,
    /// (3) LLaMA decoder for text generation conditioned on aligned tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(p);

        var visionOut = p;
        foreach (var l in Layers) visionOut = l.Forward(visionOut);

        var abstractorOut = visionOut;
        foreach (var l in _abstractorLayers) abstractorOut = l.Forward(abstractorOut);

        Tensor<T>? promptTokens = null;
        if (prompt is not null) promptTokens = TokenizeText(prompt);

        var decoderInput = abstractorOut;
        if (promptTokens is not null) decoderInput = abstractorOut.ConcatenateTensors(promptTokens);

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
            Layers.AddRange(Architecture.Layers);
            return;
        }

        int blockSize = _options.DropoutRate > 0 ? 6 : 5;
        int aBlockSize = _options.DropoutRate > 0 ? 8 : 7;
        int visionLayerEnd = 1 + _options.NumVisionLayers * blockSize;
        int aProj = _options.VisionDim != _options.AbstractorDim ? 1 : 0;
        int abstractorLayerEnd = visionLayerEnd + aProj + _options.NumAbstractorLayers * aBlockSize;

        var allLayers = LayerHelper<T>.CreateDefaultPerceiverResamplerLayers(
            _options.VisionDim, _options.AbstractorDim, _options.DecoderDim,
            _options.NumVisionLayers, _options.NumAbstractorLayers, _options.NumDecoderLayers,
            _options.MaxVisualTokens, _options.NumHeads, _options.NumAbstractorHeads,
            _options.DropoutRate);

        int idx = 0;
        foreach (var layer in allLayers)
        {
            if (idx < visionLayerEnd) Layers.Add(layer);
            else if (idx < abstractorLayerEnd) _abstractorLayers.Add(layer);
            else _decoderLayers.Add(layer);
            idx++;
        }

        RegisterAuxiliaryEncoderStream(_abstractorLayers);
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
            Name = _useNativeMode ? "mPLUG-Owl-Native" : "mPLUG-Owl-ONNX",
            Description = "mPLUG-Owl: Modular VLM with Visual Abstractor (Ye et al., 2023)",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumAbstractorLayers + _options.NumDecoderLayers
        };
        m.AdditionalInfo["Architecture"] = "mPLUG-Owl";
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
        writer.Write(_options.AbstractorDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumAbstractorLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumAbstractorHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.AbstractorDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumAbstractorLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumAbstractorHeads = reader.ReadInt32();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new MPLUGOwl<T>(Architecture, mp, _options);
        return new MPLUGOwl<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MPLUGOwl<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
