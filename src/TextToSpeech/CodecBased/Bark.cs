using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Beginner-friendly text façade over the shared <see cref="BarkModel{T}"/> foundation model.
/// </summary>
/// <remarks>
/// This type adds tokenizer loading and text-oriented synthesis. It inherits the one Bark neural
/// implementation rather than constructing another layer stack, so the low-level and high-level
/// APIs cannot drift in architecture, parameters, caching behavior, or checkpoint layout.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.TextToSpeech)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Bark: Text-Prompted Generative Audio Model", "https://github.com/suno-ai/bark")]
public partial class Bark<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly ITokenizer? _configuredTokenizer;
    private ITokenizer? _loadedTokenizer;

    /// <summary>Creates Bark with full checkpoint defaults and lazy tokenizer loading.</summary>
    public Bark(
        BarkOptions? options = null,
        IAudioCodec<T>? codec = null,
        ITokenizer? tokenizer = null,
        int? seed = null)
        : base(options, codec, seed)
    {
        _configuredTokenizer = tokenizer;
    }

    /// <summary>Creates Bark with an explicit architecture descriptor.</summary>
    public Bark(
        NeuralNetworkArchitecture<T> architecture,
        BarkOptions? options = null,
        IAudioCodec<T>? codec = null,
        ITokenizer? tokenizer = null)
        : base(architecture, options, codec)
    {
        _options = options ?? new BarkOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        // Bark's paper defaults build an approximately 100M-parameter transformer,
        // whose classic tape peak can exceed a 16 GB host. Let the framework's
        // footprint estimator select streaming training at that scale while keeping
        // reduced/custom Bark configurations on the faster classic/fused path.
        StreamingTraining = StreamingTrainingMode.Auto;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => _options.NumCodebooks;
    public int CodebookSize => _options.CodebookSize;

    /// <inheritdoc />
    /// <remarks>
    /// TRACED, not read off an observed shape: InitializeLayers passes
    /// <c>NumCodebooks * CodebookSize</c> as the codec vocabulary to CreateDefaultCodecLMLayers, and
    /// that is the width the last layer emits. Recording the 8192 the sweep measured would have been
    /// right for the default options and wrong for any other codebook configuration.
    /// </remarks>
    protected override int OutputFeatureWidth => _options.NumCodebooks * _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <inheritdoc />
    public int MaxTextLength => BarkConfiguration.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => NumberOfCodebooks;

    /// <inheritdoc />
    int ICodecTts<T>.CodebookSize => CodebookSize;

    /// <inheritdoc />
    int ICodecTts<T>.CodecFrameRate => CodecFrameRate;

    /// <summary>Synthesizes a 24 kHz waveform from text using all four Bark stages.</summary>
    public Tensor<T> Synthesize(string text)
        => SynthesizeDetailed(text).Audio;

    /// <summary>Synthesizes text and returns semantic, coarse, fine, audio, and timing outputs.</summary>
    public BarkGenerationResult<T> SynthesizeDetailed(
        string text,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        var tokenIds = Tokenize(text);
        return Generate(tokenIds, generationOptions, history, cancellationToken);
    }

    /// <summary>Asynchronously synthesizes text with cooperative cancellation.</summary>
    public async Task<Tensor<T>> SynthesizeAsync(
        string text,
        BarkGenerationOptions? generationOptions = null,
        BarkHistoryPrompt? history = null,
        CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        var tokenIds = Tokenize(text);
        var result = await GenerateAsync(tokenIds, generationOptions, history, cancellationToken)
            .ConfigureAwait(false);
        return result.Audio;
    }

    /// <inheritdoc />
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
        => ToTensor(EncodeAudio(audio));

    /// <inheritdoc />
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
        => DecodeAudioTokens(ToArray(tokens));

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.Name = "Bark-Text";
        metadata.SetProperty("tokenizer", BarkConfiguration.TokenizerModelName);
        metadata.SetProperty("api", "beginner-text-facade");
        return metadata;
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultCodecLMLayers(
                    _options.TextEncoderDim,
                    _options.LLMDim,
                    _options.NumCodebooks * _options.CodebookSize,
                    _options.NumEncoderLayers,
                    _options.NumLLMLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    _options.VocabSize
                )
            );
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Bark-Native" : "Bark-ONNX",
            Description =
                "Bark: GPT-based text-to-audio model generating speech, music, and sound effects from text prompts.",
            FeatureCount = _options.LLMDim,
        };
        m.AdditionalInfo["Architecture"] = "Bark";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.LLMDim);
        writer.Write(_options.CodebookSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLLMLayers);
        writer.Write(_options.TextEncoderDim);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.NumCodebooks = reader.ReadInt32();
        _options.LLMDim = reader.ReadInt32();
        _options.CodebookSize = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumLLMLayers = reader.ReadInt32();
        _options.TextEncoderDim = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    private void ThrowIfDisposed()
    {
        var tensor = new Tensor<T>([tokens.Count]);
        for (int index = 0; index < tokens.Count; index++)
            tensor[index] = NumOps.FromDouble(tokens[index]);
        return tensor;
    }

    private Tensor<T> ToTensor(int[,] tokens)
    {
        var tensor = new Tensor<T>([tokens.GetLength(0), tokens.GetLength(1)]);
        for (int codebook = 0; codebook < tokens.GetLength(0); codebook++)
            for (int frame = 0; frame < tokens.GetLength(1); frame++)
                tensor[codebook, frame] = NumOps.FromDouble(tokens[codebook, frame]);
        return tensor;
    }

    private int[,] ToArray(Tensor<T> tokens)
    {
        if (tokens.Shape.Length != 2)
            throw new ArgumentException("Bark codec tokens must have shape [codebook, frame].", nameof(tokens));
        var result = new int[tokens.Shape[0], tokens.Shape[1]];
        for (int codebook = 0; codebook < result.GetLength(0); codebook++)
            for (int frame = 0; frame < result.GetLength(1); frame++)
                result[codebook, frame] = Convert.ToInt32(NumOps.ToDouble(tokens[codebook, frame]));
        return result;
    }
}
