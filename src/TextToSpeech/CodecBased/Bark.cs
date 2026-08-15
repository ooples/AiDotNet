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
public class Bark<T> : BarkModel<T>, ICodecTts<T>
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
        _configuredTokenizer = tokenizer;
    }

    /// <inheritdoc />
    int ITtsModel<T>.SampleRate => SampleRate;

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
        => ToTokenTensorForFacade(Tokenize(text));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new Bark<T>(
            Architecture,
            BarkConfiguration,
            CreateCodecForNewInstance(),
            _configuredTokenizer ?? _loadedTokenizer);

    private IReadOnlyList<int> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Bark synthesis text cannot be empty.", nameof(text));

        var tokenizer = _configuredTokenizer
            ?? (_loadedTokenizer ??= AutoTokenizer.FromPretrained(BarkConfiguration.TokenizerModelName));
        var encoded = tokenizer.Encode(text, new EncodingOptions
        {
            AddSpecialTokens = false,
            MaxLength = MaxTextLength,
            Truncation = true,
            Padding = false,
        });
        if (encoded.TokenIds.Count == 0)
            throw new InvalidOperationException("The configured Bark tokenizer produced no tokens.");
        return encoded.TokenIds;
    }

    private Tensor<T> ToTokenTensorForFacade(IReadOnlyList<int> tokens)
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
