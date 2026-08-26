using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Compatibility surface for the paper-faithful Tacotron 2 acoustic model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// This type previously contained a second, unrelated Transformer acoustic stack. It now reuses
/// <see cref="AiDotNet.Audio.TextToSpeech.Tacotron2Model{T}"/>, the repository's shared
/// implementation of the character embedding, convolutional/BiLSTM encoder, location-sensitive
/// attention, autoregressive decoder, stop head, teacher forcing, and residual post-net described
/// by Shen et al. This preserves the established Classic namespace without maintaining two
/// mathematically different models under the Tacotron 2 citation.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions",
    "https://arxiv.org/abs/1712.05884",
    Year = 2018,
    Authors = "Shen et al.")]
public partial class Tacotron2<T> : AiDotNet.Audio.TextToSpeech.Tacotron2Model<T>, IAcousticModel<T>
{
    private readonly Tacotron2Options _options;
    private readonly ITokenizer _tokenizer;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates the native paper architecture.</summary>
    public Tacotron2(
        NeuralNetworkArchitecture<T> architecture,
        Tacotron2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : this(architecture, options ?? new Tacotron2Options(), optimizer, nativeMarker: true)
    {
    }

    private Tacotron2(
        NeuralNetworkArchitecture<T> architecture,
        Tacotron2Options options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        bool nativeMarker)
        : base(
            architecture,
            sampleRate: options.SampleRate,
            numMels: options.MelChannels,
            vocabSize: options.VocabSize,
            embeddingDim: options.EncoderDim,
            encoderDim: options.EncoderDim,
            decoderDim: options.DecoderRnnDim,
            attentionDim: options.AttentionDimension,
            attentionFilters: options.AttentionLocationChannels,
            prenetDim: options.PrenetDim,
            postnetEmbeddingDim: options.PostnetDim,
            numEncoderConvLayers: options.NumEncoderLayers,
            numPostnetConvLayers: options.PostnetLayers,
            numMelsPerFrame: options.OutputsPerStep,
            maxDecoderSteps: GetDecoderStepLimit(options),
            stopThreshold: options.StopThreshold,
            fftSize: options.FftSize,
            hopLength: options.HopSize,
            optimizer: optimizer,
            options: new Tacotron2ModelOptions())
    {
        _ = nativeMarker;
        _options = options;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: options.VocabSize);
    }

    /// <summary>Creates an ONNX-backed Tacotron 2 acoustic model.</summary>
    public Tacotron2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        Tacotron2Options? options = null)
        : this(architecture, modelPath, options ?? new Tacotron2Options(), onnxMarker: true)
    {
    }

    private Tacotron2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        Tacotron2Options options,
        bool onnxMarker)
        : base(
            architecture,
            acousticModelPath: modelPath,
            sampleRate: options.SampleRate,
            numMels: options.MelChannels,
            maxDecoderSteps: GetDecoderStepLimit(options),
            stopThreshold: options.StopThreshold,
            fftSize: options.FftSize,
            hopLength: options.HopSize,
            onnxOptions: options.OnnxOptions,
            options: new Tacotron2ModelOptions())
    {
        _ = onnxMarker;
        _options = options;
        _options.ModelPath = modelPath;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: options.VocabSize);
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int MelChannels => _options.MelChannels;

    /// <inheritdoc />
    public int HopSize => _options.HopSize;

    /// <inheritdoc />
    public int FftSize => _options.FftSize;

    /// <inheritdoc />
    public Tensor<T> TextToMel(string text) => Predict(CreateTokenTensor(text));

    /// <inheritdoc />
    public Tensor<T> Synthesize(string text) => TextToMel(text);

    private Tensor<T> CreateTokenTensor(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            throw new ArgumentException("Text must not be empty.", nameof(text));

        var encoding = _tokenizer.Encode(text);
        int length = Math.Min(encoding.TokenIds.Count, _options.MaxTextLength);
        if (length == 0)
            throw new ArgumentException("Text did not produce any tokens.", nameof(text));

        var tokens = new Tensor<T>([1, length]);
        for (int i = 0; i < length; i++)
            tokens[0, i] = NumOps.FromDouble(encoding.TokenIds[i] % _options.VocabSize);
        return tokens;
    }

    private static int GetDecoderStepLimit(Tacotron2Options options)
        => Math.Max(1, (options.MaxMelLength + options.OutputsPerStep - 1) / options.OutputsPerStep);
}
