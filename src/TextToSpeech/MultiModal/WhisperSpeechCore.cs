using AiDotNet.Audio.Generation;
using AiDotNet.Audio.Speaker;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>WhisperSpeech's layers by part, so each training stage steps its own and saved models keep all of them.</summary>
internal sealed class WhisperSpeechLayers<T>
{
    public List<LayerBase<T>> TextToSemantic { get; } = new();
    public List<LayerBase<T>> SemanticToAcoustic { get; } = new();
    public List<LayerBase<T>> Tokenizer { get; } = new();
    public List<LayerBase<T>> Whisper { get; } = new();
    public List<LayerBase<T>> Speaker { get; } = new();
}

/// <summary>
/// The WhisperSpeech pipeline (the release's <c>Pipeline</c>): text through the T2S model to semantic tokens, those through
/// the S2A model to EnCodec codes in a speaker's voice, the codes through Vocos to 24 kHz audio; recordings through
/// Whisper and the VQ bottleneck to semantic tokens, through EnCodec to acoustic tokens and through ECAPA-TDNN to speaker
/// embeddings, for training.
/// </summary>
internal sealed class WhisperSpeechCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly WhisperSpeechOptions _o;
    private OpenAiWhisperTranscriber<T>? _transcriber;

    /// <summary>The release's <c>Pipeline.default_speaker</c>, used when no speaker is given.</summary>
    public static readonly double[] DefaultSpeaker =
    {
        -0.2929, -0.4503, 0.4155, -0.1417, 0.0473, -0.1624, -0.2322, 0.7071, 0.4800, 0.5496, 0.0410, 0.6236, 0.4729, 0.0587, 0.2194, -0.0466,
        -0.3036, 0.0497, 0.5028, -0.1703, 0.5039, -0.6464, 0.3857, -0.7350, -0.1605, 0.4808, 0.5397, -0.4851, 0.1774, -0.8712, 0.5789, 0.1785,
        -0.1417, 0.3039, 0.4232, -0.0186, 0.2685, 0.6153, -0.3103, -0.5706, -0.4494, 0.3394, -0.6184, -0.3617, 1.1041, -0.1178, -0.1885, 0.1997,
        0.5571, -0.2906, -0.0477, -0.4048, -0.1062, 1.4779, 0.1639, -0.3712, -0.1776, -0.0568, -0.6162, 0.0110, -0.0207, -0.1319, -0.3854, 0.7248,
        0.0343, 0.5724, 0.0670, 0.0486, -0.3813, 0.1738, 0.3017, 1.0502, 0.1550, 0.5708, 0.0366, 0.5093, 0.0294, -0.7091, -0.8220, -0.1583,
        -0.2343, 0.1366, 0.7372, -0.0631, 0.1505, 0.4600, -0.1252, -0.5245, 0.7523, -0.0386, -0.2587, 1.0066, -0.2037, 0.1617, -0.3800, 0.2790,
        0.0184, -0.5111, -0.7291, 0.1627, 0.2367, -0.0192, 0.4822, -0.4458, 0.1457, -0.5884, 0.1909, 0.2563, -0.2035, -0.0377, 0.7771, 0.2139,
        0.3801, 0.6047, -0.6043, -0.2563, -0.0726, 0.3856, 0.3217, 0.0823, -0.1302, 0.3287, 0.5693, 0.2453, 0.8231, 0.0072, 1.0327, 0.6065,
        -0.0620, -0.5572, 0.5220, 0.2485, 0.1520, 0.0222, -0.2179, -0.7392, -0.3855, 0.1822, 0.1042, 0.7133, 0.3583, 0.0606, -0.0424, -0.9189,
        -0.4882, -0.5480, -0.5719, -0.1660, -0.3439, -0.5814, -0.2542, 0.0197, 0.4942, 0.0915, -0.0420, -0.0035, 0.5578, 0.1051, -0.0891, 0.2348,
        0.6876, -0.6685, 0.8215, -0.3692, -0.3150, -0.0462, -0.6806, -0.2661, -0.0308, -0.0050, 0.6756, -0.1647, 1.0734, 0.0049, 0.4969, 0.0259,
        -0.8949, 0.0731, 0.0886, 0.3442, -0.1433, -0.6804, 0.2204, 0.1859, 0.2702, 0.1699, -0.1443, -0.9614, 0.3261, 0.1718, 0.3545, -0.0686,
    };

    public WhisperSpeechCore(IEngine engine, WhisperSpeechOptions options, WhisperSpeechLayers<T> layers, int? seed)
    {
        _engine = engine;
        _o = options;
        var o = options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed ?? o.Seed ?? 42);
        if (o.WhisperAudioContext % o.VqDownsample != 0) throw new ArgumentException("The VQ downsampling must divide Whisper's audio context.");
        if (o.WhisperWidth % o.VqHeads != 0) throw new ArgumentException("The VQ heads must divide Whisper's width.");
        int semanticLength = o.WhisperAudioContext / o.VqDownsample, semanticCodes = o.SemanticCodes + 1;
        var whisperDimensions = new WhisperDimensions(o.WhisperMels, o.WhisperAudioContext, o.WhisperWidth, o.WhisperHeads, o.WhisperLayers,
            o.WhisperVocabulary, o.WhisperTextContext, o.WhisperWidth, o.WhisperHeads, o.WhisperLayers);
        Whisper = new OpenAiWhisperNetwork<T>(engine, layers.Whisper, whisperDimensions, random);
        Tokenizer = new WhisperSpeechVq<T>(engine, layers.Tokenizer, new WhisperSpeechVqConfiguration
        {
            Codes = o.SemanticCodes, Depth = o.VqDepth, Heads = o.VqHeads, HeadWidth = o.WhisperWidth / o.VqHeads, CodebookWidth = o.SemanticWidth,
            Downsample = o.VqDownsample, DeadCodeThreshold = o.VqDeadCodeThreshold, QueryMult = o.VqQueryMult, InitStd = o.VqInitStd,
            EmbeddingsStd = o.VqEmbeddingsStd, CodebookDecay = o.VqCodebookDecay,
        }, Whisper, random);
        TextToSemantic = new WhisperSpeechT2S<T>(engine, layers.TextToSemantic, new WhisperSpeechT2SConfiguration
        {
            Depth = o.T2SDepth, Heads = o.T2SHeads, HeadWidth = o.T2SHeadWidth, FfnMult = o.T2SFfnMult, TextLength = o.MaxTextLength,
            SemanticLength = semanticLength, SemanticCodes = semanticCodes, SemanticWidth = o.SemanticWidth, Languages = WhisperLanguages,
            QueryMult = o.T2SQueryMult, OutputMult = o.T2SOutputMult, EncoderDepthRatio = o.T2SEncoderDepthRatio, CausalEncoder = o.T2SCausalEncoder,
            CpsInput = o.T2SCharactersPerSecondInput, PaddingTokenOffset = o.T2SPaddingTokenOffset, InitStd = o.T2SInitStd, EmbeddingsStd = o.T2SEmbeddingsStd,
        }, random);
        var codecOptions = CodecOptions(o);
        Codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = seed }, codecOptions);
        int codecRate = ((AiDotNet.Interfaces.IAudioCodec<T>)Codec).SampleRate;
        if (codecRate != o.SampleRate || Codec.HopLength != o.HopSize)
            throw new ArgumentException($"WhisperSpeech's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's ({codecRate} Hz, {Codec.HopLength}-sample hop).");
        int quantizers = Codec.QuantizersForBandwidth(o.Codec.TargetBandwidthKbps);
        SemanticToAcoustic = new WhisperSpeechS2A<T>(engine, layers.SemanticToAcoustic, new WhisperSpeechS2AConfiguration
        {
            Depth = o.S2ADepth, Context = 3 * semanticLength, SemanticLength = semanticLength, SemanticCodes = semanticCodes, SemanticWidth = o.SemanticWidth,
            SpeakerWidth = o.SpeakerEmbeddingSize, Heads = o.S2AHeads, HeadWidth = o.S2AHeadWidth, FfnMult = o.S2AFfnMult, Quantizers = quantizers,
            AcousticCodes = o.Codec.CodebookSize, InitStd = o.S2AInitStd, EmbeddingsStd = o.S2AEmbeddingsStd, OutputMult = o.S2AOutputMult,
            QueryMult = o.S2AQueryMult, EncoderDepthRatio = o.S2AEncoderDepthRatio, CausalEncoder = o.S2ACausalEncoder,
            HiddenToEmbedding = o.S2AHiddenToEmbedding, Conditioned = o.S2AConditioned, Languages = WhisperLanguages,
        }, random);
        Speaker = new SpeechBrainEcapa<T>(engine, layers.Speaker, new SpeechBrainEcapaConfiguration
        {
            Channels = o.SpeakerChannels, AttentionChannels = o.SpeakerAttentionChannels, SeChannels = o.SpeakerSeChannels,
            EmbeddingSize = o.SpeakerEmbeddingSize,
        }, random);
        // Vocos's bandwidth classes are 2, 4, 8 and 16 codebooks of the codec's frame rate.
        double perCodebook = (double)o.SampleRate / o.HopSize * Math.Log(o.Codec.CodebookSize, 2) / 1000.0;
        Vocos = new VocosEncodecDecoder<T>(engine, random, new VocosEncodecConfiguration(new[] { 2, 4, 8, 16 }.Select(n => n * perCodebook).ToArray(),
            CodebookSize: o.Codec.CodebookSize, LatentDim: o.Codec.Dimension, FrameRate: o.SampleRate / o.HopSize, Dim: o.VocosDim,
            IntermediateDim: o.VocosIntermediateDim, Layers: o.VocosLayers, FftSize: 4 * o.HopSize, HopSize: o.HopSize));
        SyncSemanticTables();
    }

    private static EnCodecOptions CodecOptions(WhisperSpeechOptions o)
    {
        var codec = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codec.IncludeDiscriminators = false;
        return codec;
    }

    /// <summary>The language table's rows: Whisper's 99 languages.</summary>
    public const int WhisperLanguages = 99;

    public OpenAiWhisperNetwork<T> Whisper { get; }
    public WhisperSpeechVq<T> Tokenizer { get; }
    public WhisperSpeechT2S<T> TextToSemantic { get; }
    public WhisperSpeechS2A<T> SemanticToAcoustic { get; }
    public SpeechBrainEcapa<T> Speaker { get; }
    public EnCodec<T> Codec { get; }
    public VocosEncodecDecoder<T> Vocos { get; }

    /// <summary>Semantic tokens a second (Whisper's 50 frames over the downsampling: 25).</summary>
    public double SemanticRate => 50.0 / _o.VqDownsample;

    /// <summary>The transcript tokenizer of the Whisper the semantic tokenizer trains through.</summary>
    public OpenAiWhisperTranscriber<T> Transcriber => _transcriber ??= new OpenAiWhisperTranscriber<T>(_engine, Whisper);

    /// <summary>A language code's row (<c>languages.to_id</c>: its index among Whisper's languages).</summary>
    public static int LanguageId(string code)
    {
        int id = WhisperTiktoken.LanguageCodes.Take(WhisperLanguages).ToList().IndexOf(code ?? throw new ArgumentNullException(nameof(code)));
        if (id < 0) throw new ArgumentException($"'{code}' is not one of Whisper's language codes.", nameof(code));
        return id;
    }

    /// <summary>The T2S and S2A models read the tokenizer's codebook as their frozen semantic table
    /// (<c>load_frozen_semantic_embeddings</c>); copies it into both.</summary>
    public void SyncSemanticTables()
    {
        var codebook = Tokenizer.Embed[0].Weights;
        foreach (var table in new[] { TextToSemantic.SemanticEmbeddings.Main, SemanticToAcoustic.SemanticEmbedding })
        {
            var w = table.Weights;
            if (w.Length != codebook.Length) throw new InvalidOperationException("The semantic tables and the tokenizer's codebook differ in size.");
            codebook.Data.Span.CopyTo(w.AsWritableSpan());
            _engine.InvalidatePersistentTensor(w);
        }
    }

    // ---------------------------------------------------------------- recordings

    private static double[] Samples(Tensor<T> audio)
    {
        var values = new double[audio.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(audio.GetFlat(i));
        return values;
    }

    /// <summary>Mono audio at <paramref name="sampleRate"/> resampled as torchaudio's <c>Resample</c> does.</summary>
    public double[] Resampled(Tensor<T> audio, int sampleRate, int target)
    {
        var flat = _engine.Reshape(audio, new[] { audio.Length });
        return Samples(sampleRate == target ? flat : AudioHelper<T>.ResampleBandLimited(_engine, flat, sampleRate, target));
    }

    /// <summary>The 30-second window's log-mel <c>[mels, 2 × audio context]</c> of 16 kHz audio zero-padded to the window, as
    /// the release's training chunks are, and the encoder frames holding audio (<c>add_masks</c>).</summary>
    public (Tensor<T> Mel, bool[] Mask) TrainingWindow(double[] wave16k)
    {
        int frames = 2 * _o.WhisperAudioContext, samples = frames * OpenAiWhisperTranscriber<T>.HopLength;
        if (wave16k.Length > samples) throw new ArgumentException($"The recording is longer than the {samples / 16000.0:0.##} s window.");
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(wave16k, _o.WhisperMels, samples - wave16k.Length);
        var tensor = new Tensor<T>(new[] { _o.WhisperMels, frames });
        for (int m = 0; m < _o.WhisperMels; m++)
            for (int t = 0; t < frames; t++) tensor[m, t] = NumOps.FromDouble(mel[m, t]);
        var mask = new bool[_o.WhisperAudioContext];
        for (int i = 0; i < Math.Min(mask.Length, wave16k.Length / 320); i++) mask[i] = true;
        return (tensor, mask);
    }

    /// <summary>The semantic tokens of 16 kHz audio (<c>encode_audio</c>: its log-mel padded with −1.5 to the window, the
    /// tokens of the audio's frames).</summary>
    public int[] SemanticTokens(double[] wave16k)
    {
        int window = 2 * _o.WhisperAudioContext;
        // The STFT's reflect padding needs more than 200 samples (whisper raises below that); shorter audio is
        // zero-padded to 201.
        if (wave16k.Length <= OpenAiWhisperTranscriber<T>.FftSize / 2)
            wave16k = wave16k.Concat(new double[OpenAiWhisperTranscriber<T>.FftSize / 2 + 1 - wave16k.Length]).ToArray();
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(wave16k, _o.WhisperMels);
        int frames = Math.Min(mel.GetLength(1), window);
        var tensor = new Tensor<T>(new[] { _o.WhisperMels, frames });
        for (int m = 0; m < _o.WhisperMels; m++)
            for (int t = 0; t < frames; t++) tensor[m, t] = NumOps.FromDouble(mel[m, t]);
        using var scope = new NoGradScope<T>();
        return Tokenizer.EncodeMel(tensor);
    }

    /// <summary>The acoustic tokens <c>[quantizers][frames]</c> of audio at the model's rate (EnCodec at the configured
    /// bandwidth).</summary>
    public int[][] AcousticTokens(Tensor<T> audio)
    {
        using var scope = new NoGradScope<T>();
        var codes = Codec.Encode(_engine.Reshape(audio, new[] { audio.Length }));
        int quantizers = SemanticToAcoustic.Configuration.Quantizers, frames = codes.GetLength(1);
        return Enumerable.Range(0, quantizers).Select(q => Enumerable.Range(0, frames).Select(f => codes[q, f]).ToArray()).ToArray();
    }

    /// <summary>The speaker embedding of a recording (<c>extract_spk_emb</c>: its first 30 seconds at 16 kHz through
    /// ECAPA-TDNN).</summary>
    public Tensor<T> SpeakerEmbedding(Tensor<T> recording, int sampleRate)
    {
        var wave = Resampled(recording, sampleRate, 16000);
        if (wave.Length > 30 * 16000) wave = wave.Take(30 * 16000).ToArray();
        using var scope = new NoGradScope<T>();
        return Speaker.Embed(wave);
    }

    /// <summary>The default speaker (the release's vector, or zeros for another embedding size).</summary>
    public Tensor<T> DefaultSpeakerEmbedding()
    {
        int size = _o.SpeakerEmbeddingSize;
        var values = size == DefaultSpeaker.Length ? DefaultSpeaker : new double[size];
        return new Tensor<T>(new[] { size }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
    }

    // ---------------------------------------------------------------- synthesis

    /// <summary>Speech for text (<c>Pipeline.generate</c>): newlines to spaces, T2S sampling, S2A sampling, Vocos.</summary>
    public Tensor<T> Speak(string text, Tensor<T> speaker, string language, double cps, Random random, WhisperSpeechConditions? conditions = null)
    {
        var semantic = SemanticFor(text.Replace("\n", " "), language, cps, random);
        var acoustic = SemanticToAcoustic.Generate(semantic, speaker, null, _o.Temperature, _o.TopK, random, conditions);
        return Vocode(acoustic);
    }

    /// <summary>The T2S model's semantic tokens for text (<c>generate</c>, at most a window's).</summary>
    public int[] SemanticFor(string text, string language, double cps, Random random) =>
        TextToSemantic.Generate(text, LanguageId(language), cps, null, TextToSemantic.Configuration.SemanticLength, _o.Temperature, _o.TopK, random);

    /// <summary>Vocos's waveform of acoustic tokens (<c>Vocoder.decode</c>: the bandwidth class of the codebook count).</summary>
    public Tensor<T> Vocode(int[][] acoustic)
    {
        int quantizers = acoustic.Length, frames = acoustic[0].Length;
        if (frames == 0) return new Tensor<T>(new[] { 0 });
        var codes = new int[quantizers, frames];
        for (int q = 0; q < quantizers; q++)
            for (int f = 0; f < frames; f++) codes[q, f] = acoustic[q][f];
        return Vocos.Decode(codes, _o.VocosBandwidthId ?? Vocos.BandwidthId(quantizers));
    }
}
