using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny AudioPaLM ready to run: a 16-wide PaLM decoder over SpeechT5's 81-character SentencePiece vocabulary (the
/// library's embedded <c>spm_char.model</c>, written to a file) plus 8 audio tokens, a one-block-per-module w2v-BERT whose
/// tokenizer is fitted on two seconds of tones, and a 16-wide SoundStorm over a 25-frame-a-second SoundStream; the voice
/// is a quarter second of the same tones.
/// </summary>
internal static class AudioPaLMFixture
{
    private static readonly Lazy<string> CharacterModel = new(() =>
    {
        const string name = "AiDotNet.TextToSpeech.MultiModal.speecht5_spm_char.model.gz";
        using var stream = typeof(SpeechT5<>).Assembly.GetManifestResourceStream(name) ?? throw new InvalidDataException(name);
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        string path = Path.Combine(Path.GetTempPath(), $"aidotnet_audiopalm_spm_char_{System.Diagnostics.Process.GetCurrentProcess().Id}.model");
        using (var file = File.Create(path)) gzip.CopyTo(file);
        return path;
    });

    public static AudioPaLMOptions Options() => new()
    {
        HiddenDim = 16, Layers = 2, NumHeads = 2, KeyValueHeads = 1, HeadDim = 8, IntermediateSize = 32, VocabSize = 81, AudioTokens = 8,
        BeginId = 0, EndId = 2, MaxTextLength = 160, MaxNewTokens = 12, DropoutRate = 0.0, LearningRate = 1e-2,
        SentencePieceModelPath = CharacterModel.Value,
        W2vBert = new W2vBertOptions
        {
            ModelDim = 16, NumHeads = 2, FeedForwardMultiplier = 2, ContrastiveLayers = 1, MaskedLayers = 1, CodebookSize = 16, CodeDim = 16,
            Negatives = 4, SemanticLayer = 1, Seed = 3,
        },
        SoundStorm = new SoundStormOptions
        {
            IncludeTextToSemantic = false, HiddenDim = 16, NumHeads = 2, ConformerLayers = 1, ConformerHeadDim = 8, FeedForwardMultiplier = 2,
            NumCodebooks = 2, CodebookSize = 16, SemanticCodes = 8, SemanticFrameRate = 25, FirstLevelIterations = 2, DropoutRate = 0.0,
            HopSize = 640,
            SoundStream = new SoundStreamOptions { SampleRate = 16000, Filters = 4, Dimension = 8, Ratios = [8, 5, 4, 4], NumQuantizers = 2, CodebookSize = 16, TargetBandwidthKbps = 0.2 },
        },
        AudioLM = new AudioLMOptions
        {
            SemanticCodes = 8, SemanticFrameRate = 25, NumCodebooks = 2, CoarseCodebooks = 1, CodebookSize = 16, ModelDim = 16, NumLayers = 1,
            NumHeads = 2, KeyValueDim = 8, FeedForwardDim = 32, DropoutRate = 0.0, FineCropSeconds = 0.2, Seed = 42,
            SoundStream = new SoundStreamOptions { SampleRate = 16000, Filters = 4, Dimension = 8, Ratios = [8, 5, 4, 4], NumQuantizers = 2, CodebookSize = 16, TargetBandwidthKbps = 0.2 },
        },
    };

    /// <summary>Two tones at 16 kHz, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 180.0, double high = 470.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(16000 * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * low * i / 16000) + 0.2 * Math.Sin(2 * Math.PI * high * i / 16000));
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }

    private static NeuralNetworkArchitecture<T> Architecture<T>() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    public static AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> Tts<T>(AudioPaLMOptions? options = null)
    {
        var model = new AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T>(Architecture<T>(), options ?? Options());
        Prepare(model);
        return model;
    }

    internal static void Prepare<T>(AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> model)
    {
        model.FitSpeechTokenizer(new[] { Tones<T>(1.0), Tones<T>(1.0, 260.0, 610.0) });
        model.Voice = new TtsVoice<T> { ReferenceAudio = Tones<T>(0.25) };
    }

    public static AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> Asr<T>()
    {
        var model = new AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T>(Architecture<T>(), Options());
        Prepare(model);
        return model;
    }

    internal static void Prepare<T>(AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> model)
        => model.FitSpeechTokenizer(new[] { Tones<T>(1.0), Tones<T>(1.0, 260.0, 610.0) });

    /// <summary>
    /// The ASR model with its tokenizer fitted on a broad corpus rather than two tones: the tones, offset noise and
    /// steady levels across a hundredfold loudness range.
    /// </summary>
    /// <remarks>
    /// The audio tokens are the nearest k-means centroids of w2v-BERT's embeddings (§3.1), so the codebook only separates
    /// audio like what it was fitted on. Fitted on two tones alone, every steady level and every noise recording lies
    /// far from all eight centroids and quantizes to the same token, so distinct recordings give one prompt and one
    /// transcript. The paper fits on a large speech corpus; this corpus stands in for that coverage, so the generic
    /// probes (steady levels, offset noise and scaled noise) get distinct audio tokens.
    /// </remarks>
    public static AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> AsrOverBroadAudio<T>()
    {
        var model = new AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T>(Architecture<T>(), Options());
        model.FitSpeechTokenizer(BroadCorpus<T>());
        return model;
    }

    /// <summary>One-second 16 kHz recordings: two tone pairs, offset noise at three loudnesses and three steady levels.</summary>
    public static Tensor<T>[] BroadCorpus<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        Tensor<T> Filled(Func<double> sample)
        {
            var audio = new T[16000];
            for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(sample());
            return new Tensor<T>(new[] { audio.Length }, new Vector<T>(audio));
        }

        var corpus = new System.Collections.Generic.List<Tensor<T>> { Tones<T>(1.0), Tones<T>(1.0, 260.0, 610.0) };
        foreach (double loudness in new[] { 0.1, 1.0, 10.0 })
            corpus.Add(Filled(() => loudness * random.NextDouble()));
        foreach (double level in new[] { 0.05, 0.5, 5.0 })
            corpus.Add(Filled(() => level));
        return corpus.ToArray();
    }
}
