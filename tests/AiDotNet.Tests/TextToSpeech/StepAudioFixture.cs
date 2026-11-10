using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny Step-Audio ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Text: Step-1's real 74,752-piece tokenizer (embedded in stepaudio_tokenizer_reference.json).</item>
/// <item>Chat and TTS models: one-layer, 16-wide Step-1 decoders (2 heads in 1 group, ALiBi).</item>
/// <item>Tokenizer: a two-block, 16-wide streaming Paraformer over 8 bands with the release's 1,024 linguistic codes (dither
/// off), and a one-layer, 16-wide S3 v1 with 16 semantic codes.</item>
/// <item>Decoders: a one-block, 16-wide flow and a 16-wide HiFT over 8 mel bands at 22.05 kHz, CAM++ of 8; the music decoder
/// too.</item>
/// <item>Generation: at most 40 TTS tokens and 6 chat tokens; the TTS head speaks audio codes only (see Speaking).</item>
/// </list>
/// The TTS speaks as a speaker made from half a second of tone.
/// </summary>
internal static class StepAudioFixture
{
    public const int SampleRate = 22050;

    private static readonly Lazy<string> TokenizerPath = new(() =>
    {
        const string name = "stepaudio_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        string model = Path.Combine(Path.GetTempPath(), $"aidotnet_step1_{System.Diagnostics.Process.GetCurrentProcess().Id}.model");
        File.WriteAllBytes(model, Convert.FromBase64String((string)JObject.Parse(File.ReadAllText(path))["model_base64"]!));
        return model;
    });

    public static StepAudioOptions Options() => new()
    {
        TokenizerPath = TokenizerPath.Value,
        HiddenDim = 16, NumHeads = 2, ChatGroups = 1, ChatLayers = 1, ChatIntermediate = 32,
        TtsWidth = 16, TtsHeads = 2, TtsGroups = 1, TtsLayers = 1, TtsIntermediate = 32,
        ParaformerMels = 8, ParaformerWidth = 16, ParaformerHeads = 2, ParaformerFeedForward = 32, ParaformerBlocks = 2, LinguisticDither = 0,
        SemanticWidth = 16, SemanticHeads = 2, SemanticLayers = 1, SemanticCodes = 16,
        FlowWidth = 16, FlowHeads = 2, FlowFeedForward = 32, FlowEncoderLayers = 1, DecoderChannels = new[] { 16 }, DecoderHeads = 2,
        DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2, MelChannelCount = 8,
        HiftBaseChannels = 16, HiftHarmonics = 2, SpeakerEmbeddingDimension = 8, SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 4,
        ChatMaxNewTokens = 6, PpoResponseLength = 6, LearningRate = 1e-2, TtsMaxNewTokens = 40,
        // Plain training samples are TTS fine-tuning steps (text and its recording).
        Stage = StepAudioTrainingStage.TtsFineTuning,
    };

    public static StepAudio<T> Tts<T>(StepAudioOptions? options = null)
    {
        var o = options ?? Options();
        var model = new StepAudio<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), o);
        Speaking(model, o);
        SpreadCodes(model);
        model.SpeakerVoice = model.CreateSpeaker(Tones<T>(0.5), SampleRate, "你好。", "Tingting");
        return model;
    }

    /// <summary>
    /// Makes the untrained TTS model speak, as a trained one does: its head rows for every token outside the audio codes
    /// (text, &lt;|EOT|&gt;) are zero and the audio rows N(0, 0.5²), so top-k sampling stays within the audio codes for
    /// the whole budget instead of ending after one token. An untrained head ends at once, and every text then sounds the
    /// same.
    /// </summary>
    private static void Speaking<T>(StepAudio<T> model, StepAudioOptions o)
    {
        int width = o.TtsWidth, vocabulary = o.VocabSize, first = StepAudioTokens.AudioBase, last = first + o.LinguisticCodes + o.SemanticCodes;
        var random = new Random(11);
        var weights = new double[vocabulary * width];
        for (int v = first; v < last; v++)
            for (int j = 0; j < width; j++)
                weights[v * width + j] = 0.5 * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(model.Core.Tts.LmHead!, width, vocabulary, weights);
    }

    /// <summary>
    /// Spreads the linguistic k-means centroids over the encoder's own frames of three seconds of chirp, as trained
    /// centroids cover real speech: centroid k is frame k mod n plus noise growing with k / n. Random centroids give every
    /// frame of any audio the same code, so different recordings would tokenize alike.
    /// </summary>
    private static void SpreadCodes<T>(StepAudio<T> model)
    {
        var audio = new double[3 * 16000];
        for (int i = 0; i < audio.Length; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * (120 + 300 * t) * t) + 0.2 * Math.Sin(2 * Math.PI * (700 - 150 * t) * t);
        }
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> frames;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) frames = model.Core.Linguistic.Encode(audio);
        int n = frames.Shape[0], width = frames.Shape[1], codes = model.Core.Centroids.Length / width;
        var random = new Random(13);
        var centroids = new double[codes * width];
        for (int k = 0; k < codes; k++)
            for (int j = 0; j < width; j++)
                centroids[k * width + j] = ops.ToDouble(frames[k % n, j]) + 0.05 * (k / n) * (2 * random.NextDouble() - 1);
        model.Core.Centroids = centroids;
    }

    /// <summary>A chirp and a tone at 22.05 kHz.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 140.0, double high = 420.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(SampleRate * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = (double)i / SampleRate;
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * low * (1 + t) * t) + 0.2 * Math.Sin(2 * Math.PI * high * t));
        }
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }
}
