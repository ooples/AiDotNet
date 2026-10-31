using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny GLM-4-Voice ready to run. The tokenizer is a tiktoken model of the 256 single bytes (no merges), with
/// GLM-4's 18 named special tokens and the speech tokens after them in the release's order. The language model is
/// 16 wide with 2 layers. The speech tokenizer is 16 wide with the pooling and quantizer (12 codes) after layer 2. The
/// decoder's flow is 8 wide with 2 Euler steps and its HiFT is 16 wide.
/// </summary>
internal static class GLM4VoiceFixture
{
    public const int Codes = 12;

    private static readonly string[] Specials =
    {
        "<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
        "<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>", "<|begin_of_audio|>", "<|end_of_audio|>",
        "<|begin_of_transcription|>", "<|end_of_transcription|>",
    };

    public static int VocabularySize => 256 + Specials.Length + Codes;

    private static readonly Lazy<string> TokenizerDirectory = new(() =>
    {
        string directory = Path.Combine(Path.GetTempPath(), $"aidotnet_glm4voice_tokenizer_{System.Diagnostics.Process.GetCurrentProcess().Id}");
        Directory.CreateDirectory(directory);
        File.WriteAllLines(Path.Combine(directory, "tokenizer.model"),
            Enumerable.Range(0, 256).Select(b => Convert.ToBase64String(new[] { (byte)b }) + " " + b));
        var added = new JObject();
        var names = Specials.Concat(Enumerable.Range(0, Codes).Select(c => $"<|audio_{c}|>")).ToList();
        for (int i = 0; i < names.Count; i++) added[(256 + i).ToString(System.Globalization.CultureInfo.InvariantCulture)] = new JObject { ["content"] = names[i] };
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), new JObject { ["added_tokens_decoder"] = added }.ToString());
        return directory;
    });

    public static GLM4VoiceOptions Options() => new()
    {
        TokenizerDirectory = TokenizerDirectory.Value,
        HiddenDim = 16, Layers = 2, NumHeads = 2, KeyValueGroups = 1, HeadDim = 8, FeedForwardDim = 32, VocabSize = VocabularySize,
        MaxTextLength = 1024, Stage = GlmVoiceTrainingStage.PreTraining, DropoutRate = 0.0, PreTrainingLearningRate = 1e-2,
        SpeechTokenizerDim = 16, SpeechTokenizerHeads = 2, SpeechTokenizerFeedForwardDim = 32, SpeechTokenizerLayers = 4,
        SpeechTokenizerMaxPositions = 256, PoolingPosition = 2, QuantizePosition = 2, SpeechCodebookSize = Codes, CausalBlockSize = 8,
        AsrDecoderLayers = 2, AsrVocabulary = 40, AsrMaxTargetPositions = 16, AsrDecoderStartId = 1,
        FlowDim = 8, FlowHeads = 2, FlowFeedForwardDim = 16, FlowLayers = 1, DecoderChannels = new[] { 8, 8 }, DecoderHeads = 2,
        DecoderHeadDim = 4, DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2, HiftBaseChannels = 16, HiftF0Channels = 16,
        MaxNewTokens = 24,
    };

    /// <summary>Two tones at <paramref name="rate"/> Hz, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, int rate, double low = 180.0, double high = 470.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(rate * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * low * i / rate) + 0.2 * Math.Sin(2 * Math.PI * high * i / rate));
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }

    public static GLM4Voice<T> Tts<T>(GLM4VoiceOptions? options = null)
    {
        var model = new GLM4Voice<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            options ?? Options());
        SpreadCodebook(model);
        return model;
    }

    /// <summary>
    /// Makes the untrained tokenizer read the audio, as a trained one does. At random initialization the tiny encoder's
    /// convolutions are so small against Whisper's sinusoidal positions that a frame's code depends on its position
    /// only; they are scaled 40×. The codebook is then spread over the encoder's states at evenly spaced frames of a
    /// second of noisy tones.
    /// </summary>
    private static void SpreadCodebook<T>(GLM4Voice<T> model)
    {
        var tokenizer = model.SpeechTokenizer;
        var scale = MathHelper.GetNumericOperations<T>().FromDouble(40);
        foreach (var conv in new[] { tokenizer.Conv1, tokenizer.Conv2 })
        {
            var parameters = conv.GetParameters();
            for (int i = 0; i < parameters.Length; i++) parameters[i] = MathHelper.GetNumericOperations<T>().Multiply(parameters[i], scale);
            conv.SetParameters(parameters);
        }
        var random = new Random(7);
        var audio = Enumerable.Range(0, 16000).Select(i => 0.3 * (2 * random.NextDouble() - 1) + 0.3 * Math.Sin(2 * Math.PI * (150 + i / 40.0) * i / 16000)).ToArray();
        var (features, valid) = tokenizer.Features(audio);
        GlmSpeechTokenizer<T>.EncoderOutput output;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) output = tokenizer.Encode(features, valid);
        var ops = MathHelper.GetNumericOperations<T>();
        var states = output.PreQuantized.ToArray();
        int rows = output.Codes.Length, width = states.Length / rows;
        var codebook = new double[Codes * width];
        for (int j = 0; j < Codes; j++)
            for (int i = 0; i < width; i++) codebook[j * width + i] = ops.ToDouble(states[j * rows / Codes * width + i]);
        AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(tokenizer.Codebook, 1, codebook.Length, codebook);
        Array.Copy(codebook, tokenizer.EmaWeight, codebook.Length);
    }
}
