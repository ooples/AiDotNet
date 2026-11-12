using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// A tiny Moshi ready to run, every part shrunk:
/// <list type="bullet">
/// <item>Text: the SpiritLM fixture's 96-piece SentencePiece model, PAD the spare id 96 and EPAD <c>&lt;unk&gt;</c> (0), as
/// Helium's EPAD is.</item>
/// <item>RQ-Transformer: a one-layer, 16-wide Temporal and Depth Transformer over 2 codebooks of 8 per speaker, the
/// Depth Transformer predicting both speakers (the paper's layout).</item>
/// <item>Mimi: 1600 Hz, SEANet ratios 4·2 (100 frames a second), width 16, one Transformer layer per side, 4 codebooks of
/// 8 in 8 dimensions, k-means-initialized on a tone so different audio gives different codes.</item>
/// <item>Delays: 0.05 s (5 frames) between text and audio for the TTS and ASR variants.</item>
/// <item>Training: both Transformers at 1e-2 without weight decay, so a few steps move a tiny model.</item>
/// </list>
/// </summary>
internal static class MoshiFixture
{
    public const int SampleRate = 1600;

    public static MoshiOptions Options(MoshiVariant variant = MoshiVariant.Dialogue) => new()
    {
        SampleRate = SampleRate, HiddenDim = 16, NumHeads = 2, Layers = 1, FeedForward = 48, Context = 96,
        DepthDim = 16, DepthHeads = 2, DepthLayers = 1, DepthFeedForward = 48,
        Codebooks = 2, CodebookSize = 8, VocabSize = SpiritLmFixture.VocabularySize + 1, TextPadding = SpiritLmFixture.VocabularySize,
        TextEndPadding = 0, TokenizerPath = SpiritLmFixture.TinyTokenizerPath,
        MimiFilters = 4, MimiDimension = 16, MimiTransformerLayers = 1, MimiCodebooks = 4, MimiRatios = new[] { 4, 2 }, MimiHeads = 2,
        MimiFeedForward = 32, MimiContext = 6, MimiCodebookDim = 8,
        WavLmWidth = 16, WavLmLayers = 1, WavLmHeads = 2, WavLmConvChannels = 8, MimiDiscriminatorWindows = new[] { 256, 128 },
        MimiDiscriminatorFilters = 4,
        TextAudioDelaySeconds = 0.05, Variant = variant, WarmupSteps = 0, TotalSteps = 1000, LearningRate = 1e-2, DepthLearningRate = 1e-2,
        WeightDecay = 0.0,
    };

    private static NeuralNetworkArchitecture<T> Architecture<T>() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    public static Moshi<T> Tts<T>(MoshiOptions? options = null)
    {
        var model = new Moshi<T>(Architecture<T>(), options ?? Options(MoshiVariant.Tts));
        SpreadCodes(model.Core);
        return model;
    }

    public static AiDotNet.SpeechRecognition.Streaming.Moshi<T> Asr<T>(MoshiOptions? options = null)
    {
        var model = new AiDotNet.SpeechRecognition.Streaming.Moshi<T>(Architecture<T>(), options ?? Options(MoshiVariant.Asr));
        SpreadCodes(model.Core);
        return model;
    }

    /// <summary>
    /// Initializes Mimi's codebooks as training does, by k-means on the residuals of a chirp, one more codebook each pass
    /// (a codebook initializes only once the one before it has). A fresh Mimi's empty codebooks give every frame the same
    /// codes.
    /// </summary>
    internal static void SpreadCodes<T>(MoshiCore<T> core)
    {
        var audio = Tones<T>(1.0);
        var wave = new Tensor<T>(new[] { 1, 1, audio.Length }, audio.ToVector());
        var random = new Random(3);
        int acoustic = core.Mimi.Configuration.Codebooks - core.Mimi.Configuration.SemanticCodebooks;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        for (int pass = 0; pass <= core.Mimi.Configuration.Codebooks; pass++)
            core.Mimi.TrainingForward(wave, acoustic, true, true, random);
    }

    /// <summary>A chirp and a tone at the fixture's rate, <paramref name="seconds"/> long.</summary>
    public static Tensor<T> Tones<T>(double seconds, double low = 90.0, double high = 330.0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int samples = (int)(SampleRate * seconds);
        var audio = new T[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = (double)i / SampleRate;
            double f = low * (1 + t);
            audio[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * f * t) + 0.2 * Math.Sin(2 * Math.PI * high * t));
        }
        return new Tensor<T>(new[] { samples }, new Vector<T>(audio));
    }

    /// <summary>A word list spread evenly over <paramref name="seconds"/>.</summary>
    public static MoshiWord[] Words(string text, double seconds)
    {
        var words = text.Split(' ');
        double each = seconds / words.Length;
        return words.Select((w, i) => new MoshiWord(w, i * each, (i + 1) * each)).ToArray();
    }
}
