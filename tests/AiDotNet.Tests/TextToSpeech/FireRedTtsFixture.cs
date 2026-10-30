using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The generated suite's FireRedTTS: every network at width 8-64 with one layer or block, two sub-codebooks of 8 codes
/// (64 semantic tokens), the release's text vocabulary (its tokenizer is bundled), and a GPT taught to speak.
/// </summary>
/// <remarks>An untrained GPT (GPT-2's N(0, 0.02) initialization, as the release's) barely reads its text: two texts gave the
/// same tokens under every decoding setting tried. So the GPT is first trained, with the model's own training step, on two
/// texts (the suite's sentences) paired with different token sequences, 200 steps at 1e-2, once per process; each fixture copies the trained
/// parameters into a fresh model. Decoding keeps to semantic tokens, samples two candidates of at most eight tokens at
/// temperature 0.1 and runs the flow for two steps.</remarks>
internal static class FireRedTtsFixture
{
    internal static FireRedTTSOptions Options() => new()
    {
        HiddenDim = 16, NumHeads = 2, NumDecoderLayers = 1, MaxTextTokens = 64, MaxTextLength = 60, MaxAudioTokens = 40,
        CodebookSize = 8, CodebookDim = 4, AudioTokens = 66, ReferenceHidden = 8, SpeakerDim = 32, TokenizerDim = 32,
        SpeakerChannels = new[] { 8, 8, 8, 8, 24 }, SpeakerAttentionChannels = 4, SpeakerSeChannels = 4,
        FlowDim = 8, FlowHeads = 2, FlowLinearUnits = 16, FlowBlocks = 1, FlowCrossAttentionBlocks = 1, FlowRegulatorBlocks = 1,
        PromptEncoderHidden = 8, PromptEncoderLayers = 1, PromptEncoderBlocks = 1, FlowDecoderChannels = new[] { 8, 8 },
        FlowDecoderHeads = 2, FlowDecoderHeadDim = 4, FlowDecoderBlocksPerLevel = 1, FlowDecoderMidBlocks = 1, FlowSteps = 2,
        VocoderInitialChannels = 64, VocoderResBlockKernels = new[] { 3 }, VocoderResBlockDilations = new[] { new[] { 1, 3, 5 } },
        SslDim = 16, HubertDim = 16, HubertLayers = 1, HubertHeads = 2, HubertIntermediate = 32,
        HubertConvDims = new[] { 8, 8, 8, 8, 8, 8, 8 }, HubertPositionalKernel = 16, HubertPositionalGroups = 2,
        CodecDim = 32, CodecConvGroups = 4, AcousticLmDim = 16, AcousticLmHeads = 2, AcousticLmLayers = 1, AcousticLmFeedForward = 32,
        MaxNewTokens = 8, Candidates = 2, ConstrainToSemanticTokens = true, Temperature = 0.1, LearningRate = 1e-3,
    };

    private static FireRedTTS<T> Build<T>(double? learningRate = null)
    {
        var options = Options();
        if (learningRate is { } rate) options.LearningRate = rate;
        return new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1),
            options);
    }

    // The suite's two-tone second of audio at 24 kHz.
    private static Tensor<T> Recording<T>()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var recording = new Tensor<T>(new[] { 24000 });
        for (int i = 0; i < 24000; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 24000) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 24000));
        return recording;
    }

    private static readonly Lazy<double[]> Trained = new(() =>
    {
        // Taught at 1e-2 (the suite's own steps run at the fixture's 1e-3).
        var model = Build<double>(1e-2);
        var reference = Recording<double>();
        var pairs = new[]
        {
            (Text: "the quick brown fox", Tokens: new[] { 3, 17, 29, 41, 53, 5, 11, 23 }),
            (Text: "a completely different sentence", Tokens: new[] { 60, 48, 36, 24, 12, 58, 46, 34 }),
        };
        var samples = pairs.Select(p =>
        {
            var codes = new Tensor<double>(new[] { p.Tokens.Length, 1 });
            for (int i = 0; i < p.Tokens.Length; i++) codes[i, 0] = p.Tokens[i];
            var ids = model.TokenizeText(p.Text);
            return new TtsTrainingSample<double>
            {
                Tokens = new Tensor<double>(new[] { ids.Length }, new Vector<double>(ids.Select(v => (double)v).ToArray())),
                CodecTokens = codes,
                SpeakerReference = reference,
            };
        }).ToArray();
        for (int step = 0; step < 200; step++) model.Train(samples[step % 2]);
        return model.GetParameters().ToArray();
    });

    public static FireRedTTS<T> Create<T>()
    {
        var model = Build<T>();
        var ops = MathHelper.GetNumericOperations<T>();
        model.SetParameters(new Vector<T>(Trained.Value.Select(v => ops.FromDouble(v)).ToArray()));
        return model;
    }
}
