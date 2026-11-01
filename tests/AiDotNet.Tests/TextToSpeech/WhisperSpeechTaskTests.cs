using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>WhisperSpeech's tasks on the tiny fixture: synthesis, the token paths, and a training step of each stage.</summary>
public class WhisperSpeechTaskTests
{
    [Fact(Timeout = 300000)]
    public async Task Synthesize_GivesAudioOfWholeCodecFrames()
    {
        await Task.Yield();
        using var model = WhisperSpeechFixture.Tts<double>();
        var audio = model.Synthesize("hello");
        Assert.Equal(0, audio.Length % 2560);
        Assert.All(audio.ToArray(), v => Assert.True(double.IsFinite(v)));
        // The same input gives the same output (a fresh sampler per call).
        Assert.Equal(audio.ToArray(), model.Synthesize("hello").ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Tokens_RoundTripTheirShapes()
    {
        await Task.Yield();
        using var model = WhisperSpeechFixture.Tts<double>();
        var recording = WhisperSpeechFixture.Tones<double>(0.32);
        var semantic = model.SemanticTokens(recording, WhisperSpeechFixture.SampleRate);
        // 0.32 s at 16 kHz is 5120 samples: 33 mel frames cut to the 32-frame window, eight tokens.
        Assert.Equal(8, semantic.Length);
        Assert.All(semantic, t => Assert.InRange(t, 0, 7));
        // The S2A input holds a start token before the semantic tokens, so a window takes one fewer (as in the release).
        var acoustic = model.SemanticToAcousticTokens(semantic.Take(7).ToArray());
        Assert.Equal(4, acoustic.Length);
        Assert.All(acoustic, row => Assert.Equal(7 * 3 - 4, row.Length));
        var codes = model.EncodeToTokens(WhisperSpeechFixture.Tones<double>(0.32));
        Assert.Equal(new[] { 4, 3 }, codes.Shape.ToArray());
        Assert.Equal(3 * 2560, model.DecodeFromTokens(codes).Length);
    }

    [Theory(Timeout = 600000)]
    [InlineData(WhisperSpeechTrainingStage.TextToSemantic)]
    [InlineData(WhisperSpeechTrainingStage.SemanticToAcoustic)]
    [InlineData(WhisperSpeechTrainingStage.SemanticTokenizer)]
    public async Task EachStage_TrainsOnlyItsModel_AndLowersItsLoss(WhisperSpeechTrainingStage stage)
    {
        await Task.Yield();
        var options = WhisperSpeechFixture.Options();
        options.Stage = stage;
        using var model = WhisperSpeechFixture.Tts<double>(options);
        var recording = WhisperSpeechFixture.Tones<double>(0.32);
        var sample = new AiDotNet.TextToSpeech.TtsTrainingSample<double>
        {
            Tokens = AiDotNet.TextToSpeech.Latest.IndexTtsOps.Ids<double>(System.Text.Encoding.UTF8.GetBytes("a tone").Select(b => (int)b).ToArray()),
            Audio = recording,
        };
        var core = model.Core;
        double[] Snapshot(System.Collections.Generic.IEnumerable<AiDotNet.NeuralNetworks.Layers.LayerBase<double>> layers) =>
            layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
        var whisper = Snapshot(model.CoreLayers.Whisper);
        var others = stage switch
        {
            WhisperSpeechTrainingStage.TextToSemantic => model.CoreLayers.SemanticToAcoustic,
            _ => model.CoreLayers.TextToSemantic,
        };
        var untouched = Snapshot(others);
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 6; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"{stage}: {before} -> {after}");
        Assert.Equal(whisper, Snapshot(model.CoreLayers.Whisper));
        Assert.Equal(untouched, Snapshot(others));
        // The T2S and S2A semantic tables stay the tokenizer's codebook.
        var codebook = core.Tokenizer.Embed[0].Weights.ToArray();
        if (stage != WhisperSpeechTrainingStage.SemanticTokenizer)
            Assert.Equal(codebook, core.TextToSemantic.SemanticEmbeddings.Main.Weights.ToArray());
    }

    [Fact]
    public void OneCycle_MatchesTorch()
    {
        // torch.optim.lr_scheduler.OneCycleLR(max_lr=1, total_steps=100, pct_start=0.3, div_factor=25, final_div_factor=25)'s
        // first, peak and last learning rates and momenta.
        var start = WhisperSpeech<double>.OneCycle(0, 100, 30);
        Assert.Equal(0.04, start.Factor, 12);
        Assert.Equal(0.95, start.Beta1, 12);
        var peak = WhisperSpeech<double>.OneCycle(29, 100, 30);
        Assert.Equal(1.0, peak.Factor, 12);
        Assert.Equal(0.85, peak.Beta1, 12);
        var end = WhisperSpeech<double>.OneCycle(99, 100, 30);
        Assert.Equal(0.04 / 25, end.Factor, 12);
        Assert.Equal(0.95, end.Beta1, 12);
    }
}
