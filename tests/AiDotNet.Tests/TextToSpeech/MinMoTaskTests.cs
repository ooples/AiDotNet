using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>MinMo's tasks and training stages on the tiny fixture, and its input projector against FunASR's adaptor
/// (<c>tools/reference-data/minmo_projector_reference.py</c>).</summary>
public class MinMoTaskTests
{
    // double.IsFinite is not on .NET Framework.
    private static bool Finite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);

    [Fact(Timeout = 120000)]
    public async Task InputProjector_MatchesFunAsr()
    {
        await Task.Yield();
        const string name = "minmo_projector_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var projector = new MinMoInputProjector<double>(new CpuEngine(), new List<LayerBase<double>>(), 6, 2, 10, 8, 2, 2, new Random(0));
        var used = new HashSet<string>(StringComparer.Ordinal);
        projector.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        }, "");
        Assert.Empty(state.Keys.Where(k => !used.Contains(k)));
        var output = projector.Forward(new Tensor<double>(new[] { 7, 6 }, new Vector<double>(Doubles(f["input"]!)))).ToArray();
        var expected = Doubles(f["output"]!);
        Assert.Equal(expected.Length, output.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - output[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"[{i}] {expected[i]:R} vs {output[i]:R}");
    }

    [Fact(Timeout = 300000)]
    public async Task Understand_AndSpeak_RunEveryPart()
    {
        await Task.Yield();
        using var model = MinMoFixture.Tts<double>();
        var transcript = model.Transcribe(MinMoFixture.Tones<double>(0.5));
        Assert.NotNull(transcript);
        var (audio, text) = model.Speak("hello there");
        Assert.NotNull(text);
        Assert.True(audio.Length > 0 || text.Length == 0);
        // The same input gives the same output.
        Assert.Equal(audio.ToArray(), model.Speak("hello there").Audio.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task SenseVoiceSmall_EncoderOption_Runs()
    {
        await Task.Yield();
        using var model = MinMoFixture.Tts<double>(MinMoFixture.Options(MinMoVoiceEncoder.SenseVoiceSmall));
        Assert.True(model.SpeechPositions(MinMoFixture.Tones<double>(0.5)) > 0);
        Assert.NotNull(model.Transcribe(MinMoFixture.Tones<double>(0.5)));
    }

    [Fact(Timeout = 900000)]
    public async Task EveryStage_TrainsItsModulesAndLowersTheLoss()
    {
        await Task.Yield();
        var speech = MinMoFixture.Tones<double>(0.4);
        var answer = MinMoFixture.Tones<double>(1.2, 200, 300);
        foreach (MinMoTrainingStage stage in Enum.GetValues(typeof(MinMoTrainingStage)))
        {
            var options = MinMoFixture.Options();
            options.Stage = stage;
            using var model = MinMoFixture.Tts<double>(options);
            Func<double> step = stage switch
            {
                MinMoTrainingStage.PreAlign or MinMoTrainingStage.FullAlign or MinMoTrainingStage.InstructionTuning =>
                    () => model.TrainSpeechToText(speech, "Speech Transcription", "a tone"),
                MinMoTrainingStage.TextToSpeechProjector or MinMoTrainingStage.TextToSpeech =>
                    () => model.TrainTextToSpeech("a tone", answer),
                MinMoTrainingStage.SpeechToSpeech => () => model.TrainSpeechToSpeech(speech, "a tone", answer),
                _ => () => model.TrainDuplex(speech, Enumerable.Range(0, model.SpeechPositions(speech)).Select(i => i % 3).ToArray()),
            };
            double first = step(), last = first;
            for (int i = 0; i < 4; i++) last = step();
            Assert.True(Finite(first) && Finite(last), $"{stage}: {first} → {last}");
            Assert.True(last < first, $"{stage}: {first} → {last}");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Stages_RefuseTheWrongTask()
    {
        await Task.Yield();
        using var model = MinMoFixture.Tts<double>();
        Assert.Throws<InvalidOperationException>(() => model.TrainDuplex(MinMoFixture.Tones<double>(0.4), new[] { 0 }));
    }

    [Fact(Timeout = 300000)]
    public async Task DuplexActions_OnePerSpeechPosition()
    {
        await Task.Yield();
        using var model = MinMoFixture.Tts<double>();
        var speech = MinMoFixture.Tones<double>(0.5);
        Assert.Equal(model.SpeechPositions(speech), model.DuplexActions(speech).Length);
    }
}
