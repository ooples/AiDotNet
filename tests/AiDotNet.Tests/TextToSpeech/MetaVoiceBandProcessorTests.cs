using System.Text.Json;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceBandProcessorTests
{
    [Theory]
    [InlineData(2, true)]
    [InlineData(2, false)]
    [InlineData(3, true)]
    [InlineData(3, false)]
    public void OriginalBandProcessor_MatchesTrainedBuffersOnlineFitAndInputGradient(int band, bool trained)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_band_processor.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(x =>
            x.GetProperty("band").GetInt32() == band && x.GetProperty("trained").GetBoolean() == trained);
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        Tensor<double> Tensor(string key) => new(new[] { 2, 2, 37 }, new Vector<double>(Values(reference.GetProperty(key))));
        var engine = AiDotNetEngine.Current;
        var processor = new MetaVoiceBandProcessor<double>(engine, reference.GetProperty("power").GetDouble());
        if (trained)
        {
            var consumed = new HashSet<string>();
            processor.LoadTorch((name, shape) =>
            {
                Assert.True(consumed.Add(name));
                var entry = reference.GetProperty("state").GetProperty(name);
                Assert.Equal(shape, entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
                return Values(entry.GetProperty("values"));
            });
            Assert.Equal(5, consumed.Count);
        }
        var waveform = Tensor("waveform"); int draws = 0;
        using var tape = new GradientTape<double>();
        var projected = processor.ProjectSample(waveform, () => { draws++; return Tensor("gaussian"); });
        var returned = processor.ReturnSample(waveform);
        Close(projected, Values(reference.GetProperty("projected")), "projected");
        Close(returned, Values(reference.GetProperty("returned")), "returned");
        Assert.Equal(reference.GetProperty("draws").GetInt32(), draws);
        Assert.Equal(reference.GetProperty("final_count").GetDouble(), processor.FittedCount);
        var loss = engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(projected, Tensor("weights")), new[] { 0, 1, 2 }, true),
            engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(returned, Tensor("return_weights")), new[] { 0, 1, 2 }, true), .19));
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(engine.Reshape(loss, new[] { 1 }), sources: new[] { waveform });
        Close(gradients[waveform], Values(reference.GetProperty("gradient")), "gradient");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"MBD processor {name} error {error:R}.");
    }
}
