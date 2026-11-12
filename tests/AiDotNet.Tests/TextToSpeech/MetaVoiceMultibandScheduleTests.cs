using System.Text.Json;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceMultibandScheduleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AuthorSubsampledSampler_MatchesEveryIterateDrawAndFinalBoundary(bool custom)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_multiband_schedule.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray()
            .Single(x => x.GetProperty("custom").GetBoolean() == custom);
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        Tensor<double> Tensor(JsonElement value) => new(new[] { 2, 1, 19 }, new Vector<double>(Values(value)));
        var engine = AiDotNetEngine.Current;
        var releasedBetas = MetaVoiceDiffusionBetas.CreateReleaseSchedule();
        Assert.Equal(Values(reference.GetProperty("betas")), releasedBetas);
        var schedule = new MetaVoiceDiffusionSchedule<double>(engine, releasedBetas);
        var visited = new List<int>(); int drawCount = 0;
        var output = schedule.Generate(Tensor(reference.GetProperty("initial")), Tensor(reference.GetProperty("condition")),
            (current, step, condition) =>
            {
                visited.Add(step);
                return engine.TensorAddScalar(engine.TensorAdd(engine.TensorMultiplyScalar(current, .17),
                    engine.TensorMultiplyScalar(condition, .23)), step * .0001);
            }, index =>
            {
                Assert.Equal(drawCount++, index);
                return Tensor(reference.GetProperty("draws")[index]);
            }, custom ? reference.GetProperty("steps").EnumerateArray().Select(x => x.GetInt32()).ToArray() : null,
            reference.GetProperty("scale").GetDouble(), reference.GetProperty("clip").GetDouble());
        Assert.Equal(reference.GetProperty("visited").EnumerateArray().Select(x => x.GetInt32()), visited);
        Assert.DoesNotContain(0, visited);
        Assert.Equal(reference.GetProperty("draws").GetArrayLength(), drawCount);
        var expected = reference.GetProperty("matched_float64_trajectory");
        Assert.Equal(expected.GetArrayLength(), output.Count);
        for (int i = 0; i < output.Count; i++)
        {
            double error = output[i].ToArray().Zip(Values(expected[i]), (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 1e-10, $"MBD iterate {i} error {error:R}.");
        }
    }
}
