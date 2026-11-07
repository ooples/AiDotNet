using System.Text.Json;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceBandFilterTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(32)]
    public void JuliusFiltersAndAuthorEqualization_MatchShortBatchedStereoAndBothGradients(int count)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_band_filters.json")));
        var reference = document.RootElement.GetProperty("cases").EnumerateArray().Single(x => x.GetProperty("bands").GetInt32() == count);
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var filters = new MetaVoiceBandFilters<double>(AiDotNetEngine.Current, count);
        Assert.Equal(reference.GetProperty("filter_shape").EnumerateArray().Select(x => x.GetInt32()), filters.TorchFilterShape);
        Tensor<double> Tensor(string name) => new(new[] { 2, 2, 37 }, new Vector<double>(Values(reference.GetProperty(name))));
        var waveform = Tensor("waveform"); var target = Tensor("reference"); var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var split = filters.Split(waveform);
        for (int i = 0; i < count; i++) Close(split[i], Values(reference.GetProperty("split")).Skip(i * waveform.Length).Take(waveform.Length).ToArray(), "split " + i);
        var sum = split.Aggregate((a, b) => engine.TensorAdd(a, b));
        Close(sum, waveform.ToArray(), "telescoping reconstruction");
        var equalized = filters.MatchEqualization(waveform, target, .8);
        Close(equalized, Values(reference.GetProperty("equalized")), "author re_eq");
        var loss = engine.ReduceSum(engine.TensorMultiply(equalized, Tensor("weights")), new[] { 0, 1, 2 }, true);
        var bandWeights = Values(reference.GetProperty("band_weights"));
        for (int i = 0; i < count; i++)
        {
            var weights = new Tensor<double>(waveform.Shape.ToArray(), new Vector<double>(bandWeights.Skip(i * waveform.Length).Take(waveform.Length).ToArray()));
            loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(split[i], weights), new[] { 0, 1, 2 }, true), .13));
        }
        Close(loss, new[] { reference.GetProperty("loss").GetDouble() }, "objective");
        var gradients = tape.ComputeGradients(engine.Reshape(loss, new[] { 1 }), sources: new[] { waveform, target });
        Close(gradients[waveform], Values(reference.GetProperty("waveform_gradient")), "audio gradient");
        Close(gradients[target], Values(reference.GetProperty("reference_gradient")), "reference gradient");
    }

    private static void Close(Tensor<double> actual, double[] expected, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 1e-8, $"Julius {name} error {error:R}.");
    }
}
