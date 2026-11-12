using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class XttsMelReferenceTests
{
    [Fact]
    public void ConditioningAndDvaeMels_MatchOriginalHtkSpacingAndSlaneyNormalization()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_mel_reference.json")));
        var root = document.RootElement;
        using var model = new XTTSv2<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), XTTSv2PaperTests.SmallOptions());
        model.SetMelStats(root.GetProperty("norms").EnumerateArray().Select(x => x.GetDouble()).ToArray());
        var transform = typeof(XTTSv2<double>).GetMethod("PowerMel", BindingFlags.NonPublic | BindingFlags.Instance)!;
        foreach (var item in root.GetProperty("cases").EnumerateArray())
        {
            var values = item.GetProperty("audio").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            var audio = new Tensor<double>(new[] { values.Length }, new AiDotNet.Tensors.LinearAlgebra.Vector<double>(values));
            var actual = (Tensor<double>)transform.Invoke(model, new object[] { audio,
                item.GetProperty("fft").GetInt32(), item.GetProperty("hop").GetInt32(), item.GetProperty("window").GetInt32() })!;
            Assert.Equal(new[] { 80, item.GetProperty("frames").GetInt32() }, actual.Shape.ToArray());
            var expected = item.GetProperty("mel").EnumerateArray().Select(x => x.GetDouble()).ToArray();
            double error = actual.ToArray().Zip(expected, (a,b) => Math.Abs(a-b)).Max();
            Assert.True(error < 2e-8, $"Original XTTS mel differs by {error:R} for FFT {item.GetProperty("fft").GetInt32()}.");
        }
    }
}
