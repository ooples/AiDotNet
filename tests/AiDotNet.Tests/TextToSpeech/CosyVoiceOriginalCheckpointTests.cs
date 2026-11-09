using System.Reflection;
using System.Text.Json;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class CosyVoiceOriginalCheckpointTests
{
    [Theory]
    [InlineData("pth")]
    [InlineData("safetensors")]
    public void OriginalLegacyWeightNorm_NamesResolveWithoutChangingValuesAndRejectWrongSizes(string extension)
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "cosyvoice_legacy_weightnorm_reference.json")));
        Action<Func<string, int[], double[]>> verify = reader =>
        {
            foreach (var pair in new[] { ("conv_pre.weight_g", "conv_pre.parametrizations.weight.original0"),
                ("conv_pre.weight_v", "conv_pre.parametrizations.weight.original1"), ("conv_pre.bias", "conv_pre.bias") })
            {
                var shape = document.RootElement.GetProperty("shapes").GetProperty(pair.Item1).EnumerateArray().Select(x => x.GetInt32()).ToArray();
                var expected = document.RootElement.GetProperty("weights").GetProperty(pair.Item1).EnumerateArray().Select(x => x.GetDouble()).ToArray();
                Assert.Equal(expected, reader(pair.Item2, shape));
            }
            Assert.Throws<InvalidDataException>(() => reader("conv_pre.parametrizations.weight.original1", [3, 2, 4]));
            Assert.Throws<InvalidDataException>(() => reader("conv_pre.parametrizations.weight.original1", [3, 3, 2]));
            Assert.Throws<InvalidDataException>(() => reader("does_not_exist.weight", [1]));
        };
        typeof(CosyVoiceBase<double>).GetMethod("LoadCheckpoint", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, new object[] { Path.Combine(directory, "cosyvoice_legacy_weightnorm_reference." + extension), verify });
    }
}
