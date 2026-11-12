using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class SeanetCheckpointReferenceTests
{
    [Fact]
    public void OriginalNamedCodecFile_ReproducesCodesAndWaveformIncludingShortAndUnalignedInputs()
    {
        string root = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData");
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(root, "seanet_checkpoint_reference.json")));
        var options = new EnCodecOptions
        {
            SampleRate = 16000, Filters = 2, Dimension = 8, Ratios = [2, 2], ResidualLayers = 1,
            ResidualKernelSizes = [3, 1], LstmLayers = 1, NumQuantizers = 3, CodebookSize = 8,
            TargetBandwidthKbps = 36, TargetBandwidths = [36], DiscriminatorWindows = [16],
            DiscriminatorFilters = 2, DiscriminatorDilations = [1], MelScales = [4], MelBins = 4
        };
        using var codec = new EnCodec<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        codec.LoadAuthorWeights(Path.Combine(root, "seanet_checkpoint_reference.safetensors"));
        // Loading must survive the public parameter surface's lazy seed/materialization pass.
        _ = codec.GetParameters();
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            var samples = item.GetProperty("audio").EnumerateArray().Select(value => value.GetDouble()).ToArray();
            var codes = codec.Encode(new Tensor<double>(new[] { samples.Length }, new Vector<double>(samples)));
            Assert.Equal(item.GetProperty("frames").GetInt32(), codes.GetLength(1));
            Assert.Equal(3, codes.GetLength(0));
            Assert.Equal(item.GetProperty("codes").EnumerateArray().Select(value => value.GetInt32()).ToArray(), codes.Cast<int>().ToArray());
            var waveform = codec.Decode(codes).ToArray();
            var expected = item.GetProperty("waveform").EnumerateArray().Select(value => value.GetDouble()).ToArray();
            Assert.Equal(expected.Length, waveform.Length);
            double error = waveform.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 2e-6, $"Input length {samples.Length}: waveform error {error:R}.");
        }
    }
}
