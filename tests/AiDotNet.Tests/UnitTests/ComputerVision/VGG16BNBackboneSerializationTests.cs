using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// The native backbone format writes each layer's GetParameters(). A batch-normalized backbone is only restored
/// faithfully if that vector also carries the running mean and variance used at inference; otherwise the reloaded
/// backbone normalizes with default statistics and produces different feature maps.
/// </summary>
public class VGG16BNBackboneSerializationTests
{
    private static Tensor<double> Image(int seed)
    {
        var tensor = new Tensor<double>(new[] { 1, 3, 32, 32 });
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ((i * 31 + seed * 17) % 97) / 97.0 - 0.5;
        return tensor;
    }

    [Fact]
    public void RoundTrip_RestoresBatchNormRunningStatistics()
    {
        var source = new VGG16BNBackbone<double>();
        // Training-mode forwards move every BN layer's running statistics away from their defaults.
        source.SetTrainingMode(true);
        for (int step = 0; step < 3; step++)
            source.ExtractFeatures(Image(step));
        source.SetTrainingMode(false);

        var probe = Image(42);
        var expected = source.ExtractFeatures(probe);

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            source.WriteParameters(writer);
        stream.Position = 0;

        var copy = new VGG16BNBackbone<double>();
        copy.ExtractFeatures(probe); // materialize lazily sized weights before loading into them
        using (var reader = new BinaryReader(stream))
            copy.ReadParameters(reader);
        copy.SetTrainingMode(false);
        var actual = copy.ExtractFeatures(probe);

        Assert.Equal(expected.Count, actual.Count);
        for (int level = 0; level < expected.Count; level++)
            Assert.Equal(expected[level].ToArray(), actual[level].ToArray());
    }
}
