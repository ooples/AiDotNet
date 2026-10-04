using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// A seeded detector leaves the thread's layer-initialization scope armed after construction. A standalone backbone
/// built afterwards must not draw its seed from that leftover scope: it was not asked to be reproducible, and its
/// weights would otherwise depend on whichever detector happened to be built before it on the thread.
/// </summary>
public class BackboneSeedScopeTests
{
    private static double[] StandaloneBackboneAfterSeededDetector()
    {
        _ = new CRAFT<double>(new TextDetectionOptions<double> { InputSize = new[] { 32, 32 }, Size = ModelSize.Nano, RandomSeed = 5 });
        var backbone = new VGG16BNBackbone<double>();
        var image = new Tensor<double>(new[] { 1, 3, 32, 32 });
        backbone.ExtractFeatures(image); // materialize the lazily sized weights
        return backbone.GetParameters().ToArray();
    }

    [Fact]
    public void StandaloneBackbone_DoesNotInheritAFinishedDetectorsSeed()
    {
        var first = StandaloneBackboneAfterSeededDetector();
        var second = StandaloneBackboneAfterSeededDetector();
        // With the leak both backbones drew the same seed from detector 5's leftover scope and came out identical.
        Assert.NotEqual(first, second);
    }
}
