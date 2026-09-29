using System.IO;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Invariants for YOLO11's PAN-FPN neck. The neck has no options-only constructor the scaffold generator
/// can build, so these are its tests (AIDN040 requires every model to have one).
/// </summary>
public sealed class YOLOv11NeckTests
{
    public YOLOv11NeckTests() => TestModuleInitializer.EnsureInitialized();

    private static List<Tensor<double>> Features(int seed)
    {
        LayerInitializationSeedScope.ResetForModelConstruction(seed);
        var backbone = new YOLOv11Backbone<double>(ModelSize.Nano);
        var image = new Tensor<double>(new[] { 1, 3, 64, 64 });
        for (int i = 0; i < image.Length; i++) image[i] = (i % 23) / 23.0;
        return backbone.ExtractFeatures(image);
    }

    private static YOLOv11Neck<double> Neck(int seed)
    {
        LayerInitializationSeedScope.ResetForModelConstruction(seed);
        return new YOLOv11Neck<double>(ModelSize.Nano);
    }

    [Fact]
    public void Outputs_KeepEachLevelsResolution_AtItsOwnWidth()
    {
        var features = Features(3);
        var outputs = Neck(3).Forward(features);
        var neck = Neck(3);

        Assert.Equal(3, outputs.Count);
        // Nano: widths 256, 512, 1024 scaled by 0.25 -> 64, 128, 256.
        Assert.Equal(new[] { 64, 128, 256 }, neck.LevelChannels.ToArray());
        for (int level = 0; level < 3; level++)
        {
            Assert.Equal(neck.LevelChannels[level], outputs[level].Shape[1]);
            Assert.Equal(features[level].Shape[2], outputs[level].Shape[2]);
            Assert.Equal(features[level].Shape[3], outputs[level].Shape[3]);
            Assert.All(outputs[level].ToArray(), v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        }
    }

    [Fact]
    public void Parameters_AreRegistered_AndMatchTheCount()
    {
        var neck = Neck(3);
        neck.Forward(Features(3));
        var parameters = neck.GetParameters();
        Assert.True(parameters.Length > 0, "The neck's C3k2 and Conv weights reach no parameter vector.");
        Assert.Equal(neck.ParameterCount, parameters.Length);
    }

    [Fact]
    public void WriteThenRead_ReproducesTheOutputs()
    {
        var features = Features(3);
        var source = Neck(3);
        var expected = source.Forward(features);

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            source.WriteParameters(writer);
        stream.Position = 0;

        var copy = Neck(99);
        copy.Forward(features); // materialize lazily sized weights before loading into them
        using (var reader = new BinaryReader(stream))
            copy.ReadParameters(reader);
        var actual = copy.Forward(features);

        for (int level = 0; level < 3; level++)
            Assert.Equal(expected[level].ToArray(), actual[level].ToArray());
    }

    [Fact]
    public void EqualSeeds_GiveEqualWeights()
    {
        var features = Features(3);
        var a = Neck(5);
        var b = Neck(5);
        a.Forward(features);
        b.Forward(features);
        Assert.Equal(a.GetParameters().ToArray(), b.GetParameters().ToArray());
    }

    [Fact]
    public void C3k2_MatchesTheUltralyticsParameterCount_ForYolo11nLayer2()
    {
        // yolo11n.yaml layer 2 is C3k2(32 -> 64, n = 1, c3k = False, e = 0.25); Ultralytics' model summary reports
        // 6,640 parameters for it: cv1 32->32 (1,088), cv2 48->64 (3,200), and one inner Bottleneck 16->8->16
        // (2,352) built with Bottleneck's default e = 0.5. With e = 1.0 the inner block is 4,672 and the total 8,960.
        var block = new C3k2Block<double>(outChannels: 64, depth: 1, c3k: false, expansion: 0.25);
        block.Forward(new Tensor<double>(new[] { 1, 32, 8, 8 }));
        // Ultralytics counts nn.Parameters only; ParameterCount here also carries BatchNorm's running mean and
        // variance (serialized state, not trained), so compare the trainable tensors.
        Assert.Equal(6640, block.GetTrainableParameters().Sum(tensor => tensor.Length));
    }

    [Fact]
    public void DifferentSeeds_GiveDifferentWeights()
    {
        // Without this the equal-seed test also passes when the seed is ignored or the weights are constant.
        var features = Features(3);
        var a = Neck(5);
        var b = Neck(6);
        a.Forward(features);
        b.Forward(features);
        Assert.NotEqual(a.GetParameters().ToArray(), b.GetParameters().ToArray());
    }
}