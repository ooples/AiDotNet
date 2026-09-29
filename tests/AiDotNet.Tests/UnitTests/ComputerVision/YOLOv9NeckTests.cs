using System.IO;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Invariants for YOLOv9's PAN-FPN head (GELAN-t at Nano). The neck has no options-only constructor the scaffold generator
/// can build, so these are its tests (AIDN040 requires every model to have one).
/// </summary>
public sealed class YOLOv9NeckTests
{
    public YOLOv9NeckTests() => TestModuleInitializer.EnsureInitialized();

    private static List<Tensor<double>> Features(int seed)
    {
        LayerInitializationSeedScope.ResetForModelConstruction(seed);
        var backbone = new YOLOv9Backbone<double>(ModelSize.Nano);
        var image = new Tensor<double>(new[] { 1, 3, 64, 64 });
        for (int i = 0; i < image.Length; i++) image[i] = (i % 23) / 23.0;
        return backbone.ExtractFeatures(image);
    }

    private static YOLOv9Neck<double> Neck(int seed)
    {
        LayerInitializationSeedScope.ResetForModelConstruction(seed);
        return new YOLOv9Neck<double>(ModelSize.Nano);
    }

    [Fact]
    public void Outputs_KeepEachLevelsResolution_AtItsOwnWidth()
    {
        var features = Features(3);
        var outputs = Neck(3).Forward(features);
        var neck = Neck(3);

        Assert.Equal(3, outputs.Count);
        // Nano is GELAN-t: its head ends in RepNCSPELAN4 blocks of 64, 96 and 128 channels (yolov9t.yaml).
        Assert.Equal(new[] { 64, 96, 128 }, neck.LevelChannels.ToArray());
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
        Assert.True(parameters.Length > 0, "The head's GELAN and downsample weights reach no parameter vector.");
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
}
