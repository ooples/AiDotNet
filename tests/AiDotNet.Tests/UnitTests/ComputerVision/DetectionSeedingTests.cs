using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.ComputerVision.OCR.Recognition;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Issue #2201: one model from each of the three detection/OCR bases, built twice from equal options,
/// must start from equal weights and predict identically; a different seed must give different weights.
/// </summary>
/// <remarks>
/// Before the fix none of the bases read a seed: two YOLOv8 builds differed in 96% of their weights, DBNet
/// and CRNN in 99.9%. Parameters are compared after one forward pass because several layers size their
/// weights lazily on the first input.
/// </remarks>
public sealed class DetectionSeedingTests
{
    public DetectionSeedingTests() => TestModuleInitializer.EnsureInitialized();

    public static TheoryData<string> Models => new() { "YOLOv8", "DBNet", "CRNN" };

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Models))]
    public async Task EqualSeeds_GiveEqualWeightsAndPredictions(string model)
    {
        await Task.Yield();
        var (a, inputA) = Build(model, seed: 7);
        var (b, inputB) = Build(model, seed: 7);
        using (a)
        using (b)
        {
            var outputA = a.Predict(inputA);
            var outputB = b.Predict(inputB);

            var pa = a.GetParameters();
            var pb = b.GetParameters();
            Assert.True(pa.Length > 0, $"{model} exposed no parameters.");
            Assert.Equal(pa.Length, pb.Length);
            Assert.Equal(pa.ToArray(), pb.ToArray());
            Assert.Equal(outputA.ToArray(), outputB.ToArray());
        }
    }

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Models))]
    public async Task DifferentSeeds_GiveDifferentWeights(string model)
    {
        await Task.Yield();
        var (a, inputA) = Build(model, seed: 7);
        var (b, inputB) = Build(model, seed: 8);
        using (a)
        using (b)
        {
            a.Predict(inputA);
            b.Predict(inputB);
            var pa = a.GetParameters().ToArray();
            var pb = b.GetParameters().ToArray();
            Assert.Equal(pa.Length, pb.Length);
            int differing = 0;
            for (int i = 0; i < pa.Length; i++)
                if (pa[i] != pb[i]) differing++;
            // Zero-initialized biases and unit batch-norm scales legitimately agree, so require a clear
            // majority rather than every entry.
            Assert.True(differing > pa.Length / 2,
                $"{model}: only {differing} of {pa.Length} weights differ between seeds 7 and 8.");
        }
    }

    [Fact]
    public void NullSeed_ArmsNoSeedOfItsOwn()
    {
        // A null option seed must not invent one. The backbone architecture then carries no explicit
        // seed, so it falls back to the process default (null in production; the test harness pins one
        // for reproducibility, which is why this checks the explicit seed rather than comparing two
        // models built under that harness default). The ambient fallback is cleared for the same reason.
        var ambient = LayerInitializationSeedScope.AmbientFallbackSeed;
        LayerInitializationSeedScope.AmbientFallbackSeed = null;
        try
        {
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            Assert.Null(LayerInitializationSeedScope.NextSeedOrNull());
            Assert.False(AiDotNet.ComputerVision.Detection.Backbones.DetectionBackboneArchitecture<double>
                .Create(3).HasExplicitRandomSeed);

            // Inside a seeded detector's construction (the scope armed AND offered, as the detector bases do),
            // the backbone derives its seed from it.
            LayerInitializationSeedScope.ResetForModelConstruction(7);
            LayerInitializationSeedScope.OfferSeedToNestedBackbone();
            Assert.True(AiDotNet.ComputerVision.Detection.Backbones.DetectionBackboneArchitecture<double>
                .Create(3).HasExplicitRandomSeed);

            // The offer is spent on that one backbone: a scope left armed after the detector finished must not
            // seed a standalone backbone built afterwards (#2269 review).
            Assert.False(AiDotNet.ComputerVision.Detection.Backbones.DetectionBackboneArchitecture<double>
                .Create(3).HasExplicitRandomSeed);
        }
        finally
        {
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            LayerInitializationSeedScope.AmbientFallbackSeed = ambient;
        }
    }

    [Fact]
    public void A_seeded_recognizer_leaves_no_offer_for_a_later_standalone_backbone()
    {
        // A recognizer builds no detection backbone, so an offer from its base was never spent and the next standalone
        // backbone on the thread inherited the recognizer's seed (#2269 review).
        var ambient = LayerInitializationSeedScope.AmbientFallbackSeed;
        LayerInitializationSeedScope.AmbientFallbackSeed = null;
        try
        {
            using var recognizer = (IDisposable)Build("CRNN", 7).Model;
            Assert.False(AiDotNet.ComputerVision.Detection.Backbones.DetectionBackboneArchitecture<double>
                .Create(3).HasExplicitRandomSeed);
        }
        finally
        {
            LayerInitializationSeedScope.ResetForModelConstruction(null);
            LayerInitializationSeedScope.AmbientFallbackSeed = ambient;
        }
    }
    private static (AiDotNet.Models.ModelBase<double, Tensor<double>, Tensor<double>> Model, Tensor<double> Input)
        Build(string model, int? seed) => model switch
    {
        "YOLOv8" => (new YOLOv8<double>(new ObjectDetectionOptions<double>
        {
            InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, NumClasses = 2, Seed = seed
        }), Image(3, 64, 64)),
        "DBNet" => (new DBNet<double>(new TextDetectionOptions<double>
        {
            InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, RandomSeed = seed
        }), Image(3, 64, 64)),
        "CRNN" => (new CRNN<double>(new OCROptions<double>
        {
            RecognitionHeight = 32, MaxRecognitionWidth = 64, CharacterSet = "0123456789", RandomSeed = seed
        }), Image(3, 32, 64)),
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown model.")
    };

    private static Tensor<double> Image(int channels, int height, int width)
    {
        var image = new Tensor<double>(new[] { 1, channels, height, width });
        for (int i = 0; i < image.Length; i++)
            image[i] = (i % 17) / 17.0;
        return image;
    }
}