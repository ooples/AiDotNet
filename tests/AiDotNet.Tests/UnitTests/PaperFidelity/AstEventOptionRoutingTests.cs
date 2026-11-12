using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class AstEventOptionRoutingTests
{
    [Theory]
    [InlineData("Threshold")]
    [InlineData("WindowSize")]
    [InlineData("WindowOverlap")]
    public async System.Threading.Tasks.Task GeneratedTheoryHonorsDeclaredEventOnlyExemptions(string option)
    {
        var fixture = new AiDotNet.Tests.ModelFamilyTests.Generated.ASTModelTests();
        var skipped = await Assert.ThrowsAsync<Xunit.SkipException>(() =>
            fixture.Options_EachShouldChangeObservableBehaviour(option));
        Assert.Contains("[OptionBehaviourExempt]", skipped.Message);
        Assert.Contains("AstEventOptionRoutingTests", skipped.Message);
    }

    private static ASTOptions Options() => new()
    {
        EmbeddingDim = 8, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 16,
        NumMels = 4, TargetLength = 8, PatchSize = 2, FrequencyStride = 2, TimeStride = 2,
        CustomLabels = new[] { "tone" }, Threshold = 0, WindowSize = 0.05, WindowOverlap = 0,
    };

    private static AST<double> Detector(ASTOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiLabelClassification, inputHeight: 8, inputWidth: 4,
            inputDepth: 1, outputSize: 1), options);

    [Theory]
    [InlineData(0.05, 0.0, 5)]
    [InlineData(0.05, 0.5, 9)]
    [InlineData(0.10, 0.0, 2)]
    public void StreamingWindowsRespectDurationAndOverlap(double window, double overlap, int expected)
    {
        var options = Options();
        options.WindowSize = window;
        options.WindowOverlap = overlap;
        using var detector = Detector(options);
        using var session = detector.StartStreamingSession();
        session.FeedAudio(new Tensor<double>(new[] { 4000 })); // 250 ms at 16 kHz.
        Assert.Equal(expected, session.GetNewEvents().Count);
    }

    [Fact]
    public void DefaultDetectionThresholdFiltersEventsWithoutChangingTheClassifier()
    {
        var options = Options();
        using var detector = Detector(options);
        var audio = new Tensor<double>(new[] { 4000 });
        Assert.NotEmpty(detector.Detect(audio).Events);
        options.Threshold = 1;
        Assert.Empty(detector.Detect(audio).Events);
    }
}
