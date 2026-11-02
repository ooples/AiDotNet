using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public sealed class AstOptionsRoutingTests
{
    [Fact]
    public void CopyPreservesInheritedTrainingAndTranscriptionOptions()
    {
        var original = new ASTModelOptions { MaxGradNorm = 0.03, ReturnTimestamps = true };
        var copy = new ASTModelOptions(original);
        Assert.Equal(original.MaxGradNorm, copy.MaxGradNorm);
        Assert.Equal(original.ReturnTimestamps, copy.ReturnTimestamps);
    }

    [Fact]
    public void GradientClipOptionReachesTheTrainingBase()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiLabelClassification,
            inputHeight: 8, inputWidth: 4, inputDepth: 1, outputSize: 3);
        using var model = new ASTModel<double>(architecture, new ASTModelOptions
        {
            EmbeddingDim = 8, NumLayers = 1, NumHeads = 2, FeedForwardDim = 16,
            NumClasses = 3, PatchSize = 2, TargetLength = 8, NumMelBands = 4,
            MaxGradNorm = 0.03,
        });
        Assert.Equal(0.03, model.MaxGradNormValue);
    }
}
