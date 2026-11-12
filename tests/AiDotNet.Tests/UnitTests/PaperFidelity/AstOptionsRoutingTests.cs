using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public sealed class AstOptionsRoutingTests
{
    [Fact]
    public void CopyPreservesTrainingAndFrontendOptions()
    {
        var original = new ASTOptions { MaxGradNorm = 0.03, NumMels = 4 };
        var copy = new ASTOptions(original);
        Assert.Equal(original.MaxGradNorm, copy.MaxGradNorm);
        Assert.Equal(original.NumMels, copy.NumMels);
    }

    [Fact]
    public void GradientClipOptionReachesTheTrainingBase()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiLabelClassification,
            inputHeight: 8, inputWidth: 4, inputDepth: 1, outputSize: 3);
        var options = new ASTOptions
        {
            EmbeddingDim = 8, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 16,
            CustomLabels = new[] { "a", "b", "c" }, PatchSize = 2, TargetLength = 8, NumMels = 4,
            FrequencyStride = 2, TimeStride = 2,
            MaxGradNorm = 0.03,
        };
        using var model = new ASTModel<double>(architecture, options, optimizer: null);
        using var classifier = new AST<double>(architecture, options);
        Assert.Equal(0.03, model.MaxGradNormValue);
        Assert.Equal(0.03, classifier.MaxGradNormValue);
    }
}
