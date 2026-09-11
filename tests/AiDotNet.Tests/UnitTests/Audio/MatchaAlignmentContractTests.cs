using AiDotNet.Audio.TextToSpeech;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class MatchaAlignmentContractTests
{
    private readonly ITestOutputHelper _output;
    public MatchaAlignmentContractTests(ITestOutputHelper output)
    {
        _output = output;
        TestModuleInitializer.EnsureInitialized();
    }

    public enum DurationArchitectureChange { Width, Depth }

    [Theory]
    [InlineData(DurationArchitectureChange.Width)]
    [InlineData(DurationArchitectureChange.Depth)]
    public void DurationPredictorOptionsChangeActualTrainableParameters(DurationArchitectureChange change)
    {
        var changed = SmallOptions();
        switch (change)
        {
            case DurationArchitectureChange.Width:
                changed.DurationPredictorDim = 16;
                break;
            case DurationArchitectureChange.Depth:
                changed.NumDurationPredictorLayers = 2;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(change));
        }

        using var baseline = new MatchaTTS<float>(FrameArchitecture(), SmallOptions());
        using var configured = new MatchaTTS<float>(FrameArchitecture(), changed);
        int baselineCount = baseline.GetParameters().Length;
        int configuredCount = configured.GetParameters().Length;
        _output.WriteLine($"Actual duration {change} parameter counts: {baselineCount} -> {configuredCount}.");

        Assert.True(baselineCount > 0);
        Assert.Equal(baselineCount, baseline.ParameterCount);
        Assert.Equal(configuredCount, configured.ParameterCount);
        Assert.True(configuredCount > baselineCount,
            $"Changing duration-predictor {change} must change its materialized trainable parameters; " +
            $"baseline={baselineCount}, configured={configuredCount}.");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void FramePredictRetainsItsExistingBatchFeatureContract(int batchSize)
    {
        using var model = new MatchaTTS<float>(FrameArchitecture(), SmallOptions());
        var input = new Tensor<float>(new[] { batchSize, 4 });
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.125f;

        Tensor<float> output = model.Predict(input);

        Assert.Equal(new[] { batchSize, 4 }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++)
            Assert.False(float.IsNaN(output[i]) || float.IsInfinity(output[i]));
    }

    private static MatchaTTSOptions SmallOptions() => new()
    {
        NumMels = 4,
        PhonemeVocabSize = 32,
        TextEncoderDim = 8,
        NumTextEncoderLayers = 1,
        NumTextEncoderHeads = 2,
        DecoderDim = 8,
        NumDecoderLayers = 1,
        DurationPredictorDim = 8,
        NumDurationPredictorLayers = 1,
        DropoutRate = 0
    };

    private static NeuralNetworkArchitecture<float> FrameArchitecture() => new(
        inputType: InputType.OneDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputSize: 4, outputSize: 4);
}
