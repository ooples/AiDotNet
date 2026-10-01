using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video.Generation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video;

/// <summary>
/// OpenSora.Train used to compute a loss and a gradient by hand, never backpropagate it, and then call
/// UpdateParameters on every layer, so no parameter ever moved. It now trains the DDPM epsilon-prediction objective
/// on the tape, with the DiT blocks gradient-checkpointed.
/// </summary>
public class OpenSoraTrainingTests
{
    // Four blocks: sqrt(4) = 2 blocks per checkpoint segment, so the checkpointed path really has more than one segment.
    private static OpenSora<double> CreateModel() =>
        new(new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Generative,
                inputHeight: 16,
                inputWidth: 16,
                inputDepth: 3),
            new AiDotNet.Video.Options.OpenSoraOptions
            {
                NumFrames = 1,
                HiddenDim = 32,
                NumLayers = 4,
                NumInferenceSteps = 4,
            });

    private static Tensor<double> Frames(int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<double>([1, 3, 16, 16]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    [Fact]
    public void Train_MovesTheParameters_AndKeepsThemFinite()
    {
        var model = CreateModel();
        var frames = Frames(11);
        model.Predict(frames);   // materialize any lazily shaped layers before reading the parameters
        var before = model.GetParameters().ToArray();

        for (int step = 0; step < 3; step++) model.Train(frames, frames);

        var after = model.GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);
        int moved = 0;
        for (int i = 0; i < after.Length; i++)
        {
            Assert.False(double.IsNaN(after[i]) || double.IsInfinity(after[i]), $"parameter {i} is not finite after training");
            if (after[i] != before[i]) moved++;
        }

        Assert.True(moved > before.Length / 2,
            $"Train moved only {moved} of {before.Length} parameters; a real optimizer step reaches almost all of them.");
        double loss = Convert.ToDouble(model.GetLastLoss());
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"training loss is not finite: {loss}");
    }

    [Fact]
    public void TrainingObjective_IsDeterministic_AndFinite()
    {
        var model = CreateModel();
        var frames = Frames(12);
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        Assert.Equal(AiDotNet.Enums.TrainingObjectiveKind.DiffusionDenoising, provider.TrainingObjectiveKind);
        var target = provider.ResolveTrainingTarget(frames, frames);
        double first = Convert.ToDouble(provider.EvaluateTrainingObjective(frames, target));
        double second = Convert.ToDouble(provider.EvaluateTrainingObjective(frames, target));
        Assert.False(double.IsNaN(first) || double.IsInfinity(first), $"objective is not finite: {first}");
        Assert.Equal(first, second);
    }
}