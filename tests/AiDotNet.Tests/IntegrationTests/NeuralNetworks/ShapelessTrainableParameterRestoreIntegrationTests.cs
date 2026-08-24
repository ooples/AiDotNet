using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Exercises the real checkpoint path for layers whose trainable declarations intentionally omit Shape.
/// </summary>
public class ShapelessTrainableParameterRestoreIntegrationTests
{
    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SubpixelConvolutional_Checkpoint_FirstForwardPreservesEveryRestoredValue()
    {
        await Task.Yield();
        var input = Ramp([1, 2, 4, 4]);
        VerifyCheckpointFirstForward(
            () => new SubpixelConvolutionalLayer<float>(
                2, 2, 3, new ReLUActivation<float>() as IActivationFunction<float>),
            input);
    }

    [Fact(Timeout = 120000)]
    [Trait("Category", "Integration")]
    public async Task SvtrThinPlateSpline_Checkpoint_FirstForwardPreservesEveryRestoredValue()
    {
        await Task.Yield();
        var input = Ramp([1, 3, 32, 100]);
        VerifyCheckpointFirstForward(() => new SVTRThinPlateSplineLayer<float>(), input);
    }

    private static void VerifyCheckpointFirstForward(
        Func<LayerBase<float>> createLayer,
        Tensor<float> input)
    {
        var donor = createLayer();
        donor.SetTrainingMode(false);
        donor.Forward(input);

        var distinctive = donor.GetParameters();
        for (int i = 0; i < distinctive.Length; i++)
            distinctive[i] = 0.03125f + (i % 29) * 0.00390625f;
        donor.SetParameters(distinctive);

        var expectedOutput = donor.Forward(input).Clone();
        var expectedParameters = donor.GetParameters();

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            donor.Serialize(writer);

        var restored = createLayer();
        restored.SetTrainingMode(false);
        stream.Position = 0;
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            restored.Deserialize(reader);

        // This must be the recipient's first forward after Deserialize. It is the lifecycle edge
        // that previously replaced the restored tensors with fresh random initialization.
        var restoredOutput = restored.Forward(input);
        var restoredParameters = restored.GetParameters();

        Assert.Equal(expectedParameters.Length, restoredParameters.Length);
        for (int i = 0; i < expectedParameters.Length; i++)
        {
            Assert.True(expectedParameters[i] == restoredParameters[i],
                $"Restored parameter {i} changed on first forward: " +
                $"expected {expectedParameters[i]:G9}, actual {restoredParameters[i]:G9}.");
        }

        Assert.Equal(expectedOutput.Shape.ToArray(), restoredOutput.Shape.ToArray());
        for (int i = 0; i < expectedOutput.Length; i++)
        {
            Assert.True(Math.Abs(expectedOutput[i] - restoredOutput[i]) <= 1e-5f,
                $"Output {i} changed across the checkpoint round trip: " +
                $"expected {expectedOutput[i]:G9}, actual {restoredOutput[i]:G9}.");
        }
    }

    private static Tensor<float> Ramp(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = 0.125f + (i % 17) * 0.015625f;
        return tensor;
    }
}
