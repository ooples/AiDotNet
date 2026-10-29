using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// Predict skips its detach copy when the forward rented no host scratch (#2385). A forward that returns its input
/// unchanged (only an InputLayer) rents nothing either, and must still not hand the caller its own tensor back: the
/// caller would alias its input, and the streaming PredictInBatches path disposes each result after its callback.
/// </summary>
[Collection("NonParallelIntegration")]
public class PredictOutputOwnershipTests
{
    private static FeedForwardNeuralNetwork<float> PassThrough(bool reshape = false)
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 4,
            layers: reshape
                // The reshape returns a view of its input: an output distinct from the input that shares its storage.
                ? new List<ILayer<float>> { new InputLayer<float>(new[] { 4 }), new ReshapeLayer<float>(new[] { 2, 2 }) }
                : new List<ILayer<float>> { new InputLayer<float>(new[] { 4 }) });
        return new FeedForwardNeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Predict_never_returns_the_callers_input_or_a_view_of_it(bool reshape)
    {
        // The detach this guards runs on the inference-arena path, so the arena is on whatever the environment says.
        var prior = AiDotNetEngine.Current;
        bool priorArena = InferenceArenaSettings.Enabled;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            InferenceArenaSettings.Enabled = true;
            var network = PassThrough(reshape);
            var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, 4 });
            network.Predict(input);   // warm: chain validation

            var output = network.Predict(input);
            Assert.NotSame(input, output);
            output[0] = 99f;
            Assert.Equal(1f, input[0]);

            // Neither writing to the result nor releasing it may reach the input's storage.
            output.Dispose();
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, input.ToArray());

            network.PredictInBatches(input, 2, (_, _) => { });
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, input.ToArray());
        }
        finally
        {
            InferenceArenaSettings.Enabled = priorArena;
            AiDotNetEngine.Current = prior;
        }
    }
}