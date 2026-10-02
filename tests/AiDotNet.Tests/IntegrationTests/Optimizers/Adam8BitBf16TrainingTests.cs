using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// #1767: Adam8BitOptimizer's BF16 moment path threw "Source array was not long enough" from the tensor allocator
/// partway through ParaformerLarge training, after the training arena had accumulated state. Nothing exercised that
/// path across several steps. This trains through the eager tape step (compilation off, so the fused kernel cannot
/// stand in for it) for 20 steps, with two weight matrices of equal element count and transposed shapes, so the
/// arena's count-keyed ring reissues one parameter's buffers under the other's shape.
/// </summary>
[Collection("FusedTrainingSerial")]
public class Adam8BitBf16TrainingTests
{
    [Fact]
    public void Bf16MomentPath_TrainsAcrossManyStepsThroughTheArena()
    {
        var originalCodec = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            var layers = new List<ILayer<float>>
            {
                new DenseLayer<float>(64, activationFunction: new AiDotNet.ActivationFunctions.ReLUActivation<float>()),
                new DenseLayer<float>(32, activationFunction: new AiDotNet.ActivationFunctions.ReLUActivation<float>()),
                new DenseLayer<float>(64, activationFunction: new AiDotNet.ActivationFunctions.ReLUActivation<float>()),
                new DenseLayer<float>(4, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            };
            var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 32, outputSize: 4, layers: layers) { RandomSeed = 1767 };
            var optimizer = new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null,
                new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    UseBFloat16MomentStorage = true,
                    UseAMSGrad = false,
                    InitialLearningRate = 1e-2,
                });
            var network = new FeedForwardNeuralNetwork<float>(architecture, optimizer);

            var rng = RandomHelper.CreateSeededRandom(1767);
            var input = new Tensor<float>([16, 32]);
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            var target = new Tensor<float>([16, 4]);
            for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() * 2 - 1);

            var initial = network.GetParameters().ToArray();
            network.Train(input, target);
            float firstLoss = Convert.ToSingle(network.GetLastLoss());
            for (int step = 1; step < 20; step++) network.Train(input, target);
            float lastLoss = Convert.ToSingle(network.GetLastLoss());
            var trained = network.GetParameters().ToArray();

            Assert.All(trained, value => Assert.True(float.IsFinite(value), "a parameter became non-finite"));
            Assert.NotEqual(initial, trained);
            Assert.True(lastLoss < firstLoss, $"20 BF16-moment Adam steps did not reduce the loss ({firstLoss} -> {lastLoss}).");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(originalCodec);
        }
    }
}
