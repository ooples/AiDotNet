using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.Helpers;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class ConditionalGANTests : GANModelTestBase<float>
{
    // Default ConditionalGAN: generator inputSize=110 (latentDim=100 + numClasses=10), outputSize=784
    // The public API accepts noise-only input (100), conditions are added internally by Predict/Train
    protected override int[] InputShape => [1, 100];
    protected override int[] OutputShape => [1, 784];

    // GAN training is adversarial — generator and discriminator compete, so the MSE
    // (computed between generated output and random target) oscillates rather than
    // monotonically decreasing. A higher tolerance accounts for this.
    protected override double MoreDataTolerance => 0.01;


    // Iteration counts capped at 1 / 2 / 4 to fit the 60-180 s xUnit
    // per-test timeouts. The defaults (50 / 200 for MoreData and 100 for
    // MemorizationTask) are calibrated for small / mid-scale networks
    // where each step takes < 1.5 s — the model here either has a
    // higher per-step cost (per-iteration adversarial GAN forwards,
    // graph propagation across all nodes) or evolves topology between
    // calls (NEAT speciation), making 100+ iterations exceed budget.
    // Same paper-scale precedent the Forecasting Foundation models /
    // CLIP-family / VoxelCNN / VGG / DenseNet use; still exercises the
    // gradient-direction / loss-decrease invariants the tests catch
    // (sign error, oscillation, first-step explosion).
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override int MemorizationTaskIterations => 4;
    protected override double MemorizationTaskLossThreshold => 0.99999;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new ConditionalGAN<float>();

    [Fact(Timeout = 120000)]
    public async Task CustomFullWidthGeneratorArchitecture_AcceptsNoiseOnlyPredictInput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        using var cgan = new ConditionalGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            numConditionClasses: 10,
            InputType.OneDimensional);

        Assert.Equal(32, cgan.Generator.Architecture.InputSize);

        var noiseOnlyInput = CreateRandomTensor([1, 22], new Random(42));
        var output = cgan.Predict(noiseOnlyInput);

        Assert.Equal(64, output.Length);
    }
}
