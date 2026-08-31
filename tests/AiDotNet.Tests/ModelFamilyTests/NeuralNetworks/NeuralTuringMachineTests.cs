using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class NeuralTuringMachineTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new NeuralTuringMachine<float>();

    [Fact]
    public async Task DefaultTrainingRecipe_MatchesOriginalNtmPapers()
    {
        await Task.Yield();

        var network = new InspectableNeuralTuringMachine();
        var optimizer = Assert.IsType<RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>>(
            network.TrainingOptimizer);
        var options = Assert.IsType<RootMeanSquarePropagationOptimizerOptions<float, Tensor<float>, Tensor<float>>>(
            optimizer.GetOptions());

        Assert.True(options.Centered);
        Assert.Equal(0.95, options.Decay, 12);
        Assert.Equal(0.9, options.InitialMomentum, 12);
        Assert.Equal(1e-4, options.InitialLearningRate, 12);
        Assert.Equal(1e-4, options.Epsilon, 12);
        Assert.False(options.UseAdaptiveLearningRate);
        Assert.False(options.UseAdaptiveMomentum);
        Assert.True(network.ClipsGradientsElementWise);
        Assert.Equal(10.0, network.MaxGradNormValue, 12);
    }

    private sealed class InspectableNeuralTuringMachine : NeuralTuringMachine<float>
    {
        public IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> TrainingOptimizer
            => GetOrCreateBaseOptimizer();

        public bool ClipsGradientsElementWise => UsesElementWiseGradientClipping;
    }
}
