using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNetVector = AiDotNet.Tensors.LinearAlgebra.Vector<float>;

namespace AiDotNet.Tests.IntegrationTests.MixedPrecision;

/// <summary>
/// Integration test for AiDotNet#1362 — verifies the FACADE path
/// (<c>AiModelBuilder.ConfigureMixedPrecision(...).BuildAsync()</c>)
/// actually wires the configured mixed-precision settings onto the
/// constructed neural network. The unit tests in
/// <c>MixedPrecisionTrainWithTapeWiringTests</c> already cover the
/// internal <c>EnableMixedPrecision</c> direct path via
/// InternalsVisibleTo; these tests exercise the user-visible facade
/// surface end-to-end.
///
/// <para>The wire-up under test lives at
/// <c>AiModelBuilder.BuildAsync</c> (around line 2502):</para>
/// <code>
/// if (_model is NeuralNetworkBase&lt;T&gt; neuralNet) {
///     neuralNet.EnableMixedPrecision(_mixedPrecisionConfig);
/// }
/// </code>
///
/// <para>If the wire-up is removed or broken, <c>IsMixedPrecisionEnabled</c>
/// on the returned model will be <c>false</c> even though the caller
/// configured MP through the facade — silently degrading mixed-precision
/// to FP32 training. These tests catch that regression.</para>
/// </summary>
public class MixedPrecisionFacadeBuildAsyncTests
{
    private const int InputSize = 4;
    private const int OutputSize = 3;
    private const int NumSamples = 12;

    private static NeuralNetwork<float> BuildTinyNetwork()
    {
        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(InputSize),
            new DenseLayer<float>(OutputSize, activationFunction: new IdentityActivation<float>())
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: InputSize,
            outputSize: OutputSize,
            layers: layers);
        return new NeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    private static (Tensor<float> x, Tensor<float> y) BuildSyntheticData()
    {
        var x = new Tensor<float>(new[] { NumSamples, InputSize });
        var y = new Tensor<float>(new[] { NumSamples, OutputSize });
        var rng = new Random(42);
        for (int i = 0; i < NumSamples; i++)
        {
            for (int j = 0; j < InputSize; j++)
                x[i, j] = (float)(rng.NextDouble() * 2 - 1);
            for (int k = 0; k < OutputSize; k++)
                y[i, k] = (float)(rng.NextDouble() * 2 - 1);
        }
        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureMixedPrecision_BuildAsync_WiresContextOntoModel()
    {
        var network = BuildTinyNetwork();
        var (x, y) = BuildSyntheticData();

        Assert.False(network.IsMixedPrecisionEnabled,
            "Network must start without mixed-precision enabled.");

        var mpConfig = new MixedPrecisionConfig
        {
            PrecisionType = MixedPrecisionType.FP16,
            InitialLossScale = 256.0,
            EnableDynamicScaling = false
        };

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureModel(network)
            .ConfigureMixedPrecision(mpConfig);

        var result = await builder.BuildAsync();

        Assert.NotNull(result);
        Assert.NotNull(result.Model);

        // The MP wire-up under test: the network handed to BuildAsync should
        // have IsMixedPrecisionEnabled == true after the build completes.
        Assert.True(network.IsMixedPrecisionEnabled,
            "BuildAsync must apply ConfigureMixedPrecision(config) to the constructed model.");

        var ctx = network.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.Equal(MixedPrecisionType.FP16, ctx!.Config.PrecisionType);
        Assert.Equal(256.0, ctx.LossScaler.Scale);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureMixedPrecision_BF16_BuildAsync_WiresBF16Config()
    {
        var network = BuildTinyNetwork();
        var (x, y) = BuildSyntheticData();

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureModel(network)
            .ConfigureMixedPrecision(MixedPrecisionConfig.ForBF16());

        var result = await builder.BuildAsync();
        Assert.NotNull(result);

        Assert.True(network.IsMixedPrecisionEnabled,
            "BuildAsync must wire BF16 MP config when ConfigureMixedPrecision uses ForBF16().");
        var ctx = network.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
        Assert.Equal(MixedPrecisionType.BF16, ctx!.Config.PrecisionType);
        // BF16 default — no loss scaling required.
        Assert.Equal(1.0, ctx.LossScaler.Scale);
    }

    [Fact(Timeout = 120000)]
    public async Task BuildAsync_WithoutConfigureMixedPrecision_LeavesModelInFP32()
    {
        // Sanity check: when the caller does NOT invoke ConfigureMixedPrecision,
        // the resulting model must not have MP enabled (i.e. BuildAsync does
        // not silently enable it via some default).
        var network = BuildTinyNetwork();
        var (x, y) = BuildSyntheticData();

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureModel(network);

        var result = await builder.BuildAsync();
        Assert.NotNull(result);

        Assert.False(network.IsMixedPrecisionEnabled,
            "Without ConfigureMixedPrecision the resulting model must remain in FP32 mode.");
        Assert.Null(network.GetMixedPrecisionContext());
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureMixedPrecision_DefaultArgument_EnablesDefaultMpConfig()
    {
        // ConfigureMixedPrecision() with no argument is documented as enabling
        // mixed-precision with a default MixedPrecisionConfig. Verify this
        // contract is preserved through BuildAsync.
        var network = BuildTinyNetwork();
        var (x, y) = BuildSyntheticData();

        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureDataLoader(
                new InMemoryDataLoader<float, Tensor<float>, Tensor<float>>(x, y))
            .ConfigureModel(network)
            .ConfigureMixedPrecision();

        var result = await builder.BuildAsync();
        Assert.NotNull(result);

        Assert.True(network.IsMixedPrecisionEnabled,
            "ConfigureMixedPrecision() with no argument should still wire a default MP config.");
        var ctx = network.GetMixedPrecisionContext();
        Assert.NotNull(ctx);
    }
}
