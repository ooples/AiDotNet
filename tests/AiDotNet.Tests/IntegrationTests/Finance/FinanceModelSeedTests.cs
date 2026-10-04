using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// Regression for #2290. Before it, 81 of the 90 neural finance models ignored <c>Options.Seed</c>: they took
/// their seed only from the architecture, and the options of several declared a second, separate
/// <c>RandomSeed</c> that nothing read either. Setting a model's <c>Options</c> now applies the seed before its
/// layers are built, and the layers that drew from unseeded generators (MambaBlock's dt init, RWKV-7's
/// orthogonal LoRA init) now draw from their own seeded stream.
/// </summary>
public class FinanceModelSeedTests
{
    /// <summary>Every neural finance model; the GARCH-family volatility models have no weights to seed.</summary>
    public static IEnumerable<object[]> NeuralFinanceModels() =>
        FinanceModelTestFactory.GetFinancialModelTypes<double>()
            .Where(t => typeof(NeuralNetworkBase<double>).IsAssignableFrom(t))
            .Select(t => new object[] { t });

    /// <summary>The finance transformers the issue named.</summary>
    public static IEnumerable<object[]> Transformers() => new[]
    {
        typeof(Crossformer<double>), typeof(ETSformer<double>), typeof(FEDformer<double>), typeof(ITransformer<double>),
        typeof(NonStationaryTransformer<double>), typeof(PatchTST<double>), typeof(TSMixer<double>), typeof(TimesNet<double>),
    }.Select(t => new object[] { t });

    private static double[] InitialWeights(Type modelType, int? optionsSeed, int? architectureSeed = null)
    {
        bool configured = false;
        var model = (NeuralNetworkBase<double>)FinanceModelTestFactory.CreateNativeModel<double>(
            modelType,
            configureOptions: options =>
            {
                configured = true;
                ((ModelOptions)options).Seed = optionsSeed;
            },
            configureArchitecture: architecture => ((NeuralNetworkArchitecture<double>)architecture).RandomSeed = architectureSeed);
        Assert.True(configured, $"{modelType.Name} was built without options, so the seed never reached it.");

        model.MaterializeParameters();
        var parameters = model.GetParameters();
        if (parameters.Length == 0)
        {
            // Layers that size their weights on the first forward own no parameters until one runs. Feed an
            // input the model's own published contract accepts.
            var requested = InputContractShapeResolver.Conform(new[] { 1, 8, 8 }, model.GetInputShapeConstraint());
            var input = InputContractTensorFactory.CreateValid<double>(
                model.BindInputContract(requested), RandomHelper.CreateSeededRandom(42));
            model.Predict(input);
            parameters = model.GetParameters();
        }

        Assert.True(parameters.Length > 0, $"{modelType.Name} has no parameters to compare.");
        return parameters.ToArray();
    }

    [Theory(Timeout = 600000)]
    [MemberData(nameof(NeuralFinanceModels))]
    public async Task SameOptionsSeed_GivesIdenticalInitialWeights(Type modelType)
    {
        await Task.Yield();
        Assert.Equal(InitialWeights(modelType, 7), InitialWeights(modelType, 7));
    }

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Transformers))]
    public async Task DifferentOptionsSeeds_GiveDifferentInitialWeights(Type modelType)
    {
        await Task.Yield();
        Assert.NotEqual(InitialWeights(modelType, 7), InitialWeights(modelType, 8));
    }

    [Theory(Timeout = 300000)]
    [MemberData(nameof(Transformers))]
    public async Task AnExplicitArchitectureSeed_WinsOverTheOptionsSeed(Type modelType)
    {
        await Task.Yield();
        Assert.Equal(
            InitialWeights(modelType, optionsSeed: 7, architectureSeed: 123),
            InitialWeights(modelType, optionsSeed: 8, architectureSeed: 123));
    }
}
