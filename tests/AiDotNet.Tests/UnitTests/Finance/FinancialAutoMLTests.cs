using System;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Finance.AutoML;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// AutoML over the finance model families (He et al. 2021 survey): random search over the domain's candidate
/// models, returning the best trained model.
/// </summary>
public class FinancialAutoMLTests
{
    [Fact]
    public void OnlyRandomSearch_IsAccepted()
    {
        Assert.Throws<NotSupportedException>(() => new FinancialAutoML<double>(
            new FinancialAutoMLOptions<double> { SearchStrategy = AutoMLSearchStrategy.BayesianOptimization }));
    }

    [Theory]
    [InlineData(FinancialDomain.Forecasting)]
    [InlineData(FinancialDomain.Risk)]
    public async Task Trials_ProposeOnlyTheDomainsCandidateModels(FinancialDomain domain)
    {
        var automl = new FinancialAutoML<double>(new FinancialAutoMLOptions<double> { Domain = domain },
            RandomHelper.CreateSeededRandom(3));
        string expectedNamespace = domain == FinancialDomain.Risk ? "AiDotNet.Finance.Risk" : "AiDotNet.Finance.Forecasting";

        for (int i = 0; i < 8; i++)
        {
            var trial = await automl.SuggestNextTrialAsync();
            var modelType = Assert.IsAssignableFrom<Type>(trial["ModelType"]);
            Assert.StartsWith(expectedNamespace, modelType.Namespace);
        }
    }

    // Every candidate used to fail before training: its options kept their own default sizes (10 risk features,
    // a 96 x 7 window) against the search's architecture, DeepAR was offered likelihoods it does not implement, and
    // TFT drew hidden sizes its head count could not split. A one-trial search per candidate must now finish.
    [Theory]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.PatchTST<>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.ITransformer<>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Neural.DeepAR<>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Neural.NBEATSFinance<>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.TFT<>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Risk.NeuralVaR<>), FinancialDomain.Risk)]
    [InlineData(typeof(AiDotNet.Finance.Risk.TabNet<>), FinancialDomain.Risk)]
    public async Task OneTrialSearch_TrainsTheCandidate(Type candidate, FinancialDomain domain)
    {
        var options = new FinancialAutoMLOptions<double>
        {
            Domain = domain,
            CandidateModels = new() { candidate },
            Budget = new AutoMLBudgetOptions { TrialLimitOverride = 1 },
        };
        var automl = new FinancialAutoML<double>(options, RandomHelper.CreateSeededRandom(5));
        var random = RandomHelper.CreateSeededRandom(9);
        // The default architecture is 16 inputs and 1 output: 16 features for a risk model, a univariate
        // 16-step window for a forecaster.
        Tensor<double> Inputs(int rows)
        {
            var t = new Tensor<double>(domain == FinancialDomain.Forecasting ? new[] { rows, 16, 1 } : new[] { rows, 16 });
            for (int i = 0; i < t.Length; i++)
                t.SetFlat(i, random.NextDouble() - 0.5);
            return t;
        }
        Tensor<double> Targets(int rows)
        {
            var t = new Tensor<double>(new[] { rows, 1 });
            for (int i = 0; i < t.Length; i++)
                t.SetFlat(i, random.NextDouble() - 0.5);
            return t;
        }

        var best = await automl.SearchAsync(Inputs(24), Targets(24), Inputs(8), Targets(8), TimeSpan.FromMinutes(2));

        Assert.Equal(AutoMLStatus.Completed, automl.Status);
        Assert.All(best.Predict(Inputs(4)).ToArray(), v => Assert.False(double.IsNaN(v)));
    }

    // The applicator sets an option only when a property has the key's exact name and skips the rest silently,
    // so a key that names no option is a hyperparameter the search claims to tune and never does.
    [Theory]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.PatchTST<>), typeof(PatchTSTOptions<double>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.ITransformer<>), typeof(ITransformerOptions<double>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Neural.DeepAR<>), typeof(DeepAROptions<double>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Neural.NBEATSFinance<>), typeof(NBEATSModelOptions<double>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Forecasting.Transformers.TFT<>), typeof(TemporalFusionTransformerOptions<double>), FinancialDomain.Forecasting)]
    [InlineData(typeof(AiDotNet.Finance.Risk.NeuralVaR<>), typeof(NeuralVaROptions<double>), FinancialDomain.Risk)]
    [InlineData(typeof(AiDotNet.Finance.Risk.TabNet<>), typeof(TabNetOptions<double>), FinancialDomain.Risk)]
    [InlineData(typeof(AiDotNet.Finance.Risk.TabTransformer<>), typeof(TabTransformerOptions<double>), FinancialDomain.Risk)]
    public void EverySearchSpaceKey_NamesAWritableOption(Type candidate, Type optionsType, FinancialDomain domain)
    {
        foreach (var key in new FinancialSearchSpace(domain).GetSearchSpace(candidate).Keys)
        {
            var property = optionsType.GetProperty(key, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            Assert.True(property is { CanWrite: true }, $"{candidate.Name} search key '{key}' sets no option.");
        }
    }
}
