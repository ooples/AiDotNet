using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Models.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Finance;

public class FinanceCategoryIntegrationTests
{
    private static IEnumerable<object[]> WrapTypes(IEnumerable<Type> types)
    {
        return types.Select(type => new object[] { type });
    }

    public static IEnumerable<object[]> ForecastingTransformerTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Forecasting.Transformers"));

    public static IEnumerable<object[]> ForecastingTransformerTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Forecasting.Transformers"));

    public static IEnumerable<object[]> ForecastingNeuralTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Forecasting.Neural"));

    public static IEnumerable<object[]> ForecastingNeuralTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Forecasting.Neural"));

    public static IEnumerable<object[]> ForecastingFoundationTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Forecasting.Foundation"));

    public static IEnumerable<object[]> ForecastingFoundationTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Forecasting.Foundation"));

    public static IEnumerable<object[]> ForecastingStateSpaceTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Forecasting.StateSpace"));

    public static IEnumerable<object[]> ForecastingStateSpaceTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Forecasting.StateSpace"));

    public static IEnumerable<object[]> ProbabilisticTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Probabilistic"));

    public static IEnumerable<object[]> ProbabilisticTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Probabilistic"));

    public static IEnumerable<object[]> GraphTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Graph"));

    public static IEnumerable<object[]> GraphTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Graph"));

    public static IEnumerable<object[]> NlpTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.NLP"));

    public static IEnumerable<object[]> NlpTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.NLP"));

    public static IEnumerable<object[]> RiskTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Risk"));

    public static IEnumerable<object[]> RiskTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Risk"));

    public static IEnumerable<object[]> PortfolioTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Portfolio"));

    public static IEnumerable<object[]> PortfolioTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Portfolio"));

    public static IEnumerable<object[]> VolatilityTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Volatility"));

    public static IEnumerable<object[]> VolatilityTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Volatility"));

    public static IEnumerable<object[]> FactorTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.Trading.Factors"));

    public static IEnumerable<object[]> FactorTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.Trading.Factors"));

    public static IEnumerable<object[]> AutoMlTypesFloat =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<float>("AiDotNet.Finance.AutoML"));

    public static IEnumerable<object[]> AutoMlTypesDouble =>
        WrapTypes(FinanceModelTestFactory.GetFinanceModelTypesByNamespace<double>("AiDotNet.Finance.AutoML"));

    [Fact(Timeout = 120000)]
    public async Task FinanceCategories_HaveModels()
    {
        Assert.NotEmpty(ForecastingTransformerTypesFloat);
        Assert.NotEmpty(ForecastingNeuralTypesFloat);
        Assert.NotEmpty(ForecastingFoundationTypesFloat);
        Assert.NotEmpty(ForecastingStateSpaceTypesFloat);
        Assert.NotEmpty(ProbabilisticTypesFloat);
        Assert.NotEmpty(GraphTypesFloat);
        Assert.NotEmpty(NlpTypesFloat);
        Assert.NotEmpty(RiskTypesFloat);
        Assert.NotEmpty(PortfolioTypesFloat);
        Assert.NotEmpty(VolatilityTypesFloat);
        Assert.NotEmpty(FactorTypesFloat);
        Assert.NotEmpty(AutoMlTypesFloat);
        Assert.NotEmpty(FinanceModelTestFactory.GetFinancialModelTypes<float>());
        Assert.NotEmpty(FinanceModelTestFactory.GetForecastingModelTypes<float>());
        Assert.NotEmpty(FinanceModelTestFactory.GetRiskModelTypes<float>());
        Assert.NotEmpty(FinanceModelTestFactory.GetPortfolioModelTypes<float>());
        Assert.NotEmpty(FinanceModelTestFactory.GetVolatilityModelTypes<float>());
        Assert.NotEmpty(FinanceModelTestFactory.GetFactorModelTypes<float>());
    }

    [Fact]
    public void CsdiInstanceNormalization_RankThreeInput_PreservesShape()
    {
        var options = new CSDIOptions<double>
        {
            SequenceLength = 4,
            NumFeatures = 2,
            HiddenDimension = 8,
            NumResidualLayers = 1,
            NumDiffusionSteps = 1,
            NumHeads = 1,
            TimeEmbeddingDim = 4
        };
        var model = new CSDI<double>(FinanceTestHelpers.CreateArchitecture<double>(8, 8), options);
        var input = FinanceTestHelpers.CreateTimeSeriesInput<double>(2, 4, 2);

        var normalized = model.ApplyInstanceNormalization(input);

        Assert.Equal(input.Shape, normalized.Shape);
        Assert.Equal(input.Length, normalized.Length);
    }

    [Fact]
    public void TfcTraining_RankThreeInput_BackpropagatesThroughTemporalFft()
    {
        var options = new TFCOptions<double>
        {
            ContextLength = 8,
            ForecastHorizon = 4,
            HiddenDimension = 8,
            ProjectionDimension = 4,
            NumTimeLayers = 1,
            NumFreqLayers = 1,
            DropoutRate = 0.0
        };
        var model = new TFC<double>(FinanceTestHelpers.CreateArchitecture<double>(8, 4), options);
        var input = FinanceTestHelpers.CreateTimeSeriesInput<double>(1, 8, 1);
        var target = model.Predict(input);

        model.Train(input, target);

        Assert.True(model.GetFinancialMetrics().ContainsKey("LastLoss"));
    }

    [Theory]
    [MemberData(nameof(ForecastingTransformerTypesFloat))]
    public void ForecastingTransformers_Float_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingTransformerTypesDouble))]
    public void ForecastingTransformers_Double_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingNeuralTypesFloat))]
    public void ForecastingNeural_Float_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingNeuralTypesDouble))]
    public void ForecastingNeural_Double_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingFoundationTypesFloat))]
    public void ForecastingFoundation_Float_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingFoundationTypesDouble))]
    public void ForecastingFoundation_Double_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingStateSpaceTypesFloat))]
    public void ForecastingStateSpace_Float_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ForecastingStateSpaceTypesDouble))]
    public void ForecastingStateSpace_Double_CanForecastWithQuantiles(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ProbabilisticTypesFloat))]
    public void ProbabilisticModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(ProbabilisticTypesDouble))]
    public void ProbabilisticModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(GraphTypesFloat))]
    public void GraphModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(GraphTypesDouble))]
    public void GraphModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType, includeQuantileForecast: true);
    }

    [Theory]
    [MemberData(nameof(NlpTypesFloat))]
    public void NlpModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(NlpTypesDouble))]
    public void NlpModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(RiskTypesFloat))]
    public void RiskModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(RiskTypesDouble))]
    public void RiskModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(PortfolioTypesFloat))]
    public void PortfolioModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(PortfolioTypesDouble))]
    public void PortfolioModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(VolatilityTypesFloat))]
    public void VolatilityModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(VolatilityTypesDouble))]
    public void VolatilityModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(FactorTypesFloat))]
    public void FactorModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(FactorTypesDouble))]
    public void FactorModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(AutoMlTypesFloat))]
    public void AutoMlModels_Float_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(AutoMlTypesDouble))]
    public void AutoMlModels_Double_Smoke(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }
}
