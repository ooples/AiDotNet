using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.AutoML;
using AiDotNet.Finance.Forecasting.Neural;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Risk;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

public class FinanceAutoMLIntegrationTests
{
    public static IEnumerable<object[]> SupportedAutoMlModelTypes =>
        new Type[]
        {
            typeof(PatchTST<>),
            typeof(ITransformer<>),
            typeof(DeepAR<>),
            typeof(NBEATSFinance<>),
            typeof(TFT<>),
            typeof(NeuralVaR<>),
            typeof(TabNet<>),
            typeof(TabTransformer<>)
        }.Select(modelType => new object[] { modelType });

    [Theory]
    [MemberData(nameof(SupportedAutoMlModelTypes))]
    public void FinancialModelFactory_CreatesSupportedModels(Type modelType)
    {
        var architecture = FinanceTestHelpers.CreateArchitecture<double>(inputSize: 4, outputSize: 4);
        var factory = new FinancialModelFactory<double>(architecture);

        var model = factory.Create(modelType, new Dictionary<string, object>());
        Assert.NotNull(model);
    }

    [Fact]
    public void FinancialModelFactory_RejectsUnsupportedModels()
    {
        var architecture = FinanceTestHelpers.CreateArchitecture<double>(inputSize: 4, outputSize: 4);
        var factory = new FinancialModelFactory<double>(architecture);

        Assert.Throws<NotSupportedException>(() => factory.Create(typeof(string), new Dictionary<string, object>()));
    }

    [Theory]
    [MemberData(nameof(SupportedAutoMlModelTypes))]
    public void FinancialSearchSpace_ProvidesRanges(Type modelType)
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var ranges = searchSpace.GetSearchSpace(modelType);

        Assert.NotNull(ranges);
        Assert.NotEmpty(ranges);
    }
}
