using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

public class FinanceModelsIntegrationTests
{
    public static IEnumerable<object[]> FinanceModelTypesFloat =>
        FinanceModelTestFactory.GetFinanceModelTypes<float>()
            .Select(type => new object[] { type });

    public static IEnumerable<object[]> FinanceModelTypesDouble =>
        FinanceModelTestFactory.GetFinanceModelTypes<double>()
            .Select(type => new object[] { type });

    [Theory]
    [MemberData(nameof(FinanceModelTypesFloat))]
    public void FinanceModels_Float_Native_Predict_Train_Serialize(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(FinanceModelTypesDouble))]
    public void FinanceModels_Double_Native_Predict_Train_Serialize(Type modelType)
    {
        FinanceModelTestFactory.RunFullModelSmokeTest<double>(modelType);
    }

    [Theory]
    [MemberData(nameof(FinanceModelTypesFloat))]
    public void FinanceModels_Float_OnnxConstructor_ThrowsWhenMissingFile(Type modelType)
    {
        FinanceModelTestFactory.AssertOnnxConstructorFails<float>(modelType);
    }

    [Theory]
    [MemberData(nameof(FinanceModelTypesDouble))]
    public void FinanceModels_Double_OnnxConstructor_ThrowsWhenMissingFile(Type modelType)
    {
        FinanceModelTestFactory.AssertOnnxConstructorFails<double>(modelType);
    }
}
