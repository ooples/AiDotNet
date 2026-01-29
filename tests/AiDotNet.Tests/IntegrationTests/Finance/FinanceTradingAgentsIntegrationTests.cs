using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

public class FinanceTradingAgentsIntegrationTests
{
    public static IEnumerable<object[]> TradingAgentTypesFloat =>
        FinanceModelTestFactory.GetTradingAgentTypes<float>()
            .Select(type => new object[] { type });

    public static IEnumerable<object[]> TradingAgentTypesDouble =>
        FinanceModelTestFactory.GetTradingAgentTypes<double>()
            .Select(type => new object[] { type });

    [Theory]
    [MemberData(nameof(TradingAgentTypesFloat))]
    public void TradingAgents_Float_CanSelectActionsAndTrain(Type agentType)
    {
        FinanceModelTestFactory.RunTradingAgentSmokeTest<float>(agentType);
    }

    [Theory]
    [MemberData(nameof(TradingAgentTypesDouble))]
    public void TradingAgents_Double_CanSelectActionsAndTrain(Type agentType)
    {
        FinanceModelTestFactory.RunTradingAgentSmokeTest<double>(agentType);
    }
}
