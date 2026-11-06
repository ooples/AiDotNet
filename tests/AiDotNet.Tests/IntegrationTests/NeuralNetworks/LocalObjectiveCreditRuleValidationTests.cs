using System;
using AiDotNet.NeuralNetworks.CreditAssignment;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public partial class LocalObjectiveCreditRuleTests
{
    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-1.0)]
    public void DistillationRejectsInvalidCoefficients(double value)
    {
        Assert.Equal("hiddenWeight", Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), hiddenWeight: value, kdAlpha: 0)).ParamName);
        Assert.Equal("temperature", Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), temperature: value, kdAlpha: 0)).ParamName);
        Assert.Equal("projectionLearningRate", Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), projectionLearningRate: value, kdAlpha: 0)).ParamName);
        Assert.Equal("localWeight", Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TiedReadoutLocalErrorCreditRule<double>(localWeight: value)).ParamName);
    }

    [Fact]
    public void ZeroIsAllowedOnlyForHiddenWeight()
    {
        var rule = new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), hiddenWeight: 0, kdAlpha: 0);
        Assert.Null(rule.BuildLocalObjective(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), temperature: 0, kdAlpha: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), projectionLearningRate: 0, kdAlpha: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TiedReadoutLocalErrorCreditRule<double>(localWeight: 0));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    public void DistillationRejectsInvalidBlendWeights(double value)
    {
        Assert.Equal("kdAlpha", Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), _ => throw new Exception(), kdAlpha: value)).ParamName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(double.Epsilon)]
    public void DistillationAcceptsFiniteBlendBoundaries(double value)
    {
        _ = new LayerwiseDistillationCreditRule<double>(_ => throw new Exception(), _ => throw new Exception(), kdAlpha: value);
    }
}
