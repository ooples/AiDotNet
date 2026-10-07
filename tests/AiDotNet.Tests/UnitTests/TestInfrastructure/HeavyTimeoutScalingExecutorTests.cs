using System;
using System.Collections.Generic;
using AiDotNet.Tests.TestInfrastructure;
using Xunit;
using Xunit.Sdk;

namespace AiDotNet.Tests.UnitTests.TestInfrastructure;

/// <summary>
/// The nightly heavy lane lengthens HeavyTimeout tests' xUnit timeout through <see cref="HeavyTimeoutScalingExecutor"/>
/// (#2087). These pin what that relies on, so an xUnit upgrade or a trait rename fails here instead of silently
/// leaving the lane on the PR gate's 120 s budget.
/// </summary>
public class HeavyTimeoutScalingExecutorTests
{
    [Fact]
    public void XunitStillLetsTheTimeoutBeSet()
    {
        var setter = typeof(XunitTestCase).GetProperty(nameof(XunitTestCase.Timeout))?.GetSetMethod(nonPublic: true);
        Assert.NotNull(setter);
    }

    [Theory]
    [InlineData(null, 1)]
    [InlineData("", 1)]
    [InlineData("abc", 1)]
    [InlineData("0", 1)]
    [InlineData("-3", 1)]
    [InlineData("1", 1)]
    [InlineData("4", 4)]
    public void Scale_IsOneUnlessAPositiveMultiplierIsSet(string? raw, int expected)
        => Assert.Equal(expected, HeavyTimeoutScalingExecutor.ParseScale(raw));

    [Theory]
    [InlineData("HeavyTimeout", true)]
    [InlineData("GPU", false)]
    [InlineData("heavytimeout", false)]
    public void OnlyHeavyTimeoutCasesAreScaled(string category, bool expected)
    {
        var traits = new Dictionary<string, List<string>> { ["Category"] = new List<string> { category } };
        Assert.Equal(expected, HeavyTimeoutScalingExecutor.IsHeavyTimeout(traits));
    }
}
