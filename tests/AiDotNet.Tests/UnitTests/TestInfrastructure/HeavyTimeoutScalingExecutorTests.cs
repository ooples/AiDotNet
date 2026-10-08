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
    [InlineData("2147483647", int.MaxValue)]
    [InlineData("2147483648", 1)]
    public void Scale_IsOneUnlessAPositiveMultiplierIsSet(string? raw, int expected)
        => Assert.Equal(expected, HeavyTimeoutScalingExecutor.ParseScale(raw));

    [Theory]
    [InlineData(true, "HeavyTimeout")]
    [InlineData(true, "GPU", "HeavyTimeout")]
    [InlineData(false, "GPU")]
    [InlineData(false, "heavytimeout")]
    public void OnlyHeavyTimeoutCasesAreScaled(bool expected, params string[] categories)
    {
        var traits = new Dictionary<string, List<string>> { ["Category"] = new List<string>(categories) };
        Assert.Equal(expected, HeavyTimeoutScalingExecutor.IsHeavyTimeout(traits));
    }

    [Fact]
    public void ACaseWithNoCategoryIsNotScaled()
    {
        var traits = new Dictionary<string, List<string>> { ["Owner"] = new List<string> { "someone" } };
        Assert.False(HeavyTimeoutScalingExecutor.IsHeavyTimeout(traits));
    }

    [Fact]
    public void ApplyScale_MultipliesOnlyTheHeavyCase_AndClampsAtIntMax()
    {
        var heavy = CaseFor(typeof(HeavyTimeoutScalingHeavyFixture), nameof(HeavyTimeoutScalingHeavyFixture.Heavy));
        var ordinary = CaseFor(typeof(HeavyTimeoutScalingOrdinaryFixture), nameof(HeavyTimeoutScalingOrdinaryFixture.Ordinary));
        Assert.Equal(1500, heavy.Timeout);
        Assert.Equal(1500, ordinary.Timeout);

        Assert.True(HeavyTimeoutScalingExecutor.ApplyScale(new IXunitTestCase[] { heavy, ordinary }, 4));
        Assert.Equal(6000, heavy.Timeout);
        Assert.Equal(1500, ordinary.Timeout);

        Assert.True(HeavyTimeoutScalingExecutor.ApplyScale(new IXunitTestCase[] { heavy }, int.MaxValue));
        Assert.Equal(int.MaxValue, heavy.Timeout);
    }

    private static XunitTestCase CaseFor(Type fixture, string method)
    {
        var assembly = new TestAssembly(Reflector.Wrap(fixture.Assembly));
        var collection = new TestCollection(assembly, null, "HeavyTimeoutScalingExecutorTests fixtures");
        var testClass = new TestClass(collection, Reflector.Wrap(fixture));
        var testMethod = new TestMethod(testClass, Reflector.Wrap(fixture.GetMethod(method)
            ?? throw new InvalidOperationException($"{fixture.Name}.{method} not found.")));
        return new XunitTestCase(new NullMessageSink(), TestMethodDisplay.ClassAndMethod, TestMethodDisplayOptions.None, testMethod);
    }
}

// Fixtures the executor test builds xUnit cases from. Skipped so they never execute as tests themselves.
[Trait("Category", "HeavyTimeout")]
public class HeavyTimeoutScalingHeavyFixture
{
    [Fact(Timeout = 1500, Skip = "Fixture for HeavyTimeoutScalingExecutorTests; never runs.")]
    public void Heavy() { }
}

public class HeavyTimeoutScalingOrdinaryFixture
{
    [Fact(Timeout = 1500, Skip = "Fixture for HeavyTimeoutScalingExecutorTests; never runs.")]
    public void Ordinary() { }
}
