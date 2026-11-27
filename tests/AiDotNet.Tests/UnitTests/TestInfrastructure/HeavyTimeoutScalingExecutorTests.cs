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

    internal static XunitTestCase CaseFor(Type fixture, string method)
    {
        var assembly = new TestAssembly(Reflector.Wrap(fixture.Assembly));
        var collection = new TestCollection(assembly, null, "HeavyTimeoutScalingExecutorTests fixtures");
        var testClass = new TestClass(collection, Reflector.Wrap(fixture));
        var testMethod = new TestMethod(testClass, Reflector.Wrap(fixture.GetMethod(method)
            ?? throw new InvalidOperationException($"{fixture.Name}.{method} not found.")));
        return new XunitTestCase(new NullMessageSink(), TestMethodDisplay.ClassAndMethod, TestMethodDisplayOptions.None, testMethod);
    }
}

// Fixtures the executor test builds xUnit cases from. They are discovered like any class, so they run as tests
// too: a skipped one would leave the heavy lane a chunk in which nothing ran, which that lane reports as a
// failure (#2447), and the heavy one doubles as a canary that the lane runs what it discovers. Each checks its
// own case's scaling, so wherever it runs it asserts something about itself.
[Trait("Category", "HeavyTimeout")]
public class HeavyTimeoutScalingHeavyFixture
{
    [Fact(Timeout = 1500)]
    public System.Threading.Tasks.Task Heavy()
    {
        var testCase = HeavyTimeoutScalingExecutorTests.CaseFor(typeof(HeavyTimeoutScalingHeavyFixture), nameof(Heavy));
        Assert.True(HeavyTimeoutScalingExecutor.ApplyScale(new IXunitTestCase[] { testCase }, 4));
        Assert.Equal(6000, testCase.Timeout);
        return System.Threading.Tasks.Task.CompletedTask;
    }
}

public class HeavyTimeoutScalingOrdinaryFixture
{
    [Fact(Timeout = 1500)]
    public System.Threading.Tasks.Task Ordinary()
    {
        var testCase = HeavyTimeoutScalingExecutorTests.CaseFor(
            typeof(HeavyTimeoutScalingOrdinaryFixture), nameof(Ordinary));
        HeavyTimeoutScalingExecutor.ApplyScale(new IXunitTestCase[] { testCase }, 4);
        Assert.Equal(1500, testCase.Timeout);
        return System.Threading.Tasks.Task.CompletedTask;
    }
}
