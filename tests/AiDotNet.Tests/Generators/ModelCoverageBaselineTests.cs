using System;
using System.Collections.Generic;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Phase 2 of #2091: the measured coverage figure is a gate, so the gap cannot silently widen.
///
/// Phase 1 established that a coverage figure is only meaningful in the compilation that can see
/// test classes. This fixture reads the figure THIS assembly published and holds it to a recorded
/// baseline, so a change that stops the scaffold generating for some family fails here rather than
/// going unnoticed among the build's thousands of warnings - which is exactly how the report sat at
/// a self-matched 0.1% without anyone acting on it.
/// </summary>
public class ModelCoverageBaselineTests
{
    /// <summary>
    /// Models with model-family test coverage, measured on the branch that fixed the instrument
    /// (#2091 Phase 1) at 1466 of 1816, then raised to 1482
    /// once ResolveTestBaseClass stopped routing models into families they cannot satisfy. Independently corroborated at 1443 by a whole-word type
    /// reference count over the hand-written AND generated test corpus - the generated half being
    /// what the original ~36% measurement missed, since those classes exist only at compile time.
    ///
    /// RAISING this number is the normal direction and needs no ceremony. LOWERING it means models
    /// that were covered no longer are: establish why before editing this constant, because the
    /// point of the gate is that the number cannot quietly fall.
    /// </summary>
    private const int BaselineTestedCount = 1482;

    /// <summary>
    /// Total annotated models at the same measurement. Only a floor: the census grows as models are
    /// added, and a growing census must not by itself trip the tested-count gate.
    /// </summary>
    private const int BaselineTotalModels = 1816;

    [Fact]
    public void ThisAssemblyPublishesAMeasuredCoverageFigure()
    {
        var report = LoadReport();

        // If this flips false, the generator has stopped being able to see test classes in the
        // test compilation - the Phase 1 defect, reintroduced from the other side.
        Assert.True(
            report.IsMeasurable,
            "The test assembly's TestCoverage reports IsMeasurable=false, so the coverage figure "
            + "in this compilation is not a measurement. See #2091.");
    }

    [Fact]
    public void CoverageHasNotRegressedBelowTheBaseline()
    {
        var report = LoadReport();

        Assert.True(
            report.TestedCount >= BaselineTestedCount,
            $"Model-family coverage fell from {BaselineTestedCount} to {report.TestedCount} of "
            + $"{report.TotalModels} models. Find which models lost their scaffold before editing "
            + "BaselineTestedCount - a falling number is the regression this gate exists to catch.");
    }

    [Fact]
    public void CensusHasNotShrunk()
    {
        var report = LoadReport();

        Assert.True(
            report.TotalModels >= BaselineTotalModels,
            $"The annotated model census fell from {BaselineTotalModels} to {report.TotalModels}. "
            + "Models losing their metadata attributes drop out of the census entirely, which would "
            + "otherwise let coverage 'improve' by shrinking the denominator.");
    }

    [Fact]
    public void ReportIsInternallyConsistent()
    {
        var report = LoadReport();

        Assert.Equal(report.TotalModels, report.TestedCount + report.UntestedCount);

        var expected = report.TotalModels > 0
            ? Math.Round(report.TestedCount * 100.0 / report.TotalModels, 1)
            : 0.0;
        Assert.Equal(expected, report.CoveragePercent, 1);
    }

    private readonly record struct Report(
        bool IsMeasurable, int TotalModels, int TestedCount, int UntestedCount, double CoveragePercent);

    /// <summary>
    /// Reads the report by reflection rather than by naming the type directly. AiDotNet sets
    /// InternalsVisibleTo for this assembly and the generator emits TestCoverage into BOTH
    /// compilations, so the name AiDotNet.Generated.TestCoverage is in scope twice here. A direct
    /// reference silently binds to this assembly's copy and warns (CS0436); going through this
    /// assembly's own Type makes the choice explicit instead of incidental.
    /// </summary>
    private static Report LoadReport()
    {
        var type = typeof(ModelCoverageBaselineTests).Assembly.GetType("AiDotNet.Generated.TestCoverage");
        if (type is null)
        {
            Assert.Fail(
                "AiDotNet.Generated.TestCoverage was not generated into the test assembly. The "
                + "TestScaffoldGenerator only runs for compilations named 'AiDotNet' or "
                + "'AiDotNetTests'; if this assembly was renamed, that gate needs updating too.");
        }

        return new Report(
            Read<bool>(type, "IsMeasurable"),
            Read<int>(type, "TotalModels"),
            Read<int>(type, "TestedCount"),
            Read<int>(type, "UntestedCount"),
            Read<double>(type, "CoveragePercent"));
    }

    private static TValue Read<TValue>(Type type, string name)
    {
        var field = type.GetField(name, BindingFlags.Public | BindingFlags.Static);
        if (field is null)
        {
            Assert.Fail($"TestCoverage.{name} is missing. The coverage report's shape changed.");
        }

        var value = field.GetRawConstantValue();
        if (value is not TValue typed)
        {
            Assert.Fail($"TestCoverage.{name} is not a {typeof(TValue).Name}.");
            return default;
        }

        return typed;
    }
}
