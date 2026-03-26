using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for safety modules implementing ISafetyModule.
/// Tests mathematical invariants: finite confidence scores, valid severity levels,
/// consistent properties, and input handling.
/// </summary>
public abstract class SafetyModuleTestBase
{
    /// <summary>Factory method — subclasses return their concrete module instance.</summary>
    protected abstract ISafetyModule<double> CreateModule();

    /// <summary>Size of test content vector.</summary>
    protected virtual int ContentSize => 100;

    /// <summary>Creates test content vector with safe content (should produce no/few findings).</summary>
    protected virtual Vector<double> CreateSafeContent()
    {
        var content = new Vector<double>(ContentSize);
        // Fill with ASCII for "hello world test safe content"
        var text = "hello world test safe content";
        for (int i = 0; i < ContentSize; i++)
        {
            content[i] = i < text.Length ? text[i] : 32.0; // space padding
        }

        return content;
    }

    /// <summary>Creates test content vector with random data.</summary>
    protected virtual Vector<double> CreateRandomContent()
    {
        var rng = new Random(42);
        var content = new Vector<double>(ContentSize);
        for (int i = 0; i < ContentSize; i++)
        {
            content[i] = rng.NextDouble() * 255.0;
        }

        return content;
    }

    // =========================================================================
    // INVARIANT 1: Module name is not empty
    // =========================================================================

    [Fact]
    public void ModuleName_IsNotNullOrEmpty()
    {
        var module = CreateModule();
        Assert.False(string.IsNullOrWhiteSpace(module.ModuleName),
            "Safety module name should not be null or empty.");
    }

    // =========================================================================
    // INVARIANT 2: Evaluate returns non-null list
    // =========================================================================

    [Fact]
    public void Evaluate_ReturnsNonNullList()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateSafeContent();
        var findings = module.Evaluate(content);

        Assert.NotNull(findings);
    }

    // =========================================================================
    // INVARIANT 3: All finding confidences are in [0, 1]
    // =========================================================================

    [Fact]
    public void Evaluate_FindingConfidencesAreValid()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateRandomContent();
        var findings = module.Evaluate(content);

        foreach (var finding in findings)
        {
            Assert.True(finding.Confidence >= 0.0 && finding.Confidence <= 1.0,
                $"Finding confidence {finding.Confidence} is outside valid range [0, 1]. " +
                $"Module: {module.ModuleName}, Category: {finding.Category}");
        }
    }

    // =========================================================================
    // INVARIANT 4: All finding descriptions are non-null
    // =========================================================================

    [Fact]
    public void Evaluate_FindingDescriptionsAreNonNull()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateRandomContent();
        var findings = module.Evaluate(content);

        foreach (var finding in findings)
        {
            Assert.NotNull(finding.Description);
        }
    }

    // =========================================================================
    // INVARIANT 5: All findings reference the source module
    // =========================================================================

    [Fact]
    public void Evaluate_FindingsReferenceSourceModule()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateRandomContent();
        var findings = module.Evaluate(content);

        foreach (var finding in findings)
        {
            Assert.False(string.IsNullOrWhiteSpace(finding.SourceModule),
                "Finding should reference the source module name.");
        }
    }

    // =========================================================================
    // INVARIANT 6: Null input throws ArgumentNullException
    // =========================================================================

    [Fact]
    public void Evaluate_NullInput_Throws()
    {
        var module = CreateModule();

        Assert.ThrowsAny<ArgumentException>(() => module.Evaluate(null!));
    }

    // =========================================================================
    // INVARIANT 7: Empty content does not crash
    // =========================================================================

    [Fact]
    public void Evaluate_EmptyContent_DoesNotCrash()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var emptyContent = new Vector<double>(0);

        // Should not throw — either returns empty findings or handles gracefully
        try
        {
            var findings = module.Evaluate(emptyContent);
            Assert.NotNull(findings);
        }
        catch (ArgumentException)
        {
            // Acceptable to reject empty input
        }
    }

    // =========================================================================
    // INVARIANT 8: Repeated evaluation is consistent
    // =========================================================================

    [Fact]
    public void Evaluate_IsConsistentAcrossRepeatedCalls()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateSafeContent();
        var findings1 = module.Evaluate(content);
        var findings2 = module.Evaluate(content);

        Assert.Equal(findings1.Count, findings2.Count);
    }
}
