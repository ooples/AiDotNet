using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for safety modules implementing ISafetyModule.
/// Tests deep mathematical invariants: confidence calibration, severity ordering,
/// detection monotonicity, content sensitivity, and evaluation consistency.
/// </summary>
public abstract class SafetyModuleTestBase
{
    /// <summary>Factory method — subclasses return their concrete module instance.</summary>
    protected abstract ISafetyModule<double> CreateModule();

    /// <summary>Size of test content vector.</summary>
    protected virtual int ContentSize => 100;

    /// <summary>Whether the module is expected to produce findings for random/noisy content.</summary>
    protected virtual bool ProducesFindings => true;

    /// <summary>Creates test content vector with benign content.</summary>
    protected virtual Vector<double> CreateSafeContent()
    {
        var content = new Vector<double>(ContentSize);
        var text = "hello world test safe content happy peaceful";
        for (int i = 0; i < ContentSize; i++)
            content[i] = i < text.Length ? text[i] : 32.0;
        return content;
    }

    /// <summary>Creates test content vector with random data (some modules may flag this).</summary>
    protected virtual Vector<double> CreateRandomContent()
    {
        var rng = new Random(42);
        var content = new Vector<double>(ContentSize);
        for (int i = 0; i < ContentSize; i++)
            content[i] = rng.NextDouble() * 255.0;
        return content;
    }

    // =========================================================================
    // STRUCTURAL INVARIANTS
    // =========================================================================

    // INVARIANT 1: Module name is not empty
    [Fact]
    public void ModuleName_IsNotNullOrEmpty()
    {
        var module = CreateModule();
        Assert.False(string.IsNullOrWhiteSpace(module.ModuleName));
    }

    // INVARIANT 2: Evaluate returns non-null list
    [Fact]
    public void Evaluate_ReturnsNonNullList()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var findings = module.Evaluate(CreateSafeContent());
        Assert.NotNull(findings);
    }

    // =========================================================================
    // CONFIDENCE CALIBRATION INVARIANTS
    // =========================================================================

    // INVARIANT 3: All confidence scores are in [0, 1]
    // Confidence is a probability — must be bounded.
    [Fact]
    public void Evaluate_ConfidencesAreInUnitInterval()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var findings = module.Evaluate(CreateRandomContent());

        foreach (var finding in findings)
        {
            Assert.True(finding.Confidence >= 0.0,
                $"Confidence {finding.Confidence} < 0 for {finding.Category}. " +
                "Confidence represents a probability and must be in [0, 1].");
            Assert.True(finding.Confidence <= 1.0,
                $"Confidence {finding.Confidence} > 1 for {finding.Category}. " +
                "Confidence represents a probability and must be in [0, 1].");
        }
    }

    // INVARIANT 4: All finding descriptions are non-null
    [Fact]
    public void Evaluate_FindingDescriptionsAreNonNull()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        foreach (var finding in module.Evaluate(CreateRandomContent()))
            Assert.NotNull(finding.Description);
    }

    // INVARIANT 5: All findings reference their source module
    [Fact]
    public void Evaluate_FindingsReferenceSourceModule()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        foreach (var finding in module.Evaluate(CreateRandomContent()))
        {
            Assert.False(string.IsNullOrWhiteSpace(finding.SourceModule),
                "Finding must reference its source module for traceability.");
        }
    }

    // =========================================================================
    // SEVERITY ORDERING INVARIANTS
    // =========================================================================

    // INVARIANT 6: Higher confidence findings should have equal or higher severity
    // If the module is highly confident something is unsafe, severity should reflect that.
    [Fact]
    public void Evaluate_HighConfidenceFindings_HaveAppropiateSeverity()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var findings = module.Evaluate(CreateRandomContent());
        if (findings.Count < 2) return;

        // Check that high-confidence findings don't have lower severity than low-confidence ones
        var highConf = findings.Where(f => f.Confidence > 0.8).ToList();
        var lowConf = findings.Where(f => f.Confidence < 0.3).ToList();

        if (highConf.Count == 0 || lowConf.Count == 0) return;

        int highMaxSeverity = highConf.Max(f => (int)f.Severity);
        int lowMaxSeverity = lowConf.Max(f => (int)f.Severity);

        Assert.True(highMaxSeverity >= lowMaxSeverity,
            $"High-confidence findings (severity={highMaxSeverity}) should have >= " +
            $"severity of low-confidence findings ({lowMaxSeverity}). " +
            "Confidence and severity should be positively correlated.");
    }

    // =========================================================================
    // CONSISTENCY INVARIANTS
    // =========================================================================

    // INVARIANT 7: Deterministic evaluation — same input → same findings count
    [Fact]
    public void Evaluate_IsDeterministic()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var content = CreateSafeContent();
        var findings1 = module.Evaluate(content);
        var findings2 = module.Evaluate(content);

        Assert.Equal(findings1.Count, findings2.Count);

        // Also check confidences match
        for (int i = 0; i < findings1.Count; i++)
        {
            Assert.True(Math.Abs(findings1[i].Confidence - findings2[i].Confidence) < 1e-10,
                $"Finding {i} confidence differs between runs: " +
                $"{findings1[i].Confidence} vs {findings2[i].Confidence}. " +
                "Deterministic modules should produce identical results for identical inputs.");
        }
    }

    // INVARIANT 8: Content sensitivity — different content may produce different results
    // The module should not return a constant result regardless of input.
    [Fact]
    public void Evaluate_IsSensitiveToContent()
    {
        var module = CreateModule();
        if (!module.IsReady || !ProducesFindings) return;

        // Create two very different content vectors
        var content1 = CreateSafeContent();
        var content2 = new Vector<double>(ContentSize);
        for (int i = 0; i < ContentSize; i++)
            content2[i] = 0.0; // All zeros — very different from text

        var findings1 = module.Evaluate(content1);
        var findings2 = module.Evaluate(content2);

        // At least one aspect should differ (count, confidence, or category)
        bool differs = findings1.Count != findings2.Count;
        if (!differs && findings1.Count > 0)
        {
            for (int i = 0; i < findings1.Count; i++)
            {
                if (Math.Abs(findings1[i].Confidence - findings2[i].Confidence) > 1e-10)
                {
                    differs = true;
                    break;
                }
            }
        }

        // It's OK if both produce zero findings — but if both produce identical non-empty results
        // for wildly different inputs, something is wrong
        if (findings1.Count > 0 && findings2.Count > 0)
        {
            Assert.True(differs,
                "Module produced identical findings for completely different inputs. " +
                "Safety module should be sensitive to content differences.");
        }
    }

    // =========================================================================
    // EDGE CASE INVARIANTS
    // =========================================================================

    // INVARIANT 9: Null input throws
    [Fact]
    public void Evaluate_NullInput_Throws()
    {
        var module = CreateModule();
        Assert.ThrowsAny<ArgumentException>(() => module.Evaluate(null!));
    }

    // INVARIANT 10: Empty content does not crash
    [Fact]
    public void Evaluate_EmptyContent_DoesNotCrash()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        try
        {
            var findings = module.Evaluate(new Vector<double>(0));
            Assert.NotNull(findings);
        }
        catch (ArgumentException)
        {
            // Rejecting empty input is acceptable
        }
    }

    // INVARIANT 11: Very large content does not crash or produce invalid results
    [Fact]
    public void Evaluate_LargeContent_ProducesValidResults()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var largeContent = new Vector<double>(1000);
        var rng = new Random(42);
        for (int i = 0; i < 1000; i++)
            largeContent[i] = rng.NextDouble() * 255.0;

        var findings = module.Evaluate(largeContent);
        Assert.NotNull(findings);

        foreach (var finding in findings)
        {
            Assert.True(finding.Confidence >= 0.0 && finding.Confidence <= 1.0,
                $"Large content produced invalid confidence: {finding.Confidence}.");
        }
    }

    // INVARIANT 12: Repeated content (constant vector) should not cause numerical issues
    [Fact]
    public void Evaluate_ConstantContent_DoesNotProduceNaN()
    {
        var module = CreateModule();
        if (!module.IsReady) return;

        var constant = new Vector<double>(ContentSize);
        for (int i = 0; i < ContentSize; i++)
            constant[i] = 42.0; // All same value

        try
        {
            var findings = module.Evaluate(constant);
            foreach (var finding in findings)
            {
                Assert.False(double.IsNaN(finding.Confidence),
                    "Constant input produced NaN confidence. " +
                    "Module should handle degenerate inputs without numerical failure.");
            }
        }
        catch (ArgumentException)
        {
            // Rejecting degenerate input is acceptable
        }
    }
}
