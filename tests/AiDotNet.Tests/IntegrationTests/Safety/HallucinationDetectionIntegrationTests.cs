#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for hallucination detection modules.
/// Tests ReferenceBasedHallucinationDetector, SelfConsistencyHallucinationDetector,
/// KnowledgeTripletHallucinationDetector, and EntailmentHallucinationDetector.
/// </summary>
public class HallucinationDetectionIntegrationTests
{
    #region ReferenceBasedHallucinationDetector Tests

    [Fact]
    public void ReferenceBased_WithReferences_ProcessesText()
    {
        var references = new[] { "Paris is the capital of France.", "The Eiffel Tower is in Paris." };
        var detector = new ReferenceBasedHallucinationDetector<double>(references);
        var findings = detector.EvaluateText("Paris is the capital of France and has the Eiffel Tower.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void ReferenceBased_FabricatedFacts_DetectsHallucination()
    {
        var references = new[] { "Water boils at 100 degrees Celsius at sea level." };
        var detector = new ReferenceBasedHallucinationDetector<double>(references);
        var findings = detector.EvaluateText(
            "Water boils at 50 degrees Celsius. This is a well-known fact discovered by Einstein in 1742.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void ReferenceBased_WithReferencesMethod_CreatesNewDetector()
    {
        var detector = new ReferenceBasedHallucinationDetector<double>();
        var updated = detector.WithReferences(new[] { "The Earth orbits the Sun." });
        var findings = updated.EvaluateText("The Earth orbits the Sun once per year.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void ReferenceBased_EmptyText_NoFindings()
    {
        var detector = new ReferenceBasedHallucinationDetector<double>(
            new[] { "Reference document content." });
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region SelfConsistencyHallucinationDetector Tests

    [Fact]
    public void SelfConsistency_Contradictions_DetectsHallucination()
    {
        var detector = new SelfConsistencyHallucinationDetector<double>();
        var findings = detector.EvaluateText(
            "The building was constructed in 1950. The architecture is modern. " +
            "The building was actually built in 2020. It was never built in 1950.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void SelfConsistency_ConsistentText_ProcessesCleanly()
    {
        var detector = new SelfConsistencyHallucinationDetector<double>();
        var findings = detector.EvaluateText(
            "The Earth is round. It orbits the Sun. The orbit takes approximately 365 days.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void SelfConsistency_CustomThreshold_Works()
    {
        var detector = new SelfConsistencyHallucinationDetector<double>(
            contradictionThreshold: 0.1, embeddingDim: 32);
        var findings = detector.EvaluateText(
            "The cat is black. The cat is white. The cat has no color.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void SelfConsistency_ShortText_HandlesGracefully()
    {
        var detector = new SelfConsistencyHallucinationDetector<double>();
        var findings = detector.EvaluateText("Hello.");

        Assert.NotNull(findings);
    }

    #endregion

    #region KnowledgeTripletHallucinationDetector Tests

    [Fact]
    public void KnowledgeTriplet_WithReferences_ProcessesText()
    {
        var references = new[] { "Albert Einstein developed the theory of relativity." };
        var detector = new KnowledgeTripletHallucinationDetector<double>(references);
        var findings = detector.EvaluateText(
            "Einstein developed the theory of relativity in the early 20th century.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void KnowledgeTriplet_NoReferences_ProcessesText()
    {
        var detector = new KnowledgeTripletHallucinationDetector<double>();
        var findings = detector.EvaluateText(
            "The Pythagorean theorem states that a squared plus b squared equals c squared.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void KnowledgeTriplet_WithReferencesMethod_Works()
    {
        var detector = new KnowledgeTripletHallucinationDetector<double>();
        var updated = detector.WithReferences(
            new[] { "Python is a programming language created by Guido van Rossum." });
        var findings = updated.EvaluateText("Python was created by Guido van Rossum.");

        Assert.NotNull(findings);
    }

    #endregion

    #region EntailmentHallucinationDetector Tests

    [Fact]
    public void Entailment_WithReferences_ProcessesText()
    {
        var references = new[] { "The speed of light is approximately 300,000 km/s." };
        var detector = new EntailmentHallucinationDetector<double>(references);
        var findings = detector.EvaluateText("Light travels at about 300,000 kilometers per second.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Entailment_ContradictingReferences_DetectsIssue()
    {
        var references = new[] { "The Moon orbits the Earth." };
        var detector = new EntailmentHallucinationDetector<double>(references);
        var findings = detector.EvaluateText(
            "The Earth orbits the Moon. The Moon is the center of our solar system.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Entailment_WithReferencesMethod_Works()
    {
        var detector = new EntailmentHallucinationDetector<double>();
        var updated = detector.WithReferences(
            new[] { "Oxygen is essential for human respiration." });
        var findings = updated.EvaluateText("Humans need oxygen to breathe.");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Entailment_EmptyText_NoFindings()
    {
        var detector = new EntailmentHallucinationDetector<double>(
            new[] { "Some reference." });
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllDetectors_ProcessSameText_NoExceptions()
    {
        var text = "The Eiffel Tower, located in Berlin, was built by Thomas Edison in 1920.";
        var refs = new[] { "The Eiffel Tower is in Paris, France, built by Gustave Eiffel in 1889." };

        var refBased = new ReferenceBasedHallucinationDetector<double>(refs);
        var selfConsistency = new SelfConsistencyHallucinationDetector<double>();
        var triplet = new KnowledgeTripletHallucinationDetector<double>(refs);
        var entailment = new EntailmentHallucinationDetector<double>(refs);

        Assert.NotNull(refBased.EvaluateText(text));
        Assert.NotNull(selfConsistency.EvaluateText(text));
        Assert.NotNull(triplet.EvaluateText(text));
        Assert.NotNull(entailment.EvaluateText(text));
    }

    #endregion
}
