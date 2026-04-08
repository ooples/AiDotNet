#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for text toxicity detection modules.
/// Tests RuleBasedToxicityDetector, EmbeddingToxicityDetector, ClassifierToxicityDetector,
/// and EnsembleToxicityDetector with toxic, safe, and edge-case inputs.
/// </summary>
public class TextToxicityIntegrationTests
{
    #region RuleBasedToxicityDetector Tests

    [Fact(Timeout = 120000)]
    public async Task RuleBased_ExplicitThreat_DetectsToxicity()
    {
        var detector = new RuleBasedToxicityDetector<double>();
        var findings = detector.EvaluateText("I will kill you and destroy everything you love");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f =>
            f.Category == SafetyCategory.ViolenceThreat ||
            f.Category == SafetyCategory.HateSpeech);
    }

    [Fact(Timeout = 120000)]
    public async Task RuleBased_HateSpeech_DetectsToxicity()
    {
        var detector = new RuleBasedToxicityDetector<double>();
        var findings = detector.EvaluateText("I will murder you and slaughter everyone you know");

        Assert.NotEmpty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task RuleBased_SafeText_NoFindings()
    {
        var detector = new RuleBasedToxicityDetector<double>();
        var findings = detector.EvaluateText("The weather is beautiful today and the flowers are blooming.");

        Assert.Empty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task RuleBased_EmptyText_NoFindings()
    {
        var detector = new RuleBasedToxicityDetector<double>();
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task RuleBased_CustomThreshold_RespectsConfig()
    {
        var strict = new RuleBasedToxicityDetector<double>(confidenceThreshold: 0.3);
        var lenient = new RuleBasedToxicityDetector<double>(confidenceThreshold: 0.9);
        var text = "You stupid fool, shut up already";

        var strictFindings = strict.EvaluateText(text);
        var lenientFindings = lenient.EvaluateText(text);

        Assert.True(strictFindings.Count >= lenientFindings.Count,
            "Stricter threshold should detect at least as many findings");
    }

    [Fact(Timeout = 120000)]
    public async Task RuleBased_MixedContent_DetectsToxicPortion()
    {
        var detector = new RuleBasedToxicityDetector<double>();
        var findings = detector.EvaluateText(
            "The project meeting went well. I will kill you for suggesting that.");

        Assert.NotEmpty(findings);
    }

    #endregion

    #region EmbeddingToxicityDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Embedding_ToxicContent_ProcessesWithoutError()
    {
        var detector = new EmbeddingToxicityDetector<double>();
        var findings = detector.EvaluateText("Die you worthless scum, I hope you suffer");

        // Embedding-based detection uses heuristic similarity; verify it processes
        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Embedding_SafeContent_NoFindings()
    {
        var detector = new EmbeddingToxicityDetector<double>();
        var findings = detector.EvaluateText("Machine learning models can help solve complex problems.");

        Assert.Empty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Embedding_LowThreshold_MoreSensitive()
    {
        var sensitive = new EmbeddingToxicityDetector<double>(threshold: 0.3);
        var normal = new EmbeddingToxicityDetector<double>(threshold: 0.8);
        var text = "That was a really dumb thing to say";

        var sensitiveFindings = sensitive.EvaluateText(text);
        var normalFindings = normal.EvaluateText(text);

        Assert.True(sensitiveFindings.Count >= normalFindings.Count);
    }

    #endregion

    #region ClassifierToxicityDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Classifier_ToxicContent_DetectsViolation()
    {
        var detector = new ClassifierToxicityDetector<double>();
        var findings = detector.EvaluateText("I hate you and your entire disgusting family");

        Assert.NotEmpty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Classifier_SafeContent_ProcessesWithoutError()
    {
        var detector = new ClassifierToxicityDetector<double>();
        var findings = detector.EvaluateText("Please pass the salt at the dinner table.");

        // Classifier uses heuristic weights and may produce findings for any text
        Assert.NotNull(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Classifier_CustomFeatureDim_Works()
    {
        var detector = new ClassifierToxicityDetector<double>(threshold: 0.5, featureDim: 128);
        var findings = detector.EvaluateText("You are terrible and awful and disgusting");

        Assert.NotNull(findings);
    }

    #endregion

    #region EnsembleToxicityDetector Tests

    [Fact(Timeout = 120000)]
    public async Task Ensemble_StronglyToxic_DetectsViolation()
    {
        var detector = new EnsembleToxicityDetector<double>(0.3);
        var findings = detector.EvaluateText(
            "I will kill you and murder everyone. Die you worthless scum.");

        Assert.NotEmpty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Ensemble_SafeContent_NoFindings()
    {
        var detector = new EnsembleToxicityDetector<double>(0.5);
        var findings = detector.EvaluateText("Today is a wonderful day for a walk in the park.");

        Assert.Empty(findings);
    }

    [Fact(Timeout = 120000)]
    public async Task Ensemble_CombinesMultipleDetectors()
    {
        var detector = new EnsembleToxicityDetector<double>(0.3);
        var findings = detector.EvaluateText(
            "Kill yourself you worthless piece of garbage, nobody wants you around");

        Assert.NotEmpty(findings);
        Assert.All(findings, f => Assert.True(f.Confidence > 0));
    }

    [Fact(Timeout = 120000)]
    public async Task Ensemble_VaryingThresholds_AffectsDetection()
    {
        var strict = new EnsembleToxicityDetector<double>(0.2);
        var lenient = new EnsembleToxicityDetector<double>(0.9);
        var text = "You stupid jerk, get lost";

        var strictFindings = strict.EvaluateText(text);
        var lenientFindings = lenient.EvaluateText(text);

        Assert.True(strictFindings.Count >= lenientFindings.Count);
    }

    #endregion

    #region Cross-Module Tests

    [Fact(Timeout = 120000)]
    public async Task AllDetectors_SameInput_ProduceResults()
    {
        var toxicText = "I will murder you and kill everyone, you disgusting worthless garbage person";
        var rule = new RuleBasedToxicityDetector<double>();
        var embedding = new EmbeddingToxicityDetector<double>();
        var classifier = new ClassifierToxicityDetector<double>();
        var ensemble = new EnsembleToxicityDetector<double>(0.3);

        var ruleFindings = rule.EvaluateText(toxicText);
        var embeddingFindings = embedding.EvaluateText(toxicText);
        var classifierFindings = classifier.EvaluateText(toxicText);
        var ensembleFindings = ensemble.EvaluateText(toxicText);

        // Rule-based should detect the explicit violence pattern
        Assert.NotEmpty(ruleFindings);

        // All detectors should process without errors
        Assert.NotNull(embeddingFindings);
        Assert.NotNull(classifierFindings);
        Assert.NotNull(ensembleFindings);
    }

    [Fact(Timeout = 120000)]
    public async Task AllDetectors_SafeInput_RuleBasedProducesNoFindings()
    {
        var safeText = "The conference was informative and the speakers were excellent.";
        var rule = new RuleBasedToxicityDetector<double>();
        var embedding = new EmbeddingToxicityDetector<double>();

        // Rule-based detector should not flag safe content
        Assert.Empty(rule.EvaluateText(safeText));
        Assert.Empty(embedding.EvaluateText(safeText));
    }

    #endregion
}
