using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers carrying the judge's written criticism forward. The judge was already being asked to explain its scores
/// and already returning that text; the run parsed the numbers and threw the reasoning away, so the search kept
/// rediscovering criticism it had already been given.
/// </summary>
public sealed class LlmJudgeCritiqueTests
{
    private const string Source = "def solve(x):\n    return x * 2\n";
    private const string Reasoning = "The loop rescans the list on every pass; hoist the lookup out of it.";

    [Fact]
    public async Task TheJudgesReasoningIsAttachedAsAnArtifact()
    {
        // Artifacts are the channel the engine shows to whoever proposes this candidate's successor, so landing
        // there is what turns the criticism into something the next attempt can answer.
        EvolutionTaskResult result = await Judge(Answer(reasoning: Reasoning)).EvaluateAsync(Candidate(), Context());

        EvolutionArtifact artifact = Assert.Single(result.Artifacts);
        Assert.Equal(LlmFeedbackOptions.CritiqueArtifactKey, artifact.Key);
        Assert.Equal(Reasoning, artifact.Text);
        Assert.True(artifact.IsRedacted);
    }

    [Fact]
    public async Task TheCritiqueChangesNothingAboutTheScore()
    {
        // The prose is a bonus channel. If it could move the quality, a judge could talk a broken program up by
        // writing more, which is exactly what the blend exists to prevent.
        EvolutionTaskResult withProse = await Judge(Answer(reasoning: Reasoning)).EvaluateAsync(Candidate(), Context());
        EvolutionTaskResult withoutProse = await Judge(Answer(reasoning: null)).EvaluateAsync(Candidate(), Context());

        Assert.Equal(withoutProse.Quality.GetValueOrDefault(), withProse.Quality.GetValueOrDefault(), 10);
        Assert.Equal(withoutProse.Descriptors["llm_average"], withProse.Descriptors["llm_average"], 10);
    }

    [Fact]
    public async Task CarryingItForwardCanBeTurnedOff()
    {
        EvolutionTaskResult result = await Judge(
            Answer(reasoning: Reasoning),
            new LlmFeedbackOptions { CarryCritiqueForward = false }).EvaluateAsync(Candidate(), Context());

        Assert.Empty(result.Artifacts);
        Assert.Equal(0.65, result.Quality.GetValueOrDefault(), 10);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public async Task AJudgeThatWroteNothingUsefulIsNotAFailure(string? reasoning)
    {
        // The scores are the contract; refusing an otherwise-good answer because the judge stayed quiet would spend
        // a retry to learn nothing.
        EvolutionTaskResult result = await Judge(Answer(reasoning)).EvaluateAsync(Candidate(), Context());

        Assert.Empty(result.Artifacts);
        Assert.Equal(0.65, result.Quality.GetValueOrDefault(), 10);
        Assert.Equal(1.0, result.Descriptors["llm_average"], 10);
    }

    [Fact]
    public async Task AReasoningFieldThatIsNotTextIsIgnoredRatherThanSerialized()
    {
        // A judge answering with an object here has not followed the schema, and dumping whatever it did send would
        // put unbounded JSON into the next prompt.
        var client = new FakeChatClient(
            "{\"correctness\": 1.0, \"efficiency\": 1.0, \"readability\": 1.0, \"reasoning\": {\"a\": [1, 2, 3]}}");
        EvolutionTaskResult result = await Judge(client).EvaluateAsync(Candidate(), Context());

        Assert.Empty(result.Artifacts);
        Assert.Equal(0.65, result.Quality.GetValueOrDefault(), 10);
    }

    [Fact]
    public async Task ALongCritiqueIsCutToTheBoundAndSaysSo()
    {
        string overlong = new string('x', 500);
        EvolutionTaskResult result = await Judge(
            Answer(overlong),
            new LlmFeedbackOptions { MaxCritiqueChars = 64 }).EvaluateAsync(Candidate(), Context());

        EvolutionArtifact artifact = Assert.Single(result.Artifacts);
        Assert.True(artifact.Text.Length <= 64);
        Assert.True(artifact.IsTruncated);
    }

    [Fact]
    public async Task TheMeasuredEvaluatorsOwnMetricsAndArtifactsSurviveTheJudge()
    {
        // The judge is a wrapper. One that silently drops what it wraps makes every metric query wrong for judged
        // runs and loses the evaluator's own output before it ever reaches a prompt.
        var measured = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(
                EvolutionEvaluationStatus.Completed,
                0.5,
                EvolutionOptimizationDirection.Maximize,
                new Dictionary<string, double>(StringComparer.Ordinal) { ["passRate"] = 0.5 },
                costUnits: 3,
                metrics: new Dictionary<string, double>(StringComparer.Ordinal) { ["accuracy"] = 0.42 },
                artifacts: new[] { new EvolutionArtifact("stderr", "one test failed") })));

        EvolutionTaskResult result = await new LlmJudgeProgramFitnessEvaluator<double>(
            Answer(Reasoning), measured).EvaluateAsync(Candidate(), Context());

        Assert.Equal(0.42, result.Metrics["accuracy"], 10);
        Assert.Contains(result.Artifacts, artifact => artifact.Key == "stderr");
        Assert.Contains(result.Artifacts, artifact => artifact.Key == LlmFeedbackOptions.CritiqueArtifactKey);
    }

    [Fact]
    public void TwoRunsThatCarryTheCritiqueDifferentlyAreNotTheSameRun()
    {
        // The critique changes the next prompt, so it has to reach the version hash or a resume could quietly swap
        // one behaviour for the other.
        string carried = Judge(new FakeChatClient("{}")).VersionHash;
        string dropped = Judge(new FakeChatClient("{}"),
            new LlmFeedbackOptions { CarryCritiqueForward = false }).VersionHash;
        string renamed = Judge(new FakeChatClient("{}"),
            new LlmFeedbackOptions { CritiqueField = "explanation" }).VersionHash;

        Assert.NotEqual(carried, dropped);
        Assert.NotEqual(carried, renamed);
    }

    [Fact]
    public void ACritiqueFieldColldingWithACriterionIsRefused()
    {
        // The same key cannot mean both a score and prose; the judge would have to pick one, and the failure would
        // otherwise show up as an unparseable answer once per candidate.
        var options = new LlmFeedbackOptions { CritiqueField = "correctness" };

        ArgumentException failure = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("CritiqueField", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ABlankOrOversizedCritiqueSettingIsRefused()
    {
        Assert.Throws<ArgumentException>(() => new LlmFeedbackOptions { CritiqueField = "  " }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new LlmFeedbackOptions { MaxCritiqueChars = 0 }.Validate());
    }

    [Fact]
    public void TheDerivedSchemaAsksTheJudgeForTheFieldThatIsCarriedForward()
    {
        // Reading a field the judge was never asked to write would make the feature silently do nothing.
        var builder = new AiDotNet.Evolution.Prompts.ProgramPromptBuilder();
        string request = string.Join("\n", builder
            .BuildEvaluationMessages(Candidate(), new[] { "correctness" })
            .Select(message => message.Text));

        Assert.Contains(LlmFeedbackOptions.DefaultCritiqueField, request, StringComparison.Ordinal);
    }

    private static EvolutionEvaluationContext Context() => new(1, 99UL, 7UL, 1);

    private static ProgramGenome Candidate() => new(Source, ProgramLanguage.Python);

    private static LlmJudgeProgramFitnessEvaluator<double> Judge(
        FakeChatClient client, LlmFeedbackOptions? options = null) =>
        new(client, Measured(0.5), options: options);

    private static IProgramFitnessEvaluator Measured(double quality) =>
        new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(
                EvolutionEvaluationStatus.Completed,
                quality,
                EvolutionOptimizationDirection.Maximize,
                new Dictionary<string, double>(StringComparer.Ordinal) { ["passRate"] = quality },
                costUnits: 3)));

    private static FakeChatClient Answer(string? reasoning) => new(
        "{\"correctness\": 1.0, \"efficiency\": 1.0, \"readability\": 1.0" +
        (reasoning is null ? string.Empty : ", \"reasoning\": \"" + reasoning.Replace("\"", "\\\"") + "\"") +
        "}");
}
