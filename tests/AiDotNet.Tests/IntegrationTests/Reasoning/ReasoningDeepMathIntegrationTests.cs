using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Aggregation;
using AiDotNet.Reasoning.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Reasoning;

/// <summary>
/// Deep integration tests for Reasoning models and aggregation:
/// ReasoningConfig defaults, ThoughtNode tree operations (PathScores, GetPathFromRoot,
/// IsLeaf/IsRoot, PathLength, CheckIsTerminalByHeuristic), ReasoningChain (StepScores,
/// AddStep, GetMinimumScore, GetAverageScore, IsFullyVerified, TotalRefinements),
/// ReasoningStep, ReasoningResult (GetSummary), MajorityVotingAggregator,
/// and WeightedAggregator.
/// </summary>
public class ReasoningDeepMathIntegrationTests
{
    // ============================
    // ReasoningConfig Default Tests
    // ============================

    [Fact]
    public void ReasoningConfig_Defaults_CorrectValues()
    {
        var config = new ReasoningConfig();

        Assert.Equal(10, config.MaxSteps);
        Assert.Equal(3, config.ExplorationDepth);
        Assert.Equal(3, config.BranchingFactor);
        Assert.Equal(5, config.NumSamples);
        Assert.Equal(0.7, config.Temperature);
        Assert.Equal(5, config.BeamWidth);
        Assert.False(config.EnableVerification);
        Assert.False(config.EnableSelfRefinement);
        Assert.Equal(2, config.MaxRefinementAttempts);
        Assert.Equal(0.7, config.VerificationThreshold);
        Assert.False(config.EnableExternalVerification);
        Assert.False(config.EnableTestTimeCompute);
        Assert.Equal(2.0, config.ComputeScalingFactor);
        Assert.Equal(60, config.MaxReasoningTimeSeconds);
        Assert.False(config.EnableContradictionDetection);
        Assert.False(config.EnableDiversitySampling);
    }

    // ============================
    // ThoughtNode: Tree Structure Tests
    // ============================

    [Fact]
    public void ThoughtNode_Root_IsRootAndIsLeaf()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0 };

        Assert.True(root.IsRoot());
        Assert.True(root.IsLeaf());
        Assert.Null(root.Parent);
        Assert.Empty(root.Children);
    }

    [Fact]
    public void ThoughtNode_WithChildren_NotLeaf()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0 };
        var child = new ThoughtNode<double> { Thought = "Child", Depth = 1, Parent = root };
        root.Children.Add(child);

        Assert.False(root.IsLeaf());
        Assert.True(child.IsLeaf());
        Assert.False(child.IsRoot());
    }

    [Fact]
    public void ThoughtNode_PathLength_EqualsDepthPlusOne()
    {
        var root = new ThoughtNode<double> { Depth = 0 };
        var child1 = new ThoughtNode<double> { Depth = 1 };
        var child2 = new ThoughtNode<double> { Depth = 3 };

        Assert.Equal(1, root.PathLength);
        Assert.Equal(2, child1.PathLength);
        Assert.Equal(4, child2.PathLength);
    }

    [Fact]
    public void ThoughtNode_GetPathFromRoot_ReturnsCorrectOrder()
    {
        var root = new ThoughtNode<double> { Thought = "Problem", Depth = 0 };
        var step1 = new ThoughtNode<double> { Thought = "Step 1", Depth = 1, Parent = root };
        var step2 = new ThoughtNode<double> { Thought = "Step 2", Depth = 2, Parent = step1 };
        var answer = new ThoughtNode<double> { Thought = "Answer", Depth = 3, Parent = step2 };

        var path = answer.GetPathFromRoot();

        Assert.Equal(4, path.Count);
        Assert.Equal("Problem", path[0]);
        Assert.Equal("Step 1", path[1]);
        Assert.Equal("Step 2", path[2]);
        Assert.Equal("Answer", path[3]);
    }

    [Fact]
    public void ThoughtNode_PathScores_ReturnsRootToCurrentScores()
    {
        var root = new ThoughtNode<double> { Depth = 0, EvaluationScore = 0.9 };
        var child = new ThoughtNode<double> { Depth = 1, EvaluationScore = 0.85, Parent = root };
        var grandchild = new ThoughtNode<double> { Depth = 2, EvaluationScore = 0.92, Parent = child };

        var scores = grandchild.PathScores;

        Assert.Equal(3, scores.Length);
        Assert.Equal(0.9, scores[0]);
        Assert.Equal(0.85, scores[1]);
        Assert.Equal(0.92, scores[2]);
    }

    [Fact]
    public void ThoughtNode_PathScores_RootOnly_SingleElement()
    {
        var root = new ThoughtNode<double> { Depth = 0, EvaluationScore = 0.75 };

        var scores = root.PathScores;

        Assert.Equal(1, scores.Length);
        Assert.Equal(0.75, scores[0]);
    }

    // ============================
    // ThoughtNode: Terminal Heuristic Tests
    // ============================

    [Fact]
    public void CheckIsTerminalByHeuristic_ExplicitTerminal_ReturnsTrue()
    {
        var node = new ThoughtNode<double> { Thought = "Any content", IsTerminal = true };
        Assert.True(node.CheckIsTerminalByHeuristic());
    }

    [Theory]
    [InlineData("The final answer is 42")]
    [InlineData("In conclusion, the result is positive")]
    [InlineData("Therefore, we can see that X = 5")]
    [InlineData("The answer is 36")]
    public void CheckIsTerminalByHeuristic_TerminalKeywords_ReturnsTrue(string thought)
    {
        var node = new ThoughtNode<double> { Thought = thought, IsTerminal = false };
        Assert.True(node.CheckIsTerminalByHeuristic());
    }

    [Theory]
    [InlineData("Let's calculate the next step")]
    [InlineData("Consider the following equation")]
    [InlineData("We need to evaluate")]
    public void CheckIsTerminalByHeuristic_NonTerminalContent_ReturnsFalse(string thought)
    {
        var node = new ThoughtNode<double> { Thought = thought, IsTerminal = false };
        Assert.False(node.CheckIsTerminalByHeuristic());
    }

    // ============================
    // ThoughtNode: ToString Tests
    // ============================

    [Fact]
    public void ThoughtNode_ToString_IncludesDepthAndScore()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Calculate area",
            Depth = 2,
            EvaluationScore = 0.85
        };

        var str = node.ToString();
        Assert.Contains("Depth 2", str);
        Assert.Contains("0.85", str);
        Assert.Contains("Calculate area", str);
    }

    [Fact]
    public void ThoughtNode_ToString_Terminal_IncludesTerminalMarker()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Final",
            Depth = 0,
            IsTerminal = true
        };

        var str = node.ToString();
        Assert.Contains("[TERMINAL]", str);
    }

    [Fact]
    public void ThoughtNode_ToString_IndentationProportionalToDepth()
    {
        var depth0 = new ThoughtNode<double> { Depth = 0, Thought = "root" };
        var depth2 = new ThoughtNode<double> { Depth = 2, Thought = "deep" };

        // Depth 0 = no indentation, Depth 2 = 4 spaces
        var str0 = depth0.ToString();
        var str2 = depth2.ToString();

        Assert.StartsWith("[", str0); // No leading spaces
        Assert.StartsWith("    [", str2); // 4 leading spaces (2 * 2)
    }

    // ============================
    // ReasoningStep Tests
    // ============================

    [Fact]
    public void ReasoningStep_Defaults_ZeroScoreEmptyContent()
    {
        var step = new ReasoningStep<double>();

        Assert.Equal(0.0, step.Score);
        Assert.Equal(string.Empty, step.Content);
        Assert.Equal(0, step.StepNumber);
        Assert.False(step.IsVerified);
        Assert.Null(step.CriticFeedback);
        Assert.Equal(0, step.RefinementCount);
        Assert.Null(step.OriginalContent);
        Assert.Empty(step.Metadata);
    }

    [Fact]
    public void ReasoningStep_ToString_ShowsStepNumberAndContent()
    {
        var step = new ReasoningStep<double>
        {
            StepNumber = 3,
            Content = "Multiply 0.15 by 240"
        };

        Assert.Equal("Step 3: Multiply 0.15 by 240", step.ToString());
    }

    // ============================
    // ReasoningChain Tests
    // ============================

    [Fact]
    public void ReasoningChain_Empty_ZeroScoresAndMetrics()
    {
        var chain = new ReasoningChain<double>();

        Assert.Equal(0, chain.Steps.Count);
        Assert.Equal(0, chain.StepScores.Length);
        Assert.Equal(0.0, chain.GetMinimumScore());
        Assert.Equal(0.0, chain.GetAverageScore());
        Assert.Equal(0, chain.TotalRefinements);
        Assert.False(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_AddStep_SetsStepNumber()
    {
        var chain = new ReasoningChain<double>();

        chain.AddStep(new ReasoningStep<double> { Content = "Step A" });
        chain.AddStep(new ReasoningStep<double> { Content = "Step B" });
        chain.AddStep(new ReasoningStep<double> { Content = "Step C" });

        Assert.Equal(3, chain.Steps.Count);
        Assert.Equal(1, chain.Steps[0].StepNumber);
        Assert.Equal(2, chain.Steps[1].StepNumber);
        Assert.Equal(3, chain.Steps[2].StepNumber);
    }

    [Fact]
    public void ReasoningChain_AddStep_NullThrows()
    {
        var chain = new ReasoningChain<double>();
        Assert.Throws<ArgumentNullException>(() => chain.AddStep(null!));
    }

    [Fact]
    public void ReasoningChain_StepScores_ReturnsVectorOfScores()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Score = 0.8 });
        chain.AddStep(new ReasoningStep<double> { Score = 0.95 });

        var scores = chain.StepScores;

        Assert.Equal(3, scores.Length);
        Assert.Equal(0.9, scores[0]);
        Assert.Equal(0.8, scores[1]);
        Assert.Equal(0.95, scores[2]);
    }

    [Fact]
    public void ReasoningChain_GetMinimumScore_ReturnsWeakestLink()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Score = 0.6 }); // Weakest
        chain.AddStep(new ReasoningStep<double> { Score = 0.95 });

        Assert.Equal(0.6, chain.GetMinimumScore());
    }

    [Fact]
    public void ReasoningChain_GetAverageScore_ComputesMean()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Score = 0.8 });
        chain.AddStep(new ReasoningStep<double> { Score = 1.0 });

        // Mean = (0.9 + 0.8 + 1.0) / 3 = 2.7 / 3 = 0.9
        Assert.Equal(0.9, chain.GetAverageScore(), 1e-10);
    }

    [Fact]
    public void ReasoningChain_IsFullyVerified_AllVerified_True()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { IsVerified = true });
        chain.AddStep(new ReasoningStep<double> { IsVerified = true });

        Assert.True(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_IsFullyVerified_OneUnverified_False()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { IsVerified = true });
        chain.AddStep(new ReasoningStep<double> { IsVerified = false });

        Assert.False(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_TotalRefinements_SumsAcrossSteps()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { RefinementCount = 1 });
        chain.AddStep(new ReasoningStep<double> { RefinementCount = 0 });
        chain.AddStep(new ReasoningStep<double> { RefinementCount = 3 });

        Assert.Equal(4, chain.TotalRefinements);
    }

    [Fact]
    public void ReasoningChain_Duration_ComputedFromStartAndComplete()
    {
        var chain = new ReasoningChain<double>();
        var start = new DateTime(2025, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        var end = new DateTime(2025, 1, 1, 0, 0, 30, DateTimeKind.Utc);

        chain.StartedAt = start;
        chain.CompletedAt = end;

        Assert.Equal(TimeSpan.FromSeconds(30), chain.Duration);
    }

    // ============================
    // ReasoningResult Tests
    // ============================

    [Fact]
    public void ReasoningResult_Defaults_SuccessWithZeroConfidence()
    {
        var result = new ReasoningResult<double>();

        Assert.True(result.Success);
        Assert.Equal(0.0, result.OverallConfidence);
        Assert.Equal(string.Empty, result.FinalAnswer);
        Assert.Equal(string.Empty, result.StrategyUsed);
        Assert.Null(result.ErrorMessage);
        Assert.Null(result.ConfidenceScores);
        Assert.Empty(result.VerificationFeedback);
        Assert.Empty(result.ToolsUsed);
        Assert.Empty(result.Metrics);
        Assert.Empty(result.Metadata);
    }

    [Fact]
    public void ReasoningResult_GetSummary_IncludesKeyInfo()
    {
        var result = new ReasoningResult<double>
        {
            FinalAnswer = "42",
            StrategyUsed = "Chain-of-Thought",
            OverallConfidence = 0.95,
            Success = true,
            TotalDuration = TimeSpan.FromSeconds(2.5)
        };
        result.ReasoningChain.AddStep(new ReasoningStep<double> { Content = "Step 1" });

        var summary = result.GetSummary();

        Assert.Contains("Chain-of-Thought", summary);
        Assert.Contains("Success: True", summary);
        Assert.Contains("42", summary);
        Assert.Contains("0.95", summary);
        Assert.Contains("Steps: 1", summary);
    }

    [Fact]
    public void ReasoningResult_GetSummary_FailedIncludesError()
    {
        var result = new ReasoningResult<double>
        {
            Success = false,
            ErrorMessage = "Timeout after 60 seconds"
        };

        var summary = result.GetSummary();

        Assert.Contains("Success: False", summary);
        Assert.Contains("Timeout after 60 seconds", summary);
    }

    [Fact]
    public void ReasoningResult_GetSummary_WithToolsUsed()
    {
        var result = new ReasoningResult<double>
        {
            ToolsUsed = new List<string> { "Calculator", "PythonInterpreter" }
        };

        var summary = result.GetSummary();
        Assert.Contains("Calculator", summary);
        Assert.Contains("PythonInterpreter", summary);
    }

    // ============================
    // MajorityVotingAggregator Tests
    // ============================

    [Fact]
    public void MajorityVoting_ClearWinner_ReturnsCorrectAnswer()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "36", "36", "35", "36", "37" };
        var scores = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.85, 0.6 });

        var winner = aggregator.Aggregate(answers, scores);

        Assert.Equal("36", winner);
    }

    [Fact]
    public void MajorityVoting_AllSameAnswer_ReturnsThatAnswer()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "42", "42", "42" };
        var scores = new Vector<double>(new double[] { 0.9, 0.8, 0.7 });

        Assert.Equal("42", aggregator.Aggregate(answers, scores));
    }

    [Fact]
    public void MajorityVoting_EmptyAnswers_Throws()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        Assert.Throws<ArgumentException>(() =>
            aggregator.Aggregate(new List<string>(), new Vector<double>(0)));
    }

    [Fact]
    public void MajorityVoting_CaseInsensitive()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "Yes", "yes", "YES", "No" };
        var scores = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.6 });

        var winner = aggregator.Aggregate(answers, scores);

        // "Yes"/"yes"/"YES" should all be counted together (case-insensitive)
        Assert.Equal("Yes", winner, ignoreCase: true);
    }

    [Fact]
    public void MajorityVoting_IgnoresWhitespace()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { " 36 ", "36", " 36", "37" };
        var scores = new Vector<double>(new double[] { 0.9, 0.8, 0.7, 0.6 });

        var winner = aggregator.Aggregate(answers, scores);

        Assert.Equal("36", winner);
    }

    [Fact]
    public void MajorityVoting_MethodName_Correct()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        Assert.Equal("Majority Voting", aggregator.MethodName);
    }

    // ============================
    // WeightedAggregator Tests
    // ============================

    [Fact]
    public void WeightedVoting_HighConfidenceWins()
    {
        var aggregator = new WeightedAggregator<double>();

        // "36" appears twice with high confidence, "35" once with low
        var answers = new List<string> { "36", "35", "36" };
        var scores = new Vector<double>(new double[] { 0.9, 0.3, 0.8 });

        var winner = aggregator.Aggregate(answers, scores);

        // Total weight for "36" = 0.9 + 0.8 = 1.7
        // Total weight for "35" = 0.3
        Assert.Equal("36", winner);
    }

    [Fact]
    public void WeightedVoting_LowCountHighConfidence_CanWin()
    {
        var aggregator = new WeightedAggregator<double>();

        // "A" appears once with very high confidence
        // "B" appears twice with very low confidence
        var answers = new List<string> { "A", "B", "B" };
        var scores = new Vector<double>(new double[] { 0.99, 0.1, 0.1 });

        var winner = aggregator.Aggregate(answers, scores);

        // Total weight for "A" = 0.99
        // Total weight for "B" = 0.1 + 0.1 = 0.2
        Assert.Equal("A", winner);
    }

    [Fact]
    public void WeightedVoting_EqualWeights_BehavesLikeMajority()
    {
        var aggregator = new WeightedAggregator<double>();

        var answers = new List<string> { "36", "36", "35" };
        var scores = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

        var winner = aggregator.Aggregate(answers, scores);

        // With equal weights: "36" weight = 2.0, "35" weight = 1.0
        Assert.Equal("36", winner);
    }

    [Fact]
    public void WeightedVoting_MismatchedLengths_Throws()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string> { "36", "35" };
        var scores = new Vector<double>(new double[] { 0.9 }); // Wrong length

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(answers, scores));
    }

    [Fact]
    public void WeightedVoting_EmptyAnswers_Throws()
    {
        var aggregator = new WeightedAggregator<double>();
        Assert.Throws<ArgumentException>(() =>
            aggregator.Aggregate(new List<string>(), new Vector<double>(0)));
    }

    [Fact]
    public void WeightedVoting_MethodName_Correct()
    {
        var aggregator = new WeightedAggregator<double>();
        Assert.Equal("Weighted Voting", aggregator.MethodName);
    }

    [Fact]
    public void WeightedVoting_CaseInsensitive()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string> { "Yes", "yes", "No" };
        var scores = new Vector<double>(new double[] { 0.5, 0.5, 0.8 });

        var winner = aggregator.Aggregate(answers, scores);

        // "Yes"/"yes" total = 1.0, "No" = 0.8
        Assert.Equal("Yes", winner, ignoreCase: true);
    }

    // ============================
    // Cross-Component Integration Tests
    // ============================

    [Fact]
    public void ThoughtNode_TreeBuilding_MultiLevelConsistency()
    {
        // Build a 3-level tree
        var root = new ThoughtNode<double> { Thought = "Problem", Depth = 0, EvaluationScore = 1.0 };

        var branch1 = new ThoughtNode<double> { Thought = "Approach A", Depth = 1, EvaluationScore = 0.8, Parent = root };
        var branch2 = new ThoughtNode<double> { Thought = "Approach B", Depth = 1, EvaluationScore = 0.9, Parent = root };
        root.Children.Add(branch1);
        root.Children.Add(branch2);

        var leaf1a = new ThoughtNode<double> { Thought = "Answer A1", Depth = 2, EvaluationScore = 0.7, Parent = branch1 };
        var leaf2a = new ThoughtNode<double> { Thought = "Answer B1", Depth = 2, EvaluationScore = 0.95, Parent = branch2 };
        branch1.Children.Add(leaf1a);
        branch2.Children.Add(leaf2a);

        // Verify tree structure
        Assert.True(root.IsRoot());
        Assert.False(root.IsLeaf());
        Assert.Equal(2, root.Children.Count);

        Assert.False(branch1.IsRoot());
        Assert.False(branch1.IsLeaf());

        Assert.True(leaf1a.IsLeaf());
        Assert.True(leaf2a.IsLeaf());

        // Verify paths
        var pathA = leaf1a.GetPathFromRoot();
        Assert.Equal(3, pathA.Count);
        Assert.Equal("Problem", pathA[0]);
        Assert.Equal("Approach A", pathA[1]);
        Assert.Equal("Answer A1", pathA[2]);

        // Verify path scores
        var scoresB = leaf2a.PathScores;
        Assert.Equal(3, scoresB.Length);
        Assert.Equal(1.0, scoresB[0]);  // root
        Assert.Equal(0.9, scoresB[1]);  // branch2
        Assert.Equal(0.95, scoresB[2]); // leaf2a
    }

    [Fact]
    public void ReasoningChain_FullPipeline_ScoresAndVerification()
    {
        var chain = new ReasoningChain<double> { Query = "What is 15% of 240?" };

        chain.AddStep(new ReasoningStep<double>
        {
            Content = "Convert 15% to 0.15",
            Score = 1.0,
            IsVerified = true,
            RefinementCount = 0
        });

        chain.AddStep(new ReasoningStep<double>
        {
            Content = "0.15 * 240 = 36",
            Score = 0.95,
            IsVerified = true,
            VerificationMethod = "Calculator",
            RefinementCount = 0
        });

        chain.AddStep(new ReasoningStep<double>
        {
            Content = "The answer is 36",
            Score = 1.0,
            IsVerified = true,
            RefinementCount = 0
        });

        chain.FinalAnswer = "36";

        // Verify chain metrics
        Assert.Equal(3, chain.Steps.Count);
        Assert.True(chain.IsFullyVerified);
        Assert.Equal(0, chain.TotalRefinements);

        // Scores
        Assert.Equal(0.95, chain.GetMinimumScore());
        double expectedMean = (1.0 + 0.95 + 1.0) / 3.0;
        Assert.Equal(expectedMean, chain.GetAverageScore(), 1e-10);

        // StepScores vector
        var scores = chain.StepScores;
        Assert.Equal(3, scores.Length);
        Assert.Equal(1.0, scores[0]);
        Assert.Equal(0.95, scores[1]);
        Assert.Equal(1.0, scores[2]);
    }
}
