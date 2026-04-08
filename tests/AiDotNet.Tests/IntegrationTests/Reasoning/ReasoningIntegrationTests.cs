using Xunit;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Aggregation;
using AiDotNet.Reasoning.Components;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Search;

namespace AiDotNet.Tests.IntegrationTests.Reasoning;

/// <summary>
/// Integration tests for the AiDotNet.Reasoning module.
/// Tests the Reasoning models, search algorithms, components, and aggregation strategies.
/// </summary>
public class ReasoningIntegrationTests
{
    #region ReasoningConfig Tests

    [Fact]
    public void ReasoningConfig_DefaultValues_AreCorrect()
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

    [Fact]
    public void ReasoningConfig_CustomValues_ArePreserved()
    {
        var config = new ReasoningConfig
        {
            MaxSteps = 20,
            ExplorationDepth = 5,
            BranchingFactor = 4,
            NumSamples = 10,
            Temperature = 0.5,
            BeamWidth = 8,
            EnableVerification = true,
            EnableSelfRefinement = true,
            MaxRefinementAttempts = 3,
            VerificationThreshold = 0.9,
            EnableExternalVerification = true,
            EnableTestTimeCompute = true,
            ComputeScalingFactor = 3.0,
            MaxReasoningTimeSeconds = 120,
            EnableContradictionDetection = true,
            EnableDiversitySampling = true
        };

        Assert.Equal(20, config.MaxSteps);
        Assert.Equal(5, config.ExplorationDepth);
        Assert.Equal(4, config.BranchingFactor);
        Assert.Equal(10, config.NumSamples);
        Assert.Equal(0.5, config.Temperature);
        Assert.Equal(8, config.BeamWidth);
        Assert.True(config.EnableVerification);
        Assert.True(config.EnableSelfRefinement);
        Assert.Equal(3, config.MaxRefinementAttempts);
        Assert.Equal(0.9, config.VerificationThreshold);
        Assert.True(config.EnableExternalVerification);
        Assert.True(config.EnableTestTimeCompute);
        Assert.Equal(3.0, config.ComputeScalingFactor);
        Assert.Equal(120, config.MaxReasoningTimeSeconds);
        Assert.True(config.EnableContradictionDetection);
        Assert.True(config.EnableDiversitySampling);
    }

    #endregion

    #region ReasoningStep Tests

    [Fact]
    public void ReasoningStep_DefaultValues_AreCorrect()
    {
        var step = new ReasoningStep<double>();

        Assert.Equal(0, step.StepNumber);
        Assert.Equal(string.Empty, step.Content);
        Assert.Null(step.OriginalContent);
        Assert.Equal(0.0, step.Score);
        Assert.False(step.IsVerified);
        Assert.Null(step.CriticFeedback);
        Assert.Equal(0, step.RefinementCount);
        Assert.Null(step.VerificationMethod);
        Assert.Null(step.ExternalVerificationResult);
        Assert.NotNull(step.Metadata);
        Assert.Empty(step.Metadata);
    }

    [Fact]
    public void ReasoningStep_WithContent_PreservesValues()
    {
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "Convert 15% to decimal: 0.15",
            Score = 0.95,
            IsVerified = true,
            CriticFeedback = "Correct conversion",
            VerificationMethod = "Calculator",
            ExternalVerificationResult = "0.15"
        };

        Assert.Equal(1, step.StepNumber);
        Assert.Equal("Convert 15% to decimal: 0.15", step.Content);
        Assert.Equal(0.95, step.Score);
        Assert.True(step.IsVerified);
        Assert.Equal("Correct conversion", step.CriticFeedback);
        Assert.Equal("Calculator", step.VerificationMethod);
        Assert.Equal("0.15", step.ExternalVerificationResult);
    }

    [Fact]
    public void ReasoningStep_Refinement_TracksOriginalContent()
    {
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "Calculate 0.15 × 240 = 36",
            OriginalContent = "Calculate 0.15 × 240 = 35",
            RefinementCount = 1
        };

        Assert.Equal("Calculate 0.15 × 240 = 35", step.OriginalContent);
        Assert.Equal("Calculate 0.15 × 240 = 36", step.Content);
        Assert.Equal(1, step.RefinementCount);
    }

    [Fact]
    public void ReasoningStep_ToString_ReturnsFormattedString()
    {
        var step = new ReasoningStep<double>
        {
            StepNumber = 3,
            Content = "Final answer: 36"
        };

        var result = step.ToString();

        Assert.Equal("Step 3: Final answer: 36", result);
    }

    [Fact]
    public void ReasoningStep_Metadata_CanStoreCustomData()
    {
        var step = new ReasoningStep<double>();
        step.Metadata["calculation_type"] = "multiplication";
        step.Metadata["confidence_level"] = "high";

        Assert.Equal("multiplication", step.Metadata["calculation_type"]);
        Assert.Equal("high", step.Metadata["confidence_level"]);
    }

    [Fact]
    public void ReasoningStep_CreatedAt_IsSet()
    {
        var before = DateTime.UtcNow;
        var step = new ReasoningStep<double>();
        var after = DateTime.UtcNow;

        Assert.True(step.CreatedAt >= before);
        Assert.True(step.CreatedAt <= after);
    }

    #endregion

    #region ReasoningChain Tests

    [Fact]
    public void ReasoningChain_DefaultValues_AreCorrect()
    {
        var chain = new ReasoningChain<double>();

        Assert.Equal(string.Empty, chain.Query);
        Assert.NotNull(chain.Steps);
        Assert.Empty(chain.Steps);
        Assert.Equal(string.Empty, chain.FinalAnswer);
        Assert.Equal(0.0, chain.OverallScore);
        Assert.NotNull(chain.Metadata);
        Assert.Empty(chain.Metadata);
    }

    [Fact]
    public void ReasoningChain_AddStep_IncrementsStepNumber()
    {
        var chain = new ReasoningChain<double>();

        chain.AddStep(new ReasoningStep<double> { Content = "Step 1" });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2" });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3" });

        Assert.Equal(3, chain.Steps.Count);
        Assert.Equal(1, chain.Steps[0].StepNumber);
        Assert.Equal(2, chain.Steps[1].StepNumber);
        Assert.Equal(3, chain.Steps[2].StepNumber);
    }

    [Fact]
    public void ReasoningChain_StepScores_ReturnsVector()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", Score = 0.85 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", Score = 0.95 });

        var scores = chain.StepScores;

        Assert.Equal(3, scores.Length);
        Assert.Equal(0.9, scores[0]);
        Assert.Equal(0.85, scores[1]);
        Assert.Equal(0.95, scores[2]);
    }

    [Fact]
    public void ReasoningChain_StepScores_EmptyChain_ReturnsEmptyVector()
    {
        var chain = new ReasoningChain<double>();

        var scores = chain.StepScores;

        Assert.Equal(0, scores.Length);
    }

    [Fact]
    public void ReasoningChain_GetMinimumScore_ReturnsLowestScore()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", Score = 0.7 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", Score = 0.85 });

        var minScore = chain.GetMinimumScore();

        Assert.Equal(0.7, minScore);
    }

    [Fact]
    public void ReasoningChain_GetMinimumScore_EmptyChain_ReturnsZero()
    {
        var chain = new ReasoningChain<double>();

        var minScore = chain.GetMinimumScore();

        Assert.Equal(0.0, minScore);
    }

    [Fact]
    public void ReasoningChain_GetAverageScore_CalculatesCorrectly()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", Score = 0.8 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", Score = 1.0 });

        var avgScore = chain.GetAverageScore();

        Assert.Equal(0.9, avgScore, 5);
    }

    [Fact]
    public void ReasoningChain_IsFullyVerified_AllVerified_ReturnsTrue()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", IsVerified = true });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", IsVerified = true });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", IsVerified = true });

        Assert.True(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_IsFullyVerified_NotAllVerified_ReturnsFalse()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", IsVerified = true });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", IsVerified = false });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", IsVerified = true });

        Assert.False(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_IsFullyVerified_EmptyChain_ReturnsFalse()
    {
        var chain = new ReasoningChain<double>();

        Assert.False(chain.IsFullyVerified);
    }

    [Fact]
    public void ReasoningChain_TotalRefinements_SumsCorrectly()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", RefinementCount = 0 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", RefinementCount = 2 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", RefinementCount = 1 });

        Assert.Equal(3, chain.TotalRefinements);
    }

    [Fact]
    public void ReasoningChain_Duration_CalculatesCorrectly()
    {
        var chain = new ReasoningChain<double>
        {
            StartedAt = DateTime.UtcNow.AddSeconds(-10)
        };

        // Duration should be approximately 10 seconds
        Assert.True(chain.Duration.TotalSeconds >= 9);
        Assert.True(chain.Duration.TotalSeconds <= 11);
    }

    [Fact]
    public void ReasoningChain_Duration_WhenCompleted_UsesCompletedAt()
    {
        var start = new DateTime(2024, 1, 1, 12, 0, 0, DateTimeKind.Utc);
        var end = new DateTime(2024, 1, 1, 12, 0, 15, DateTimeKind.Utc);

        var chain = new ReasoningChain<double>
        {
            StartedAt = start,
            CompletedAt = end
        };

        Assert.Equal(15, chain.Duration.TotalSeconds);
    }

    [Fact]
    public void ReasoningChain_ToString_FormatsCorrectly()
    {
        var chain = new ReasoningChain<double>
        {
            Query = "What is 15% of 240?",
            FinalAnswer = "36",
            OverallScore = 0.95
        };
        chain.AddStep(new ReasoningStep<double> { Content = "Convert percentage", Score = 1.0, IsVerified = true });

        var result = chain.ToString();

        Assert.Contains("Query: What is 15% of 240?", result);
        Assert.Contains("Final Answer: 36", result);
        Assert.Contains("Overall Score: 0.95", result);
    }

    #endregion

    #region ThoughtNode Tests

    [Fact]
    public void ThoughtNode_DefaultValues_AreCorrect()
    {
        var node = new ThoughtNode<double>();

        Assert.Equal(string.Empty, node.Thought);
        Assert.Null(node.Parent);
        Assert.NotNull(node.Children);
        Assert.Empty(node.Children);
        Assert.Equal(0, node.Depth);
        Assert.Equal(0.0, node.EvaluationScore);
        Assert.False(node.IsVisited);
        Assert.False(node.IsTerminal);
        Assert.NotNull(node.Metadata);
        Assert.Empty(node.Metadata);
    }

    [Fact]
    public void ThoughtNode_WithValues_PreservesData()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Consider using renewable energy",
            Depth = 1,
            EvaluationScore = 0.85,
            IsVisited = true,
            IsTerminal = false
        };

        Assert.Equal("Consider using renewable energy", node.Thought);
        Assert.Equal(1, node.Depth);
        Assert.Equal(0.85, node.EvaluationScore);
        Assert.True(node.IsVisited);
        Assert.False(node.IsTerminal);
    }

    [Fact]
    public void ThoughtNode_ParentChild_Relationship_IsCorrect()
    {
        var root = new ThoughtNode<double>
        {
            Thought = "Root problem",
            Depth = 0
        };

        var child1 = new ThoughtNode<double>
        {
            Thought = "Approach 1",
            Parent = root,
            Depth = 1
        };

        var child2 = new ThoughtNode<double>
        {
            Thought = "Approach 2",
            Parent = root,
            Depth = 1
        };

        root.Children.Add(child1);
        root.Children.Add(child2);

        Assert.Equal(2, root.Children.Count);
        Assert.Same(root, child1.Parent);
        Assert.Same(root, child2.Parent);
    }

    [Fact]
    public void ThoughtNode_IsRoot_ReturnsTrue_WhenNoParent()
    {
        var node = new ThoughtNode<double> { Thought = "Root" };

        Assert.True(node.IsRoot());
    }

    [Fact]
    public void ThoughtNode_IsRoot_ReturnsFalse_WhenHasParent()
    {
        var parent = new ThoughtNode<double> { Thought = "Parent" };
        var child = new ThoughtNode<double> { Thought = "Child", Parent = parent };

        Assert.False(child.IsRoot());
    }

    [Fact]
    public void ThoughtNode_IsLeaf_ReturnsTrue_WhenNoChildren()
    {
        var node = new ThoughtNode<double> { Thought = "Leaf" };

        Assert.True(node.IsLeaf());
    }

    [Fact]
    public void ThoughtNode_IsLeaf_ReturnsFalse_WhenHasChildren()
    {
        var parent = new ThoughtNode<double> { Thought = "Parent" };
        parent.Children.Add(new ThoughtNode<double> { Thought = "Child" });

        Assert.False(parent.IsLeaf());
    }

    [Fact]
    public void ThoughtNode_PathLength_EqualsDepthPlusOne()
    {
        var node = new ThoughtNode<double> { Depth = 3 };

        Assert.Equal(4, node.PathLength);
    }

    [Fact]
    public void ThoughtNode_GetPathFromRoot_ReturnsCorrectPath()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0 };
        var level1 = new ThoughtNode<double> { Thought = "Level 1", Parent = root, Depth = 1 };
        var level2 = new ThoughtNode<double> { Thought = "Level 2", Parent = level1, Depth = 2 };

        var path = level2.GetPathFromRoot();

        Assert.Equal(3, path.Count);
        Assert.Equal("Root", path[0]);
        Assert.Equal("Level 1", path[1]);
        Assert.Equal("Level 2", path[2]);
    }

    [Fact]
    public void ThoughtNode_PathScores_ReturnsCorrectVector()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0, EvaluationScore = 0.9 };
        var level1 = new ThoughtNode<double> { Thought = "Level 1", Parent = root, Depth = 1, EvaluationScore = 0.85 };
        var level2 = new ThoughtNode<double> { Thought = "Level 2", Parent = level1, Depth = 2, EvaluationScore = 0.95 };

        var scores = level2.PathScores;

        Assert.Equal(3, scores.Length);
        Assert.Equal(0.9, scores[0]);
        Assert.Equal(0.85, scores[1]);
        Assert.Equal(0.95, scores[2]);
    }

    [Fact]
    public void ThoughtNode_CheckIsTerminalByHeuristic_DetectsFinalAnswer()
    {
        var node1 = new ThoughtNode<double> { Thought = "The final answer is 42" };
        var node2 = new ThoughtNode<double> { Thought = "In conclusion, we should choose option A" };
        var node3 = new ThoughtNode<double> { Thought = "Therefore, the result is positive" };
        var node4 = new ThoughtNode<double> { Thought = "The answer is correct" };

        Assert.True(node1.CheckIsTerminalByHeuristic());
        Assert.True(node2.CheckIsTerminalByHeuristic());
        Assert.True(node3.CheckIsTerminalByHeuristic());
        Assert.True(node4.CheckIsTerminalByHeuristic());
    }

    [Fact]
    public void ThoughtNode_CheckIsTerminalByHeuristic_ReturnsFalse_ForNonTerminal()
    {
        var node = new ThoughtNode<double> { Thought = "Let's consider option B" };

        Assert.False(node.CheckIsTerminalByHeuristic());
    }

    [Fact]
    public void ThoughtNode_CheckIsTerminalByHeuristic_ReturnsTrue_WhenMarkedTerminal()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Some intermediate thought",
            IsTerminal = true
        };

        Assert.True(node.CheckIsTerminalByHeuristic());
    }

    [Fact]
    public void ThoughtNode_ToString_FormatsCorrectly()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Consider option A",
            Depth = 2,
            EvaluationScore = 0.75
        };

        var result = node.ToString();

        Assert.Contains("Depth 2", result);
        Assert.Contains("Score 0.75", result);
        Assert.Contains("Consider option A", result);
    }

    [Fact]
    public void ThoughtNode_ToString_IncludesTerminalMarker()
    {
        var node = new ThoughtNode<double>
        {
            Thought = "Final answer",
            IsTerminal = true
        };

        var result = node.ToString();

        Assert.Contains("[TERMINAL]", result);
    }

    [Fact]
    public void ThoughtNode_Metadata_CanStoreCustomData()
    {
        var node = new ThoughtNode<double>();
        node.Metadata["evaluation_response"] = "Good reasoning";
        node.Metadata["evaluation_score"] = 0.85;

        Assert.Equal("Good reasoning", node.Metadata["evaluation_response"]);
        Assert.Equal(0.85, node.Metadata["evaluation_score"]);
    }

    #endregion

    #region ReasoningResult Tests

    [Fact]
    public void ReasoningResult_DefaultValues_AreCorrect()
    {
        var result = new ReasoningResult<double>();

        Assert.Equal(string.Empty, result.FinalAnswer);
        Assert.NotNull(result.ReasoningChain);
        Assert.NotNull(result.AlternativeChains);
        Assert.Empty(result.AlternativeChains);
        Assert.Equal(0.0, result.OverallConfidence);
        Assert.Null(result.ConfidenceScores);
        Assert.Equal(string.Empty, result.StrategyUsed);
        Assert.True(result.Success);
        Assert.Null(result.ErrorMessage);
        Assert.NotNull(result.VerificationFeedback);
        Assert.Empty(result.VerificationFeedback);
        Assert.NotNull(result.ToolsUsed);
        Assert.Empty(result.ToolsUsed);
        Assert.NotNull(result.Metrics);
        Assert.Empty(result.Metrics);
        Assert.NotNull(result.Metadata);
        Assert.Empty(result.Metadata);
    }

    [Fact]
    public void ReasoningResult_WithValues_PreservesData()
    {
        var result = new ReasoningResult<double>
        {
            FinalAnswer = "42",
            OverallConfidence = 0.95,
            StrategyUsed = "Chain-of-Thought",
            Success = true,
            TotalDuration = TimeSpan.FromSeconds(5.5)
        };

        Assert.Equal("42", result.FinalAnswer);
        Assert.Equal(0.95, result.OverallConfidence);
        Assert.Equal("Chain-of-Thought", result.StrategyUsed);
        Assert.True(result.Success);
        Assert.Equal(5.5, result.TotalDuration.TotalSeconds);
    }

    [Fact]
    public void ReasoningResult_FailedResult_HasErrorMessage()
    {
        var result = new ReasoningResult<double>
        {
            Success = false,
            ErrorMessage = "Reasoning timeout after 60 seconds"
        };

        Assert.False(result.Success);
        Assert.Equal("Reasoning timeout after 60 seconds", result.ErrorMessage);
    }

    [Fact]
    public void ReasoningResult_VerificationFeedback_CanAddItems()
    {
        var result = new ReasoningResult<double>();
        result.VerificationFeedback.Add("Step 1 verified: Logic is sound");
        result.VerificationFeedback.Add("Step 2 verified: Calculation correct");

        Assert.Equal(2, result.VerificationFeedback.Count);
    }

    [Fact]
    public void ReasoningResult_ToolsUsed_CanAddItems()
    {
        var result = new ReasoningResult<double>();
        result.ToolsUsed.Add("Calculator");
        result.ToolsUsed.Add("WebSearch");

        Assert.Equal(2, result.ToolsUsed.Count);
        Assert.Contains("Calculator", result.ToolsUsed);
    }

    [Fact]
    public void ReasoningResult_Metrics_CanStoreData()
    {
        var result = new ReasoningResult<double>();
        result.Metrics["llm_calls"] = 5;
        result.Metrics["tokens_used"] = 1500;
        result.Metrics["nodes_explored"] = 12;

        Assert.Equal(5, result.Metrics["llm_calls"]);
        Assert.Equal(1500, result.Metrics["tokens_used"]);
    }

    [Fact]
    public void ReasoningResult_AlternativeChains_CanAddChains()
    {
        var result = new ReasoningResult<double>();
        result.AlternativeChains.Add(new ReasoningChain<double> { FinalAnswer = "Option A" });
        result.AlternativeChains.Add(new ReasoningChain<double> { FinalAnswer = "Option B" });

        Assert.Equal(2, result.AlternativeChains.Count);
    }

    [Fact]
    public void ReasoningResult_ConfidenceScores_CanSetVector()
    {
        var result = new ReasoningResult<double>();
        result.ConfidenceScores = new Vector<double>(new[] { 0.9, 0.85, 0.92, 0.88 });

        Assert.NotNull(result.ConfidenceScores);
        Assert.Equal(4, result.ConfidenceScores.Length);
    }

    [Fact]
    public void ReasoningResult_GetSummary_FormatsCorrectly()
    {
        var result = new ReasoningResult<double>
        {
            StrategyUsed = "Tree-of-Thoughts",
            Success = true,
            FinalAnswer = "The answer is 42",
            OverallConfidence = 0.92,
            TotalDuration = TimeSpan.FromSeconds(3.5)
        };

        var summary = result.GetSummary();

        Assert.Contains("Tree-of-Thoughts", summary);
        Assert.Contains("Success: True", summary);
        Assert.Contains("Final Answer: The answer is 42", summary);
        Assert.Contains("Confidence: 0.92", summary);
    }

    [Fact]
    public void ReasoningResult_GetSummary_IncludesToolsUsed()
    {
        var result = new ReasoningResult<double>
        {
            StrategyUsed = "Test",
            Success = true,
            FinalAnswer = "42"
        };
        result.ToolsUsed.Add("Calculator");
        result.ToolsUsed.Add("WebSearch");

        var summary = result.GetSummary();

        Assert.Contains("Tools Used:", summary);
        Assert.Contains("Calculator", summary);
    }

    [Fact]
    public void ReasoningResult_GetSummary_IncludesErrorWhenFailed()
    {
        var result = new ReasoningResult<double>
        {
            StrategyUsed = "Test",
            Success = false,
            ErrorMessage = "Timeout occurred"
        };

        var summary = result.GetSummary();

        Assert.Contains("Success: False", summary);
        Assert.Contains("Error: Timeout occurred", summary);
    }

    [Fact]
    public void ReasoningResult_ToString_ReturnsSummary()
    {
        var result = new ReasoningResult<double>
        {
            StrategyUsed = "Test",
            Success = true,
            FinalAnswer = "Result"
        };

        var toString = result.ToString();
        var summary = result.GetSummary();

        Assert.Equal(summary, toString);
    }

    #endregion

    #region MajorityVotingAggregator Tests

    [Fact]
    public void MajorityVotingAggregator_MethodName_IsCorrect()
    {
        var aggregator = new MajorityVotingAggregator<double>();

        Assert.Equal("Majority Voting", aggregator.MethodName);
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_ReturnsMostCommonAnswer()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "36", "36", "35", "36", "37", "36", "36" };
        var scores = new Vector<double>(answers.Count);

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("36", result);
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_IsCaseInsensitive()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "Yes", "yes", "YES", "No", "no" };
        var scores = new Vector<double>(answers.Count);

        var result = aggregator.Aggregate(answers, scores);

        Assert.True(result.Equals("Yes", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_TrimsWhitespace()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { " answer ", "answer", "  answer", "other" };
        var scores = new Vector<double>(answers.Count);

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("answer", result);
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_SkipsEmptyAnswers()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "", "valid", " ", "valid", null! };
        var scores = new Vector<double>(answers.Count);

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("valid", result);
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_ThrowsOnEmptyList()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string>();
        var scores = new Vector<double>(0);

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(answers, scores));
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_ThrowsOnNullList()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var scores = new Vector<double>(0);

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(null!, scores));
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_ThrowsWhenAllAnswersEmpty()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "", " ", "  " };
        var scores = new Vector<double>(answers.Count);

        Assert.Throws<InvalidOperationException>(() => aggregator.Aggregate(answers, scores));
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_SingleAnswer_ReturnsIt()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        var answers = new List<string> { "only answer" };
        var scores = new Vector<double>(1);

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("only answer", result);
    }

    [Fact]
    public void MajorityVotingAggregator_Aggregate_Tie_ReturnsFirst()
    {
        var aggregator = new MajorityVotingAggregator<double>();
        // Equal counts - should return one of them (implementation detail: returns highest count first encountered)
        var answers = new List<string> { "A", "B", "A", "B" };
        var scores = new Vector<double>(answers.Count);

        var result = aggregator.Aggregate(answers, scores);

        Assert.True(result == "A" || result == "B");
    }

    #endregion

    #region WeightedAggregator Tests

    [Fact]
    public void WeightedAggregator_MethodName_IsCorrect()
    {
        var aggregator = new WeightedAggregator<double>();

        Assert.Equal("Weighted Voting", aggregator.MethodName);
    }

    [Fact]
    public void WeightedAggregator_Aggregate_ConsidersConfidenceScores()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string> { "A", "B", "B" };
        // A has higher confidence (0.95) than B's combined (0.4 + 0.4 = 0.8)
        var scores = new Vector<double>(new[] { 0.95, 0.4, 0.4 });

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("A", result);
    }

    [Fact]
    public void WeightedAggregator_Aggregate_SumsWeightsForSameAnswer()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string> { "A", "B", "A" };
        // A's combined weight (0.5 + 0.5 = 1.0) > B's weight (0.9)
        var scores = new Vector<double>(new[] { 0.5, 0.9, 0.5 });

        var result = aggregator.Aggregate(answers, scores);

        Assert.Equal("A", result);
    }

    [Fact]
    public void WeightedAggregator_Aggregate_ThrowsOnEmptyList()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string>();
        var scores = new Vector<double>(0);

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(answers, scores));
    }

    [Fact]
    public void WeightedAggregator_Aggregate_ThrowsOnMismatchedCounts()
    {
        var aggregator = new WeightedAggregator<double>();
        var answers = new List<string> { "A", "B", "C" };
        var scores = new Vector<double>(new[] { 0.5, 0.5 }); // Only 2 scores for 3 answers

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(answers, scores));
    }

    #endregion

    #region Search Algorithm Tests

    [Fact]
    public void BeamSearch_AlgorithmName_IsCorrect()
    {
        var search = new BeamSearch<double>();

        Assert.Equal("Beam Search", search.AlgorithmName);
    }

    [Fact]
    public void BeamSearch_Description_IsNotEmpty()
    {
        var search = new BeamSearch<double>();

        Assert.False(string.IsNullOrEmpty(search.Description));
        Assert.Contains("beam", search.Description.ToLower());
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ThrowsOnNullRoot()
    {
        var search = new BeamSearch<double>();
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(null!, generator, evaluator, config));
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ThrowsOnNullGenerator()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(root, null!, evaluator, config));
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ThrowsOnNullEvaluator()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var config = new ReasoningConfig();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(root, generator, null!, config));
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ThrowsOnNullConfig()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(root, generator, evaluator, null!));
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ThrowsOnInvalidBeamWidth()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig { BeamWidth = 0 };

        await Assert.ThrowsAsync<ArgumentException>(() =>
            search.SearchAsync(root, generator, evaluator, config));
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_ReturnsPathIncludingRoot()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root problem" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            BeamWidth = 2,
            BranchingFactor = 2,
            ExplorationDepth = 2
        };

        var path = await search.SearchAsync(root, generator, evaluator, config);

        Assert.NotEmpty(path);
        Assert.Equal("Root problem", path[0].Thought);
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_MarksRootAsVisited()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig { BeamWidth = 2, ExplorationDepth = 1 };

        await search.SearchAsync(root, generator, evaluator, config);

        Assert.True(root.IsVisited);
    }

    [Fact]
    public async Task BeamSearch_SearchAsync_RespectsCancellation()
    {
        var search = new BeamSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig { BeamWidth = 2, ExplorationDepth = 10 };
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            search.SearchAsync(root, generator, evaluator, config, cts.Token));
    }

    [Fact]
    public void BestFirstSearch_AlgorithmName_IsCorrect()
    {
        var search = new BestFirstSearch<double>();

        Assert.Equal("Best-First Search", search.AlgorithmName);
    }

    [Fact]
    public void BestFirstSearch_Description_IsNotEmpty()
    {
        var search = new BestFirstSearch<double>();

        Assert.False(string.IsNullOrEmpty(search.Description));
    }

    [Fact]
    public void BreadthFirstSearch_AlgorithmName_IsCorrect()
    {
        var search = new BreadthFirstSearch<double>();

        Assert.Equal("Breadth-First Search", search.AlgorithmName);
    }

    [Fact]
    public void BreadthFirstSearch_Description_IsNotEmpty()
    {
        var search = new BreadthFirstSearch<double>();

        Assert.False(string.IsNullOrEmpty(search.Description));
    }

    [Fact]
    public void DepthFirstSearch_AlgorithmName_IsCorrect()
    {
        var search = new DepthFirstSearch<double>();

        Assert.Equal("Depth-First Search", search.AlgorithmName);
    }

    [Fact]
    public void DepthFirstSearch_Description_IsNotEmpty()
    {
        var search = new DepthFirstSearch<double>();

        Assert.False(string.IsNullOrEmpty(search.Description));
    }

    [Fact]
    public void MonteCarloTreeSearch_AlgorithmName_IsCorrect()
    {
        var search = new MonteCarloTreeSearch<double>();

        Assert.Equal("Monte Carlo Tree Search (MCTS)", search.AlgorithmName);
    }

    [Fact]
    public void MonteCarloTreeSearch_Description_IsNotEmpty()
    {
        var search = new MonteCarloTreeSearch<double>();

        Assert.False(string.IsNullOrEmpty(search.Description));
        // Description mentions exploration/exploitation, not necessarily "Monte Carlo"
        Assert.Contains("exploration", search.Description.ToLower());
    }

    #endregion

    #region Mock Classes

    /// <summary>
    /// Mock implementation of IThoughtGenerator for testing search algorithms.
    /// </summary>
    private class MockThoughtGenerator<T> : IThoughtGenerator<T>
    {
        private int _callCount;

        public Task<List<ThoughtNode<T>>> GenerateThoughtsAsync(
            ThoughtNode<T> currentNode,
            int numThoughts,
            ReasoningConfig config,
            CancellationToken cancellationToken = default)
        {
            _callCount++;
            var thoughts = new List<ThoughtNode<T>>();

            for (int i = 0; i < numThoughts; i++)
            {
                var node = new ThoughtNode<T>
                {
                    Thought = $"Generated thought {_callCount}.{i + 1}",
                    Parent = currentNode,
                    Depth = currentNode.Depth + 1
                };

                // Mark some as terminal to test terminal detection
                if (_callCount >= 2 && i == 0)
                {
                    node.Thought = "Therefore, the answer is 42";
                }

                thoughts.Add(node);
            }

            return Task.FromResult(thoughts);
        }
    }

    /// <summary>
    /// Mock implementation of IThoughtEvaluator for testing search algorithms.
    /// </summary>
    private class MockThoughtEvaluator<T> : IThoughtEvaluator<T>
    {
        private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
        private readonly Random _random = new(42); // Fixed seed for reproducibility

        public Task<T> EvaluateThoughtAsync(
            ThoughtNode<T> node,
            string originalQuery,
            ReasoningConfig config,
            CancellationToken cancellationToken = default)
        {
            // Return a deterministic but varied score based on the thought content
            double score = 0.5 + (_random.NextDouble() * 0.4); // Scores between 0.5 and 0.9

            // Higher scores for terminal-looking thoughts
            if (node.Thought.Contains("answer") || node.Thought.Contains("Therefore"))
            {
                score = 0.95;
            }

            return Task.FromResult(_numOps.FromDouble(score));
        }
    }

    #endregion
}
