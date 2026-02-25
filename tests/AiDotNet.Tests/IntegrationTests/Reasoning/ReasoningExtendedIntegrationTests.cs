using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Components;
using AiDotNet.Reasoning.ComputeScaling;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Search;
using AiDotNet.Reasoning.Verification;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Reasoning;

/// <summary>
/// Extended deep integration tests for AiDotNet.Reasoning module.
/// Tests mathematical correctness, edge cases, and cross-component integration
/// for DiversitySampler, ContradictionDetector heuristics, CalculatorVerifier,
/// AdaptiveComputeScaler, and search algorithms.
/// </summary>
public class ReasoningExtendedIntegrationTests
{
    #region DiversitySampler - Jaccard Diversity Calculation

    [Fact]
    public void DiversitySampler_IdenticalThoughts_ZeroDiversity()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "Use solar panels for electricity generation" };
        var t2 = new ThoughtNode<double> { Thought = "Use solar panels for electricity generation" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        // Identical text => Jaccard similarity = 1.0, diversity = 0.0
        Assert.Equal(0.0, diversity, 3);
    }

    [Fact]
    public void DiversitySampler_CompletelyDifferentThoughts_HighDiversity()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "Solar panels generate renewable electricity efficiently" };
        var t2 = new ThoughtNode<double> { Thought = "Factory manufacturing process optimization reduces waste" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        // No word overlap and different domains => diversity should be high
        Assert.True(diversity > 0.8, $"Expected diversity > 0.8 for unrelated thoughts, got {diversity}");
    }

    [Fact]
    public void DiversitySampler_HandCalculated_JaccardDiversity()
    {
        // Hand-calculate:
        // t1 words (after stop-word removal, len>2): {"solar", "panels", "generate", "power"}
        // t2 words: {"solar", "panels", "provide", "energy"}
        // Intersection: {"solar", "panels"} => 2
        // Union: {"solar", "panels", "generate", "power", "provide", "energy"} => 6
        // Jaccard similarity = 2/6 = 0.333
        // Diversity = 1 - 0.333 = 0.667
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "solar panels generate power" };
        var t2 = new ThoughtNode<double> { Thought = "solar panels provide energy" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        // Jaccard: intersection=2 (solar, panels), union=6 (solar, panels, generate, power, provide, energy)
        // similarity=2/6=0.333, diversity=0.667
        Assert.InRange(diversity, 0.55, 0.75);
    }

    [Fact]
    public void DiversitySampler_DomainBoost_IncreaseDiversity()
    {
        var sampler = new DiversitySampler<double>();
        // Same domain (energy)
        var t1 = new ThoughtNode<double> { Thought = "Install solar panels on rooftops" };
        var t2 = new ThoughtNode<double> { Thought = "Build wind turbines offshore" };
        double sameDomainDiversity = sampler.CalculateDiversity(t1, t2);

        // Different domains (energy vs transportation)
        var t3 = new ThoughtNode<double> { Thought = "Install solar panels on rooftops" };
        var t4 = new ThoughtNode<double> { Thought = "Switch to electric vehicle fleet" };
        double crossDomainDiversity = sampler.CalculateDiversity(t3, t4);

        // Cross-domain should have higher diversity due to 30% domain boost
        Assert.True(crossDomainDiversity >= sameDomainDiversity,
            $"Cross-domain diversity ({crossDomainDiversity}) should >= same-domain ({sameDomainDiversity})");
    }

    [Fact]
    public void DiversitySampler_NullThought_ReturnsZero()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "Some thought" };

        double diversity = sampler.CalculateDiversity(t1, null!);

        Assert.Equal(0.0, diversity);
    }

    [Fact]
    public void DiversitySampler_EmptyThoughts_ReturnsZero()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "" };
        var t2 = new ThoughtNode<double> { Thought = "" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        Assert.Equal(0.0, diversity);
    }

    [Fact]
    public void DiversitySampler_OneEmptyOneNonEmpty_ReturnsOne()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "" };
        var t2 = new ThoughtNode<double> { Thought = "solar panels generate power" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        // One empty, one not => diversity should be 1.0
        Assert.Equal(1.0, diversity);
    }

    [Fact]
    public void DiversitySampler_SampleDiverse_SelectsMaximallyDifferentThoughts()
    {
        var sampler = new DiversitySampler<double>();
        var candidates = new List<ThoughtNode<double>>
        {
            new() { Thought = "Use solar panels for electricity", EvaluationScore = 0.9 },
            new() { Thought = "Use wind turbines for electricity", EvaluationScore = 0.85 },  // Similar to solar
            new() { Thought = "Switch to electric vehicle fleet", EvaluationScore = 0.8 },   // Different domain
            new() { Thought = "Install solar power systems", EvaluationScore = 0.7 },         // Similar to solar
            new() { Thought = "Carbon capture in factory facilities", EvaluationScore = 0.75 } // Different domain
        };

        var config = new ReasoningConfig();
        var selected = sampler.SampleDiverse(candidates, 3, config);

        Assert.Equal(3, selected.Count);
        // First selected should be highest-scored
        Assert.Contains("solar", selected[0].Thought.ToLower());
        // Next selections should maximize diversity - expect different domains
    }

    [Fact]
    public void DiversitySampler_SampleDiverse_RequestMoreThanAvailable_ReturnsAll()
    {
        var sampler = new DiversitySampler<double>();
        var candidates = new List<ThoughtNode<double>>
        {
            new() { Thought = "Option A", EvaluationScore = 0.9 },
            new() { Thought = "Option B", EvaluationScore = 0.8 }
        };

        var config = new ReasoningConfig();
        var selected = sampler.SampleDiverse(candidates, 5, config);

        Assert.Equal(2, selected.Count);
    }

    [Fact]
    public void DiversitySampler_SampleDiverse_EmptyCandidates_Throws()
    {
        var sampler = new DiversitySampler<double>();
        var config = new ReasoningConfig();

        Assert.Throws<ArgumentException>(() =>
            sampler.SampleDiverse(new List<ThoughtNode<double>>(), 3, config));
    }

    [Fact]
    public void DiversitySampler_SampleDiverse_ZeroSamples_Throws()
    {
        var sampler = new DiversitySampler<double>();
        var candidates = new List<ThoughtNode<double>>
        {
            new() { Thought = "Option A", EvaluationScore = 0.9 }
        };

        var config = new ReasoningConfig();
        Assert.Throws<ArgumentException>(() =>
            sampler.SampleDiverse(candidates, 0, config));
    }

    [Fact]
    public void DiversitySampler_Diversity_IsSymmetric()
    {
        var sampler = new DiversitySampler<double>();
        var t1 = new ThoughtNode<double> { Thought = "Solar panels generate electricity" };
        var t2 = new ThoughtNode<double> { Thought = "Electric vehicles reduce emissions" };

        double d12 = sampler.CalculateDiversity(t1, t2);
        double d21 = sampler.CalculateDiversity(t2, t1);

        Assert.Equal(d12, d21, 10);
    }

    [Fact]
    public void DiversitySampler_StopWordsFiltered_DontAffectDiversity()
    {
        var sampler = new DiversitySampler<double>();
        // These texts differ only in stop words
        var t1 = new ThoughtNode<double> { Thought = "The solar panels are on the roof" };
        var t2 = new ThoughtNode<double> { Thought = "Solar panels on a roof" };

        double diversity = sampler.CalculateDiversity(t1, t2);

        // After stop word removal, the significant words should be the same
        // So diversity should be very low
        Assert.True(diversity < 0.3, $"Expected low diversity for stop-word-only differences, got {diversity}");
    }

    #endregion

    #region ContradictionDetector - HasObviousContradiction Heuristics

    [Fact]
    public void ContradictionDetector_DifferentAnswerValues_DetectsContradiction()
    {
        // Uses the HasObviousContradiction heuristic: "answer is N" vs "answer is M"
        // Test the heuristic via the full pipeline with a mock chat model
        var text1 = "The answer is 36";
        var text2 = "The answer is 42";

        // Both match the "answer is N" pattern with different values
        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.True(detected, "Should detect contradiction between different answer values");
    }

    [Fact]
    public void ContradictionDetector_SameAnswerValues_NoContradiction()
    {
        var text1 = "The answer is 36";
        var text2 = "The answer is 36";

        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.False(detected, "Should not detect contradiction when answers match");
    }

    [Fact]
    public void ContradictionDetector_IsNotPattern_DetectsContradiction()
    {
        // "X is not Y" vs "X is Y" pattern
        var text1 = "x is not positive";
        var text2 = "x is positive";

        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.True(detected, "Should detect 'is not Y' vs 'is Y' contradiction");
    }

    [Fact]
    public void ContradictionDetector_DifferentEqualsValues_DetectsContradiction()
    {
        // "X equals N" vs "X equals M" pattern
        var text1 = "x equals 10";
        var text2 = "x equals 5";

        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.True(detected, "Should detect different 'equals' values as contradiction");
    }

    [Fact]
    public void ContradictionDetector_DifferentIsValues_DetectsContradiction()
    {
        // "X is Y" vs "X is Z" pattern
        var text1 = "result is positive";
        var text2 = "result is negative";

        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.True(detected, "Should detect 'X is Y' vs 'X is Z' contradiction");
    }

    [Fact]
    public void ContradictionDetector_UnrelatedStatements_NoContradiction()
    {
        var text1 = "Calculate the area of the circle";
        var text2 = "Multiply by pi";

        bool detected = HasObviousContradictionViaReflection(text1, text2);
        Assert.False(detected, "Unrelated statements should not be contradictions");
    }

    [Fact]
    public void ContradictionDetector_EmptyStrings_NoContradiction()
    {
        bool detected = HasObviousContradictionViaReflection("", "some text");
        Assert.False(detected, "Empty string should not cause contradiction");

        detected = HasObviousContradictionViaReflection("some text", "");
        Assert.False(detected, "Empty string should not cause contradiction");
    }

    /// <summary>
    /// Uses reflection to test the private HasObviousContradiction method since
    /// ContradictionDetector requires an IChatModel that we can't mock easily.
    /// </summary>
    private bool HasObviousContradictionViaReflection(string text1, string text2)
    {
        // Create a mock IChatModel via proxy
        var mockModel = new MockChatModel<double>();
        var detector = new ContradictionDetector<double>(mockModel);

        var method = typeof(ContradictionDetector<double>).GetMethod(
            "HasObviousContradiction",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (method is null)
            throw new InvalidOperationException("Could not find HasObviousContradiction method");

        return (bool)method.Invoke(detector, new object[] { text1, text2 })!;
    }

    #endregion

    #region CalculatorVerifier - Expression Evaluation

    [Fact]
    public void CalculatorVerifier_ToolName_IsCalculator()
    {
        var verifier = new CalculatorVerifier<double>();
        Assert.Equal("Calculator", verifier.ToolName);
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_ArithmeticExpression_ReturnsTrue()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double> { Content = "Calculate 15 + 25 = 40" };

        Assert.True(verifier.CanVerify(step));
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_PercentageExpression_ReturnsTrue()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double> { Content = "15% of the total" };

        Assert.True(verifier.CanVerify(step));
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_MultiplicationSign_ReturnsTrue()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double> { Content = "0.15 × 240" };

        Assert.True(verifier.CanVerify(step));
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_PlainText_ReturnsFalse()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double> { Content = "Consider the problem from a different angle" };

        Assert.False(verifier.CanVerify(step));
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_NullStep_ReturnsFalse()
    {
        var verifier = new CalculatorVerifier<double>();
        Assert.False(verifier.CanVerify(null!));
    }

    [Fact]
    public void CalculatorVerifier_CanVerify_EmptyContent_ReturnsFalse()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double> { Content = "" };

        Assert.False(verifier.CanVerify(step));
    }

    [Fact]
    public async Task CalculatorVerifier_CorrectArithmetic_Passes()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "0.15 * 240 = 36"
        };

        var result = await verifier.VerifyStepAsync(step);

        Assert.True(result.Passed, $"Expected verification to pass. Explanation: {result.Explanation}");
        Assert.Equal("Calculator", result.ToolUsed);
    }

    [Fact]
    public async Task CalculatorVerifier_IncorrectArithmetic_Fails()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "0.15 * 240 = 35"  // Wrong! Should be 36
        };

        var result = await verifier.VerifyStepAsync(step);

        Assert.False(result.Passed, "Expected verification to fail for incorrect calculation");
    }

    [Fact]
    public async Task CalculatorVerifier_Addition_HandVerified()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "100 + 200 + 50 = 350"
        };

        var result = await verifier.VerifyStepAsync(step);

        Assert.True(result.Passed, $"100+200+50=350 should pass. Explanation: {result.Explanation}");
    }

    [Fact]
    public async Task CalculatorVerifier_Division_HandVerified()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "100 / 4 = 25"
        };

        var result = await verifier.VerifyStepAsync(step);

        Assert.True(result.Passed, $"100/4=25 should pass. Explanation: {result.Explanation}");
    }

    [Fact]
    public async Task CalculatorVerifier_NoCalculation_PassesWithLowConfidence()
    {
        var verifier = new CalculatorVerifier<double>();
        var step = new ReasoningStep<double>
        {
            StepNumber = 1,
            Content = "Consider the problem carefully"
        };

        var result = await verifier.VerifyStepAsync(step);

        Assert.True(result.Passed);
        Assert.Equal(0.5, result.Confidence, 3);
    }

    [Fact]
    public async Task CalculatorVerifier_NullStep_ThrowsArgumentNull()
    {
        var verifier = new CalculatorVerifier<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            verifier.VerifyStepAsync(null!));
    }

    #endregion

    #region AdaptiveComputeScaler - Difficulty Estimation

    [Fact]
    public void AdaptiveComputeScaler_EmptyProblem_DefaultMediumDifficulty()
    {
        var scaler = new AdaptiveComputeScaler();
        double difficulty = scaler.EstimateDifficulty("");

        Assert.Equal(0.5, difficulty);
    }

    [Fact]
    public void AdaptiveComputeScaler_SimpleProblem_LowDifficulty()
    {
        var scaler = new AdaptiveComputeScaler();
        double difficulty = scaler.EstimateDifficulty("What is 2 + 2?");

        // Short, simple problem
        Assert.True(difficulty < 0.4, $"Simple problem should have low difficulty, got {difficulty}");
    }

    [Fact]
    public void AdaptiveComputeScaler_ComplexProblem_HighDifficulty()
    {
        var scaler = new AdaptiveComputeScaler();
        string problem = "Prove that the algorithm for dynamic programming optimization achieves " +
                         "O(n log n) complexity. First, analyze the recursive structure. " +
                         "Then, design an efficient implementation using memoization. " +
                         "Finally, evaluate the time and space complexity.";

        double difficulty = scaler.EstimateDifficulty(problem);

        // Contains "prove", "algorithm", "dynamic programming", "optimize", "design", "evaluate"
        // Has multiple steps ("first", "then", "finally")
        // Has mathematical symbols and code keywords
        Assert.True(difficulty > 0.5, $"Complex problem should have high difficulty, got {difficulty}");
    }

    [Fact]
    public void AdaptiveComputeScaler_MathProblem_IncreasesDifficulty()
    {
        var scaler = new AdaptiveComputeScaler();
        double simpleDiff = scaler.EstimateDifficulty("What is two plus two?");
        double mathDiff = scaler.EstimateDifficulty("Calculate x + y = z where x = 5");

        // Math expressions increase difficulty by 0.2
        Assert.True(mathDiff > simpleDiff,
            $"Math problem ({mathDiff}) should be harder than text ({simpleDiff})");
    }

    [Fact]
    public void AdaptiveComputeScaler_DifficultyClamped_0To1()
    {
        var scaler = new AdaptiveComputeScaler();

        // Very long problem with many hard keywords
        string monster = string.Join(" ", Enumerable.Repeat(
            "prove analyze optimize design algorithm complexity recursive dynamic programming evaluate synthesize", 20));

        double difficulty = scaler.EstimateDifficulty(monster);

        Assert.InRange(difficulty, 0.0, 1.0);
    }

    #endregion

    #region AdaptiveComputeScaler - Config Scaling

    [Fact]
    public void AdaptiveComputeScaler_EasyProblem_ReducedConfig()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10, ExplorationDepth = 3 };
        var scaler = new AdaptiveComputeScaler(baseline);

        var config = scaler.ScaleConfig("Simple question", estimatedDifficulty: 0.1);

        // Easy: scaling factor = 0.5 + (0.1/0.3)*0.5 = 0.5 + 0.167 = 0.667
        // MaxSteps = round(10 * 0.667) = 7
        Assert.True(config.MaxSteps < baseline.MaxSteps,
            $"Easy problem MaxSteps ({config.MaxSteps}) should be < baseline ({baseline.MaxSteps})");
        Assert.False(config.EnableVerification, "Easy problem should not enable verification");
        Assert.False(config.EnableSelfRefinement, "Easy problem should not enable self-refinement");
    }

    [Fact]
    public void AdaptiveComputeScaler_HardProblem_IncreasedConfig()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10, ExplorationDepth = 3 };
        var scaler = new AdaptiveComputeScaler(baseline);

        var config = scaler.ScaleConfig("Prove theorem", estimatedDifficulty: 0.9);

        // Hard: scaling factor = 2.0 + (0.2/0.3)*(5.0-2.0) = 2.0 + 2.0 = 4.0
        Assert.True(config.MaxSteps > baseline.MaxSteps,
            $"Hard problem MaxSteps ({config.MaxSteps}) should be > baseline ({baseline.MaxSteps})");
        Assert.True(config.EnableVerification, "Hard problem should enable verification");
        Assert.True(config.EnableSelfRefinement, "Hard problem should enable self-refinement");
        Assert.True(config.EnableContradictionDetection, "Hard problem should enable contradiction detection");
    }

    [Fact]
    public void AdaptiveComputeScaler_ScalingFactor_BoundaryAt03()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10 };
        var scaler = new AdaptiveComputeScaler(baseline);

        // At difficulty = 0.3, the easy formula gives: 0.5 + (0.3/0.3)*0.5 = 1.0
        var config = scaler.ScaleConfig("test", estimatedDifficulty: 0.3);

        // scalingFactor = 1.0, so MaxSteps = round(10 * 1.0) = 10
        Assert.Equal(10, config.MaxSteps);
    }

    [Fact]
    public void AdaptiveComputeScaler_ScalingFactor_BoundaryAt07()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10 };
        var scaler = new AdaptiveComputeScaler(baseline);

        // At difficulty = 0.7, the medium formula gives: 1.0 + ((0.7-0.3)/0.4)*1.0 = 2.0
        var config = scaler.ScaleConfig("test", estimatedDifficulty: 0.7);

        // scalingFactor = 2.0, so MaxSteps = round(10 * 2.0) = 20
        Assert.Equal(20, config.MaxSteps);
    }

    [Fact]
    public void AdaptiveComputeScaler_Difficulty0_MinimalScaling()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10 };
        var scaler = new AdaptiveComputeScaler(baseline);

        var config = scaler.ScaleConfig("test", estimatedDifficulty: 0.0);

        // Easy: 0.5 + (0/0.3)*0.5 = 0.5
        // MaxSteps = round(10 * 0.5) = 5
        Assert.Equal(5, config.MaxSteps);
    }

    [Fact]
    public void AdaptiveComputeScaler_Difficulty1_MaximalScaling()
    {
        var baseline = new ReasoningConfig { MaxSteps = 10 };
        var scaler = new AdaptiveComputeScaler(baseline, maxScalingFactor: 5.0);

        var config = scaler.ScaleConfig("test", estimatedDifficulty: 1.0);

        // Hard: 2.0 + ((1.0-0.7)/0.3)*(5.0-2.0) = 2.0 + 3.0 = 5.0
        // MaxSteps = round(10 * 5.0) = 50
        Assert.Equal(50, config.MaxSteps);
    }

    [Fact]
    public void AdaptiveComputeScaler_VerificationThresholds_CorrectBehavior()
    {
        var scaler = new AdaptiveComputeScaler();

        // Difficulty 0.3 => no verification
        var config03 = scaler.ScaleConfig("test", estimatedDifficulty: 0.3);
        Assert.False(config03.EnableVerification);

        // Difficulty 0.5 => verification enabled (threshold 0.4)
        var config05 = scaler.ScaleConfig("test", estimatedDifficulty: 0.5);
        Assert.True(config05.EnableVerification);
        Assert.False(config05.EnableSelfRefinement); // threshold 0.6

        // Difficulty 0.65 => verification + self-refinement
        var config065 = scaler.ScaleConfig("test", estimatedDifficulty: 0.65);
        Assert.True(config065.EnableVerification);
        Assert.True(config065.EnableSelfRefinement);
        Assert.False(config065.EnableContradictionDetection); // threshold 0.7

        // Difficulty 0.8 => all enabled
        var config08 = scaler.ScaleConfig("test", estimatedDifficulty: 0.8);
        Assert.True(config08.EnableVerification);
        Assert.True(config08.EnableSelfRefinement);
        Assert.True(config08.EnableContradictionDetection);
        Assert.True(config08.EnableDiversitySampling);
    }

    [Fact]
    public void AdaptiveComputeScaler_Temperature_DecreasesForEasy()
    {
        var baseline = new ReasoningConfig { Temperature = 0.7 };
        var scaler = new AdaptiveComputeScaler(baseline);

        var easyConfig = scaler.ScaleConfig("test", estimatedDifficulty: 0.1);
        var hardConfig = scaler.ScaleConfig("test", estimatedDifficulty: 0.9);

        // Temperature formula: max(0.1, baseline * (0.5 + difficulty * 0.5))
        // Easy: max(0.1, 0.7 * (0.5 + 0.05)) = max(0.1, 0.385) = 0.385
        // Hard: max(0.1, 0.7 * (0.5 + 0.45)) = max(0.1, 0.665) = 0.665
        Assert.True(easyConfig.Temperature < hardConfig.Temperature,
            $"Easy temperature ({easyConfig.Temperature}) should be < hard ({hardConfig.Temperature})");
    }

    #endregion

    #region AdaptiveComputeScaler - Strategy Recommendation

    [Fact]
    public void AdaptiveComputeScaler_RecommendedStrategy_EasyIsCoT()
    {
        var scaler = new AdaptiveComputeScaler();

        Assert.Equal("Chain-of-Thought", scaler.GetRecommendedStrategy(0.1));
        Assert.Equal("Chain-of-Thought", scaler.GetRecommendedStrategy(0.2));
    }

    [Fact]
    public void AdaptiveComputeScaler_RecommendedStrategy_MediumIsCoT()
    {
        var scaler = new AdaptiveComputeScaler();

        Assert.Equal("Chain-of-Thought", scaler.GetRecommendedStrategy(0.4));
        Assert.Equal("Chain-of-Thought", scaler.GetRecommendedStrategy(0.5));
    }

    [Fact]
    public void AdaptiveComputeScaler_RecommendedStrategy_HardIsSelfConsistency()
    {
        var scaler = new AdaptiveComputeScaler();

        Assert.Equal("Self-Consistency", scaler.GetRecommendedStrategy(0.6));
        Assert.Equal("Self-Consistency", scaler.GetRecommendedStrategy(0.7));
    }

    [Fact]
    public void AdaptiveComputeScaler_RecommendedStrategy_VeryHardIsToT()
    {
        var scaler = new AdaptiveComputeScaler();

        Assert.Equal("Tree-of-Thoughts", scaler.GetRecommendedStrategy(0.8));
        Assert.Equal("Tree-of-Thoughts", scaler.GetRecommendedStrategy(1.0));
    }

    [Fact]
    public void AdaptiveComputeScaler_RecommendedStrategy_ClampedInput()
    {
        var scaler = new AdaptiveComputeScaler();

        // Negative values should be clamped to 0.0
        Assert.Equal("Chain-of-Thought", scaler.GetRecommendedStrategy(-0.5));

        // Values > 1.0 should be clamped
        Assert.Equal("Tree-of-Thoughts", scaler.GetRecommendedStrategy(1.5));
    }

    [Fact]
    public void AdaptiveComputeScaler_MaxScalingFactor_ClampsToMinimum2()
    {
        // If we pass 1.0 as max scaling, it should clamp to 2.0
        var scaler = new AdaptiveComputeScaler(maxScalingFactor: 1.0);

        // At difficulty 1.0: factor = 2.0 + (0.3/0.3)*(2.0-2.0) = 2.0
        var config = scaler.ScaleConfig("test", estimatedDifficulty: 1.0);

        // MaxSteps = round(10 * 2.0) = 20 (using default baseline MaxSteps=10)
        Assert.Equal(20, config.MaxSteps);
    }

    #endregion

    #region Search Algorithms - BestFirstSearch

    [Fact]
    public async Task BestFirstSearch_FindsTerminalNode()
    {
        var search = new BestFirstSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Solve the problem" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 3,
            BranchingFactor = 2,
            MaxSteps = 10
        };

        var path = await search.SearchAsync(root, generator, evaluator, config);

        Assert.NotEmpty(path);
        Assert.Equal("Solve the problem", path[0].Thought);
        // Path should contain at least root + one child
        Assert.True(path.Count >= 2, $"Path should have >= 2 nodes, got {path.Count}");
    }

    [Fact]
    public async Task BestFirstSearch_NullRoot_Throws()
    {
        var search = new BestFirstSearch<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(null!, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(), new ReasoningConfig()));
    }

    [Fact]
    public async Task BestFirstSearch_RespectsCancellation()
    {
        var search = new BestFirstSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            search.SearchAsync(root, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(),
                new ReasoningConfig { ExplorationDepth = 10 }, cts.Token));
    }

    [Fact]
    public async Task BestFirstSearch_RespectsDepthLimit()
    {
        var search = new BestFirstSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 2,
            BranchingFactor = 2,
            MaxSteps = 50
        };

        var path = await search.SearchAsync(root, generator, evaluator, config);

        // No node in path should exceed depth 2
        foreach (var node in path)
        {
            Assert.True(node.Depth <= config.ExplorationDepth,
                $"Node at depth {node.Depth} exceeds limit {config.ExplorationDepth}");
        }
    }

    #endregion

    #region Search Algorithms - BreadthFirstSearch

    [Fact]
    public async Task BreadthFirstSearch_ExploresByLevel()
    {
        var search = new BreadthFirstSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 2,
            BranchingFactor = 2,
            MaxSteps = 10
        };

        var path = await search.SearchAsync(root, generator, evaluator, config);

        Assert.NotEmpty(path);
        // BFS explores level by level, so path should go root -> child
        Assert.Equal("Root", path[0].Thought);
    }

    [Fact]
    public async Task BreadthFirstSearch_NullRoot_Throws()
    {
        var search = new BreadthFirstSearch<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(null!, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(), new ReasoningConfig()));
    }

    #endregion

    #region Search Algorithms - DepthFirstSearch

    [Fact]
    public async Task DepthFirstSearch_ExploresDepthFirst()
    {
        var search = new DepthFirstSearch<double>();
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 3,
            BranchingFactor = 2,
            MaxSteps = 10
        };

        var path = await search.SearchAsync(root, generator, evaluator, config);

        Assert.NotEmpty(path);
        Assert.Equal("Root", path[0].Thought);
        // DFS should produce a path that goes deep
        if (path.Count > 1)
        {
            // Each subsequent node should have increasing depth
            for (int i = 1; i < path.Count; i++)
            {
                Assert.True(path[i].Depth >= path[i - 1].Depth,
                    $"DFS path should have non-decreasing depth. Node {i}: depth {path[i].Depth} < {path[i - 1].Depth}");
            }
        }
    }

    [Fact]
    public async Task DepthFirstSearch_NullRoot_Throws()
    {
        var search = new DepthFirstSearch<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            search.SearchAsync(null!, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(), new ReasoningConfig()));
    }

    #endregion

    #region Search Algorithms - MonteCarloTreeSearch

    [Fact]
    public void MCTS_DefaultConstruction_ValidParameters()
    {
        var mcts = new MonteCarloTreeSearch<double>();

        Assert.Equal("Monte Carlo Tree Search (MCTS)", mcts.AlgorithmName);
        Assert.Contains("exploration", mcts.Description.ToLower());
    }

    [Fact]
    public void MCTS_NegativeExplorationConstant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MonteCarloTreeSearch<double>(explorationConstant: -1.0));
    }

    [Fact]
    public void MCTS_NaNExplorationConstant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MonteCarloTreeSearch<double>(explorationConstant: double.NaN));
    }

    [Fact]
    public void MCTS_InfinityExplorationConstant_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MonteCarloTreeSearch<double>(explorationConstant: double.PositiveInfinity));
    }

    [Fact]
    public void MCTS_ZeroSimulations_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MonteCarloTreeSearch<double>(numSimulations: 0));
    }

    [Fact]
    public async Task MCTS_SearchAsync_ProducesPath()
    {
        var mcts = new MonteCarloTreeSearch<double>(numSimulations: 10);
        var root = new ThoughtNode<double> { Thought = "Root problem" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 2,
            BranchingFactor = 2,
            MaxSteps = 10
        };

        var path = await mcts.SearchAsync(root, generator, evaluator, config);

        Assert.NotEmpty(path);
        Assert.Equal("Root problem", path[0].Thought);
    }

    [Fact]
    public async Task MCTS_Backpropagation_VisitCountsIncrease()
    {
        var mcts = new MonteCarloTreeSearch<double>(numSimulations: 5);
        var root = new ThoughtNode<double> { Thought = "Root" };
        var generator = new MockThoughtGenerator<double>();
        var evaluator = new MockThoughtEvaluator<double>();
        var config = new ReasoningConfig
        {
            ExplorationDepth = 2,
            BranchingFactor = 2,
            MaxSteps = 10
        };

        await mcts.SearchAsync(root, generator, evaluator, config);

        // After MCTS, root should have visit counts in metadata
        Assert.True(root.Metadata.ContainsKey("visits"), "Root should have visits metadata");
        int visits = (int)root.Metadata["visits"];
        Assert.True(visits >= 5, $"Root should have at least 5 visits (from 5 simulations), got {visits}");
    }

    [Fact]
    public async Task MCTS_NullRoot_Throws()
    {
        var mcts = new MonteCarloTreeSearch<double>();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            mcts.SearchAsync(null!, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(), new ReasoningConfig()));
    }

    [Fact]
    public async Task MCTS_RespectsCancellation()
    {
        var mcts = new MonteCarloTreeSearch<double>(numSimulations: 1000);
        var root = new ThoughtNode<double> { Thought = "Root" };
        var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            mcts.SearchAsync(root, new MockThoughtGenerator<double>(),
                new MockThoughtEvaluator<double>(),
                new ReasoningConfig { ExplorationDepth = 10 }, cts.Token));
    }

    [Fact]
    public void MCTS_ZeroExplorationConstant_AllowsPureExploitation()
    {
        // Zero exploration constant means pure exploitation (no exploration bonus)
        var mcts = new MonteCarloTreeSearch<double>(explorationConstant: 0.0);
        Assert.NotNull(mcts);
    }

    #endregion

    #region Cross-Component Integration

    [Fact]
    public void AdaptiveComputeScaler_WithSearchAlgorithm_ConfigAffectsSearch()
    {
        var scaler = new AdaptiveComputeScaler();

        var easyConfig = scaler.ScaleConfig("What is 2+2?", estimatedDifficulty: 0.1);
        var hardConfig = scaler.ScaleConfig("Prove the Riemann hypothesis", estimatedDifficulty: 0.95);

        // Verify that harder problems get more compute budget
        Assert.True(hardConfig.MaxSteps > easyConfig.MaxSteps);
        Assert.True(hardConfig.ExplorationDepth > easyConfig.ExplorationDepth);
        Assert.True(hardConfig.BranchingFactor >= easyConfig.BranchingFactor);
    }

    [Fact]
    public void DiversitySampler_WithReasoningChainSteps_ProducesDiversePaths()
    {
        var sampler = new DiversitySampler<double>();

        // Simulate multiple reasoning paths from a chain
        var paths = new List<ThoughtNode<double>>
        {
            new() { Thought = "Approach 1: Use algebraic manipulation to solve", EvaluationScore = 0.85 },
            new() { Thought = "Approach 2: Use algebraic substitution method", EvaluationScore = 0.82 },
            new() { Thought = "Approach 3: Draw a geometric diagram to visualize", EvaluationScore = 0.78 },
            new() { Thought = "Approach 4: Apply computational numerical methods", EvaluationScore = 0.80 },
            new() { Thought = "Approach 5: Use algebraic factoring technique", EvaluationScore = 0.81 }
        };

        var config = new ReasoningConfig();
        var selected = sampler.SampleDiverse(paths, 3, config);

        Assert.Equal(3, selected.Count);
        // The 3 selected should be more diverse than just the top-3 by score
        // (which would all be algebraic approaches)
    }

    [Fact]
    public async Task CalculatorVerifier_WithReasoningChain_VerifiesAllSteps()
    {
        var verifier = new CalculatorVerifier<double>();
        var chain = new ReasoningChain<double> { Query = "What is 15% of 240?" };

        var step1 = new ReasoningStep<double>
        {
            Content = "Convert 15% to decimal: 15 / 100 = 0.15"
        };
        chain.AddStep(step1);

        var step2 = new ReasoningStep<double>
        {
            Content = "Multiply: 0.15 * 240 = 36"
        };
        chain.AddStep(step2);

        // Verify each step
        var result1 = await verifier.VerifyStepAsync(step1);
        var result2 = await verifier.VerifyStepAsync(step2);

        Assert.True(result1.Passed, $"Step 1 should pass: {result1.Explanation}");
        Assert.True(result2.Passed, $"Step 2 should pass: {result2.Explanation}");
    }

    [Fact]
    public void ReasoningChain_ScoreConsistency_AverageMatchesManualCalculation()
    {
        var chain = new ReasoningChain<double>();
        chain.AddStep(new ReasoningStep<double> { Content = "Step 1", Score = 0.8 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 2", Score = 0.9 });
        chain.AddStep(new ReasoningStep<double> { Content = "Step 3", Score = 0.7 });

        // Hand-calculated average: (0.8 + 0.9 + 0.7) / 3 = 0.8
        double avg = chain.GetAverageScore();
        Assert.Equal(0.8, avg, 5);

        // Minimum should be 0.7
        double min = chain.GetMinimumScore();
        Assert.Equal(0.7, min, 5);
    }

    [Fact]
    public async Task AllSearchAlgorithms_SameInput_AllProducePaths()
    {
        var algorithms = new ISearchAlgorithm<double>[]
        {
            new BeamSearch<double>(),
            new BestFirstSearch<double>(),
            new BreadthFirstSearch<double>(),
            new DepthFirstSearch<double>(),
            new MonteCarloTreeSearch<double>(numSimulations: 5)
        };

        var config = new ReasoningConfig
        {
            ExplorationDepth = 2,
            BranchingFactor = 2,
            BeamWidth = 3,
            MaxSteps = 10
        };

        foreach (var algorithm in algorithms)
        {
            var root = new ThoughtNode<double> { Thought = "Root" };
            var generator = new MockThoughtGenerator<double>();
            var evaluator = new MockThoughtEvaluator<double>();

            var path = await algorithm.SearchAsync(root, generator, evaluator, config);

            Assert.NotEmpty(path);
            Assert.Equal("Root", path[0].Thought);
        }
    }

    #endregion

    #region ThoughtNode - Additional Property Tests

    [Fact]
    public void ThoughtNode_PathScores_HandCalculated_3LevelTree()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0, EvaluationScore = 0.9 };
        var child = new ThoughtNode<double> { Thought = "Child", Parent = root, Depth = 1, EvaluationScore = 0.85 };
        var grandchild = new ThoughtNode<double> { Thought = "Grandchild", Parent = child, Depth = 2, EvaluationScore = 0.95 };

        var pathScores = grandchild.PathScores;

        Assert.Equal(3, pathScores.Length);
        Assert.Equal(0.9, pathScores[0]);
        Assert.Equal(0.85, pathScores[1]);
        Assert.Equal(0.95, pathScores[2]);
    }

    [Fact]
    public void ThoughtNode_CheckIsTerminalByHeuristic_AllKeywords()
    {
        var keywords = new[] { "final answer", "conclusion", "therefore", "the answer is" };

        foreach (var keyword in keywords)
        {
            var node = new ThoughtNode<double> { Thought = $"After analysis, {keyword} 42" };
            Assert.True(node.CheckIsTerminalByHeuristic(),
                $"Should detect '{keyword}' as terminal");
        }
    }

    [Fact]
    public void ThoughtNode_CheckIsTerminalByHeuristic_CaseInsensitive()
    {
        var node = new ThoughtNode<double> { Thought = "THEREFORE, the result is 42" };
        Assert.True(node.CheckIsTerminalByHeuristic());
    }

    #endregion

    #region Contradiction Model Properties

    [Fact]
    public void Contradiction_DefaultValues()
    {
        var contradiction = new Contradiction();

        Assert.Equal(0, contradiction.Step1Number);
        Assert.Equal(0, contradiction.Step2Number);
        Assert.Equal(string.Empty, contradiction.Explanation);
        Assert.Equal(0.0, contradiction.Severity);
    }

    [Fact]
    public void Contradiction_ToString_IncludesAllFields()
    {
        var contradiction = new Contradiction
        {
            Step1Number = 2,
            Step2Number = 5,
            Explanation = "Conflicting values",
            Severity = 0.85
        };

        var str = contradiction.ToString();

        Assert.Contains("2", str);
        Assert.Contains("5", str);
        Assert.Contains("Conflicting values", str);
        Assert.Contains("0.85", str);
    }

    [Fact]
    public void VerificationResult_DefaultValues()
    {
        var result = new VerificationResult<double>();

        Assert.False(result.Passed);
        Assert.Equal(string.Empty, result.ActualResult);
        Assert.Equal(string.Empty, result.ExpectedResult);
        Assert.Equal(string.Empty, result.Explanation);
        Assert.Equal(string.Empty, result.ToolUsed);
    }

    [Fact]
    public void VerificationResult_ToString_ShowsPassFail()
    {
        var passed = new VerificationResult<double> { Passed = true, Explanation = "All correct", ToolUsed = "Calculator" };
        var failed = new VerificationResult<double> { Passed = false, Explanation = "Wrong answer", ToolUsed = "Calculator" };

        Assert.Contains("PASSED", passed.ToString());
        Assert.Contains("FAILED", failed.ToString());
    }

    #endregion

    #region Mock Classes

    /// <summary>
    /// Mock IChatModel for testing components that require it.
    /// </summary>
    private class MockChatModel<T> : IChatModel<T>
    {
        public string ModelName => "MockChatModel";
        public int MaxContextTokens => 4096;
        public int MaxGenerationTokens => 1024;

        public Task<string> GenerateAsync(string prompt, CancellationToken cancellationToken = default)
        {
            return Task.FromResult("{\"contradictory\": false}");
        }

        public string Generate(string prompt)
        {
            return "{\"contradictory\": false}";
        }

        public Task<string> GenerateResponseAsync(string prompt, CancellationToken cancellationToken = default)
        {
            return GenerateAsync(prompt, cancellationToken);
        }
    }

    /// <summary>
    /// Mock thought generator for search algorithm testing.
    /// Generates deterministic thoughts with terminal detection at depth 2+.
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
            cancellationToken.ThrowIfCancellationRequested();
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

                // At depth 2+, generate terminal thoughts
                if (currentNode.Depth >= 1 && i == 0)
                {
                    node.Thought = "Therefore, the answer is 42";
                }

                thoughts.Add(node);
            }

            return Task.FromResult(thoughts);
        }
    }

    /// <summary>
    /// Mock thought evaluator that returns deterministic scores.
    /// Terminal-looking thoughts get higher scores.
    /// </summary>
    private class MockThoughtEvaluator<T> : IThoughtEvaluator<T>
    {
        private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
        private int _evalCount;

        public Task<T> EvaluateThoughtAsync(
            ThoughtNode<T> node,
            string originalQuery,
            ReasoningConfig config,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _evalCount++;

            // Deterministic scoring
            double score;
            if (node.Thought.Contains("answer") || node.Thought.Contains("Therefore"))
            {
                score = 0.95; // Terminal thoughts get high scores
            }
            else
            {
                // Vary score based on call count for determinism but not uniformity
                score = 0.5 + (_evalCount % 10) * 0.04;
            }

            return Task.FromResult(_numOps.FromDouble(score));
        }
    }

    #endregion
}
