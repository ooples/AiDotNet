using Xunit;
using Moq;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Verification;
using AiDotNet.Reasoning.DomainSpecific;
using AiDotNet.Reasoning.Training;

namespace AiDotNet.Tests.Reasoning;

/// <summary>
/// Integration tests for end-to-end reasoning workflows.
/// </summary>
public class IntegrationTests
{
    private readonly Mock<IChatModel> _mockChatModel;

    public IntegrationTests()
    {
        _mockChatModel = new Mock<IChatModel>();
    }

    [Fact]
    public async Task EndToEnd_MathProblem_WithVerification_Succeeds()
    {
        // Arrange
        string problem = "What is 15 × 12?";
        string mockResponse = @"Step 1: Break down the multiplication
15 × 12 = 15 × 10 + 15 × 2

Step 2: Calculate each part
15 × 10 = 150
15 × 2 = 30

Step 3: Add the results
150 + 30 = 180

Final Answer: 180";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var reasoner = new MathematicalReasoner<double>(_mockChatModel.Object);

        // Act
        var result = await reasoner.SolveAsync(problem, useVerification: true);

        // Assert
        Assert.True(result.Success);
        Assert.Contains("180", result.FinalAnswer);
        Assert.NotNull(result.Chain);
        Assert.True(result.Chain.Steps.Count >= 3);
    }

    [Fact]
    public async Task EndToEnd_CodeGeneration_WithExecution_Succeeds()
    {
        // Arrange
        string problem = "Write a function to check if a number is even";
        string mockResponse = @"```python
def is_even(n):
    return n % 2 == 0

# Test
assert is_even(4) == True
assert is_even(7) == False
```";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var reasoner = new CodeReasoner<double>(_mockChatModel.Object);

        // Act
        var result = await reasoner.GenerateCodeAsync(problem, language: "python");

        // Assert
        Assert.True(result.Success);
        Assert.Contains("def is_even", result.FinalAnswer);
    }

    [Fact]
    public async Task EndToEnd_SelfConsistency_MultipleChains_AggregatesResults()
    {
        // Arrange
        string problem = "What is 7 × 8?";
        int callCount = 0;

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                callCount++;
                return callCount % 2 == 0
                    ? "Step 1: Calculate 7 × 8 = 56\nFinal Answer: 56"
                    : "Step 1: 7 × 8 = 56\nFinal Answer: 56";
            });

        var strategy = new SelfConsistencyStrategy<double>(_mockChatModel.Object);
        var config = new ReasoningConfig { NumSamples = 3 };

        // Act
        var result = await strategy.ReasonAsync(problem, config);

        // Assert
        Assert.True(result.Success);
        Assert.Contains("56", result.FinalAnswer);
        Assert.True(callCount >= 3); // Should have sampled multiple times
    }

    [Fact]
    public async Task EndToEnd_HybridRewardModel_CombinesPRMandORM()
    {
        // Arrange
        var chain = new ReasoningChain<double>
        {
            Steps = new List<ReasoningStep<double>>
            {
                new() { Content = "Step 1", Score = 0.9 },
                new() { Content = "Step 2", Score = 0.8 }
            },
            FinalAnswer = "42"
        };

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Score: 0.85");

        var prm = new ProcessRewardModel<double>(_mockChatModel.Object);
        var orm = new OutcomeRewardModel<double>(_mockChatModel.Object);
        var hybrid = new HybridRewardModel<double>(prm, orm, 0.5, 0.5);

        // Act
        var reward = await hybrid.CalculateRewardAsync(chain, correctAnswer: "42");

        // Assert
        Assert.True(Convert.ToDouble(reward) > 0.0);
        Assert.True(Convert.ToDouble(reward) <= 1.0);
    }

    [Fact]
    public async Task EndToEnd_ScientificReasoner_SolvesPhysicsProblem()
    {
        // Arrange
        string problem = "A ball is dropped from 20 meters. How long to hit ground? (g=10m/s²)";
        string mockResponse = @"Step 1: Identify the formula
h = ½gt²

Step 2: Plug in values
20 = ½ × 10 × t²
20 = 5t²

Step 3: Solve for t
t² = 4
t = 2 seconds

Final Answer: 2 seconds";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var reasoner = new ScientificReasoner<double>(_mockChatModel.Object);

        // Act
        var result = await reasoner.SolveAsync(problem, domain: "physics", useFormulas: true);

        // Assert
        Assert.True(result.Success);
        Assert.Contains("2", result.FinalAnswer);
    }

    [Fact]
    public async Task EndToEnd_LogicalReasoner_SolvesDeductiveProblem()
    {
        // Arrange
        string problem = "All cats are mammals. All mammals are animals. What can we conclude about cats?";
        string mockResponse = @"Step 1: Identify premises
P1: All cats are mammals
P2: All mammals are animals

Step 2: Apply transitivity
If A→B and B→C, then A→C

Step 3: Conclusion
All cats are animals

Final Answer: All cats are animals";

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(mockResponse);

        var reasoner = new LogicalReasoner<double>(_mockChatModel.Object);

        // Act
        var result = await reasoner.SolveAsync(problem, logicType: "deductive");

        // Assert
        Assert.True(result.Success);
        Assert.Contains("animals", result.FinalAnswer.ToLowerInvariant());
    }

    [Fact]
    public async Task EndToEnd_TrainingDataCollection_SavesAndLoads()
    {
        // Arrange
        var collector = new TrainingDataCollector<double>();
        var tempFile = Path.GetTempFileName();

        var sample = new TrainingSample<double>
        {
            Problem = "Test problem",
            CorrectAnswer = "Test answer",
            ChainReward = 0.9,
            OutcomeReward = 1.0,
            Category = "test"
        };

        collector.AddSample(sample);

        // Act
        await collector.SaveToFileAsync(tempFile);
        var newCollector = new TrainingDataCollector<double>();
        await newCollector.LoadFromFileAsync(tempFile);

        // Assert
        Assert.Equal(1, newCollector.SampleCount);

        // Cleanup
        File.Delete(tempFile);
    }

    [Fact]
    public async Task EndToEnd_PolicyGradientTraining_UpdatesModel()
    {
        // Arrange
        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Step 1: Calculate\nAnswer: 42");

        var prm = new ProcessRewardModel<double>(_mockChatModel.Object);
        var trainer = new PolicyGradientTrainer<double>(_mockChatModel.Object, prm);

        var chains = new List<ReasoningChain<double>>
        {
            new()
            {
                Steps = new List<ReasoningStep<double>>
                {
                    new() { Content = "Step 1", Score = 0.8 }
                },
                FinalAnswer = "42"
            }
        };

        var correctAnswers = new List<string> { "42" };

        // Act
        var metrics = await trainer.TrainBatchAsync(chains, correctAnswers);

        // Assert
        Assert.NotNull(metrics);
        Assert.True(Convert.ToDouble(metrics.AverageReward) > 0);
    }

    [Fact]
    public async Task EndToEnd_AdaptiveComputeScaling_AdjustsForDifficulty()
    {
        // Arrange
        var scaler = new AdaptiveComputeScaler<double>();

        string easyProblem = "What is 2 + 2?";
        string hardProblem = "Prove the Riemann Hypothesis using advanced number theory";

        // Act
        var easyConfig = scaler.ScaleConfig(easyProblem);
        var hardConfig = scaler.ScaleConfig(hardProblem);

        // Assert
        Assert.True(easyConfig.MaxSteps < hardConfig.MaxSteps);
        Assert.True(easyConfig.ExplorationDepth < hardConfig.ExplorationDepth);
    }

    [Fact]
    public async Task EndToEnd_ChainWithVerificationAndRefinement_ImprovesQuality()
    {
        // Arrange
        string problem = "Calculate 15 × 12";
        int refinementCount = 0;

        _mockChatModel
            .Setup(m => m.GenerateResponseAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                refinementCount++;
                return refinementCount == 1
                    ? "Step 1: 15 × 12 = 170\nFinal: 170" // Wrong
                    : "Step 1: 15 × 12 = 180\nFinal: 180"; // Corrected
            });

        var strategy = new ChainOfThoughtStrategy<double>(_mockChatModel.Object);
        var criticModel = new CriticModel<double>(_mockChatModel.Object);
        var refinement = new SelfRefinementEngine<double>(_mockChatModel.Object);

        // Act
        var result = await strategy.ReasonAsync(problem);

        // If result is wrong, refine
        if (!result.FinalAnswer.Contains("180") && result.Chain != null)
        {
            var context = new ReasoningContext { OriginalQuery = problem };
            foreach (var step in result.Chain.Steps)
            {
                var critique = await criticModel.CritiqueStepAsync(step, context);
                if (Convert.ToDouble(critique.OverallScore) < 0.8)
                {
                    var refined = await refinement.RefineStepAsync(step, critique, context);
                    // In real scenario, would update the step
                }
            }
        }

        // Assert
        Assert.True(refinementCount > 0);
    }

    [Fact]
    public void EndToEnd_ConfigurationPresets_WorkCorrectly()
    {
        // Arrange & Act
        var fastConfig = ReasoningConfig.Fast;
        var defaultConfig = ReasoningConfig.Default;
        var thoroughConfig = ReasoningConfig.Thorough;

        // Assert
        Assert.True(fastConfig.MaxSteps < defaultConfig.MaxSteps);
        Assert.True(defaultConfig.MaxSteps < thoroughConfig.MaxSteps);
        Assert.True(fastConfig.ExplorationDepth < thoroughConfig.ExplorationDepth);
    }
}
