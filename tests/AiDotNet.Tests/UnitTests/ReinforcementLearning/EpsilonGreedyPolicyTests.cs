using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Policies;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public class EpsilonGreedyPolicyTests
{
    [Fact]
    public void Constructor_WithValidParameters_CreatesPolicy()
    {
        // Arrange & Act
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 1.0,
            epsilonMin: 0.01,
            epsilonDecay: 0.995
        );

        // Assert
        Assert.Equal(1.0, policy.Epsilon);
    }

    [Fact]
    public void Constructor_WithInvalidActionSpace_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 0
        ));
    }

    [Fact]
    public void SelectAction_WithHighEpsilon_ReturnsRandomActions()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 1.0, // Always explore
            epsilonMin: 1.0,
            epsilonDecay: 1.0,
            seed: 42
        );

        var state = new Tensor<double>(new[] { 1.0, 2.0 }, [2]);
        var qValues = new Tensor<double>(new[] { 0.1, 0.9, 0.2, 0.3 }, [4]);

        // Act - select multiple times
        var actions = new HashSet<int>();
        for (int i = 0; i < 100; i++)
        {
            actions.Add(policy.SelectAction(state, qValues));
        }

        // Assert - should see multiple different actions (not always best)
        Assert.True(actions.Count > 1);
    }

    [Fact]
    public void SelectAction_WithZeroEpsilon_SelectsBestAction()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 0.0, // Never explore
            epsilonMin: 0.0,
            epsilonDecay: 1.0,
            seed: 42
        );

        var state = new Tensor<double>(new[] { 1.0, 2.0 }, [2]);
        var qValues = new Tensor<double>(new[] { 0.1, 0.9, 0.2, 0.3 }, [4]);

        // Act - select multiple times
        for (int i = 0; i < 10; i++)
        {
            int action = policy.SelectAction(state, qValues);

            // Assert - should always select action 1 (highest Q-value of 0.9)
            Assert.Equal(1, action);
        }
    }

    [Fact]
    public void Update_DecaysEpsilon()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 1.0,
            epsilonMin: 0.01,
            epsilonDecay: 0.9
        );

        double initialEpsilon = policy.Epsilon;

        // Act
        policy.Update();

        // Assert
        Assert.True(policy.Epsilon < initialEpsilon);
        Assert.Equal(0.9, policy.Epsilon, precision: 5);
    }

    [Fact]
    public void Update_DoesNotDecayBelowMinimum()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 0.02,
            epsilonMin: 0.01,
            epsilonDecay: 0.5 // Aggressive decay
        );

        // Act - update multiple times
        for (int i = 0; i < 10; i++)
        {
            policy.Update();
        }

        // Assert - should not go below minimum
        Assert.Equal(0.01, policy.Epsilon, precision: 5);
    }

    [Fact]
    public void GetActionProbabilities_SumToOne()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(
            actionSpaceSize: 4,
            epsilonStart: 0.1,
            epsilonMin: 0.1,
            epsilonDecay: 1.0
        );

        var state = new Tensor<double>(new[] { 1.0, 2.0 }, [2]);
        var qValues = new Tensor<double>(new[] { 0.1, 0.9, 0.2, 0.3 }, [4]);

        // Act
        var probabilities = policy.GetActionProbabilities(state, qValues);

        // Assert
        double sum = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            sum += probabilities[i];
        }
        Assert.Equal(1.0, sum, precision: 5);
    }

    [Fact]
    public void SetEpsilon_WithValidValue_UpdatesEpsilon()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(actionSpaceSize: 4);

        // Act
        policy.SetEpsilon(0.5);

        // Assert
        Assert.Equal(0.5, policy.Epsilon);
    }

    [Fact]
    public void SetEpsilon_WithInvalidValue_ThrowsException()
    {
        // Arrange
        var policy = new EpsilonGreedyPolicy<double>(actionSpaceSize: 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => policy.SetEpsilon(-0.1));
        Assert.Throws<ArgumentException>(() => policy.SetEpsilon(1.1));
    }
}
