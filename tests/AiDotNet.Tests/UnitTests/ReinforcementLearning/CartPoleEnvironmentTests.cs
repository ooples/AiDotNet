using AiDotNet.ReinforcementLearning.Environments;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public class CartPoleEnvironmentTests
{
    [Fact]
    public void Constructor_CreatesEnvironment()
    {
        // Arrange & Act
        var env = new CartPoleEnvironment<double>();

        // Assert
        Assert.Equal(4, env.ObservationSpaceDimension);
        Assert.Equal(2, env.ActionSpaceSize);
    }

    [Fact]
    public void Reset_ReturnsValidState()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>(seed: 42);

        // Act
        var state = env.Reset();

        // Assert
        Assert.NotNull(state);
        Assert.Equal(4, state.Length);

        // State values should be small (initial random values)
        for (int i = 0; i < state.Length; i++)
        {
            Assert.True(Math.Abs(state[i]) < 0.1);
        }
    }

    [Fact]
    public void Step_WithValidAction_ReturnsValidTransition()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>(seed: 42);
        env.Reset();

        // Act
        var (nextState, reward, done, info) = env.Step(0); // Push left

        // Assert
        Assert.NotNull(nextState);
        Assert.Equal(4, nextState.Length);
        Assert.True(reward >= 0); // Reward should be 0 or 1
        Assert.NotNull(info);
    }

    [Fact]
    public void Step_WithInvalidAction_ThrowsException()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>();
        env.Reset();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => env.Step(-1));
        Assert.Throws<ArgumentException>(() => env.Step(2));
    }

    [Fact]
    public void Episode_EventuallyTerminates()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>(maxSteps: 100, seed: 42);
        env.Reset();

        bool done = false;
        int steps = 0;
        int maxSteps = 1000; // Safety limit

        // Act - take random actions until done
        var random = new Random(42);
        while (!done && steps < maxSteps)
        {
            int action = random.Next(2);
            (_, _, done, _) = env.Step(action);
            steps++;
        }

        // Assert
        Assert.True(done); // Episode should terminate
        Assert.True(steps <= 100); // Should terminate before max steps
    }

    [Fact]
    public void Seed_MakesEnvironmentDeterministic()
    {
        // Arrange
        var env1 = new CartPoleEnvironment<double>();
        var env2 = new CartPoleEnvironment<double>();

        // Act - seed both environments
        env1.Seed(42);
        env2.Seed(42);

        var state1 = env1.Reset();
        var state2 = env2.Reset();

        // Take same actions
        var (nextState1, _, _, _) = env1.Step(0);
        var (nextState2, _, _, _) = env2.Step(0);

        // Assert - states should be identical
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(nextState1[i], nextState2[i], precision: 10);
        }
    }

    [Fact]
    public void Close_DoesNotThrow()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>();

        // Act & Assert
        env.Close(); // Should not throw
    }
}
