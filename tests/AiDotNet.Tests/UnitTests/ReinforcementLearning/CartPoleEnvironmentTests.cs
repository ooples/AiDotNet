using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.Tensors.LinearAlgebra;
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
        var action = new Vector<double>(new double[] { 0 }); // Push left
        var (nextState, reward, done, info) = env.Step(action);

        // Assert
        Assert.NotNull(nextState);
        Assert.Equal(4, nextState.Length);
        Assert.True(reward >= 0); // Reward should be 0 or 1
        Assert.False(done); // Episode should not be done after one step
        Assert.NotNull(info);
    }

    [Fact]
    public void Step_WithInvalidAction_ThrowsException()
    {
        // Arrange
        var env = new CartPoleEnvironment<double>();
        env.Reset();

        // Act & Assert
        var invalidAction1 = new Vector<double>(new double[] { -1 });
        var invalidAction2 = new Vector<double>(new double[] { 2 });
        Assert.Throws<ArgumentException>(() => env.Step(invalidAction1));
        Assert.Throws<ArgumentException>(() => env.Step(invalidAction2));
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
        var random = RandomHelper.CreateSeededRandom(42);
        while (!done && steps < maxSteps)
        {
            int actionIndex = random.Next(2);
            var action = new Vector<double>(new double[] { actionIndex });
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

        // Assert - initial states should be identical
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(state1[i], state2[i], precision: 10);
        }

        // Take same actions
        var action = new Vector<double>(new double[] { 0 });
        var (nextState1, _, _, _) = env1.Step(action);
        var (nextState2, _, _, _) = env2.Step(action);

        // Assert - next states should be identical
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
