using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public class UniformReplayBufferTests
{
    [Fact]
    public void Constructor_WithValidCapacity_CreatesBuffer()
    {
        // Arrange & Act
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 100);

        // Assert
        Assert.Equal(100, buffer.Capacity);
        Assert.Equal(0, buffer.Count);
    }

    [Fact]
    public void Constructor_WithInvalidCapacity_ThrowsException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() => new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 0));
        Assert.Throws<ArgumentException>(() => new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: -1));
    }

    [Fact]
    public void Add_WithValidExperience_IncreasesCount()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 10);
        var state = new Vector<double>(new double[] { 1.0, 2.0 });
        var nextState = new Vector<double>(new double[] { 3.0, 4.0 });
        var action = new Vector<double>(new double[] { 0.0 });
        var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);

        // Act
        buffer.Add(experience);

        // Assert
        Assert.Equal(1, buffer.Count);
    }

    [Fact]
    public void Add_BeyondCapacity_ReplacesOldest()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 3);

        // Add 4 experiences
        for (int i = 0; i < 4; i++)
        {
            var state = new Vector<double>(new double[] { (double)i });
            var nextState = new Vector<double>(new double[] { (double)(i + 1) });
            var action = new Vector<double>(new double[] { 0.0 });
            var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
            buffer.Add(experience);
        }

        // Assert
        Assert.Equal(3, buffer.Count); // Should still be at capacity
    }

    [Fact]
    public void Sample_WithEnoughExperiences_ReturnsBatch()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 100, seed: 42);

        // Add 50 experiences
        for (int i = 0; i < 50; i++)
        {
            var state = new Vector<double>(new double[] { (double)i });
            var nextState = new Vector<double>(new double[] { (double)(i + 1) });
            var action = new Vector<double>(new double[] { 0.0 });
            var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
            buffer.Add(experience);
        }

        // Act
        var batch = buffer.Sample(batchSize: 10);

        // Assert
        Assert.Equal(10, batch.Count);
    }

    [Fact]
    public void Sample_WithInsufficientExperiences_ThrowsException()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 100);

        // Add only 5 experiences
        for (int i = 0; i < 5; i++)
        {
            var state = new Vector<double>(new double[] { (double)i });
            var nextState = new Vector<double>(new double[] { (double)(i + 1) });
            var action = new Vector<double>(new double[] { 0.0 });
            var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
            buffer.Add(experience);
        }

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => buffer.Sample(batchSize: 10));
    }

    [Fact]
    public void CanSample_ReturnsCorrectValue()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 100);

        // Add 5 experiences
        for (int i = 0; i < 5; i++)
        {
            var state = new Vector<double>(new double[] { (double)i });
            var nextState = new Vector<double>(new double[] { (double)(i + 1) });
            var action = new Vector<double>(new double[] { 0.0 });
            var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
            buffer.Add(experience);
        }

        // Assert
        Assert.True(buffer.CanSample(5));
        Assert.False(buffer.CanSample(6));
    }

    [Fact]
    public void Clear_RemovesAllExperiences()
    {
        // Arrange
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 100);

        // Add experiences
        for (int i = 0; i < 10; i++)
        {
            var state = new Vector<double>(new double[] { (double)i });
            var nextState = new Vector<double>(new double[] { (double)(i + 1) });
            var action = new Vector<double>(new double[] { 0.0 });
            var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
            buffer.Add(experience);
        }

        // Act
        buffer.Clear();

        // Assert
        Assert.Equal(0, buffer.Count);
    }
}
