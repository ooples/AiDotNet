using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the TeacherModelWrapper class.
/// </summary>
public class TeacherModelWrapperTests
{
    [Fact]
    public void Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);

        // Assert
        Assert.NotNull(wrapper);
        Assert.Equal(3, wrapper.OutputDimension);
    }

    [Fact]
    public void Constructor_WithNullForwardFunc_ThrowsArgumentNullException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new TeacherModelWrapper<double>(null!, outputDimension: 3));
    }

    [Fact]
    public void Constructor_WithInvalidOutputDimension_ThrowsArgumentException()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new TeacherModelWrapper<double>(forwardFunc, outputDimension: 0));
        Assert.Throws<ArgumentException>(() =>
            new TeacherModelWrapper<double>(forwardFunc, outputDimension: -1));
    }

    [Fact]
    public void GetLogits_WithValidInput_ReturnsCorrectOutput()
    {
        // Arrange
        var expectedLogits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        Func<Vector<double>, Vector<double>> forwardFunc = input => expectedLogits;
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var logits = wrapper.GetLogits(input);

        // Assert
        Assert.NotNull(logits);
        Assert.Equal(expectedLogits.Length, logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(expectedLogits[i], logits[i]);
        }
    }

    [Fact]
    public void GetLogits_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => wrapper.GetLogits(null!));
    }
}
