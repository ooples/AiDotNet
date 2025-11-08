using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
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

    [Fact]
    public void GetSoftPredictions_WithTemperature1_ReturnsSoftmaxProbabilities()
    {
        // Arrange
        var logits = new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        Func<Vector<double>, Vector<double>> forwardFunc = input => logits;
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var softPredictions = wrapper.GetSoftPredictions(input, temperature: 1.0);

        // Assert
        Assert.NotNull(softPredictions);
        Assert.Equal(logits.Length, softPredictions.Length);

        // Probabilities should sum to approximately 1
        double sum = 0;
        for (int i = 0; i < softPredictions.Length; i++)
        {
            Assert.True(softPredictions[i] >= 0 && softPredictions[i] <= 1,
                $"Probability at index {i} should be between 0 and 1, got {softPredictions[i]}");
            sum += softPredictions[i];
        }
        Assert.True(Math.Abs(sum - 1.0) < 1e-6, $"Probabilities should sum to 1, got {sum}");
    }

    [Fact]
    public void GetSoftPredictions_WithHighTemperature_ProducesSofterDistribution()
    {
        // Arrange
        var logits = new Vector<double>(new[] { 10.0, 1.0, 0.1 }); // Very peaked logits
        Func<Vector<double>, Vector<double>> forwardFunc = input => logits;
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var lowTempProbs = wrapper.GetSoftPredictions(input, temperature: 1.0);
        var highTempProbs = wrapper.GetSoftPredictions(input, temperature: 5.0);

        // Assert
        // With high temperature, non-max classes should get higher probabilities
        Assert.True(highTempProbs[1] > lowTempProbs[1],
            "High temperature should increase probability of non-max class");
        Assert.True(highTempProbs[2] > lowTempProbs[2],
            "High temperature should increase probability of non-max class");

        // Max class probability should decrease with higher temperature
        Assert.True(lowTempProbs[0] > highTempProbs[0],
            "High temperature should decrease probability of max class");
    }

    [Fact]
    public void GetSoftPredictions_WithInvalidTemperature_ThrowsArgumentException()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => wrapper.GetSoftPredictions(input, temperature: 0));
        Assert.Throws<ArgumentException>(() => wrapper.GetSoftPredictions(input, temperature: -1));
    }

    [Fact]
    public void GetFeatures_WithoutExtractor_ReturnsNull()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var features = wrapper.GetFeatures(input, "layer1");

        // Assert
        Assert.Null(features);
    }

    [Fact]
    public void GetFeatures_WithExtractor_ReturnsFeatures()
    {
        // Arrange
        var expectedFeatures = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        Func<Vector<double>, string, object?> featureExtractor = (input, layerName) => expectedFeatures;

        var wrapper = new TeacherModelWrapper<double>(
            forwardFunc,
            outputDimension: 3,
            featureExtractor: featureExtractor);

        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var features = wrapper.GetFeatures(input, "layer1");

        // Assert
        Assert.NotNull(features);
        Assert.IsType<Vector<double>>(features);
        var featureVector = (Vector<double>)features;
        Assert.Equal(expectedFeatures.Length, featureVector.Length);
    }

    [Fact]
    public void GetAttentionWeights_WithoutExtractor_ReturnsNull()
    {
        // Arrange
        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var wrapper = new TeacherModelWrapper<double>(forwardFunc, outputDimension: 3);
        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var attention = wrapper.GetAttentionWeights(input, "attention1");

        // Assert
        Assert.Null(attention);
    }

    [Fact]
    public void GetAttentionWeights_WithExtractor_ReturnsWeights()
    {
        // Arrange
        var expectedWeights = new Matrix<double>(2, 2);
        expectedWeights[0, 0] = 0.8; expectedWeights[0, 1] = 0.2;
        expectedWeights[1, 0] = 0.3; expectedWeights[1, 1] = 0.7;

        Func<Vector<double>, Vector<double>> forwardFunc = input =>
            new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        Func<Vector<double>, string, object?> attentionExtractor = (input, layerName) => expectedWeights;

        var wrapper = new TeacherModelWrapper<double>(
            forwardFunc,
            outputDimension: 3,
            attentionExtractor: attentionExtractor);

        var input = new Vector<double>(new[] { 0.5, 0.3, 0.2 });

        // Act
        var attention = wrapper.GetAttentionWeights(input, "attention1");

        // Assert
        Assert.NotNull(attention);
        Assert.IsType<Matrix<double>>(attention);
    }
}
