using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.KnowledgeDistillation;

/// <summary>
/// Unit tests for the KnowledgeDistillationTrainer class.
/// </summary>
public class KnowledgeDistillationTrainerTests
{
    private class MockTeacher : ITeacherModel<Vector<double>, Vector<double>>
    {
        public int OutputDimension => 3;

        public Vector<double> GetLogits(Vector<double> input)
        {
            // Simple mock: return fixed logits
            return new Vector<double>(new[] { 2.0, 1.0, 0.5 });
        }

        public Vector<double> GetSoftPredictions(Vector<double> input, double temperature = 1.0)
        {
            // Not used in basic tests
            return GetLogits(input);
        }

        public object? GetFeatures(Vector<double> input, string layerName)
        {
            return null;
        }

        public object? GetAttentionWeights(Vector<double> input, string layerName)
        {
            return null;
        }
    }

    [Fact]
    public void Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();

        // Act
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        // Assert
        Assert.NotNull(trainer);
    }

    [Fact]
    public void Constructor_WithNullTeacher_ThrowsArgumentNullException()
    {
        // Arrange
        var distillationLoss = new DistillationLoss<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new KnowledgeDistillationTrainer<double>(null!, distillationLoss));
    }

    [Fact]
    public void Constructor_WithNullDistillationStrategy_ThrowsArgumentNullException()
    {
        // Arrange
        var teacher = new MockTeacher();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new KnowledgeDistillationTrainer<double>(teacher, null!));
    }

    [Fact]
    public void TrainBatch_WithValidInputs_ReturnsPositiveLoss()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var inputs = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 0.5, 0.3, 0.2 }),
            new Vector<double>(new[] { 0.4, 0.4, 0.2 })
        });

        var labels = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 })
        });

        // Simple student model: returns fixed logits
        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 1.0, 1.5, 0.8 });

        // Track backward calls
        int backwardCalls = 0;
        Action<Vector<double>> studentBackward = gradient => backwardCalls++;

        // Act
        var loss = trainer.TrainBatch(studentForward, studentBackward, inputs, labels);

        // Assert
        Assert.True(loss > 0, "Loss should be positive");
        Assert.Equal(inputs.Length, backwardCalls); // Should call backward for each sample
    }

    [Fact]
    public void TrainBatch_WithNullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 1.0, 1.5, 0.8 });
        Action<Vector<double>> studentBackward = gradient => { };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            trainer.TrainBatch(studentForward, studentBackward, null!, null));
    }

    [Fact]
    public void TrainBatch_WithEmptyInputs_ThrowsArgumentException()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 1.0, 1.5, 0.8 });
        Action<Vector<double>> studentBackward = gradient => { };

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            trainer.TrainBatch(studentForward, studentBackward, new Vector<Vector<double>>(0), null));
    }

    [Fact]
    public void Train_WithValidParameters_CompletesSuccessfully()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>(temperature: 3.0, alpha: 0.3);
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var trainInputs = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 0.5, 0.3, 0.2 }),
            new Vector<double>(new[] { 0.4, 0.4, 0.2 }),
            new Vector<double>(new[] { 0.6, 0.2, 0.2 }),
            new Vector<double>(new[] { 0.3, 0.5, 0.2 })
        });

        var trainLabels = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 }),
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 1.0, 0.0 })
        });

        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 1.0, 1.5, 0.8 });

        Action<Vector<double>> studentBackward = gradient => { };

        int epochsCompleted = 0;
        Action<int, double> onEpochComplete = (epoch, loss) => epochsCompleted++;

        // Act
        trainer.Train(
            studentForward,
            studentBackward,
            trainInputs,
            trainLabels,
            epochs: 2,
            batchSize: 2,
            onEpochComplete: onEpochComplete);

        // Assert
        Assert.Equal(2, epochsCompleted);
    }

    [Fact]
    public void Train_WithInvalidEpochs_ThrowsArgumentException()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var trainInputs = new Vector<Vector<double>>(new[] { new Vector<double>(new[] { 0.5, 0.3, 0.2 }) });
        var trainLabels = new Vector<Vector<double>>(new[] { new Vector<double>(new[] { 1.0, 0.0, 0.0 }) });

        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 1.0, 1.5, 0.8 });
        Action<Vector<double>> studentBackward = gradient => { };

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            trainer.Train(studentForward, studentBackward, trainInputs, trainLabels, epochs: 0));
        Assert.Throws<ArgumentException>(() =>
            trainer.Train(studentForward, studentBackward, trainInputs, trainLabels, epochs: -1));
    }

    [Fact]
    public void Evaluate_ReturnsAccuracyBetweenZeroAndOne()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var testInputs = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 0.5, 0.3, 0.2 }),
            new Vector<double>(new[] { 0.4, 0.4, 0.2 })
        });

        var testLabels = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }), // Class 0
            new Vector<double>(new[] { 0.0, 1.0, 0.0 })  // Class 1
        });

        // Student predicts class 1 for all inputs
        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 0.5, 2.0, 0.3 });

        // Act
        var accuracy = trainer.Evaluate(studentForward, testInputs, testLabels);

        // Assert
        Assert.True(accuracy >= 0 && accuracy <= 1, "Accuracy should be between 0 and 1");
        Assert.Equal(0.5, accuracy); // Should get 1 out of 2 correct (predicts class 1, second is class 1)
    }

    [Fact]
    public void Evaluate_WithPerfectPredictions_ReturnsOne()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var testInputs = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 0.5, 0.3, 0.2 }),
            new Vector<double>(new[] { 0.4, 0.4, 0.2 })
        };

        var testLabels = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 1.0, 0.0, 0.0 })
        };

        // Student always predicts class 0 correctly
        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 3.0, 1.0, 0.5 }); // argmax = 0

        // Act
        var accuracy = trainer.Evaluate(studentForward, testInputs, testLabels);

        // Assert
        Assert.Equal(1.0, accuracy);
    }

    [Fact]
    public void Evaluate_WithNoCorrectPredictions_ReturnsZero()
    {
        // Arrange
        var teacher = new MockTeacher();
        var distillationLoss = new DistillationLoss<double>();
        var trainer = new KnowledgeDistillationTrainer<double>(teacher, distillationLoss);

        var testInputs = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 0.5, 0.3, 0.2 }),
            new Vector<double>(new[] { 0.4, 0.4, 0.2 })
        };

        var testLabels = new Vector<Vector<double>>(new[]
        {
            new Vector<double>(new[] { 1.0, 0.0, 0.0 }), // True class: 0
            new Vector<double>(new[] { 1.0, 0.0, 0.0 })  // True class: 0
        };

        // Student always predicts class 2 (wrong)
        Func<Vector<double>, Vector<double>> studentForward = input =>
            new Vector<double>(new[] { 0.5, 1.0, 3.0 }); // argmax = 2

        // Act
        var accuracy = trainer.Evaluate(studentForward, testInputs, testLabels);

        // Assert
        Assert.Equal(0.0, accuracy);
    }
}
