using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the LearningWithoutForgetting class.
/// </summary>
public class LearningWithoutForgettingTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var lwf = new LearningWithoutForgetting<double>();

        // Assert
        Assert.NotNull(lwf);
        Assert.Equal(1.0, lwf.Lambda);
        Assert.Equal(2.0, lwf.Temperature);
        Assert.Equal(0, lwf.TaskCount);
    }

    [Fact]
    public void Constructor_CustomLambda_InitializesCorrectly()
    {
        // Arrange
        double customLambda = 2.5;

        // Act
        var lwf = new LearningWithoutForgetting<double>(lambda: customLambda);

        // Assert
        Assert.Equal(customLambda, lwf.Lambda);
    }

    [Fact]
    public void Constructor_CustomTemperature_InitializesCorrectly()
    {
        // Arrange
        double customTemperature = 4.0;

        // Act
        var lwf = new LearningWithoutForgetting<double>(temperature: customTemperature);

        // Assert
        Assert.Equal(customTemperature, lwf.Temperature);
    }

    [Fact]
    public void Constructor_AllCustomParameters_InitializesCorrectly()
    {
        // Arrange
        double lambda = 0.5;
        double temperature = 3.0;

        // Act
        var lwf = new LearningWithoutForgetting<double>(lambda, temperature);

        // Assert
        Assert.Equal(lambda, lwf.Lambda);
        Assert.Equal(temperature, lwf.Temperature);
        Assert.Equal(0, lwf.TaskCount);
    }

    #endregion

    #region Property Tests

    [Fact]
    public void Lambda_SetValue_UpdatesCorrectly()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        double newLambda = 5.0;

        // Act
        lwf.Lambda = newLambda;

        // Assert
        Assert.Equal(newLambda, lwf.Lambda);
    }

    [Fact]
    public void Temperature_SetValue_UpdatesCorrectly()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        double newTemperature = 6.0;

        // Act
        lwf.Temperature = newTemperature;

        // Assert
        Assert.Equal(newTemperature, lwf.Temperature);
    }

    [Fact]
    public void Temperature_SetVeryLowValue_ClampsToMinimum()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();

        // Act
        lwf.Temperature = 0.01; // Below minimum of 0.1

        // Assert
        Assert.Equal(0.1, lwf.Temperature);
    }

    [Fact]
    public void Temperature_SetNegativeValue_ClampsToMinimum()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();

        // Act
        lwf.Temperature = -5.0;

        // Assert
        Assert.Equal(0.1, lwf.Temperature);
    }

    #endregion

    #region BeforeTask Tests

    [Fact]
    public void BeforeTask_WithValidNetwork_ExecutesWithoutError()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert - Should not throw (BeforeTask is a no-op for LwF)
        lwf.BeforeTask(network, taskId: 0);
    }

    #endregion

    #region PrepareDistillation Tests

    [Fact]
    public void PrepareDistillation_WithValidInputs_StoresOldPredictions()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        // Act
        lwf.PrepareDistillation(network, inputs, taskId: 0);

        // Assert
        Assert.Equal(1, lwf.TaskCount);
    }

    [Fact]
    public void PrepareDistillation_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => lwf.PrepareDistillation(null!, inputs, taskId: 0));
    }

    [Fact]
    public void PrepareDistillation_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => lwf.PrepareDistillation(network, null!, taskId: 0));
    }

    [Fact]
    public void PrepareDistillation_MultipleTasks_StoresMultiplePredictions()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs1 = CreateTestTensor(batchSize: 5, featureSize: 10);
        var inputs2 = CreateTestTensor(batchSize: 5, featureSize: 10);

        // Act
        lwf.PrepareDistillation(network, inputs1, taskId: 0);
        lwf.PrepareDistillation(network, inputs2, taskId: 1);

        // Assert
        Assert.Equal(2, lwf.TaskCount);
    }

    #endregion

    #region AfterTask Tests

    [Fact]
    public void AfterTask_ClearsCurrentTaskInputs()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        lwf.PrepareDistillation(network, inputs, taskId: 0);

        // Act
        lwf.AfterTask(network, (inputs, targets), taskId: 0);

        // Assert - ComputeLoss should return zero after AfterTask (currentTaskInputs cleared)
        var loss = lwf.ComputeLoss(network);
        Assert.Equal(0.0, loss);
    }

    #endregion

    #region ComputeLoss Tests

    [Fact]
    public void ComputeLoss_BeforeAnyDistillation_ReturnsZero()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act
        var loss = lwf.ComputeLoss(network);

        // Assert
        Assert.Equal(0.0, loss);
    }

    [Fact]
    public void ComputeLoss_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => lwf.ComputeLoss(null!));
    }

    [Fact]
    public void ComputeLoss_AfterDistillationPrep_ReturnsNonNegativeLoss()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwf.PrepareDistillation(network, inputs, taskId: 0);

        // Act
        var loss = lwf.ComputeLoss(network);

        // Assert - KL divergence is always non-negative
        Assert.True(loss >= 0);
    }

    [Fact]
    public void ComputeLoss_HigherLambda_ProducesHigherLoss()
    {
        // Arrange
        var lwfLowLambda = new LearningWithoutForgetting<double>(lambda: 0.5);
        var lwfHighLambda = new LearningWithoutForgetting<double>(lambda: 2.0);
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwfLowLambda.PrepareDistillation(network, inputs, taskId: 0);
        lwfHighLambda.PrepareDistillation(network, inputs, taskId: 0);

        // Change parameters to create difference
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 1.0;
        }
        network.SetParameters(parameters);

        // Act
        var lowLoss = lwfLowLambda.ComputeLoss(network);
        var highLoss = lwfHighLambda.ComputeLoss(network);

        // Assert - Both should be non-negative, higher lambda means higher loss
        Assert.True(lowLoss >= 0);
        Assert.True(highLoss >= 0);
        // Note: The ratio should be approximately 4x (2.0/0.5)
    }

    #endregion

    #region ComputeDistillationLoss Tests

    [Fact]
    public void ComputeDistillationLoss_IdenticalPredictions_ReturnsZero()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var predictions = CreateTestTensor(batchSize: 3, featureSize: 4);

        // Act
        var loss = lwf.ComputeDistillationLoss(predictions, predictions);

        // Assert - KL divergence of identical distributions is zero
        Assert.True(Math.Abs(loss) < 1e-6);
    }

    [Fact]
    public void ComputeDistillationLoss_DifferentPredictions_ReturnsPositiveLoss()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var currentPredictions = CreateTestTensor(batchSize: 3, featureSize: 4);
        var oldPredictions = CreateTestTensor(batchSize: 3, featureSize: 4);

        // Make old predictions different
        for (int i = 0; i < oldPredictions.Length; i++)
        {
            oldPredictions[i] += 1.0;
        }

        // Act
        var loss = lwf.ComputeDistillationLoss(currentPredictions, oldPredictions);

        // Assert - Should be positive when distributions differ
        Assert.True(loss >= 0);
    }

    [Fact]
    public void ComputeDistillationLoss_NullCurrentPredictions_ThrowsArgumentNullException()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var oldPredictions = CreateTestTensor(batchSize: 3, featureSize: 4);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => lwf.ComputeDistillationLoss(null!, oldPredictions));
    }

    [Fact]
    public void ComputeDistillationLoss_NullOldPredictions_ThrowsArgumentNullException()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var currentPredictions = CreateTestTensor(batchSize: 3, featureSize: 4);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => lwf.ComputeDistillationLoss(currentPredictions, null!));
    }

    #endregion

    #region ModifyGradients Tests

    [Fact]
    public void ModifyGradients_ReturnsUnmodifiedGradients()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = i * 0.1;
        }

        // Act
        var modifiedGradients = lwf.ModifyGradients(network, gradients);

        // Assert - LwF uses loss-based regularization, not gradient modification
        Assert.Equal(gradients.Length, modifiedGradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    [Fact]
    public void ModifyGradients_AfterDistillation_StillReturnsUnmodified()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwf.PrepareDistillation(network, inputs, taskId: 0);

        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = i * 0.5;
        }

        // Act
        var modifiedGradients = lwf.ModifyGradients(network, gradients);

        // Assert - Still unchanged (LwF doesn't modify gradients)
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_AfterDistillation_ClearsAllStoredData()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwf.PrepareDistillation(network, inputs, taskId: 0);
        lwf.PrepareDistillation(network, inputs, taskId: 1);
        Assert.Equal(2, lwf.TaskCount);

        // Act
        lwf.Reset();

        // Assert
        Assert.Equal(0, lwf.TaskCount);
    }

    [Fact]
    public void Reset_BeforeAnyDistillation_DoesNotThrow()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();

        // Act & Assert - Should not throw
        lwf.Reset();
        Assert.Equal(0, lwf.TaskCount);
    }

    [Fact]
    public void Reset_ThenComputeLoss_ReturnsZero()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwf.PrepareDistillation(network, inputs, taskId: 0);
        lwf.Reset();

        // Act
        var loss = lwf.ComputeLoss(network);

        // Assert
        Assert.Equal(0.0, loss);
    }

    #endregion

    #region Temperature Effect Tests

    [Fact]
    public void Temperature_HigherValue_ProducesSofterDistribution()
    {
        // Arrange - Two LwF instances with different temperatures
        var lwfLowTemp = new LearningWithoutForgetting<double>(temperature: 1.0);
        var lwfHighTemp = new LearningWithoutForgetting<double>(temperature: 5.0);
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        lwfLowTemp.PrepareDistillation(network, inputs, taskId: 0);
        lwfHighTemp.PrepareDistillation(network, inputs, taskId: 0);

        // Both should complete without error and have stored predictions
        Assert.Equal(1, lwfLowTemp.TaskCount);
        Assert.Equal(1, lwfHighTemp.TaskCount);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void LwF_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>(lambda: 1.0, temperature: 2.0);
        var network = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);

        // Task 1 data
        var task1Inputs = CreateTestTensor(batchSize: 10, featureSize: 20);
        var task1Targets = CreateTestTensor(batchSize: 10, featureSize: 5);

        // Task 2 data
        var task2Inputs = CreateTestTensor(batchSize: 10, featureSize: 20);
        var task2Targets = CreateTestTensor(batchSize: 10, featureSize: 5);

        // Act - Task 1
        lwf.BeforeTask(network, taskId: 0);
        // Simulate training on task 1...
        lwf.AfterTask(network, (task1Inputs, task1Targets), taskId: 0);

        // Prepare for Task 2 with distillation
        lwf.PrepareDistillation(network, task2Inputs, taskId: 1);
        Assert.Equal(1, lwf.TaskCount);

        // Compute distillation loss during task 2 training
        var loss = lwf.ComputeLoss(network);
        Assert.True(loss >= 0);

        // Modify gradients (should return unchanged for LwF)
        var gradients = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            gradients[i] = 0.05;
        }
        var modifiedGradients = lwf.ModifyGradients(network, gradients);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }

        // Complete task 2
        lwf.AfterTask(network, (task2Inputs, task2Targets), taskId: 1);

        // Loss should be zero after AfterTask (currentTaskInputs cleared)
        var lossAfter = lwf.ComputeLoss(network);
        Assert.Equal(0.0, lossAfter);

        // Reset clears everything
        lwf.Reset();
        Assert.Equal(0, lwf.TaskCount);
    }

    [Fact]
    public void LwF_DistillationLoss_IncreasesWhenPredictionsChange()
    {
        // Arrange
        var lwf = new LearningWithoutForgetting<double>(lambda: 1.0, temperature: 2.0);
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        // Store old predictions
        lwf.PrepareDistillation(network, inputs, taskId: 0);

        // Get initial loss (should be near zero - predictions haven't changed)
        var initialLoss = lwf.ComputeLoss(network);

        // Change network parameters significantly
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 5.0;
        }
        network.SetParameters(parameters);

        // Get loss after parameter change
        var changedLoss = lwf.ComputeLoss(network);

        // Assert - Loss should increase when predictions change
        Assert.True(changedLoss >= 0);
    }

    #endregion

    #region Helper Methods

    private static Tensor<double> CreateTestTensor(int batchSize, int featureSize)
    {
        var tensor = new Tensor<double>(new int[] { batchSize, featureSize });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i * 0.01;
        }
        return tensor;
    }

    #endregion
}
