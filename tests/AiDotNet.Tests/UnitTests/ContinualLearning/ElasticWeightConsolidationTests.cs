using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the ElasticWeightConsolidation class.
/// </summary>
public class ElasticWeightConsolidationTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultLambda_InitializesCorrectly()
    {
        // Arrange & Act
        var ewc = new ElasticWeightConsolidation<double>();

        // Assert
        Assert.NotNull(ewc);
        Assert.Equal(400.0, ewc.Lambda);
    }

    [Fact]
    public void Constructor_CustomLambda_InitializesCorrectly()
    {
        // Arrange
        double customLambda = 1000.0;

        // Act
        var ewc = new ElasticWeightConsolidation<double>(lambda: customLambda);

        // Assert
        Assert.Equal(customLambda, ewc.Lambda);
    }

    [Fact]
    public void Constructor_ZeroLambda_AllowsZeroValue()
    {
        // Arrange & Act
        var ewc = new ElasticWeightConsolidation<double>(lambda: 0.0);

        // Assert
        Assert.Equal(0.0, ewc.Lambda);
    }

    #endregion

    #region Lambda Property Tests

    [Fact]
    public void Lambda_SetValue_UpdatesCorrectly()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        double newLambda = 500.0;

        // Act
        ewc.Lambda = newLambda;

        // Assert
        Assert.Equal(newLambda, ewc.Lambda);
    }

    #endregion

    #region BeforeTask Tests

    [Fact]
    public void BeforeTask_WithValidNetwork_ExecutesWithoutError()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert - Should not throw
        ewc.BeforeTask(network, taskId: 0);
    }

    [Fact]
    public void BeforeTask_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ewc.BeforeTask(null!, taskId: 0));
    }

    #endregion

    #region AfterTask Tests

    [Fact]
    public void AfterTask_FirstTask_StoresParametersAndComputesFisher()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Assert - ComputeLoss should now return non-zero
        var loss = ewc.ComputeLoss(network);
        // The loss should be computed (may be zero if parameters haven't changed)
        Assert.True(loss >= 0);
    }

    [Fact]
    public void AfterTask_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ewc.AfterTask(null!, (inputs, targets), taskId: 0));
    }

    [Fact]
    public void AfterTask_MultipleTasks_StoresMultipleSnapshots()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Modify parameters slightly to simulate training
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 0.1;
        }
        network.SetParameters(parameters);

        ewc.AfterTask(network, (inputs, targets), taskId: 1);

        // Assert - ComputeLoss should work with multiple task snapshots
        var loss = ewc.ComputeLoss(network);
        Assert.True(loss >= 0);
    }

    #endregion

    #region ComputeLoss Tests

    [Fact]
    public void ComputeLoss_BeforeAnyTask_ReturnsZero()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act
        var loss = ewc.ComputeLoss(network);

        // Assert
        Assert.Equal(0.0, loss);
    }

    [Fact]
    public void ComputeLoss_AfterTaskWithUnchangedParameters_ReturnsZero()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Act - Parameters unchanged
        var loss = ewc.ComputeLoss(network);

        // Assert - Should be zero or very small since parameters haven't changed
        Assert.True(loss < 1e-10);
    }

    [Fact]
    public void ComputeLoss_AfterTaskWithChangedParameters_ReturnsPositiveLoss()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Change parameters to simulate training on new task
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 1.0;
        }
        network.SetParameters(parameters);

        // Act
        var loss = ewc.ComputeLoss(network);

        // Assert - Should be positive since parameters have changed
        Assert.True(loss > 0);
    }

    [Fact]
    public void ComputeLoss_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ewc.ComputeLoss(null!));
    }

    [Fact]
    public void ComputeLoss_HigherLambda_ProducesHigherLoss()
    {
        // Arrange
        var ewcLowLambda = new ElasticWeightConsolidation<double>(lambda: 100.0);
        var ewcHighLambda = new ElasticWeightConsolidation<double>(lambda: 1000.0);
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        ewcLowLambda.AfterTask(network, (inputs, targets), taskId: 0);
        ewcHighLambda.AfterTask(network, (inputs, targets), taskId: 0);

        // Change parameters
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 1.0;
        }
        network.SetParameters(parameters);

        // Act
        var lowLoss = ewcLowLambda.ComputeLoss(network);
        var highLoss = ewcHighLambda.ComputeLoss(network);

        // Assert - Higher lambda should produce higher loss
        Assert.True(highLoss > lowLoss);
    }

    #endregion

    #region ModifyGradients Tests

    [Fact]
    public void ModifyGradients_BeforeAnyTask_ReturnsUnmodifiedGradients()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = i * 0.1;
        }

        // Act
        var modifiedGradients = ewc.ModifyGradients(network, gradients);

        // Assert - Should return same gradients (no modification before any task)
        Assert.Equal(gradients.Length, modifiedGradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    [Fact]
    public void ModifyGradients_AfterTask_AddsEWCGradient()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Change parameters to simulate training
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 1.0;
        }
        network.SetParameters(parameters);

        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = 0.1;
        }

        // Save original gradient values before modification
        var originalGradients = new double[10];
        for (int i = 0; i < 10; i++)
        {
            originalGradients[i] = gradients[i];
        }

        // Act
        var modifiedGradients = ewc.ModifyGradients(network, gradients);

        // Assert - Gradients should be modified (EWC gradient added)
        Assert.Equal(originalGradients.Length, modifiedGradients.Length);
        // At least some gradient should be different (due to EWC penalty gradient)
        bool anyDifferent = false;
        for (int i = 0; i < originalGradients.Length; i++)
        {
            if (Math.Abs(modifiedGradients[i] - originalGradients[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void ModifyGradients_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var gradients = new Vector<double>(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ewc.ModifyGradients(null!, gradients));
    }

    [Fact]
    public void ModifyGradients_NullGradients_ThrowsArgumentNullException()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ewc.ModifyGradients(network, null!));
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_AfterTasks_ClearsAllStoredData()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);
        ewc.AfterTask(network, (inputs, targets), taskId: 0);

        // Change parameters
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 1.0;
        }
        network.SetParameters(parameters);

        // Verify loss is non-zero before reset
        var lossBefore = ewc.ComputeLoss(network);
        Assert.True(lossBefore > 0);

        // Act
        ewc.Reset();

        // Assert - Loss should be zero after reset
        var lossAfter = ewc.ComputeLoss(network);
        Assert.Equal(0.0, lossAfter);
    }

    [Fact]
    public void Reset_BeforeAnyTask_DoesNotThrow()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>();

        // Act & Assert - Should not throw
        ewc.Reset();
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void EWC_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var ewc = new ElasticWeightConsolidation<double>(lambda: 500.0);
        var network = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);

        // Task 1 data
        var task1Inputs = CreateTestTensor(batchSize: 10, featureSize: 20);
        var task1Targets = CreateTestTensor(batchSize: 10, featureSize: 5);

        // Task 2 data
        var task2Inputs = CreateTestTensor(batchSize: 10, featureSize: 20);
        var task2Targets = CreateTestTensor(batchSize: 10, featureSize: 5);

        // Act - Task 1
        ewc.BeforeTask(network, taskId: 0);
        // Simulate training...
        ewc.AfterTask(network, (task1Inputs, task1Targets), taskId: 0);

        // Verify no loss immediately after task (parameters unchanged)
        var lossAfterTask1 = ewc.ComputeLoss(network);
        Assert.True(lossAfterTask1 < 1e-10);

        // Task 2 - Parameters will change
        ewc.BeforeTask(network, taskId: 1);

        // Simulate training on task 2 (parameters change)
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] += 0.5;
        }
        network.SetParameters(parameters);

        // Compute EWC loss during task 2
        var lossDuringTask2 = ewc.ComputeLoss(network);
        Assert.True(lossDuringTask2 > 0);

        // Modify gradients during training
        var gradients = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            gradients[i] = 0.05;
        }
        var modifiedGradients = ewc.ModifyGradients(network, gradients);

        // Gradients should be modified to include EWC penalty
        Assert.Equal(20, modifiedGradients.Length);

        ewc.AfterTask(network, (task2Inputs, task2Targets), taskId: 1);

        // Assert - Reset clears everything
        ewc.Reset();
        var lossAfterReset = ewc.ComputeLoss(network);
        Assert.Equal(0.0, lossAfterReset);
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
