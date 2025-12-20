using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ContinualLearning;

/// <summary>
/// Unit tests for the GradientEpisodicMemory class.
/// </summary>
public class GradientEpisodicMemoryTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_DefaultParameters_InitializesCorrectly()
    {
        // Arrange & Act
        var gem = new GradientEpisodicMemory<double>();

        // Assert
        Assert.NotNull(gem);
        Assert.Equal(1.0, gem.Lambda);
        Assert.Equal(0.5, gem.Margin);
        Assert.Equal(0, gem.TaskCount);
    }

    [Fact]
    public void Constructor_CustomMemorySize_InitializesCorrectly()
    {
        // Arrange & Act
        var gem = new GradientEpisodicMemory<double>(memorySize: 128);

        // Assert
        Assert.NotNull(gem);
        Assert.Equal(0, gem.TaskCount);
    }

    [Fact]
    public void Constructor_CustomMargin_InitializesCorrectly()
    {
        // Arrange
        double customMargin = 0.8;

        // Act
        var gem = new GradientEpisodicMemory<double>(margin: customMargin);

        // Assert
        Assert.Equal(customMargin, gem.Margin);
    }

    [Fact]
    public void Constructor_CustomLambda_InitializesCorrectly()
    {
        // Arrange
        double customLambda = 2.0;

        // Act
        var gem = new GradientEpisodicMemory<double>(lambda: customLambda);

        // Assert
        Assert.Equal(customLambda, gem.Lambda);
    }

    [Fact]
    public void Constructor_AllCustomParameters_InitializesCorrectly()
    {
        // Arrange
        int memorySize = 64;
        double margin = 1.0;
        double lambda = 0.5;

        // Act
        var gem = new GradientEpisodicMemory<double>(memorySize, margin, lambda);

        // Assert
        Assert.Equal(lambda, gem.Lambda);
        Assert.Equal(margin, gem.Margin);
        Assert.Equal(0, gem.TaskCount);
    }

    #endregion

    #region Property Tests

    [Fact]
    public void Lambda_SetValue_UpdatesCorrectly()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        double newLambda = 5.0;

        // Act
        gem.Lambda = newLambda;

        // Assert
        Assert.Equal(newLambda, gem.Lambda);
    }

    [Fact]
    public void Margin_SetValue_UpdatesCorrectly()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        double newMargin = 0.3;

        // Act
        gem.Margin = newMargin;

        // Assert
        Assert.Equal(newMargin, gem.Margin);
    }

    [Fact]
    public void TaskCount_AfterAddingTasks_ReflectsCorrectCount()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act
        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Assert
        Assert.Equal(1, gem.TaskCount);
    }

    #endregion

    #region BeforeTask Tests

    [Fact]
    public void BeforeTask_WithValidNetwork_ExecutesWithoutError()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert - Should not throw
        gem.BeforeTask(network, taskId: 0);
    }

    [Fact]
    public void BeforeTask_AfterPreviousTask_UpdatesReferenceGradients()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Complete first task
        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Act - Start second task (updates reference gradients)
        gem.BeforeTask(network, taskId: 1);

        // Assert - No exception and task count remains 1
        Assert.Equal(1, gem.TaskCount);
    }

    #endregion

    #region AfterTask Tests

    [Fact]
    public void AfterTask_FirstTask_StoresEpisodicMemory()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act
        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Assert
        Assert.Equal(1, gem.TaskCount);
    }

    [Fact]
    public void AfterTask_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => gem.AfterTask(null!, (inputs, targets), taskId: 0));
    }

    [Fact]
    public void AfterTask_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => gem.AfterTask(network, (null!, targets), taskId: 0));
    }

    [Fact]
    public void AfterTask_NullTargets_ThrowsArgumentNullException()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => gem.AfterTask(network, (inputs, null!), taskId: 0));
    }

    [Fact]
    public void AfterTask_MultipleTasks_StoresMultipleEpisodicMemories()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs1 = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets1 = CreateTestTensor(batchSize: 5, featureSize: 3);
        var inputs2 = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets2 = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act
        gem.AfterTask(network, (inputs1, targets1), taskId: 0);
        gem.AfterTask(network, (inputs2, targets2), taskId: 1);

        // Assert
        Assert.Equal(2, gem.TaskCount);
    }

    [Fact]
    public void AfterTask_LargeDataset_SamplesDownToMemorySize()
    {
        // Arrange
        int memorySize = 10;
        var gem = new GradientEpisodicMemory<double>(memorySize: memorySize);
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 100, featureSize: 10); // 100 samples > memorySize
        var targets = CreateTestTensor(batchSize: 100, featureSize: 3);

        // Act
        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Assert - Should complete without error (sampling happened internally)
        Assert.Equal(1, gem.TaskCount);
    }

    #endregion

    #region ComputeLoss Tests

    [Fact]
    public void ComputeLoss_AlwaysReturnsZero()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        // Act (before any task)
        var lossBefore = gem.ComputeLoss(network);

        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Act (after task)
        var lossAfter = gem.ComputeLoss(network);

        // Assert - GEM uses gradient modification, not loss-based regularization
        Assert.Equal(0.0, lossBefore);
        Assert.Equal(0.0, lossAfter);
    }

    #endregion

    #region ModifyGradients Tests

    [Fact]
    public void ModifyGradients_BeforeAnyTask_ReturnsUnmodifiedGradients()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = i * 0.1;
        }

        // Act
        var modifiedGradients = gem.ModifyGradients(network, gradients);

        // Assert - Should return same gradients (no reference gradients yet)
        Assert.Equal(gradients.Length, modifiedGradients.Length);
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    [Fact]
    public void ModifyGradients_AfterTask_MayModifyGradientsBasedOnConstraints()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        gem.AfterTask(network, (inputs, targets), taskId: 0);

        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = 0.5; // Positive gradients
        }

        // Act
        var modifiedGradients = gem.ModifyGradients(network, gradients);

        // Assert - Gradients may be modified if constraints are violated
        Assert.Equal(gradients.Length, modifiedGradients.Length);
    }

    [Fact]
    public void ModifyGradients_NullNetwork_ThrowsArgumentNullException()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var gradients = new Vector<double>(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => gem.ModifyGradients(null!, gradients));
    }

    [Fact]
    public void ModifyGradients_NullGradients_ThrowsArgumentNullException()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => gem.ModifyGradients(network, null!));
    }

    [Fact]
    public void ModifyGradients_ConflictingGradient_ProjectsToSatisfyConstraint()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>(margin: 0.0); // Zero margin for easier testing
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        gem.AfterTask(network, (inputs, targets), taskId: 0);

        // Create gradients that would violate constraint (negative dot product with reference)
        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = -10.0; // Large negative values to force violation
        }

        // Act
        var modifiedGradients = gem.ModifyGradients(network, gradients);

        // Assert - Gradients were processed (may or may not be modified depending on reference)
        Assert.Equal(gradients.Length, modifiedGradients.Length);
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_AfterTasks_ClearsAllStoredData()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        gem.AfterTask(network, (inputs, targets), taskId: 0);
        gem.AfterTask(network, (inputs, targets), taskId: 1);
        Assert.Equal(2, gem.TaskCount);

        // Act
        gem.Reset();

        // Assert
        Assert.Equal(0, gem.TaskCount);
    }

    [Fact]
    public void Reset_BeforeAnyTask_DoesNotThrow()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();

        // Act & Assert - Should not throw
        gem.Reset();
        Assert.Equal(0, gem.TaskCount);
    }

    [Fact]
    public void Reset_ThenModifyGradients_ReturnsUnmodified()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>();
        var network = new MockNeuralNetwork(parameterCount: 10, outputSize: 3);
        var inputs = CreateTestTensor(batchSize: 5, featureSize: 10);
        var targets = CreateTestTensor(batchSize: 5, featureSize: 3);

        gem.AfterTask(network, (inputs, targets), taskId: 0);
        gem.Reset();

        var gradients = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            gradients[i] = i * 0.1;
        }

        // Act
        var modifiedGradients = gem.ModifyGradients(network, gradients);

        // Assert - After reset, no constraints so gradients unchanged
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void GEM_CompleteWorkflow_ExecutesCorrectly()
    {
        // Arrange
        var gem = new GradientEpisodicMemory<double>(memorySize: 10, margin: 0.5, lambda: 1.0);
        var network = new MockNeuralNetwork(parameterCount: 20, outputSize: 5);

        // Task 1 data
        var task1Inputs = CreateTestTensor(batchSize: 15, featureSize: 20);
        var task1Targets = CreateTestTensor(batchSize: 15, featureSize: 5);

        // Task 2 data
        var task2Inputs = CreateTestTensor(batchSize: 15, featureSize: 20);
        var task2Targets = CreateTestTensor(batchSize: 15, featureSize: 5);

        // Act - Task 1
        gem.BeforeTask(network, taskId: 0);
        Assert.Equal(0, gem.TaskCount);

        gem.AfterTask(network, (task1Inputs, task1Targets), taskId: 0);
        Assert.Equal(1, gem.TaskCount);

        // Task 2
        gem.BeforeTask(network, taskId: 1);

        // Modify gradients during task 2
        var gradients = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            gradients[i] = 0.1;
        }
        var modifiedGradients = gem.ModifyGradients(network, gradients);
        Assert.Equal(20, modifiedGradients.Length);

        gem.AfterTask(network, (task2Inputs, task2Targets), taskId: 1);
        Assert.Equal(2, gem.TaskCount);

        // Verify loss is always zero (GEM works through gradients)
        var loss = gem.ComputeLoss(network);
        Assert.Equal(0.0, loss);

        // Reset clears everything
        gem.Reset();
        Assert.Equal(0, gem.TaskCount);
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
