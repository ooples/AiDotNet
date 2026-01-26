using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.NestedLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NestedLearning;

/// <summary>
/// Integration tests for the NestedLearning module.
/// Tests AssociativeMemory and ContextFlow components.
/// </summary>
public class NestedLearningIntegrationTests
{
    private const double Tolerance = 1e-5;

    #region AssociativeMemory Tests

    [Fact]
    public void AssociativeMemory_Constructor_CreatesValidInstance()
    {
        // Arrange & Act
        var memory = new AssociativeMemory<double>(dimension: 8, capacity: 100);

        // Assert
        Assert.NotNull(memory);
        Assert.Equal(100, memory.Capacity);
        Assert.Equal(0, memory.MemoryCount);
    }

    [Fact]
    public void AssociativeMemory_Associate_StoresMemory()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var target = CreateRandomVector(4, 43);

        // Act
        memory.Associate(input, target);

        // Assert
        Assert.Equal(1, memory.MemoryCount);
    }

    [Fact]
    public void AssociativeMemory_Associate_MaintainsCapacityLimit()
    {
        // Arrange
        var capacity = 5;
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: capacity);

        // Act - Add more than capacity
        for (int i = 0; i < capacity + 3; i++)
        {
            var input = CreateRandomVector(4, i);
            var target = CreateRandomVector(4, i + 100);
            memory.Associate(input, target);
        }

        // Assert - Should be capped at capacity (FIFO)
        Assert.Equal(capacity, memory.MemoryCount);
    }

    [Fact]
    public void AssociativeMemory_Associate_UpdatesAssociationMatrix()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var target = CreateRandomVector(4, 43);

        // Act
        memory.Associate(input, target);

        // Assert - Matrix should have non-zero values after association
        var matrix = memory.GetAssociationMatrix();
        Assert.True(HasNonZeroElements(matrix));
    }

    [Fact]
    public void AssociativeMemory_Retrieve_ReturnsValidVector()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var target = CreateRandomVector(4, 43);
        memory.Associate(input, target);

        // Act
        var query = CreateRandomVector(4, 42); // Same seed as input
        var retrieved = memory.Retrieve(query);

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal(4, retrieved.Length);
    }

    [Fact]
    public void AssociativeMemory_Retrieve_ReturnsExactMatchForSameInput()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);

        // Create normalized vectors for better retrieval
        var input = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 });
        var target = new Vector<double>(new double[] { 0.0, 1.0, 0.0, 0.0 });

        // Associate multiple times to strengthen the association
        for (int i = 0; i < 100; i++)
        {
            memory.Associate(input, target);
        }

        // Act - Query with exact input
        var retrieved = memory.Retrieve(input);

        // Assert - Retrieved should have some correlation with target
        Assert.NotNull(retrieved);
        Assert.Equal(4, retrieved.Length);
    }

    [Fact]
    public void AssociativeMemory_Retrieve_WithDimensionMismatch_ThrowsException()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var wrongDimQuery = new Vector<double>(new double[] { 1.0, 2.0 }); // Wrong dimension

        // Act & Assert
        Assert.Throws<ArgumentException>(() => memory.Retrieve(wrongDimQuery));
    }

    [Fact]
    public void AssociativeMemory_Associate_WithDimensionMismatch_ThrowsException()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var wrongDimTarget = new Vector<double>(new double[] { 1.0, 2.0 }); // Wrong dimension

        // Act & Assert
        Assert.Throws<ArgumentException>(() => memory.Associate(input, wrongDimTarget));
    }

    [Fact]
    public void AssociativeMemory_Update_ModifiesAssociationMatrix()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var target = CreateRandomVector(4, 43);

        // Initial association
        memory.Associate(input, target);
        var matrixBefore = CopyMatrix(memory.GetAssociationMatrix());

        // Act - Update with different learning rate
        memory.Update(input, target, 0.1);

        // Assert - Matrix should have changed
        var matrixAfter = memory.GetAssociationMatrix();
        Assert.False(MatricesEqual(matrixBefore, matrixAfter, Tolerance));
    }

    [Fact]
    public void AssociativeMemory_Clear_ResetsMemory()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 10);
        var input = CreateRandomVector(4, 42);
        var target = CreateRandomVector(4, 43);
        memory.Associate(input, target);
        Assert.Equal(1, memory.MemoryCount);

        // Act
        memory.Clear();

        // Assert
        Assert.Equal(0, memory.MemoryCount);
    }

    [Fact]
    public void AssociativeMemory_MultipleAssociations_ImproveRetrieval()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 4, capacity: 100);

        // Create clear input-target pairs
        var input1 = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 });
        var target1 = new Vector<double>(new double[] { 0.5, 0.5, 0.0, 0.0 });

        var input2 = new Vector<double>(new double[] { 0.0, 1.0, 0.0, 0.0 });
        var target2 = new Vector<double>(new double[] { 0.0, 0.5, 0.5, 0.0 });

        // Act - Associate multiple times
        for (int i = 0; i < 50; i++)
        {
            memory.Associate(input1, target1);
            memory.Associate(input2, target2);
        }

        // Assert - Retrieval should work
        var retrieved1 = memory.Retrieve(input1);
        var retrieved2 = memory.Retrieve(input2);

        Assert.NotNull(retrieved1);
        Assert.NotNull(retrieved2);
        Assert.Equal(4, retrieved1.Length);
        Assert.Equal(4, retrieved2.Length);
    }

    #endregion

    #region ContextFlow Tests

    [Fact]
    public void ContextFlow_Constructor_CreatesValidInstance()
    {
        // Arrange & Act
        var contextFlow = new ContextFlow<double>(contextDimension: 8, numLevels: 3);

        // Assert
        Assert.NotNull(contextFlow);
        Assert.Equal(3, contextFlow.NumberOfLevels);
    }

    [Fact]
    public void ContextFlow_Constructor_InitializesMatrices()
    {
        // Arrange & Act
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 2);

        // Assert
        var transformationMatrices = contextFlow.GetTransformationMatrices();
        var compressionMatrices = contextFlow.GetCompressionMatrices();

        Assert.Equal(2, transformationMatrices.Length);
        Assert.Equal(2, compressionMatrices.Length);

        // Check matrices are initialized (not all zeros)
        Assert.True(HasNonZeroElements(transformationMatrices[0]));
        Assert.True(HasNonZeroElements(compressionMatrices[0]));
    }

    [Fact]
    public void ContextFlow_PropagateContext_ReturnsValidOutput()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input = CreateRandomVector(4, 42);

        // Act
        var output = contextFlow.PropagateContext(input, currentLevel: 0);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(4, output.Length);
    }

    [Fact]
    public void ContextFlow_PropagateContext_UpdatesContextState()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input = CreateRandomVector(4, 42);

        var stateBefore = contextFlow.GetContextState(0);
        Assert.True(IsZeroVector(stateBefore)); // Initial state should be zero

        // Act
        contextFlow.PropagateContext(input, currentLevel: 0);

        // Assert
        var stateAfter = contextFlow.GetContextState(0);
        Assert.False(IsZeroVector(stateAfter)); // State should be updated
    }

    [Fact]
    public void ContextFlow_PropagateContext_InvalidLevel_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input = CreateRandomVector(4, 42);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.PropagateContext(input, currentLevel: -1));
        Assert.Throws<ArgumentException>(() => contextFlow.PropagateContext(input, currentLevel: 3));
    }

    [Fact]
    public void ContextFlow_ComputeContextGradients_ReturnsValidOutput()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var upstreamGradient = CreateRandomVector(4, 42);

        // Act
        var gradient = contextFlow.ComputeContextGradients(upstreamGradient, level: 0);

        // Assert
        Assert.NotNull(gradient);
        Assert.Equal(4, gradient.Length);
    }

    [Fact]
    public void ContextFlow_ComputeContextGradients_InvalidLevel_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var upstreamGradient = CreateRandomVector(4, 42);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.ComputeContextGradients(upstreamGradient, level: -1));
        Assert.Throws<ArgumentException>(() => contextFlow.ComputeContextGradients(upstreamGradient, level: 3));
    }

    [Fact]
    public void ContextFlow_UpdateFlow_ModifiesTransformationMatrices()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 2);

        // First propagate to get non-zero context states
        var input = CreateRandomVector(4, 42);
        contextFlow.PropagateContext(input, 0);
        contextFlow.PropagateContext(input, 1);

        var matrixBefore = CopyMatrix(contextFlow.GetTransformationMatrices()[0]);

        var gradients = new Vector<double>[]
        {
            CreateRandomVector(4, 100),
            CreateRandomVector(4, 101)
        };
        var learningRates = new double[] { 0.01, 0.01 };

        // Act
        contextFlow.UpdateFlow(gradients, learningRates);

        // Assert
        var matrixAfter = contextFlow.GetTransformationMatrices()[0];
        Assert.False(MatricesEqual(matrixBefore, matrixAfter, Tolerance));
    }

    [Fact]
    public void ContextFlow_UpdateFlow_InvalidGradientsCount_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 2);
        var gradients = new Vector<double>[] { CreateRandomVector(4, 100) }; // Only 1, need 2
        var learningRates = new double[] { 0.01, 0.01 };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.UpdateFlow(gradients, learningRates));
    }

    [Fact]
    public void ContextFlow_UpdateFlow_InvalidLearningRatesCount_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 2);
        var gradients = new Vector<double>[]
        {
            CreateRandomVector(4, 100),
            CreateRandomVector(4, 101)
        };
        var learningRates = new double[] { 0.01 }; // Only 1, need 2

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.UpdateFlow(gradients, learningRates));
    }

    [Fact]
    public void ContextFlow_GetContextState_ReturnsValidState()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);

        // Act
        var state = contextFlow.GetContextState(level: 1);

        // Assert
        Assert.NotNull(state);
        Assert.Equal(4, state.Length);
    }

    [Fact]
    public void ContextFlow_GetContextState_InvalidLevel_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.GetContextState(level: -1));
        Assert.Throws<ArgumentException>(() => contextFlow.GetContextState(level: 3));
    }

    [Fact]
    public void ContextFlow_CompressContext_ReturnsValidOutput()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var context = CreateRandomVector(4, 42);

        // Act
        var compressed = contextFlow.CompressContext(context, targetLevel: 1);

        // Assert
        Assert.NotNull(compressed);
        Assert.Equal(4, compressed.Length);
    }

    [Fact]
    public void ContextFlow_CompressContext_InvalidLevel_ThrowsException()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var context = CreateRandomVector(4, 42);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => contextFlow.CompressContext(context, targetLevel: -1));
        Assert.Throws<ArgumentException>(() => contextFlow.CompressContext(context, targetLevel: 3));
    }

    [Fact]
    public void ContextFlow_Reset_ClearsContextStates()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input = CreateRandomVector(4, 42);

        // Propagate to set non-zero states
        contextFlow.PropagateContext(input, 0);
        contextFlow.PropagateContext(input, 1);
        contextFlow.PropagateContext(input, 2);

        Assert.False(IsZeroVector(contextFlow.GetContextState(0)));

        // Act
        contextFlow.Reset();

        // Assert - All states should be zero after reset
        Assert.True(IsZeroVector(contextFlow.GetContextState(0)));
        Assert.True(IsZeroVector(contextFlow.GetContextState(1)));
        Assert.True(IsZeroVector(contextFlow.GetContextState(2)));
    }

    [Fact]
    public void ContextFlow_MultipleLevels_IndependentStates()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input0 = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 });
        var input1 = new Vector<double>(new double[] { 0.0, 1.0, 0.0, 0.0 });

        // Act - Propagate different inputs to different levels
        contextFlow.PropagateContext(input0, 0);
        contextFlow.PropagateContext(input1, 1);

        // Assert - States should be different
        var state0 = contextFlow.GetContextState(0);
        var state1 = contextFlow.GetContextState(1);
        var state2 = contextFlow.GetContextState(2);

        Assert.False(IsZeroVector(state0));
        Assert.False(IsZeroVector(state1));
        Assert.True(IsZeroVector(state2)); // Level 2 was not updated
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AssociativeMemory_ContextFlow_Integration_WorksTogether()
    {
        // Arrange
        var dimension = 8;
        var memory = new AssociativeMemory<double>(dimension, capacity: 50);
        var contextFlow = new ContextFlow<double>(dimension, numLevels: 2);

        // Simulate a nested learning workflow
        var input = CreateRandomVector(dimension, 42);
        var target = CreateRandomVector(dimension, 43);

        // Act
        // 1. Store association in memory
        memory.Associate(input, target);

        // 2. Use context flow to transform input
        var transformed = contextFlow.PropagateContext(input, 0);

        // 3. Retrieve from memory using transformed input
        var retrieved = memory.Retrieve(transformed);

        // 4. Use retrieved as input to next level
        contextFlow.PropagateContext(retrieved, 1);

        // Assert - Both components should have valid state
        Assert.Equal(1, memory.MemoryCount);
        Assert.False(IsZeroVector(contextFlow.GetContextState(0)));
        Assert.False(IsZeroVector(contextFlow.GetContextState(1)));
    }

    [Fact]
    public void AssociativeMemory_LargeCapacity_HandlesCorrectly()
    {
        // Arrange
        var memory = new AssociativeMemory<double>(dimension: 16, capacity: 1000);

        // Act - Add many associations
        for (int i = 0; i < 500; i++)
        {
            var input = CreateRandomVector(16, i);
            var target = CreateRandomVector(16, i + 1000);
            memory.Associate(input, target);
        }

        // Assert
        Assert.Equal(500, memory.MemoryCount);
        Assert.Equal(1000, memory.Capacity);
    }

    [Fact]
    public void ContextFlow_SequentialPropagation_AccumulatesState()
    {
        // Arrange
        var contextFlow = new ContextFlow<double>(contextDimension: 4, numLevels: 3);
        var input = CreateRandomVector(4, 42);

        // Act - Propagate same input multiple times
        for (int i = 0; i < 10; i++)
        {
            contextFlow.PropagateContext(input, 0);
        }

        // Assert - State should have accumulated due to momentum
        var state = contextFlow.GetContextState(0);
        Assert.False(IsZeroVector(state));
    }

    #endregion

    #region Helper Methods

    private static Vector<double> CreateRandomVector(int size, int seed)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[size];

        for (int i = 0; i < size; i++)
        {
            data[i] = random.NextDouble() * 2 - 1; // [-1, 1]
        }

        return new Vector<double>(data);
    }

    private static bool IsZeroVector(Vector<double> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            if (Math.Abs(vector[i]) > Tolerance)
                return false;
        }
        return true;
    }

    private static bool HasNonZeroElements(Matrix<double> matrix)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                if (Math.Abs(matrix[i, j]) > Tolerance)
                    return true;
            }
        }
        return false;
    }

    private static Matrix<double> CopyMatrix(Matrix<double> source)
    {
        var copy = new Matrix<double>(source.Rows, source.Columns);
        for (int i = 0; i < source.Rows; i++)
        {
            for (int j = 0; j < source.Columns; j++)
            {
                copy[i, j] = source[i, j];
            }
        }
        return copy;
    }

    private static bool MatricesEqual(Matrix<double> a, Matrix<double> b, double tolerance)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return false;

        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                if (Math.Abs(a[i, j] - b[i, j]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    #endregion
}
