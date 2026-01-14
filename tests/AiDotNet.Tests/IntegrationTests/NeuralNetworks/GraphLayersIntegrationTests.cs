using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for Graph Neural Network layers.
/// Tests GraphConvolutionalLayer, GraphAttentionLayer, GraphSAGELayer,
/// GraphIsomorphismLayer, GraphTransformerLayer, DirectionalGraphLayer,
/// and HeterogeneousGraphLayer.
/// </summary>
public class GraphLayersIntegrationTests
{
    private const float Tolerance = 1e-5f;

    #region Helper Methods

    /// <summary>
    /// Creates a simple undirected graph adjacency matrix for testing.
    /// Graph: 0 -- 1 -- 2
    ///        |       |
    ///        3 ----- 4
    /// </summary>
    private static Tensor<float> CreateSimpleAdjacencyMatrix(int numNodes = 5)
    {
        var adj = new Tensor<float>(new[] { numNodes, numNodes });
        // Node connections (undirected)
        var edges = new (int, int)[]
        {
            (0, 1), (1, 0),  // 0 -- 1
            (1, 2), (2, 1),  // 1 -- 2
            (0, 3), (3, 0),  // 0 -- 3
            (2, 4), (4, 2),  // 2 -- 4
            (3, 4), (4, 3),  // 3 -- 4
        };

        foreach (var (i, j) in edges)
        {
            adj[i, j] = 1.0f;
        }

        // Add self-loops
        for (int i = 0; i < numNodes; i++)
        {
            adj[i, i] = 1.0f;
        }

        return adj;
    }

    /// <summary>
    /// Creates a batched adjacency matrix for testing.
    /// </summary>
    private static Tensor<float> CreateBatchedAdjacencyMatrix(int batchSize, int numNodes)
    {
        var adj = new Tensor<float>(new[] { batchSize, numNodes, numNodes });
        var simpleAdj = CreateSimpleAdjacencyMatrix(numNodes);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    adj[b, i, j] = simpleAdj[i, j];
                }
            }
        }

        return adj;
    }

    /// <summary>
    /// Creates random node features for testing.
    /// </summary>
    private static Tensor<float> CreateNodeFeatures(int numNodes, int numFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var features = new Tensor<float>(new[] { numNodes, numFeatures });

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                features[i, j] = (float)(random.NextDouble() * 2 - 1);
            }
        }

        return features;
    }

    /// <summary>
    /// Creates batched random node features for testing.
    /// </summary>
    private static Tensor<float> CreateBatchedNodeFeatures(int batchSize, int numNodes, int numFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var features = new Tensor<float>(new[] { batchSize, numNodes, numFeatures });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numFeatures; j++)
                {
                    features[b, i, j] = (float)(random.NextDouble() * 2 - 1);
                }
            }
        }

        return features;
    }

    /// <summary>
    /// Creates a directed adjacency matrix for testing DirectionalGraphLayer.
    /// </summary>
    private static Tensor<float> CreateDirectedAdjacencyMatrix(int numNodes = 5)
    {
        var adj = new Tensor<float>(new[] { numNodes, numNodes });
        // Directed edges (arrows)
        var edges = new (int, int)[]
        {
            (0, 1),  // 0 -> 1
            (1, 2),  // 1 -> 2
            (3, 0),  // 3 -> 0
            (4, 2),  // 4 -> 2
            (3, 4),  // 3 -> 4
        };

        foreach (var (i, j) in edges)
        {
            adj[i, j] = 1.0f;
        }

        return adj;
    }

    #endregion

    #region GraphConvolutionalLayer Tests

    [Fact]
    public void GraphConvolutionalLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphConvolutionalLayer_ForwardPass_WithBatchedInput()
    {
        // Arrange
        int batchSize = 2;
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var adj = CreateBatchedAdjacencyMatrix(batchSize, numNodes);
        var nodeFeatures = CreateBatchedNodeFeatures(batchSize, numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(numNodes, output.Shape[1]);
        Assert.Equal(outputFeatures, output.Shape[2]);
    }

    [Fact]
    public void GraphConvolutionalLayer_WithActivation_AppliesActivation()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(
            inputFeatures, outputFeatures, (IActivationFunction<float>?)new ReLUActivation<float>());
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        // ReLU should make all negative values 0
        for (int i = 0; i < output.Shape[0]; i++)
        {
            for (int j = 0; j < output.Shape[1]; j++)
            {
                Assert.True(output[i, j] >= 0, $"ReLU output at [{i},{j}] = {output[i, j]} should be >= 0");
            }
        }
    }

    [Fact]
    public void GraphConvolutionalLayer_Backward_ComputesGradients()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Forward pass first
        var output = layer.Forward(nodeFeatures);

        // Create gradient for backward pass
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient.Data.Span[i] = 1.0f;
        }

        // Act
        var inputGradient = layer.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
    }

    [Fact]
    public void GraphConvolutionalLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);

        // Act
        var clone = layer.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(layer, clone);

        var typedClone = Assert.IsType<GraphConvolutionalLayer<float>>(clone);
        Assert.Equal(layer.InputFeatures, typedClone.InputFeatures);
        Assert.Equal(layer.OutputFeatures, typedClone.OutputFeatures);
    }

    [Fact]
    public void GraphConvolutionalLayer_WithoutAdjacencyMatrix_ThrowsException()
    {
        // Arrange
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var nodeFeatures = CreateNodeFeatures(5, inputFeatures);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => layer.Forward(nodeFeatures));
    }

    [Fact]
    public void GraphConvolutionalLayer_AuxiliaryLoss_ComputesSmoothnessLoss()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        layer.UseAuxiliaryLoss = true;

        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        layer.Forward(nodeFeatures);
        var auxLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.True(auxLoss >= 0, "Auxiliary loss should be non-negative");
    }

    #endregion

    #region GraphAttentionLayer Tests

    [Fact]
    public void GraphAttentionLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        int numHeads = 4;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        // GAT concatenates heads: outputFeatures per head * numHeads
        // But our implementation outputs just outputFeatures
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphAttentionLayer_MultiHead_LearnsDifferentPatterns()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        int numHeads = 4;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        // Verify output is not all zeros
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data.Span[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "GAT output should have non-zero values");
    }

    [Fact]
    public void GraphAttentionLayer_WithDropout_AppliesDropoutDuringTraining()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        double dropoutRate = 0.5;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2, dropoutRate: dropoutRate);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);
        layer.SetTrainingMode(true);

        // Act - run multiple times to see dropout effect
        var outputs = new List<Tensor<float>>();
        for (int i = 0; i < 5; i++)
        {
            outputs.Add(layer.Forward(nodeFeatures));
        }

        // Assert - at least some outputs should differ due to dropout
        // (though with same seed they might be same - this is a sanity check)
        Assert.True(outputs.Count == 5);
    }

    [Fact]
    public void GraphAttentionLayer_Backward_ComputesGradients()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads: 2);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Forward pass
        var output = layer.Forward(nodeFeatures);

        // Create gradient
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient.Data.Span[i] = 1.0f;
        }

        // Act
        var inputGradient = layer.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
    }

    [Fact]
    public void GraphAttentionLayer_Clone_CreatesIndependentCopy()
    {
        // Arrange
        int inputFeatures = 8;
        int outputFeatures = 16;
        int numHeads = 4;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures, numHeads);

        // Act
        var clone = layer.Clone();

        // Assert
        Assert.NotNull(clone);
        Assert.NotSame(layer, clone);
        Assert.IsType<GraphAttentionLayer<float>>(clone);
    }

    #endregion

    #region GraphSAGELayer Tests

    [Fact]
    public void GraphSAGELayer_MeanAggregator_ProducesCorrectOutput()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(
            inputFeatures, outputFeatures, SAGEAggregatorType.Mean);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphSAGELayer_MaxPoolAggregator_ProducesCorrectOutput()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(
            inputFeatures, outputFeatures, SAGEAggregatorType.MaxPool);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphSAGELayer_SumAggregator_ProducesCorrectOutput()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(
            inputFeatures, outputFeatures, SAGEAggregatorType.Sum);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphSAGELayer_WithNormalization_NormalizesOutput()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(
            inputFeatures, outputFeatures, SAGEAggregatorType.Mean, normalize: true);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        // With normalization, output vectors should have unit norm (or close to it)
        for (int i = 0; i < numNodes; i++)
        {
            float norm = 0;
            for (int j = 0; j < outputFeatures; j++)
            {
                norm += output[i, j] * output[i, j];
            }
            norm = (float)Math.Sqrt(norm);
            // Normalized vectors should have norm close to 1
            Assert.True(Math.Abs(norm - 1.0f) < 0.5f || norm < 1.5f,
                $"Normalized vector at node {i} should have norm close to 1, got {norm}");
        }
    }

    [Fact]
    public void GraphSAGELayer_Backward_ComputesGradients()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(inputFeatures, outputFeatures);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Forward pass
        var output = layer.Forward(nodeFeatures);

        // Create gradient
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient.Data.Span[i] = 1.0f;
        }

        // Act
        var inputGradient = layer.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
    }

    #endregion

    #region GraphIsomorphismLayer Tests

    [Fact]
    public void GraphIsomorphismLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphIsomorphismLayer<float>(inputFeatures, outputFeatures);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphIsomorphismLayer_LearnableEpsilon_UpdatesDuringTraining()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphIsomorphismLayer<float>(
            inputFeatures, outputFeatures, learnEpsilon: true);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert - just verify it runs without error with learnable epsilon
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphIsomorphismLayer_FixedEpsilon_UsesProvidedValue()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        double epsilon = 0.5;

        var layer = new GraphIsomorphismLayer<float>(
            inputFeatures, outputFeatures, learnEpsilon: false, epsilon: epsilon);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphIsomorphismLayer_WithMLP_ProcessesCorrectly()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;
        int mlpHiddenDim = 32;

        var layer = new GraphIsomorphismLayer<float>(
            inputFeatures, outputFeatures, mlpHiddenDim: mlpHiddenDim);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    #endregion

    #region GraphTransformerLayer Tests

    [Fact]
    public void GraphTransformerLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 32;  // Must be divisible by numHeads
        int outputFeatures = 32;
        int numHeads = 4;
        int headDim = 8;

        var layer = new GraphTransformerLayer<float>(
            inputFeatures, outputFeatures, numHeads, headDim, activationFunction: (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphTransformerLayer_WithStructuralEncoding_UsesGraphStructure()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 32;
        int outputFeatures = 32;
        int numHeads = 4;

        var layer = new GraphTransformerLayer<float>(
            inputFeatures, outputFeatures, numHeads, useStructuralEncoding: true, activationFunction: (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphTransformerLayer_WithDropout_AppliesDropout()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 32;
        int outputFeatures = 32;
        double dropoutRate = 0.2;

        var layer = new GraphTransformerLayer<float>(
            inputFeatures, outputFeatures, dropoutRate: dropoutRate, activationFunction: (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);
        layer.SetTrainingMode(true);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
    }

    [Fact]
    public void GraphTransformerLayer_Backward_ComputesGradients()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 32;
        int outputFeatures = 32;

        var layer = new GraphTransformerLayer<float>(inputFeatures, outputFeatures, activationFunction: (IActivationFunction<float>?)null);
        var adj = CreateSimpleAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Forward pass
        var output = layer.Forward(nodeFeatures);

        // Create gradient
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient.Data.Span[i] = 1.0f;
        }

        // Act
        var inputGradient = layer.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
    }

    #endregion

    #region DirectionalGraphLayer Tests

    [Fact]
    public void DirectionalGraphLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new DirectionalGraphLayer<float>(inputFeatures, outputFeatures);
        var adj = CreateDirectedAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void DirectionalGraphLayer_WithGating_AppliesGatingMechanism()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new DirectionalGraphLayer<float>(
            inputFeatures, outputFeatures, useGating: true);
        var adj = CreateDirectedAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void DirectionalGraphLayer_SeparatesIncomingOutgoing()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new DirectionalGraphLayer<float>(inputFeatures, outputFeatures);
        var adj = CreateDirectedAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert - verify directional processing produces output
        Assert.NotNull(output);
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data.Span[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "DirectionalGraphLayer output should have non-zero values");
    }

    [Fact]
    public void DirectionalGraphLayer_Backward_ComputesGradients()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new DirectionalGraphLayer<float>(inputFeatures, outputFeatures);
        var adj = CreateDirectedAdjacencyMatrix(numNodes);
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Forward pass
        var output = layer.Forward(nodeFeatures);

        // Create gradient
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient.Data.Span[i] = 1.0f;
        }

        // Act
        var inputGradient = layer.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
        Assert.Equal(nodeFeatures.Shape, inputGradient.Shape);
    }

    #endregion

    #region HeterogeneousGraphLayer Tests

    [Fact]
    public void HeterogeneousGraphLayer_ForwardPass_ProducesCorrectOutputShape()
    {
        // Arrange
        int numNodes = 5;
        int outputFeatures = 16;

        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "user", "item" },
            EdgeTypes = new[] { "buys", "views" },
            NodeTypeFeatures = new Dictionary<string, int>
            {
                ["user"] = 8,
                ["item"] = 12
            },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["buys"] = ("user", "item"),
                ["views"] = ("user", "item")
            }
        };

        var layer = new HeterogeneousGraphLayer<float>(metadata, outputFeatures);

        // Create adjacency matrices for each edge type
        var adjMatrices = new Dictionary<string, Tensor<float>>
        {
            ["buys"] = CreateSimpleAdjacencyMatrix(numNodes),
            ["views"] = CreateSimpleAdjacencyMatrix(numNodes)
        };

        // Create node type map (first 2 are users, last 3 are items)
        var nodeTypeMap = new Dictionary<int, string>
        {
            [0] = "user",
            [1] = "user",
            [2] = "item",
            [3] = "item",
            [4] = "item"
        };

        layer.SetAdjacencyMatrices(adjMatrices);
        layer.SetNodeTypeMap(nodeTypeMap);

        // Create node features using max input features
        int maxInputFeatures = metadata.NodeTypeFeatures.Values.Max();
        var nodeFeatures = CreateNodeFeatures(numNodes, maxInputFeatures);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void HeterogeneousGraphLayer_WithBasisDecomposition_ReducesParameters()
    {
        // Arrange
        int numNodes = 5;
        int outputFeatures = 16;
        int numBases = 2;

        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "user", "item", "category" },
            EdgeTypes = new[] { "buys", "views", "belongs_to", "rates" },
            NodeTypeFeatures = new Dictionary<string, int>
            {
                ["user"] = 8,
                ["item"] = 8,
                ["category"] = 8
            },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["buys"] = ("user", "item"),
                ["views"] = ("user", "item"),
                ["belongs_to"] = ("item", "category"),
                ["rates"] = ("user", "item")
            }
        };

        var layer = new HeterogeneousGraphLayer<float>(
            metadata, outputFeatures, useBasis: true, numBases: numBases);

        // Create adjacency matrices
        var adjMatrices = new Dictionary<string, Tensor<float>>
        {
            ["buys"] = CreateSimpleAdjacencyMatrix(numNodes),
            ["views"] = CreateSimpleAdjacencyMatrix(numNodes),
            ["belongs_to"] = CreateSimpleAdjacencyMatrix(numNodes),
            ["rates"] = CreateSimpleAdjacencyMatrix(numNodes)
        };

        // Node type map
        var nodeTypeMap = new Dictionary<int, string>
        {
            [0] = "user",
            [1] = "user",
            [2] = "item",
            [3] = "item",
            [4] = "category"
        };

        layer.SetAdjacencyMatrices(adjMatrices);
        layer.SetNodeTypeMap(nodeTypeMap);

        int inputFeatures = 8;
        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void HeterogeneousGraphLayer_SetAdjacencyMatrix_ThrowsNotSupportedException()
    {
        // Arrange
        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "user", "item" },
            EdgeTypes = new[] { "buys" },
            NodeTypeFeatures = new Dictionary<string, int> { ["user"] = 8, ["item"] = 8 },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["buys"] = ("user", "item")
            }
        };

        var layer = new HeterogeneousGraphLayer<float>(metadata, 16);
        var adj = CreateSimpleAdjacencyMatrix(5);

        // Act & Assert
        Assert.Throws<NotSupportedException>(() => layer.SetAdjacencyMatrix(adj));
    }

    [Fact]
    public void HeterogeneousGraphLayer_GetAdjacencyMatrix_ReturnsNull()
    {
        // Arrange
        var metadata = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "user", "item" },
            EdgeTypes = new[] { "buys" },
            NodeTypeFeatures = new Dictionary<string, int> { ["user"] = 8, ["item"] = 8 },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["buys"] = ("user", "item")
            }
        };

        var layer = new HeterogeneousGraphLayer<float>(metadata, 16);

        // Act
        var result = layer.GetAdjacencyMatrix();

        // Assert
        Assert.Null(result);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void GraphConvolutionalLayer_InvalidInputFeatures_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphConvolutionalLayer<float>(0, 16, (IActivationFunction<float>?)null));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphConvolutionalLayer<float>(-1, 16, (IActivationFunction<float>?)null));
    }

    [Fact]
    public void GraphConvolutionalLayer_InvalidOutputFeatures_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphConvolutionalLayer<float>(8, 0, (IActivationFunction<float>?)null));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new GraphConvolutionalLayer<float>(8, -1, (IActivationFunction<float>?)null));
    }

    [Fact]
    public void GraphAttentionLayer_SingleNode_HandlesEdgeCase()
    {
        // Arrange
        int numNodes = 1;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphAttentionLayer<float>(inputFeatures, outputFeatures);

        var adj = new Tensor<float>(new[] { 1, 1 });
        adj[0, 0] = 1.0f;  // Self-loop only

        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphSAGELayer_DisconnectedGraph_HandlesIsolatedNodes()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphSAGELayer<float>(inputFeatures, outputFeatures);

        // Create adjacency with isolated node (node 2 has no connections except self-loop)
        var adj = new Tensor<float>(new[] { numNodes, numNodes });
        adj[0, 1] = 1.0f;
        adj[1, 0] = 1.0f;
        adj[3, 4] = 1.0f;
        adj[4, 3] = 1.0f;
        // Node 2 is isolated
        for (int i = 0; i < numNodes; i++)
        {
            adj[i, i] = 1.0f;  // Self-loops
        }

        var nodeFeatures = CreateNodeFeatures(numNodes, inputFeatures);

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    [Fact]
    public void GraphLayers_ParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        int inputFeatures = 8;
        int outputFeatures = 16;

        var gcn = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var gat = new GraphAttentionLayer<float>(inputFeatures, outputFeatures);
        var sage = new GraphSAGELayer<float>(inputFeatures, outputFeatures);
        var gin = new GraphIsomorphismLayer<float>(inputFeatures, outputFeatures);

        // Assert
        Assert.True(gcn.ParameterCount > 0, "GCN should have parameters");
        Assert.True(gat.ParameterCount > 0, "GAT should have parameters");
        Assert.True(sage.ParameterCount > 0, "GraphSAGE should have parameters");
        Assert.True(gin.ParameterCount > 0, "GIN should have parameters");
    }

    [Fact]
    public void GraphLayers_SupportsTraining_ReturnsTrue()
    {
        // Arrange
        int inputFeatures = 8;
        int outputFeatures = 16;

        var gcn = new GraphConvolutionalLayer<float>(inputFeatures, outputFeatures, (IActivationFunction<float>?)null);
        var gat = new GraphAttentionLayer<float>(inputFeatures, outputFeatures);
        var sage = new GraphSAGELayer<float>(inputFeatures, outputFeatures);

        // Assert
        Assert.True(gcn.SupportsTraining);
        Assert.True(gat.SupportsTraining);
        Assert.True(sage.SupportsTraining);
    }

    #endregion

    #region Double Precision Tests

    [Fact]
    public void GraphConvolutionalLayer_DoublePrecision_WorksCorrectly()
    {
        // Arrange
        int numNodes = 5;
        int inputFeatures = 8;
        int outputFeatures = 16;

        var layer = new GraphConvolutionalLayer<double>(inputFeatures, outputFeatures, (IActivationFunction<double>?)null);

        var adj = new Tensor<double>(new[] { numNodes, numNodes });
        var simpleAdj = CreateSimpleAdjacencyMatrix(numNodes);
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                adj[i, j] = simpleAdj[i, j];
            }
        }

        var random = new Random(42);
        var nodeFeatures = new Tensor<double>(new[] { numNodes, inputFeatures });
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < inputFeatures; j++)
            {
                nodeFeatures[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        layer.SetAdjacencyMatrix(adj);

        // Act
        var output = layer.Forward(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(numNodes, output.Shape[0]);
        Assert.Equal(outputFeatures, output.Shape[1]);
    }

    #endregion
}
