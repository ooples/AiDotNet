using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.Layers
{
    public class GraphLayerTests
    {
        #region GraphConvolutionalLayer Tests

        [Fact]
        public void GraphConvolutionalLayer_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new GraphConvolutionalLayer<double>(inputFeatures: 10, outputFeatures: 16, (IActivationFunction<double>?)null);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.Equal(10, layer.InputFeatures);
            Assert.Equal(16, layer.OutputFeatures);
        }

        [Fact]
        public void GraphConvolutionalLayer_Forward_WithoutAdjacencyMatrix_ThrowsException()
        {
            // Arrange
            var layer = new GraphConvolutionalLayer<double>(inputFeatures: 10, outputFeatures: 16, (IActivationFunction<double>?)null);
            var input = new Tensor<double>([1, 5, 10]); // batch=1, nodes=5, features=10

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        }

        [Fact]
        public void GraphConvolutionalLayer_Forward_WithAdjacencyMatrix_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new GraphConvolutionalLayer<double>(inputFeatures: 10, outputFeatures: 16, (IActivationFunction<double>?)null);
            int batchSize = 2;
            int numNodes = 5;

            var input = new Tensor<double>([batchSize, numNodes, 10]);
            var adjacency = new Tensor<double>([batchSize, numNodes, numNodes]);

            // Initialize input
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.1;
            }

            // Simple adjacency: each node connected to itself and next node
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    adjacency[b, i, i] = 1.0; // Self-connection
                    if (i < numNodes - 1)
                    {
                        adjacency[b, i, i + 1] = 1.0; // Connect to next
                    }
                }
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(3, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(numNodes, output.Shape[1]);
            Assert.Equal(16, output.Shape[2]);
        }

        [Fact]
        public void GraphConvolutionalLayer_GetAdjacencyMatrix_ReturnsSetMatrix()
        {
            // Arrange
            var layer = new GraphConvolutionalLayer<double>(inputFeatures: 10, outputFeatures: 16, (IActivationFunction<double>?)null);
            var adjacency = new Tensor<double>([1, 5, 5]);

            // Act
            layer.SetAdjacencyMatrix(adjacency);
            var retrieved = layer.GetAdjacencyMatrix();

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(adjacency.Shape[0], retrieved.Shape[0]);
            Assert.Equal(adjacency.Shape[1], retrieved.Shape[1]);
            Assert.Equal(adjacency.Shape[2], retrieved.Shape[2]);
        }

        #endregion

        #region GraphAttentionLayer Tests

        [Fact]
        public void GraphAttentionLayer_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new GraphAttentionLayer<double>(
                inputFeatures: 10,
                outputFeatures: 16,
                numHeads: 4);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.Equal(10, layer.InputFeatures);
            Assert.Equal(16, layer.OutputFeatures);
        }

        [Fact]
        public void GraphAttentionLayer_Forward_WithoutAdjacencyMatrix_ThrowsException()
        {
            // Arrange
            var layer = new GraphAttentionLayer<double>(inputFeatures: 10, outputFeatures: 16);
            var input = new Tensor<double>([1, 5, 10]);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        }

        [Fact]
        public void GraphAttentionLayer_Forward_WithAdjacencyMatrix_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new GraphAttentionLayer<double>(
                inputFeatures: 8,
                outputFeatures: 16,
                numHeads: 2);

            int batchSize = 1;
            int numNodes = 4;

            var input = new Tensor<double>([batchSize, numNodes, 8]);
            var adjacency = new Tensor<double>([batchSize, numNodes, numNodes]);

            // Initialize input with small values
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.01 * (i % 10);
            }

            // Create simple graph: nodes connected in a chain
            for (int i = 0; i < numNodes; i++)
            {
                adjacency[0, i, i] = 1.0; // Self-connection
                if (i > 0)
                {
                    adjacency[0, i, i - 1] = 1.0;
                }
                if (i < numNodes - 1)
                {
                    adjacency[0, i, i + 1] = 1.0;
                }
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(3, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(numNodes, output.Shape[1]);
            Assert.Equal(16, output.Shape[2]);
        }

        [Fact]
        public void GraphAttentionLayer_MultipleHeads_WorksCorrectly()
        {
            // Arrange
            var layer = new GraphAttentionLayer<double>(
                inputFeatures: 4,
                outputFeatures: 8,
                numHeads: 4);

            var input = new Tensor<double>([1, 3, 4]);
            var adjacency = new Tensor<double>([1, 3, 3]);

            // Fully connected graph
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    adjacency[0, i, j] = 1.0;
                }
            }

            // Initialize input
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.5;
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert - should complete without error
            Assert.NotNull(output);
            Assert.Equal(8, output.Shape[2]);
        }

        #endregion

        #region GraphSAGELayer Tests

        [Fact]
        public void GraphSAGELayer_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new GraphSAGELayer<double>(
                inputFeatures: 10,
                outputFeatures: 16,
                aggregatorType: SAGEAggregatorType.Mean);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.Equal(10, layer.InputFeatures);
            Assert.Equal(16, layer.OutputFeatures);
        }

        [Fact]
        public void GraphSAGELayer_MeanAggregator_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new GraphSAGELayer<double>(
                inputFeatures: 8,
                outputFeatures: 12,
                aggregatorType: SAGEAggregatorType.Mean,
                normalize: true);

            int batchSize = 2;
            int numNodes = 6;

            var input = new Tensor<double>([batchSize, numNodes, 8]);
            var adjacency = new Tensor<double>([batchSize, numNodes, numNodes]);

            // Initialize input
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.1 * (i % 5);
            }

            // Create graph structure
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < numNodes; i++)
                {
                    adjacency[b, i, i] = 1.0;
                    if (i > 0)
                    {
                        adjacency[b, i, i - 1] = 1.0;
                    }
                }
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(3, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(numNodes, output.Shape[1]);
            Assert.Equal(12, output.Shape[2]);
        }

        [Fact]
        public void GraphSAGELayer_MaxPoolAggregator_WorksCorrectly()
        {
            // Arrange
            var layer = new GraphSAGELayer<double>(
                inputFeatures: 4,
                outputFeatures: 8,
                aggregatorType: SAGEAggregatorType.MaxPool,
                normalize: false);

            var input = new Tensor<double>([1, 4, 4]);
            var adjacency = new Tensor<double>([1, 4, 4]);

            // Initialize with varying values
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    input[0, i, j] = i + j * 0.1;
                }
            }

            // Create star graph (node 0 connected to all)
            for (int i = 0; i < 4; i++)
            {
                adjacency[0, 0, i] = 1.0;
                adjacency[0, i, 0] = 1.0;
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(8, output.Shape[2]);
        }

        [Fact]
        public void GraphSAGELayer_SumAggregator_WorksCorrectly()
        {
            // Arrange
            var layer = new GraphSAGELayer<double>(
                inputFeatures: 5,
                outputFeatures: 10,
                aggregatorType: SAGEAggregatorType.Sum);

            var input = new Tensor<double>([1, 3, 5]);
            var adjacency = new Tensor<double>([1, 3, 3]);

            // Simple initialization
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 1.0;
            }

            // Fully connected
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    adjacency[0, i, j] = 1.0;
                }
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(10, output.Shape[2]);
        }

        #endregion

        #region GraphIsomorphismLayer Tests

        [Fact]
        public void GraphIsomorphismLayer_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var layer = new GraphIsomorphismLayer<double>(
                inputFeatures: 10,
                outputFeatures: 16,
                mlpHiddenDim: 20);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.SupportsTraining);
            Assert.Equal(10, layer.InputFeatures);
            Assert.Equal(16, layer.OutputFeatures);
        }

        [Fact]
        public void GraphIsomorphismLayer_Forward_ReturnsCorrectShape()
        {
            // Arrange
            var layer = new GraphIsomorphismLayer<double>(
                inputFeatures: 6,
                outputFeatures: 12,
                mlpHiddenDim: 10,
                learnEpsilon: true);

            int batchSize = 1;
            int numNodes = 5;

            var input = new Tensor<double>([batchSize, numNodes, 6]);
            var adjacency = new Tensor<double>([batchSize, numNodes, numNodes]);

            // Initialize
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 0.2;
            }

            // Ring graph
            for (int i = 0; i < numNodes; i++)
            {
                adjacency[0, i, (i + 1) % numNodes] = 1.0;
                adjacency[0, i, (i - 1 + numNodes) % numNodes] = 1.0;
            }

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(3, output.Rank);
            Assert.Equal(batchSize, output.Shape[0]);
            Assert.Equal(numNodes, output.Shape[1]);
            Assert.Equal(12, output.Shape[2]);
        }

        [Fact]
        public void GraphIsomorphismLayer_WithLearnableEpsilon_WorksCorrectly()
        {
            // Arrange
            var layer = new GraphIsomorphismLayer<double>(
                inputFeatures: 4,
                outputFeatures: 8,
                learnEpsilon: true,
                epsilon: 0.5);

            var input = new Tensor<double>([1, 3, 4]);
            var adjacency = new Tensor<double>([1, 3, 3]);

            // Initialize
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = 1.0;
            }

            // Simple connected graph
            for (int i = 0; i < 3; i++)
            {
                adjacency[0, i, i] = 1.0;
            }
            adjacency[0, 0, 1] = 1.0;
            adjacency[0, 1, 2] = 1.0;

            layer.SetAdjacencyMatrix(adjacency);

            // Act
            var output = layer.Forward(input);

            // Assert - should complete without error
            Assert.NotNull(output);
            Assert.Equal(8, output.Shape[2]);
        }

        #endregion

        #region Interface Compliance Tests

        [Fact]
        public void AllGraphLayers_ImplementIGraphConvolutionLayer()
        {
            // Arrange & Act
            var gcn = new GraphConvolutionalLayer<double>(5, 10, (IActivationFunction<double>?)null);
            var gat = new GraphAttentionLayer<double>(5, 10);
            var sage = new GraphSAGELayer<double>(5, 10);
            var gin = new GraphIsomorphismLayer<double>(5, 10);

            // Assert
            Assert.IsAssignableFrom<IGraphConvolutionLayer<double>>(gcn);
            Assert.IsAssignableFrom<IGraphConvolutionLayer<double>>(gat);
            Assert.IsAssignableFrom<IGraphConvolutionLayer<double>>(sage);
            Assert.IsAssignableFrom<IGraphConvolutionLayer<double>>(gin);
        }

        #endregion
    }
}
