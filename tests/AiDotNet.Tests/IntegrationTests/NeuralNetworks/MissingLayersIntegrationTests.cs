using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for neural network layers that were missing coverage.
/// These tests focus on forward/backward shape validation and basic behavior.
/// </summary>
public class MissingLayersIntegrationTests
{
    private const float Tolerance = 1e-4f;

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = new Random(seed);
        var length = 1;
        foreach (var dim in shape)
        {
            length *= dim;
        }

        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        }

        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> CreateDenseAdjacencyMatrix(int numNodes)
    {
        var adj = new Tensor<float>(new[] { numNodes, numNodes });
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                adj[i, j] = 1.0f;
            }
        }

        return adj;
    }

    private static Tensor<float> CreateDenseAdjacencyMatrix(int batchSize, int numNodes)
    {
        var adj = new Tensor<float>(new[] { batchSize, numNodes, numNodes });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < numNodes; i++)
            {
                for (int j = 0; j < numNodes; j++)
                {
                    adj[b, i, j] = 1.0f;
                }
            }
        }

        return adj;
    }

    private static Tensor<float> CreateEdgeFeatures(int batchSize, int numEdges, int edgeFeatures, int seed = 123)
    {
        return CreateRandomTensor(new[] { batchSize, numEdges, edgeFeatures }, seed);
    }

    private static Tensor<float> CreateIdentityMatrix(int size)
    {
        var matrix = new Tensor<float>(new[] { size, size });
        for (int i = 0; i < size; i++)
        {
            matrix[i, i] = 1.0f;
        }

        return matrix;
    }

    private static int[,] CreateEdgeAdjacency(int numEdges, int numNeighbors)
    {
        var adjacency = new int[numEdges, numNeighbors];
        for (int e = 0; e < numEdges; e++)
        {
            for (int n = 0; n < numNeighbors; n++)
            {
                adjacency[e, n] = (e + n + 1) % numEdges;
            }
        }

        return adjacency;
    }

    private static int[,] CreateSpiralIndices(int numVertices, int spiralLength)
    {
        var indices = new int[numVertices, spiralLength];
        for (int v = 0; v < numVertices; v++)
        {
            for (int s = 0; s < spiralLength; s++)
            {
                indices[v, s] = (v + s) % numVertices;
            }
        }

        return indices;
    }




    [Fact(Timeout = 120000)]
    public async Task DiffusionConvLayer_Forward_BatchedInputProducesExpectedShape()
    {
        int batchSize = 2;
        int numVertices = 5;
        int inputChannels = 3;
        int outputChannels = 4;

        var layer = new DiffusionConvLayer<float>(
            inputChannels,
            outputChannels,
            numTimeScales: 2,
            numEigenvectors: 4,
            numVertices: numVertices,
            activation: (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetLaplacian(CreateIdentityMatrix(numVertices));

        var input = CreateRandomTensor(new[] { batchSize, numVertices, inputChannels });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, numVertices, outputChannels }, output.Shape.ToArray());
    }










    [Fact(Timeout = 120000)]
    public async Task MeshPoolLayer_Forward_ProducesExpectedShape()
    {
        int numEdges = 6;
        int inputChannels = 4;
        int targetEdges = 3;
        int numNeighbors = 2;

        var layer = new MeshPoolLayer<float>(inputChannels, targetEdges, numNeighbors);
        layer.SetEdgeAdjacency(CreateEdgeAdjacency(numEdges, numNeighbors));

        var input = CreateRandomTensor(new[] { numEdges, inputChannels });
        var output = layer.Forward(input);

        Assert.Equal(new[] { targetEdges, inputChannels }, output.Shape.ToArray());
        Assert.NotNull(layer.RemainingEdgeIndices);
        Assert.Equal(targetEdges, layer.RemainingEdgeIndices!.Length);
        Assert.NotNull(layer.UpdatedAdjacency);
    }








    [Fact(Timeout = 120000)]
    public async Task SpatialPoolerLayer_Forward_ProducesExpectedShape()
    {
        int inputSize = 8;
        int columnCount = 5;

        var layer = new SpatialPoolerLayer<float>(inputSize, columnCount, sparsityThreshold: 0.4);
        var input = CreateRandomTensor(new[] { inputSize });
        var output = layer.Forward(input);

        Assert.Equal(new[] { columnCount }, output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task SpiralConvLayer_Forward_BatchedInputProducesExpectedShape()
    {
        int batchSize = 2;
        int numVertices = 5;
        int inputChannels = 3;
        int outputChannels = 4;
        int spiralLength = 3;

        var layer = new SpiralConvLayer<float>(
            inputChannels,
            outputChannels,
            spiralLength,
            numVertices,
            activationFunction: new ReLUActivation<float>());
        layer.SetTrainingMode(true);
        layer.SetSpiralIndices(CreateSpiralIndices(numVertices, spiralLength));

        var input = CreateRandomTensor(new[] { batchSize, numVertices, inputChannels });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, numVertices, outputChannels }, output.Shape.ToArray());
    }


    [Fact(Timeout = 120000)]
    public async Task TemporalMemoryLayer_Forward_ProducesExpectedShape()
    {
        int columns = 4;
        int cellsPerColumn = 3;

        var layer = new TemporalMemoryLayer<float>(columns, cellsPerColumn);
        var input = CreateRandomTensor(new[] { columns });
        var output = layer.Forward(input);

        Assert.Equal(new[] { columns * cellsPerColumn }, output.Shape.ToArray());
    }
}
