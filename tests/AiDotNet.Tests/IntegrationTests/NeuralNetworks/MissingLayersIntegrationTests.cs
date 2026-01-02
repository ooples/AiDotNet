using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

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

    [Fact]
    public void ConditionalRandomFieldLayer_ForwardBackward_RetainsAnyRankShape()
    {
        int batchA = 2;
        int batchB = 3;
        int sequenceLength = 4;
        int numClasses = 5;

        var layer = new ConditionalRandomFieldLayer<float>(
            numClasses,
            sequenceLength,
            scalarActivation: (IActivationFunction<float>)new IdentityActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchA, batchB, sequenceLength, numClasses });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 7);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void ContinuumMemorySystemLayer_ForwardBackward_PreservesBatchShape()
    {
        int inputSize = 6;
        int hiddenDim = 4;

        var layer = new ContinuumMemorySystemLayer<float>(new[] { inputSize }, hiddenDim, numFrequencyLevels: 2);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { 2, 3, inputSize });
        var output = layer.Forward(input);

        Assert.Equal(new[] { 2, 3, hiddenDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 9);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void DenseBlockLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputChannels = 3;
        int growthRate = 4;
        int height = 6;
        int width = 6;

        var layer = new DenseBlockLayer<float>(inputChannels, growthRate, height, width);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputChannels, height, width });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, growthRate, height, width }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 5);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void DiffusionConvLayer_Forward_BatchedInputProducesExpectedShape()
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

        Assert.Equal(new[] { batchSize, numVertices, outputChannels }, output.Shape);
    }

    [Fact]
    public void DigitCapsuleLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputCapsules = 4;
        int inputCapsuleDim = 6;
        int numClasses = 3;
        int outputCapsuleDim = 5;

        var layer = new DigitCapsuleLayer<float>(
            inputCapsules,
            inputCapsuleDim,
            numClasses,
            outputCapsuleDim,
            routingIterations: 3);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputCapsules, inputCapsuleDim });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, numClasses * outputCapsuleDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 11);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void EdgeConditionalConvolutionalLayer_ForwardBackward_HandlesUnbatchedInput()
    {
        int numNodes = 3;
        int inputFeatures = 4;
        int outputFeatures = 2;
        int edgeFeatures = 3;

        var layer = new EdgeConditionalConvolutionalLayer<float>(inputFeatures, outputFeatures, edgeFeatures);
        layer.SetTrainingMode(true);

        var adjacency = CreateDenseAdjacencyMatrix(numNodes);
        layer.SetAdjacencyMatrix(adjacency);

        int numEdges = numNodes * numNodes;
        var edgeFeatureTensor = CreateEdgeFeatures(1, numEdges, edgeFeatures);
        layer.SetEdgeFeatures(edgeFeatureTensor);

        var input = CreateRandomTensor(new[] { numNodes, inputFeatures });
        var output = layer.Forward(input);

        Assert.Equal(new[] { numNodes, outputFeatures }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 13);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void FlashAttentionLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int sequenceLength = 6;
        int embeddingDimension = 8;
        int headCount = 2;

        var layer = new FlashAttentionLayer<float>(sequenceLength, embeddingDimension, headCount);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, sequenceLength, embeddingDimension });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 15);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void HyperbolicLinearLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 3;
        int inputFeatures = 6;
        int outputFeatures = 4;

        var layer = new HyperbolicLinearLayer<float>(inputFeatures, outputFeatures);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputFeatures });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, outputFeatures }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 17);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void LogVarianceLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int features = 5;

        var layer = new LogVarianceLayer<float>(new[] { batchSize, features }, axis: 1);
        var input = CreateRandomTensor(new[] { batchSize, features });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 19);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MeasurementLayer_ForwardBackward_ProducesNormalizedOutput()
    {
        int size = 4;
        var layer = new MeasurementLayer<float>(size);

        var input = CreateRandomTensor(new[] { size });
        var output = layer.Forward(input);

        Assert.Equal(new[] { size }, output.Shape);

        float sum = 0.0f;
        for (int i = 0; i < output.Length; i++)
        {
            sum += output[i];
        }
        Assert.True(Math.Abs(sum - 1.0f) < Tolerance, $"Expected probabilities to sum to 1, got {sum}");

        var gradient = CreateRandomTensor(output.Shape, seed: 21);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MemoryReadLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputDim = 4;
        int memoryDim = 5;
        int outputDim = 3;
        int memorySlots = 3;

        var layer = new MemoryReadLayer<float>(
            inputDim,
            memoryDim,
            outputDim,
            activationFunction: (IActivationFunction<float>)new IdentityActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputDim });
        var memory = CreateRandomTensor(new[] { memorySlots, memoryDim });
        var output = layer.Forward(input, memory);

        Assert.Equal(new[] { batchSize, outputDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 23);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MemoryWriteLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputDim = 4;
        int memoryDim = 5;
        int memorySlots = 3;

        var layer = new MemoryWriteLayer<float>(
            inputDim,
            memoryDim,
            activationFunction: (IActivationFunction<float>)new IdentityActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputDim });
        var memory = CreateRandomTensor(new[] { memorySlots, memoryDim });
        var output = layer.Forward(input, memory);

        Assert.Equal(new[] { batchSize, memoryDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 25);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MeshEdgeConvLayer_ForwardBackward_ProducesExpectedShape()
    {
        int numEdges = 6;
        int inputChannels = 4;
        int outputChannels = 5;
        int numNeighbors = 2;

        var layer = new MeshEdgeConvLayer<float>(
            inputChannels,
            outputChannels,
            numNeighbors,
            activationFunction: (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);
        layer.SetEdgeAdjacency(CreateEdgeAdjacency(numEdges, numNeighbors));

        var input = CreateRandomTensor(new[] { numEdges, inputChannels });
        var output = layer.Forward(input);

        Assert.Equal(new[] { numEdges, outputChannels }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 27);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void MeshPoolLayer_Forward_ProducesExpectedShape()
    {
        int numEdges = 6;
        int inputChannels = 4;
        int targetEdges = 3;
        int numNeighbors = 2;

        var layer = new MeshPoolLayer<float>(inputChannels, targetEdges, numNeighbors);
        layer.SetEdgeAdjacency(CreateEdgeAdjacency(numEdges, numNeighbors));

        var input = CreateRandomTensor(new[] { numEdges, inputChannels });
        var output = layer.Forward(input);

        Assert.Equal(new[] { targetEdges, inputChannels }, output.Shape);
        Assert.NotNull(layer.RemainingEdgeIndices);
        Assert.Equal(targetEdges, layer.RemainingEdgeIndices!.Length);
        Assert.NotNull(layer.UpdatedAdjacency);
    }

    [Fact]
    public void MixtureOfExpertsLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputSize = 4;
        int outputSize = 3;

        var expertLayers = new List<ILayer<float>>
        {
            new DenseLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>())
        };

        var experts = new List<ILayer<float>>
        {
            new ExpertLayer<float>(expertLayers, new[] { inputSize }, new[] { outputSize }),
            new ExpertLayer<float>(expertLayers, new[] { inputSize }, new[] { outputSize })
        };

        var router = new DenseLayer<float>(inputSize, experts.Count);
        var layer = new MixtureOfExpertsLayer<float>(
            experts,
            router,
            new[] { inputSize },
            new[] { outputSize },
            topK: 1);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputSize });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, outputSize }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 29);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void OctonionLinearLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputFeatures = 2;
        int outputFeatures = 3;

        var layer = new OctonionLinearLayer<float>(inputFeatures, outputFeatures);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputFeatures * 8 });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, outputFeatures * 8 }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 31);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void PrimaryCapsuleLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputChannels = 3;
        int capsuleChannels = 2;
        int capsuleDim = 4;
        int height = 6;
        int width = 6;

        var layer = new PrimaryCapsuleLayer<float>(
            inputChannels,
            capsuleChannels,
            capsuleDim,
            kernelSize: 3,
            stride: 1,
            scalarActivation: (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputChannels, height, width });
        var output = layer.Forward(input);

        int outputHeight = (height - 3) / 1 + 1;
        int outputWidth = (width - 3) / 1 + 1;
        Assert.Equal(new[] { batchSize, outputHeight, outputWidth, capsuleChannels, capsuleDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 33);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void PrincipalNeighbourhoodAggregationLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int numNodes = 4;
        int inputFeatures = 5;
        int outputFeatures = 6;

        var layer = new PrincipalNeighbourhoodAggregationLayer<float>(inputFeatures, outputFeatures);
        layer.SetTrainingMode(true);
        layer.SetAdjacencyMatrix(CreateDenseAdjacencyMatrix(batchSize, numNodes));

        var input = CreateRandomTensor(new[] { batchSize, numNodes, inputFeatures });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, numNodes, outputFeatures }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 35);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void ReadoutLayer_ForwardBackward_ProducesExpectedShape()
    {
        int inputSize = 6;
        int outputSize = 4;

        var layer = new ReadoutLayer<float>(inputSize, outputSize, (IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { inputSize });
        var output = layer.Forward(input);

        Assert.Equal(new[] { outputSize }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 37);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void ReconstructionLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int inputDim = 6;
        int hidden1 = 5;
        int hidden2 = 4;
        int outputDim = 3;

        var layer = new ReconstructionLayer<float>(
            inputDim,
            hidden1,
            hidden2,
            outputDim,
            hiddenActivation: new ReLUActivation<float>(),
            outputActivation: new SigmoidActivation<float>());
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, inputDim });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, outputDim }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 39);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void RepParameterizationLayer_ForwardBackward_ProducesExpectedShape()
    {
        int batchSize = 2;
        int latentSize = 3;

        var layer = new RepParameterizationLayer<float>(new[] { batchSize, latentSize * 2 });
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, latentSize * 2 });
        var output = layer.Forward(input);

        Assert.Equal(new[] { batchSize, latentSize }, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 41);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(input.Shape, inputGradient.Shape);
    }

    [Fact]
    public void SpatialPoolerLayer_Forward_ProducesExpectedShape()
    {
        int inputSize = 8;
        int columnCount = 5;

        var layer = new SpatialPoolerLayer<float>(inputSize, columnCount, sparsityThreshold: 0.4);
        var input = CreateRandomTensor(new[] { inputSize });
        var output = layer.Forward(input);

        Assert.Equal(new[] { columnCount }, output.Shape);
    }

    [Fact]
    public void SpiralConvLayer_Forward_BatchedInputProducesExpectedShape()
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

        Assert.Equal(new[] { batchSize, numVertices, outputChannels }, output.Shape);
    }

    [Fact]
    public void SynapticPlasticityLayer_ForwardBackward_PreservesShape()
    {
        int batchSize = 2;
        int size = 6;

        var layer = new SynapticPlasticityLayer<float>(size);
        layer.SetTrainingMode(true);

        var input = CreateRandomTensor(new[] { batchSize, size });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);

        var gradient = CreateRandomTensor(output.Shape, seed: 43);
        var inputGradient = layer.Backward(gradient);

        Assert.Equal(output.Shape, inputGradient.Shape);
    }

    [Fact]
    public void TemporalMemoryLayer_Forward_ProducesExpectedShape()
    {
        int columns = 4;
        int cellsPerColumn = 3;

        var layer = new TemporalMemoryLayer<float>(columns, cellsPerColumn);
        var input = CreateRandomTensor(new[] { columns });
        var output = layer.Forward(input);

        Assert.Equal(new[] { columns * cellsPerColumn }, output.Shape);
    }
}
