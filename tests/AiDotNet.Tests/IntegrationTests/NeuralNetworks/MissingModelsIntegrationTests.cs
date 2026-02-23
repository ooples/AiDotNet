using System;
using System.IO;
using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Tensors;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

public class MissingModelsIntegrationTests
{
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
            data[i] = (float)(random.NextDouble() * 2 - 1);
        }

        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> CreateAdjacencyMatrix(int numNodes)
    {
        var adjacency = new Tensor<float>(new[] { numNodes, numNodes });
        for (int i = 0; i < numNodes; i++)
        {
            adjacency[i, i] = 1f;
        }

        if (numNodes > 1)
        {
            adjacency[0, 1] = 1f;
            adjacency[1, 0] = 1f;
        }

        return adjacency;
    }

    [Fact]
    public void ClipNeuralNetwork_InvalidOnnxModels_ThrowsOnConstruction()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        string imagePath = Path.Combine(tempDir, "image.onnx");
        string textPath = Path.Combine(tempDir, "text.onnx");
        // Write empty files as invalid ONNX models
        File.WriteAllText(imagePath, string.Empty);
        File.WriteAllText(textPath, string.Empty);

        try
        {
            int imageSize = 2;
            int embeddingDim = 8;
            var tokenizer = ClipTokenizerFactory.CreateSimple();
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple,
                inputDepth: 3,
                inputHeight: imageSize,
                inputWidth: imageSize,
                outputSize: embeddingDim);

            // Invalid ONNX models should throw during construction
            Assert.ThrowsAny<Exception>(() => new ClipNeuralNetwork<float>(
                architecture,
                imagePath,
                textPath,
                tokenizer,
                embeddingDimension: embeddingDim,
                maxSequenceLength: 8,
                imageSize: imageSize));
        }
        finally
        {
            if (File.Exists(imagePath))
            {
                File.Delete(imagePath);
            }
            if (File.Exists(textPath))
            {
                File.Delete(textPath);
            }
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void DDPMModel_Generate_ReturnsExpectedShape()
    {
        var model = new DDPMModel<float>();
        var output = model.Generate(new[] { 1, 2 }, numInferenceSteps: 2);

        Assert.Equal(new[] { 1, 2 }, output.Shape);
    }

    [Fact]
    public void DDPMModel_PredictNoise_ReturnsExpectedShape()
    {
        var model = new DDPMModel<float>();
        var input = CreateRandomTensor(new[] { 1, 3 }, 17);

        var output = model.PredictNoise(input, timestep: 1);

        Assert.Equal(input.Shape, output.Shape);
    }

    [Fact]
    public void GraphClassificationModel_Predict_ProducesExpectedShape()
    {
        int numNodes = 4;
        int inputFeatures = 3;
        int numClasses = 2;

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputFeatures,
            outputSize: numClasses);

        var model = new GraphClassificationModel<float>(
            architecture,
            hiddenDim: 4,
            embeddingDim: 6,
            numGnnLayers: 2);

        model.SetAdjacencyMatrix(CreateAdjacencyMatrix(numNodes));
        var nodeFeatures = CreateRandomTensor(new[] { numNodes, inputFeatures });

        var output = model.Predict(nodeFeatures);

        Assert.Equal(new[] { 1, 6 }, output.Shape);
    }

    [Fact]
    public void LinkPredictionModel_Predict_ProducesExpectedShape()
    {
        int numNodes = 5;
        int inputFeatures = 4;
        int embeddingDim = 3;

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputFeatures,
            outputSize: 1);

        var model = new LinkPredictionModel<float>(
            architecture,
            hiddenDim: 6,
            embeddingDim: embeddingDim,
            numLayers: 2);

        model.SetAdjacencyMatrix(CreateAdjacencyMatrix(numNodes));
        var nodeFeatures = CreateRandomTensor(new[] { numNodes, inputFeatures });

        var output = model.Predict(nodeFeatures);

        Assert.Equal(new[] { numNodes, embeddingDim }, output.Shape);
    }

    [Fact]
    public void NodeClassificationModel_Predict_ProducesExpectedShape()
    {
        int numNodes = 4;
        int inputFeatures = 3;
        int numClasses = 2;

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputFeatures,
            outputSize: numClasses);

        var model = new NodeClassificationModel<float>(
            architecture,
            hiddenDim: 4,
            numLayers: 2);

        model.SetAdjacencyMatrix(CreateAdjacencyMatrix(numNodes));
        var nodeFeatures = CreateRandomTensor(new[] { numNodes, inputFeatures });

        var output = model.Predict(nodeFeatures);

        Assert.Equal(new[] { numNodes, numClasses }, output.Shape);
    }

    [Fact]
    public void GraphModels_RequireAdjacencyMatrix()
    {
        int numNodes = 3;
        int inputFeatures = 2;
        int numClasses = 2;

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: inputFeatures,
            outputSize: numClasses);

        var graphModel = new GraphClassificationModel<float>(architecture);
        var linkModel = new LinkPredictionModel<float>(architecture);
        var nodeModel = new NodeClassificationModel<float>(architecture);
        var nodeFeatures = CreateRandomTensor(new[] { numNodes, inputFeatures });

        Assert.Throws<InvalidOperationException>(() => graphModel.Predict(nodeFeatures));
        Assert.Throws<InvalidOperationException>(() => linkModel.Predict(nodeFeatures));
        Assert.Throws<InvalidOperationException>(() => nodeModel.Predict(nodeFeatures));
    }

    [Fact]
    public void NeuralNetwork_PredictAndTrain_ProduceExpectedShape()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var network = new NeuralNetwork<float>(architecture);
        var input = CreateRandomTensor(new[] { 4 });
        var target = CreateRandomTensor(new[] { 2 }, 21);

        var output = network.Predict(input);

        Assert.Equal(new[] { 2 }, output.Shape);

        network.Train(input, target);
        var trainedOutput = network.Predict(input);

        Assert.Equal(new[] { 2 }, trainedOutput.Shape);
    }
}
