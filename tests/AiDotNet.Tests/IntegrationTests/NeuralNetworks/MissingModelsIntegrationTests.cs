using System;
using System.IO;
using AiDotNet.Data.Structures;
using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Tensors;
using AiDotNet.Tokenization;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task ClipNeuralNetwork_InvalidOnnxModels_ThrowsOnConstruction()
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

    [Fact(Timeout = 120000)]
    public async Task DDPMModel_Generate_ReturnsExpectedShape()
    {
        var model = new DDPMModel<float>();
        var output = model.Generate(new[] { 1, 2 }, numInferenceSteps: 2);

        Assert.Equal(new[] { 1, 2 }, output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task DDPMModel_PredictNoise_ReturnsExpectedShape()
    {
        var model = new DDPMModel<float>();
        var input = CreateRandomTensor(new[] { 1, 3 }, 17);

        var output = model.PredictNoise(input, timestep: 1);

        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task GraphClassificationModel_Predict_ProducesExpectedShape()
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

        Assert.Equal(new[] { 1, 6 }, output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task LinkPredictionModel_Predict_ProducesExpectedShape()
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

        Assert.Equal(new[] { numNodes, embeddingDim }, output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task NodeClassificationModel_Predict_ProducesExpectedShape()
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

        Assert.Equal(new[] { numNodes, numClasses }, output.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task NodeClassificationModel_TrainOnTask_LearnsSeparableGraph()
    {
        // Regression test for the GNN training bugs (fix/gnn-node-classification-training):
        //   1. TrainOnTask threw ("Backward pass must be called before updating parameters")
        //      because it called layer.UpdateParameters without an autodiff backward pass.
        //   2. Train(input, expected) was a no-op — it read stale (zero) parameter gradients,
        //      so weights never changed and loss stayed flat.
        // Both now route through the GradientTape path, so a GCN must actually learn a
        // linearly/graph-separable node-classification problem.
        await Task.Yield();

        const int n = 60, F = 4, C = 2;
        var rng = new Random(3);
        var features = new Tensor<double>(new[] { n, F });
        var labels = new Tensor<double>(new[] { n, C });
        var adjacency = new Tensor<double>(new[] { n, n });
        var yTrue = new int[n];

        // Two communities (first/second half). Features clearly encode the class;
        // edges only connect within a community.
        var nbr = new List<int>[n];
        for (int i = 0; i < n; i++) nbr[i] = new List<int>();
        for (int i = 0; i < n; i++)
        {
            int comm = i < n / 2 ? 0 : 1;
            yTrue[i] = comm;
            labels[i, comm] = 1.0;
            for (int f = 0; f < F; f++)
                features[i, f] = rng.NextDouble() * 0.2 + (f == comm ? 1.0 : 0.0);
        }
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if ((i < n / 2) == (j < n / 2) && rng.NextDouble() < 0.3)
                {
                    nbr[i].Add(j);
                    nbr[j].Add(i);
                }

        // Symmetric-normalized adjacency with self-loops (Kipf & Welling renormalization).
        var deg = new double[n];
        for (int i = 0; i < n; i++) deg[i] = nbr[i].Count + 1.0;
        for (int i = 0; i < n; i++)
        {
            adjacency[i, i] = 1.0 / deg[i];
            foreach (int j in nbr[i])
                adjacency[i, j] = 1.0 / (Math.Sqrt(deg[i]) * Math.Sqrt(deg[j]));
        }

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: F,
            outputSize: C);
        var model = new NodeClassificationModel<double>(
            architecture, hiddenDim: 16, numLayers: 2, dropoutRate: 0.0, maxGradNorm: 0.0);

        var allNodes = Enumerable.Range(0, n).ToArray();
        var task = new NodeClassificationTask<double>
        {
            Graph = new GraphData<double> { NodeFeatures = features, AdjacencyMatrix = adjacency, NodeLabels = labels },
            Labels = labels,
            NumClasses = C,
            TrainIndices = allNodes,
            ValIndices = allNodes,
            TestIndices = allNodes,
        };

        // Must NOT throw (bug #1) and must actually learn (bug #2).
        var history = model.TrainOnTask(task, epochs: 150, learningRate: 0.02);

        double firstLoss = history["train_loss"][0];
        double lastLoss = history["train_loss"][^1];
        double finalAcc = history["train_accuracy"][^1];

        Assert.True(lastLoss < firstLoss,
            $"Training should reduce loss (first={firstLoss:F4}, last={lastLoss:F4}) — a flat loss means the no-op bug regressed.");
        Assert.True(finalAcc >= 0.9,
            $"GCN should learn the separable graph (final train accuracy {finalAcc:P0} < 90%).");

        double testAcc = model.EvaluateOnTask(task);
        Assert.True(testAcc >= 0.9, $"EvaluateOnTask accuracy {testAcc:P0} < 90%.");
    }

    [Fact(Timeout = 120000)]
    public async Task GraphModels_RequireAdjacencyMatrix()
    {
        await Task.Yield();
        // Strict PyTorch-Geometric contract: graph task models REQUIRE the graph structure — Predict
        // throws when no adjacency was set and the implicit-identity tolerance was not opted into.
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

    [Fact(Timeout = 120000)]
    public async Task NeuralNetwork_PredictAndTrain_ProduceExpectedShape()
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

        Assert.Equal(new[] { 2 }, output.Shape.ToArray());

        network.Train(input, target);
        var trainedOutput = network.Predict(input);

        Assert.Equal(new[] { 2 }, trainedOutput.Shape.ToArray());
    }
}
