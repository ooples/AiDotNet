using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Neural Network Architecture creation and configuration
/// Tests architecture setup and network instantiation performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
[SimpleJob(RuntimeMoniker.Net80)]
public class NeuralNetworkArchitecturesBenchmarks
{
    [Params(10, 50)]
    public int InputSize { get; set; }

    [Params(5, 10)]
    public int OutputSize { get; set; }

    private Tensor<double> _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness

        // Initialize input tensor (batch_size x input_size)
        int batchSize = 32;
        _input = new Tensor<double>(new[] { batchSize, InputSize });
        for (int i = 0; i < _input.Length; i++)
        {
            _input[i] = random.NextDouble() * 2 - 1;
        }
    }

    #region Architecture Creation

    [Benchmark(Baseline = true)]
    public NeuralNetworkArchitecture<double> Architecture_CreateForRegression()
    {
        return new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize
        );
    }

    [Benchmark]
    public NeuralNetworkArchitecture<double> Architecture_CreateForRegression_HighComplexity()
    {
        return new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize,
            complexity: NetworkComplexity.Deep
        );
    }

    [Benchmark]
    public NeuralNetworkArchitecture<double> Architecture_CreateForClassification()
    {
        return new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            numClasses: OutputSize
        );
    }

    [Benchmark]
    public NeuralNetworkArchitecture<double> Architecture_CreateForClassification_Binary()
    {
        return new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            numClasses: 2,
            isMultiClass: false
        );
    }

    [Benchmark]
    public NeuralNetworkArchitecture<double> Architecture_CreateDetailed()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputWidth: InputSize,
            outputSize: OutputSize
        );
    }

    #endregion

    #region Feedforward Neural Network

    [Benchmark]
    public FeedForwardNeuralNetwork<double> FeedForward_Create()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize
        );
        return new FeedForwardNeuralNetwork<double>(architecture);
    }

    [Benchmark]
    public FeedForwardNeuralNetwork<double> FeedForward_CreateWithLoss()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize
        );
        var loss = new MeanSquaredErrorLoss<double>();
        return new FeedForwardNeuralNetwork<double>(architecture, lossFunction: loss);
    }

    [Benchmark]
    public Tensor<double> FeedForward_Predict()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize
        );
        var network = new FeedForwardNeuralNetwork<double>(architecture);
        return network.Predict(_input);
    }

    #endregion

    #region LSTM Neural Network

    [Benchmark]
    public LSTMNeuralNetwork<double> LSTM_Create()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 10,  // sequence length
            inputWidth: InputSize,  // features
            outputSize: OutputSize
        );
        ILossFunction<double> loss = new MeanSquaredErrorLoss<double>();
        IActivationFunction<double>? activation = null;
        return new LSTMNeuralNetwork<double>(architecture, loss, activation);
    }

    #endregion

    #region GRU Neural Network

    [Benchmark]
    public GRUNeuralNetwork<double> GRU_Create()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 10,  // sequence length
            inputWidth: InputSize,  // features
            outputSize: OutputSize
        );
        return new GRUNeuralNetwork<double>(architecture);
    }

    #endregion

    #region RNN Neural Network

    [Benchmark]
    public RecurrentNeuralNetwork<double> RNN_Create()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 10,  // sequence length
            inputWidth: InputSize,  // features
            outputSize: OutputSize
        );
        return new RecurrentNeuralNetwork<double>(architecture);
    }

    #endregion

    #region ResNet Neural Network

    [Benchmark]
    public ResidualNeuralNetwork<double> ResNet_Create()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: InputSize,
            outputSize: OutputSize,
            complexity: NetworkComplexity.Medium
        );
        return new ResidualNeuralNetwork<double>(architecture);
    }

    #endregion

    #region Loss Functions

    [Benchmark]
    public MeanSquaredErrorLoss<double> Loss_CreateMSE()
    {
        return new MeanSquaredErrorLoss<double>();
    }

    [Benchmark]
    public CrossEntropyLoss<double> Loss_CreateCrossEntropy()
    {
        return new CrossEntropyLoss<double>();
    }

    [Benchmark]
    public BinaryCrossEntropyLoss<double> Loss_CreateBinaryCrossEntropy()
    {
        return new BinaryCrossEntropyLoss<double>();
    }

    #endregion
}
