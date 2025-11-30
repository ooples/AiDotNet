using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Optimizers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for complete Neural Network architectures
/// Tests end-to-end training and inference performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class NeuralNetworkArchitecturesBenchmarks
{
    [Params(100, 500)]
    public int TrainSize { get; set; }

    [Params(10, 50)]
    public int InputSize { get; set; }

    [Params(5, 10)]
    public int OutputSize { get; set; }

    private Matrix<double> _trainX = null!;
    private Matrix<double> _trainY = null!;
    private Matrix<double> _testX = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize training data
        _trainX = new Matrix<double>(TrainSize, InputSize);
        _trainY = new Matrix<double>(TrainSize, OutputSize);

        for (int i = 0; i < TrainSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                _trainX[i, j] = random.NextDouble() * 2 - 1;
            }

            // Generate synthetic targets
            for (int j = 0; j < OutputSize; j++)
            {
                _trainY[i, j] = random.NextDouble();
            }
        }

        // Initialize test data
        _testX = new Matrix<double>(20, InputSize);
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                _testX[i, j] = random.NextDouble() * 2 - 1;
            }
        }
    }

    #region Feedforward Neural Network

    [Benchmark(Baseline = true)]
    public FeedForwardNeuralNetwork<double> FeedForward_Train()
    {
        var network = new FeedForwardNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSizes: new[] { 64, 32 },
            outputSize: OutputSize
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        network.Train(_trainX, _trainY, loss, optimizer, epochs: 10, batchSize: 32);
        return network;
    }

    [Benchmark]
    public Matrix<double> FeedForward_Predict()
    {
        var network = new FeedForwardNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSizes: new[] { 64, 32 },
            outputSize: OutputSize
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        network.Train(_trainX, _trainY, loss, optimizer, epochs: 5, batchSize: 32);
        return network.Predict(_testX);
    }

    #endregion

    #region Recurrent Neural Network

    [Benchmark]
    public RecurrentNeuralNetwork<double> RNN_Train()
    {
        // Reshape data for RNN (batch x sequence x features)
        int seqLength = 5;
        int numSequences = TrainSize / seqLength;
        var rnnInput = new Tensor<double>(new[] { numSequences, seqLength, InputSize });
        var rnnTarget = new Tensor<double>(new[] { numSequences, OutputSize });

        for (int i = 0; i < numSequences; i++)
        {
            for (int t = 0; t < seqLength; t++)
            {
                int srcIdx = i * seqLength + t;
                if (srcIdx < TrainSize)
                {
                    for (int j = 0; j < InputSize; j++)
                    {
                        rnnInput[i, t, j] = _trainX[srcIdx, j];
                    }
                }
            }
            // Use last sample's target for sequence
            if ((i + 1) * seqLength - 1 < TrainSize)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    rnnTarget[i, j] = _trainY[(i + 1) * seqLength - 1, j];
                }
            }
        }

        var rnn = new RecurrentNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSize: 32,
            outputSize: OutputSize
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        rnn.Train(rnnInput, rnnTarget, loss, optimizer, epochs: 10);
        return rnn;
    }

    #endregion

    #region LSTM Neural Network

    [Benchmark]
    public LSTMNeuralNetwork<double> LSTM_Train()
    {
        // Reshape data for LSTM
        int seqLength = 5;
        int numSequences = TrainSize / seqLength;
        var lstmInput = new Tensor<double>(new[] { numSequences, seqLength, InputSize });
        var lstmTarget = new Tensor<double>(new[] { numSequences, OutputSize });

        for (int i = 0; i < numSequences; i++)
        {
            for (int t = 0; t < seqLength; t++)
            {
                int srcIdx = i * seqLength + t;
                if (srcIdx < TrainSize)
                {
                    for (int j = 0; j < InputSize; j++)
                    {
                        lstmInput[i, t, j] = _trainX[srcIdx, j];
                    }
                }
            }
            if ((i + 1) * seqLength - 1 < TrainSize)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    lstmTarget[i, j] = _trainY[(i + 1) * seqLength - 1, j];
                }
            }
        }

        var lstm = new LSTMNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSize: 32,
            outputSize: OutputSize
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        lstm.Train(lstmInput, lstmTarget, loss, optimizer, epochs: 10);
        return lstm;
    }

    #endregion

    #region GRU Neural Network

    [Benchmark]
    public GRUNeuralNetwork<double> GRU_Train()
    {
        // Reshape data for GRU
        int seqLength = 5;
        int numSequences = TrainSize / seqLength;
        var gruInput = new Tensor<double>(new[] { numSequences, seqLength, InputSize });
        var gruTarget = new Tensor<double>(new[] { numSequences, OutputSize });

        for (int i = 0; i < numSequences; i++)
        {
            for (int t = 0; t < seqLength; t++)
            {
                int srcIdx = i * seqLength + t;
                if (srcIdx < TrainSize)
                {
                    for (int j = 0; j < InputSize; j++)
                    {
                        gruInput[i, t, j] = _trainX[srcIdx, j];
                    }
                }
            }
            if ((i + 1) * seqLength - 1 < TrainSize)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    gruTarget[i, j] = _trainY[(i + 1) * seqLength - 1, j];
                }
            }
        }

        var gru = new GRUNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSize: 32,
            outputSize: OutputSize
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        gru.Train(gruInput, gruTarget, loss, optimizer, epochs: 10);
        return gru;
    }

    #endregion

    #region AutoEncoder

    [Benchmark]
    public AutoEncoder<double> AutoEncoder_Train()
    {
        var autoencoder = new AutoEncoder<double>(
            inputSize: InputSize,
            encoderSizes: new[] { 32, 16 },
            latentSize: 8
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        autoencoder.Train(_trainX, loss, optimizer, epochs: 10, batchSize: 32);
        return autoencoder;
    }

    [Benchmark]
    public (Matrix<double> encoded, Matrix<double> decoded) AutoEncoder_Encode_Decode()
    {
        var autoencoder = new AutoEncoder<double>(
            inputSize: InputSize,
            encoderSizes: new[] { 32, 16 },
            latentSize: 8
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        autoencoder.Train(_trainX, loss, optimizer, epochs: 5, batchSize: 32);

        var encoded = autoencoder.Encode(_testX);
        var decoded = autoencoder.Decode(encoded);
        return (encoded, decoded);
    }

    #endregion

    #region Residual Neural Network

    [Benchmark]
    public ResidualNeuralNetwork<double> ResNet_Train()
    {
        var resnet = new ResidualNeuralNetwork<double>(
            inputSize: InputSize,
            hiddenSize: 64,
            outputSize: OutputSize,
            numBlocks: 3
        );

        var loss = new MeanSquaredErrorLoss<double>();
        var optimizer = new AdamOptimizer<double>(learningRate: 0.001);

        resnet.Train(_trainX, _trainY, loss, optimizer, epochs: 10, batchSize: 32);
        return resnet;
    }

    #endregion
}
