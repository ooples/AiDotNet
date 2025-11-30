using AiDotNet.TransferLearning.DomainAdaptation;
using AiDotNet.TransferLearning.FeatureMapping;
using AiDotNet.TransferLearning.Algorithms;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for Transfer Learning
/// Tests DomainAdaptation, FeatureMapping, and TransferLearning algorithms
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class TransferLearningBenchmarks
{
    [Params(100, 500)]
    public int SourceSamples { get; set; }

    [Params(50, 200)]
    public int TargetSamples { get; set; }

    [Params(10, 30)]
    public int FeatureCount { get; set; }

    private Matrix<double> _sourceData = null!;
    private Vector<double> _sourceLabels = null!;
    private Matrix<double> _targetData = null!;
    private Vector<double> _targetLabels = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Generate source domain data
        _sourceData = new Matrix<double>(SourceSamples, FeatureCount);
        _sourceLabels = new Vector<double>(SourceSamples);

        for (int i = 0; i < SourceSamples; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _sourceData[i, j] = random.NextDouble() * 2 - 1;
            }
            _sourceLabels[i] = random.NextDouble() > 0.5 ? 1.0 : 0.0;
        }

        // Generate target domain data (slightly different distribution)
        _targetData = new Matrix<double>(TargetSamples, FeatureCount);
        _targetLabels = new Vector<double>(TargetSamples);

        for (int i = 0; i < TargetSamples; i++)
        {
            for (int j = 0; j < FeatureCount; j++)
            {
                _targetData[i, j] = random.NextDouble() * 2 - 0.5; // Shifted distribution
            }
            _targetLabels[i] = random.NextDouble() > 0.5 ? 1.0 : 0.0;
        }
    }

    #region Domain Adaptation Benchmarks

    [Benchmark(Baseline = true)]
    public Matrix<double> TransferLearning01_CORALDomainAdapter()
    {
        var adapter = new CORALDomainAdapter<double>();
        return adapter.AdaptDomain(_sourceData, _targetData);
    }

    [Benchmark]
    public Matrix<double> TransferLearning02_MMDDomainAdapter()
    {
        var adapter = new MMDDomainAdapter<double>(kernelType: "rbf", gamma: 1.0);
        return adapter.AdaptDomain(_sourceData, _targetData);
    }

    [Benchmark]
    public Matrix<double> TransferLearning03_MMDDomainAdapter_LinearKernel()
    {
        var adapter = new MMDDomainAdapter<double>(kernelType: "linear");
        return adapter.AdaptDomain(_sourceData, _targetData);
    }

    [Benchmark]
    public Matrix<double> TransferLearning04_MMDDomainAdapter_PolynomialKernel()
    {
        var adapter = new MMDDomainAdapter<double>(kernelType: "polynomial", degree: 3);
        return adapter.AdaptDomain(_sourceData, _targetData);
    }

    #endregion

    #region Feature Mapping Benchmarks

    [Benchmark]
    public Matrix<double> TransferLearning05_LinearFeatureMapper_Fit()
    {
        var mapper = new LinearFeatureMapper<double>();
        return mapper.FitTransform(_sourceData, _targetData);
    }

    [Benchmark]
    public Matrix<double> TransferLearning06_LinearFeatureMapper_Transform()
    {
        var mapper = new LinearFeatureMapper<double>();
        mapper.FitTransform(_sourceData, _targetData);
        return mapper.Transform(_targetData);
    }

    [Benchmark]
    public Matrix<double> TransferLearning07_LinearFeatureMapper_FitTransform()
    {
        var mapper = new LinearFeatureMapper<double>();
        return mapper.FitTransform(_sourceData, _targetData);
    }

    #endregion

    #region Transfer Learning Algorithms

    [Benchmark]
    public TransferNeuralNetwork<double> TransferLearning08_TransferNeuralNetwork_Train()
    {
        var transferNN = new TransferNeuralNetwork<double>(
            inputSize: FeatureCount,
            hiddenSize: 20,
            outputSize: 1,
            learningRate: 0.01
        );

        transferNN.Train(_sourceData, _sourceLabels);
        return transferNN;
    }

    [Benchmark]
    public TransferNeuralNetwork<double> TransferLearning09_TransferNeuralNetwork_FineTune()
    {
        var transferNN = new TransferNeuralNetwork<double>(
            inputSize: FeatureCount,
            hiddenSize: 20,
            outputSize: 1,
            learningRate: 0.01
        );

        transferNN.Train(_sourceData, _sourceLabels);
        transferNN.FineTune(_targetData, _targetLabels, epochs: 10);
        return transferNN;
    }

    [Benchmark]
    public Vector<double> TransferLearning10_TransferNeuralNetwork_Predict()
    {
        var transferNN = new TransferNeuralNetwork<double>(
            inputSize: FeatureCount,
            hiddenSize: 20,
            outputSize: 1,
            learningRate: 0.01
        );

        transferNN.Train(_sourceData, _sourceLabels);
        return transferNN.Predict(_targetData);
    }

    [Benchmark]
    public TransferRandomForest<double> TransferLearning11_TransferRandomForest_Train()
    {
        var transferRF = new TransferRandomForest<double>(
            numTrees: 10,
            maxDepth: 5
        );

        transferRF.Train(_sourceData, _sourceLabels);
        return transferRF;
    }

    [Benchmark]
    public TransferRandomForest<double> TransferLearning12_TransferRandomForest_FineTune()
    {
        var transferRF = new TransferRandomForest<double>(
            numTrees: 10,
            maxDepth: 5
        );

        transferRF.Train(_sourceData, _sourceLabels);
        transferRF.FineTune(_targetData, _targetLabels);
        return transferRF;
    }

    [Benchmark]
    public Vector<double> TransferLearning13_TransferRandomForest_Predict()
    {
        var transferRF = new TransferRandomForest<double>(
            numTrees: 10,
            maxDepth: 5
        );

        transferRF.Train(_sourceData, _sourceLabels);
        return transferRF.Predict(_targetData);
    }

    #endregion

    #region End-to-End Transfer Learning Scenarios

    [Benchmark]
    public Vector<double> TransferLearning14_EndToEnd_CORAL_NN()
    {
        // Domain adaptation with CORAL
        var adapter = new CORALDomainAdapter<double>();
        var adaptedSource = adapter.AdaptDomain(_sourceData, _targetData);

        // Train neural network on adapted data
        var transferNN = new TransferNeuralNetwork<double>(
            inputSize: FeatureCount,
            hiddenSize: 20,
            outputSize: 1,
            learningRate: 0.01
        );

        transferNN.Train(adaptedSource, _sourceLabels);

        // Fine-tune on target domain
        transferNN.FineTune(_targetData, _targetLabels, epochs: 5);

        // Predict
        return transferNN.Predict(_targetData);
    }

    [Benchmark]
    public Vector<double> TransferLearning15_EndToEnd_MMD_RF()
    {
        // Domain adaptation with MMD
        var adapter = new MMDDomainAdapter<double>(kernelType: "rbf", gamma: 1.0);
        var adaptedSource = adapter.AdaptDomain(_sourceData, _targetData);

        // Train random forest on adapted data
        var transferRF = new TransferRandomForest<double>(
            numTrees: 10,
            maxDepth: 5
        );

        transferRF.Train(adaptedSource, _sourceLabels);

        // Fine-tune on target domain
        transferRF.FineTune(_targetData, _targetLabels);

        // Predict
        return transferRF.Predict(_targetData);
    }

    [Benchmark]
    public Vector<double> TransferLearning16_EndToEnd_LinearMapping_NN()
    {
        // Feature mapping
        var mapper = new LinearFeatureMapper<double>();
        var mappedSource = mapper.FitTransform(_sourceData, _targetData);
        var mappedTarget = mapper.Transform(_targetData);

        // Train neural network on mapped data
        var transferNN = new TransferNeuralNetwork<double>(
            inputSize: FeatureCount,
            hiddenSize: 20,
            outputSize: 1,
            learningRate: 0.01
        );

        transferNN.Train(mappedSource, _sourceLabels);

        // Fine-tune on mapped target domain
        transferNN.FineTune(mappedTarget, _targetLabels, epochs: 5);

        // Predict
        return transferNN.Predict(mappedTarget);
    }

    #endregion
}
