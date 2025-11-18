using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LinearAlgebra;
using AiDotNet.ActivationFunctions;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Neural Network Layer operations
/// Tests forward pass and backward pass performance for various layer types
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class NeuralNetworkLayersBenchmarks
{
    [Params(32, 128)]
    public int BatchSize { get; set; }

    [Params(128, 512)]
    public int InputSize { get; set; }

    [Params(64, 256)]
    public int OutputSize { get; set; }

    private Tensor<double> _input = null!;
    private Tensor<double> _gradOutput = null!;

    private DenseLayer<double> _denseLayer = null!;
    private ActivationLayer<double> _activationLayer = null!;
    private DropoutLayer<double> _dropoutLayer = null!;
    private BatchNormalizationLayer<double> _batchNormLayer = null!;
    private LayerNormalizationLayer<double> _layerNormLayer = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize input tensor (batch_size x input_size)
        _input = new Tensor<double>(new[] { BatchSize, InputSize });
        for (int i = 0; i < _input.Length; i++)
        {
            _input[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize gradient output tensor (batch_size x output_size)
        _gradOutput = new Tensor<double>(new[] { BatchSize, OutputSize });
        for (int i = 0; i < _gradOutput.Length; i++)
        {
            _gradOutput[i] = random.NextDouble() * 0.1;
        }

        // Initialize layers
        _denseLayer = new DenseLayer<double>(InputSize, OutputSize);
        _activationLayer = new ActivationLayer<double>(new ReLUActivation<double>());
        _dropoutLayer = new DropoutLayer<double>(dropoutRate: 0.5);
        _batchNormLayer = new BatchNormalizationLayer<double>(InputSize);
        _layerNormLayer = new LayerNormalizationLayer<double>(InputSize);
    }

    #region Dense Layer

    [Benchmark(Baseline = true)]
    public Tensor<double> DenseLayer_Forward()
    {
        return _denseLayer.Forward(_input);
    }

    [Benchmark]
    public Tensor<double> DenseLayer_ForwardBackward()
    {
        var output = _denseLayer.Forward(_input);
        return _denseLayer.Backward(_gradOutput);
    }

    #endregion

    #region Activation Layer

    [Benchmark]
    public Tensor<double> ActivationLayer_Forward()
    {
        return _activationLayer.Forward(_input);
    }

    [Benchmark]
    public Tensor<double> ActivationLayer_ForwardBackward()
    {
        var output = _activationLayer.Forward(_input);
        return _activationLayer.Backward(_gradOutput);
    }

    #endregion

    #region Dropout Layer

    [Benchmark]
    public Tensor<double> DropoutLayer_Forward()
    {
        return _dropoutLayer.Forward(_input);
    }

    [Benchmark]
    public Tensor<double> DropoutLayer_ForwardBackward()
    {
        var output = _dropoutLayer.Forward(_input);
        return _dropoutLayer.Backward(_gradOutput);
    }

    #endregion

    #region Batch Normalization

    [Benchmark]
    public Tensor<double> BatchNormalization_Forward()
    {
        return _batchNormLayer.Forward(_input);
    }

    [Benchmark]
    public Tensor<double> BatchNormalization_ForwardBackward()
    {
        var output = _batchNormLayer.Forward(_input);
        return _batchNormLayer.Backward(_gradOutput);
    }

    #endregion

    #region Layer Normalization

    [Benchmark]
    public Tensor<double> LayerNormalization_Forward()
    {
        return _layerNormLayer.Forward(_input);
    }

    [Benchmark]
    public Tensor<double> LayerNormalization_ForwardBackward()
    {
        var output = _layerNormLayer.Forward(_input);
        return _layerNormLayer.Backward(_gradOutput);
    }

    #endregion

    #region Sequential Layer Processing

    [Benchmark]
    public Tensor<double> Sequential_DenseActivation()
    {
        var dense1 = new DenseLayer<double>(InputSize, OutputSize);
        var activation1 = new ActivationLayer<double>(new ReLUActivation<double>());

        var h1 = dense1.Forward(_input);
        return activation1.Forward(h1);
    }

    [Benchmark]
    public Tensor<double> Sequential_DenseNormActivation()
    {
        var dense1 = new DenseLayer<double>(InputSize, OutputSize);
        var norm1 = new BatchNormalizationLayer<double>(OutputSize);
        var activation1 = new ActivationLayer<double>(new ReLUActivation<double>());

        var h1 = dense1.Forward(_input);
        var h2 = norm1.Forward(h1);
        return activation1.Forward(h2);
    }

    [Benchmark]
    public Tensor<double> Sequential_ThreeLayerNetwork()
    {
        int hiddenSize = 128;

        var dense1 = new DenseLayer<double>(InputSize, hiddenSize);
        var activation1 = new ActivationLayer<double>(new ReLUActivation<double>());
        var dense2 = new DenseLayer<double>(hiddenSize, hiddenSize);
        var activation2 = new ActivationLayer<double>(new ReLUActivation<double>());
        var dense3 = new DenseLayer<double>(hiddenSize, OutputSize);

        var h1 = dense1.Forward(_input);
        var a1 = activation1.Forward(h1);
        var h2 = dense2.Forward(a1);
        var a2 = activation2.Forward(h2);
        return dense3.Forward(a2);
    }

    #endregion
}
