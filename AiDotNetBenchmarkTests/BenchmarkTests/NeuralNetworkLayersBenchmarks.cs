using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Neural Network Layer operations
/// Tests forward pass and backward pass performance for various layer types
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net471, baseline: true)]
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
        // Using seeded Random for reproducible benchmark data - not used for security purposes
        var random = RandomHelper.CreateSeededRandom(42); // NOSONAR S2245 - benchmarks don't need cryptographic randomness

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

        // Initialize layers with explicit activation function to avoid ambiguity
        IActivationFunction<double> relu = new ReLUActivation<double>();
        _denseLayer = new DenseLayer<double>(InputSize, OutputSize, relu);
        _activationLayer = new ActivationLayer<double>(new[] { BatchSize, InputSize }, relu);
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
        _denseLayer.Forward(_input);
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
        _activationLayer.Forward(_input);
        // Use same-shape gradient for activation layer
        var activationGrad = new Tensor<double>(new[] { BatchSize, InputSize });
        for (int i = 0; i < activationGrad.Length; i++)
        {
            activationGrad[i] = 0.1;
        }
        return _activationLayer.Backward(activationGrad);
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
        _dropoutLayer.Forward(_input);
        // Use same-shape gradient for dropout layer
        var dropoutGrad = new Tensor<double>(new[] { BatchSize, InputSize });
        for (int i = 0; i < dropoutGrad.Length; i++)
        {
            dropoutGrad[i] = 0.1;
        }
        return _dropoutLayer.Backward(dropoutGrad);
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
        _batchNormLayer.Forward(_input);
        // Use same-shape gradient for batch norm layer
        var bnGrad = new Tensor<double>(new[] { BatchSize, InputSize });
        for (int i = 0; i < bnGrad.Length; i++)
        {
            bnGrad[i] = 0.1;
        }
        return _batchNormLayer.Backward(bnGrad);
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
        _layerNormLayer.Forward(_input);
        // Use same-shape gradient for layer norm layer
        var lnGrad = new Tensor<double>(new[] { BatchSize, InputSize });
        for (int i = 0; i < lnGrad.Length; i++)
        {
            lnGrad[i] = 0.1;
        }
        return _layerNormLayer.Backward(lnGrad);
    }

    #endregion

    #region Layer Construction

    [Benchmark]
    public DenseLayer<double> DenseLayer_Create()
    {
        IActivationFunction<double> relu = new ReLUActivation<double>();
        return new DenseLayer<double>(InputSize, OutputSize, relu);
    }

    [Benchmark]
    public ActivationLayer<double> ActivationLayer_Create()
    {
        IActivationFunction<double> relu = new ReLUActivation<double>();
        return new ActivationLayer<double>(new[] { BatchSize, InputSize }, relu);
    }

    [Benchmark]
    public DropoutLayer<double> DropoutLayer_Create()
    {
        return new DropoutLayer<double>(dropoutRate: 0.5);
    }

    [Benchmark]
    public BatchNormalizationLayer<double> BatchNormLayer_Create()
    {
        return new BatchNormalizationLayer<double>(InputSize);
    }

    [Benchmark]
    public LayerNormalizationLayer<double> LayerNormLayer_Create()
    {
        return new LayerNormalizationLayer<double>(InputSize);
    }

    #endregion
}
