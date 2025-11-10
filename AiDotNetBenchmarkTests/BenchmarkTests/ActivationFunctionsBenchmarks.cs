using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for all Activation Functions in AiDotNet
/// Tests scalar, vector, and tensor operations
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class ActivationFunctionsBenchmarks
{
    [Params(100, 1000, 10000)]
    public int Size { get; set; }

    private Vector<double> _inputVector = null!;
    private Tensor<double> _inputTensor = null!;
    private double _scalarInput;

    // Common activation functions
    private ReLUActivation<double> _relu = null!;
    private LeakyReLUActivation<double> _leakyRelu = null!;
    private SigmoidActivation<double> _sigmoid = null!;
    private TanhActivation<double> _tanh = null!;
    private SoftmaxActivation<double> _softmax = null!;
    private ELUActivation<double> _elu = null!;
    private GELUActivation<double> _gelu = null!;
    private SwishActivation<double> _swish = null!;
    private MishActivation<double> _mish = null!;
    private SoftPlusActivation<double> _softplus = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);

        // Initialize scalar input
        _scalarInput = random.NextDouble() * 2 - 1; // Range [-1, 1]

        // Initialize vector
        _inputVector = new Vector<double>(Size);
        for (int i = 0; i < Size; i++)
        {
            _inputVector[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize tensor (batch_size x features)
        int batchSize = Size / 10;
        int features = 10;
        _inputTensor = new Tensor<double>(new[] { batchSize, features });
        for (int i = 0; i < _inputTensor.Length; i++)
        {
            _inputTensor[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize activation functions
        _relu = new ReLUActivation<double>();
        _leakyRelu = new LeakyReLUActivation<double>();
        _sigmoid = new SigmoidActivation<double>();
        _tanh = new TanhActivation<double>();
        _softmax = new SoftmaxActivation<double>();
        _elu = new ELUActivation<double>();
        _gelu = new GELUActivation<double>();
        _swish = new SwishActivation<double>();
        _mish = new MishActivation<double>();
        _softplus = new SoftPlusActivation<double>();
    }

    #region ReLU Activation

    [Benchmark]
    public double ReLU_Scalar()
    {
        return _relu.Activate(_scalarInput);
    }

    [Benchmark]
    public Vector<double> ReLU_Vector()
    {
        return _relu.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> ReLU_Tensor()
    {
        return _relu.Activate(_inputTensor);
    }

    [Benchmark]
    public Vector<double> ReLU_Vector_Derivative()
    {
        var activated = _relu.Activate(_inputVector);
        return activated.Transform(x => _relu.Derivative(x));
    }

    #endregion

    #region LeakyReLU Activation

    [Benchmark]
    public Vector<double> LeakyReLU_Vector()
    {
        return _leakyRelu.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> LeakyReLU_Tensor()
    {
        return _leakyRelu.Activate(_inputTensor);
    }

    #endregion

    #region Sigmoid Activation

    [Benchmark(Baseline = true)]
    public double Sigmoid_Scalar()
    {
        return _sigmoid.Activate(_scalarInput);
    }

    [Benchmark]
    public Vector<double> Sigmoid_Vector()
    {
        return _sigmoid.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> Sigmoid_Tensor()
    {
        return _sigmoid.Activate(_inputTensor);
    }

    [Benchmark]
    public Vector<double> Sigmoid_Vector_Derivative()
    {
        var activated = _sigmoid.Activate(_inputVector);
        return activated.Transform(x => _sigmoid.Derivative(x));
    }

    #endregion

    #region Tanh Activation

    [Benchmark]
    public double Tanh_Scalar()
    {
        return _tanh.Activate(_scalarInput);
    }

    [Benchmark]
    public Vector<double> Tanh_Vector()
    {
        return _tanh.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> Tanh_Tensor()
    {
        return _tanh.Activate(_inputTensor);
    }

    #endregion

    #region Softmax Activation

    [Benchmark]
    public Vector<double> Softmax_Vector()
    {
        return _softmax.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> Softmax_Tensor()
    {
        return _softmax.Activate(_inputTensor);
    }

    #endregion

    #region ELU Activation

    [Benchmark]
    public Vector<double> ELU_Vector()
    {
        return _elu.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> ELU_Tensor()
    {
        return _elu.Activate(_inputTensor);
    }

    #endregion

    #region GELU Activation

    [Benchmark]
    public Vector<double> GELU_Vector()
    {
        return _gelu.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> GELU_Tensor()
    {
        return _gelu.Activate(_inputTensor);
    }

    #endregion

    #region Swish Activation

    [Benchmark]
    public Vector<double> Swish_Vector()
    {
        return _swish.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> Swish_Tensor()
    {
        return _swish.Activate(_inputTensor);
    }

    #endregion

    #region Mish Activation

    [Benchmark]
    public Vector<double> Mish_Vector()
    {
        return _mish.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> Mish_Tensor()
    {
        return _mish.Activate(_inputTensor);
    }

    #endregion

    #region SoftPlus Activation

    [Benchmark]
    public Vector<double> SoftPlus_Vector()
    {
        return _softplus.Activate(_inputVector);
    }

    [Benchmark]
    public Tensor<double> SoftPlus_Tensor()
    {
        return _softplus.Activate(_inputTensor);
    }

    #endregion
}
