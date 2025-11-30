using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Comprehensive benchmarks for ALL 38 Activation Functions in AiDotNet
/// Each activation function tested individually on vectors
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class AllActivationFunctionsBenchmarks
{
    [Params(1000, 10000)]
    public int Size { get; set; }

    private Vector<double> _input = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _input = new Vector<double>(Size);
        for (int i = 0; i < Size; i++)
        {
            _input[i] = random.NextDouble() * 4 - 2; // Range [-2, 2]
        }
    }

    [Benchmark(Baseline = true)]
    public Vector<double> Act01_ReLU()
    {
        var act = new ReLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act02_LeakyReLU()
    {
        var act = new LeakyReLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act03_PReLU()
    {
        var act = new PReLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act04_RReLU()
    {
        var act = new RReLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act05_ELU()
    {
        var act = new ELUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act06_SELU()
    {
        var act = new SELUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act07_CELU()
    {
        var act = new CELUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act08_GELU()
    {
        var act = new GELUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act09_Sigmoid()
    {
        var act = new SigmoidActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act10_HardSigmoid()
    {
        var act = new HardSigmoidActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act11_Tanh()
    {
        var act = new TanhActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act12_HardTanh()
    {
        var act = new HardTanhActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act13_ScaledTanh()
    {
        var act = new ScaledTanhActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act14_Swish()
    {
        var act = new SwishActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act15_SiLU()
    {
        var act = new SiLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act16_Mish()
    {
        var act = new MishActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act17_SoftPlus()
    {
        var act = new SoftPlusActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act18_SoftSign()
    {
        var act = new SoftSignActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act19_Softmax()
    {
        var act = new SoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act20_Softmin()
    {
        var act = new SoftminActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act21_LogSoftmax()
    {
        var act = new LogSoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act22_LogSoftmin()
    {
        var act = new LogSoftminActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act23_Sparsemax()
    {
        var act = new SparsemaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act24_TaylorSoftmax()
    {
        var act = new TaylorSoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act25_GumbelSoftmax()
    {
        var act = new GumbelSoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act26_HierarchicalSoftmax()
    {
        var act = new HierarchicalSoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act27_SphericalSoftmax()
    {
        var act = new SphericalSoftmaxActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act28_Gaussian()
    {
        var act = new GaussianActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act29_SQRBF()
    {
        var act = new SQRBFActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act30_BentIdentity()
    {
        var act = new BentIdentityActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act31_Identity()
    {
        var act = new IdentityActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act32_Sign()
    {
        var act = new SignActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act33_BinarySpiking()
    {
        var act = new BinarySpikingActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act34_ThresholdedReLU()
    {
        var act = new ThresholdedReLUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act35_LiSHT()
    {
        var act = new LiSHTActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act36_ISRU()
    {
        var act = new ISRUActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act37_Maxout()
    {
        var act = new MaxoutActivation<double>();
        return act.Activate(_input);
    }

    [Benchmark]
    public Vector<double> Act38_Squash()
    {
        var act = new SquashActivation<double>();
        return act.Activate(_input);
    }
}
