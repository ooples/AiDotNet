using AiDotNet.Regularization;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for Regularization techniques
/// Tests L1, L2, ElasticNet regularization performance
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class RegularizationBenchmarks
{
    [Params(1000, 5000)]
    public int ParameterSize { get; set; }

    private Vector<double> _weights = null!;
    private Matrix<double> _weightsMatrix = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _weights = new Vector<double>(ParameterSize);
        _weightsMatrix = new Matrix<double>(ParameterSize / 10, 10);

        for (int i = 0; i < ParameterSize; i++)
        {
            _weights[i] = random.NextDouble() * 2 - 1;
        }

        for (int i = 0; i < _weightsMatrix.Rows; i++)
        {
            for (int j = 0; j < _weightsMatrix.Columns; j++)
            {
                _weightsMatrix[i, j] = random.NextDouble() * 2 - 1;
            }
        }
    }

    [Benchmark(Baseline = true)]
    public double L1Regularization_ComputePenalty()
    {
        var reg = new L1Regularization<double>(lambda: 0.01);
        return reg.ComputePenalty(_weights);
    }

    [Benchmark]
    public Vector<double> L1Regularization_ComputeGradient()
    {
        var reg = new L1Regularization<double>(lambda: 0.01);
        return reg.ComputeGradient(_weights);
    }

    [Benchmark]
    public Vector<double> L1Regularization_ApplyProximalOperator()
    {
        var reg = new L1Regularization<double>(lambda: 0.01);
        return reg.ApplyProximalOperator(_weights, stepSize: 0.1);
    }

    [Benchmark]
    public double L2Regularization_ComputePenalty()
    {
        var reg = new L2Regularization<double>(lambda: 0.01);
        return reg.ComputePenalty(_weights);
    }

    [Benchmark]
    public Vector<double> L2Regularization_ComputeGradient()
    {
        var reg = new L2Regularization<double>(lambda: 0.01);
        return reg.ComputeGradient(_weights);
    }

    [Benchmark]
    public Vector<double> L2Regularization_ApplyProximalOperator()
    {
        var reg = new L2Regularization<double>(lambda: 0.01);
        return reg.ApplyProximalOperator(_weights, stepSize: 0.1);
    }

    [Benchmark]
    public double ElasticRegularization_ComputePenalty()
    {
        var reg = new ElasticRegularization<double>(lambda: 0.01, l1Ratio: 0.5);
        return reg.ComputePenalty(_weights);
    }

    [Benchmark]
    public Vector<double> ElasticRegularization_ComputeGradient()
    {
        var reg = new ElasticRegularization<double>(lambda: 0.01, l1Ratio: 0.5);
        return reg.ComputeGradient(_weights);
    }

    [Benchmark]
    public Vector<double> ElasticRegularization_ApplyProximalOperator()
    {
        var reg = new ElasticRegularization<double>(lambda: 0.01, l1Ratio: 0.5);
        return reg.ApplyProximalOperator(_weights, stepSize: 0.1);
    }

    [Benchmark]
    public double NoRegularization_ComputePenalty()
    {
        var reg = new NoRegularization<double>();
        return reg.ComputePenalty(_weights);
    }
}
