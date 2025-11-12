using AiDotNet.LoRA;
using AiDotNet.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks for LoRA (Low-Rank Adaptation) techniques
/// Tests LoRA layer performance for efficient fine-tuning
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net462, baseline: true)]
[SimpleJob(RuntimeMoniker.Net60)]
[SimpleJob(RuntimeMoniker.Net70)]
[SimpleJob(RuntimeMoniker.Net80)]
public class LoRABenchmarks
{
    [Params(64, 256)]
    public int InputSize { get; set; }

    [Params(64, 256)]
    public int OutputSize { get; set; }

    [Params(4, 8)]
    public int Rank { get; set; }

    private Tensor<double> _input = null!;
    private LoRALayer<double> _loraLayer = null!;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        int batchSize = 32;

        // Initialize input tensor
        _input = new Tensor<double>(new[] { batchSize, InputSize });
        for (int i = 0; i < _input.Length; i++)
        {
            _input[i] = random.NextDouble() * 2 - 1;
        }

        // Initialize LoRA layer
        _loraLayer = new LoRALayer<double>(
            inputSize: InputSize,
            outputSize: OutputSize,
            rank: Rank,
            alpha: 1.0,
            dropout: 0.1
        );
    }

    [Benchmark(Baseline = true)]
    public LoRALayer<double> LoRA_CreateLayer()
    {
        return new LoRALayer<double>(
            inputSize: InputSize,
            outputSize: OutputSize,
            rank: Rank,
            alpha: 1.0,
            dropout: 0.1
        );
    }

    [Benchmark]
    public Tensor<double> LoRA_Forward()
    {
        return _loraLayer.Forward(_input);
    }

    [Benchmark]
    public DefaultLoRAConfiguration LoRA_CreateConfiguration()
    {
        var config = new DefaultLoRAConfiguration
        {
            Rank = Rank,
            Alpha = 1.0,
            Dropout = 0.1,
            TargetModules = new[] { "query", "key", "value" },
            MergeWeights = false
        };
        return config;
    }

    [Benchmark]
    public int LoRA_CalculateParameterReduction()
    {
        // Standard layer parameters: InputSize * OutputSize
        int standardParams = InputSize * OutputSize;

        // LoRA parameters: (InputSize * Rank) + (Rank * OutputSize)
        int loraParams = (InputSize * Rank) + (Rank * OutputSize);

        return standardParams - loraParams;
    }
}
