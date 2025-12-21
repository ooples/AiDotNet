using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Xunit;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Performance benchmarks comparing JIT compiled vs interpreted graph execution.
/// </summary>
/// <remarks>
/// These benchmarks are quarantined because they trigger GPU initialization which can fail
/// on machines without proper GPU support or drivers.
/// </remarks>
[Trait("Category", "GPU")]
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 5, iterationCount: 20)]
public class JitCompilerBenchmarks
{
    private global::AiDotNet.JitCompiler.JitCompiler? _jit;

    // Simple operations
    private ComputationNode<float>? _simpleGraph;
    private List<ComputationNode<float>>? _simpleInputs;
    private Func<Tensor<float>[], Tensor<float>[]>? _simpleCompiled;
    private Tensor<float>? _simpleData;

    // Linear layer
    private ComputationNode<float>? _linearGraph;
    private List<ComputationNode<float>>? _linearInputs;
    private Func<Tensor<float>[], Tensor<float>[]>? _linearCompiled;
    private Tensor<float>? _linearInput;
    private Tensor<float>? _linearWeights;
    private Tensor<float>? _linearBias;

    // Deep network (10 layers)
    private ComputationNode<float>? _deepGraph;
    private List<ComputationNode<float>>? _deepInputs;
    private Func<Tensor<float>[], Tensor<float>[]>? _deepCompiled;
    private Tensor<float>? _deepInput;
    private List<Tensor<float>>? _deepWeights;
    private List<Tensor<float>>? _deepBiases;

    [GlobalSetup]
    public void Setup()
    {
        _jit = new global::AiDotNet.JitCompiler.JitCompiler();

        SetupSimpleOperations();
        SetupLinearLayer();
        SetupDeepNetwork();
    }

    private void SetupSimpleOperations()
    {
        // Graph: ReLU(Exp(input))
        _simpleData = CreateRandomTensor(new[] { 64, 64 });

        var input = new ComputationNode<float>(_simpleData) { OperationType = OperationType.Input };

        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 64, 64 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Exp
        };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 64, 64 }),
            parents: new List<ComputationNode<float>> { exp })
        {
            OperationType = OperationType.ReLU
        };

        _simpleGraph = relu;
        _simpleInputs = new List<ComputationNode<float>> { input };
        _simpleCompiled = _jit!.Compile(relu, _simpleInputs);
    }

    private void SetupLinearLayer()
    {
        // Graph: ReLU(MatMul(input, weights) + bias)
        _linearInput = CreateRandomTensor(new[] { 32, 128 });
        _linearWeights = CreateRandomTensor(new[] { 128, 256 });
        _linearBias = CreateRandomTensor(new[] { 1, 256 });

        var input = new ComputationNode<float>(_linearInput) { OperationType = OperationType.Input };
        var weights = new ComputationNode<float>(_linearWeights) { OperationType = OperationType.Input };
        var bias = new ComputationNode<float>(_linearBias) { OperationType = OperationType.Input };

        var matmul = new ComputationNode<float>(
            new Tensor<float>(new[] { 32, 256 }),
            parents: new List<ComputationNode<float>> { input, weights })
        {
            OperationType = OperationType.MatMul
        };

        var add = new ComputationNode<float>(
            new Tensor<float>(new[] { 32, 256 }),
            parents: new List<ComputationNode<float>> { matmul, bias })
        {
            OperationType = OperationType.Add
        };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 32, 256 }),
            parents: new List<ComputationNode<float>> { add })
        {
            OperationType = OperationType.ReLU
        };

        _linearGraph = relu;
        _linearInputs = new List<ComputationNode<float>> { input, weights, bias };
        _linearCompiled = _jit!.Compile(relu, _linearInputs);
    }

    private void SetupDeepNetwork()
    {
        // Build a 10-layer network: input -> (Linear + ReLU) x 10 -> output
        const int numLayers = 10;
        const int layerSize = 128;
        const int batchSize = 16;

        _deepInput = CreateRandomTensor(new[] { batchSize, layerSize });
        _deepWeights = new List<Tensor<float>>();
        _deepBiases = new List<Tensor<float>>();

        for (int i = 0; i < numLayers; i++)
        {
            _deepWeights.Add(CreateRandomTensor(new[] { layerSize, layerSize }));
            _deepBiases.Add(CreateRandomTensor(new[] { 1, layerSize }));
        }

        // Build graph
        var input = new ComputationNode<float>(_deepInput) { OperationType = OperationType.Input };
        _deepInputs = new List<ComputationNode<float>> { input };

        var current = input;

        for (int i = 0; i < numLayers; i++)
        {
            var weights = new ComputationNode<float>(_deepWeights[i]) { OperationType = OperationType.Input };
            var bias = new ComputationNode<float>(_deepBiases[i]) { OperationType = OperationType.Input };
            _deepInputs.Add(weights);
            _deepInputs.Add(bias);

            var matmul = new ComputationNode<float>(
                new Tensor<float>(new[] { batchSize, layerSize }),
                parents: new List<ComputationNode<float>> { current, weights })
            {
                OperationType = OperationType.MatMul
            };

            var add = new ComputationNode<float>(
                new Tensor<float>(new[] { batchSize, layerSize }),
                parents: new List<ComputationNode<float>> { matmul, bias })
            {
                OperationType = OperationType.Add
            };

            var relu = new ComputationNode<float>(
                new Tensor<float>(new[] { batchSize, layerSize }),
                parents: new List<ComputationNode<float>> { add })
            {
                OperationType = OperationType.ReLU
            };

            current = relu;
        }

        _deepGraph = current;
        _deepCompiled = _jit!.Compile(current, _deepInputs);
    }

    // ===== Simple Operations Benchmarks =====

    [Benchmark(Description = "Simple ops - JIT Compiled")]
    public Tensor<float>[] SimpleOperationsJIT()
    {
        return _simpleCompiled!(new[] { _simpleData! });
    }

    // Note: Interpreted version would require TensorOperations execution
    // This is a placeholder - actual implementation would execute graph directly

    // ===== Linear Layer Benchmarks =====

    [Benchmark(Description = "Linear layer - JIT Compiled")]
    public Tensor<float>[] LinearLayerJIT()
    {
        return _linearCompiled!(new[] { _linearInput!, _linearWeights!, _linearBias! });
    }

    // ===== Deep Network Benchmarks =====

    [Benchmark(Description = "Deep network (10 layers) - JIT Compiled")]
    public Tensor<float>[] DeepNetworkJIT()
    {
        var inputs = new List<Tensor<float>> { _deepInput! };
        for (int i = 0; i < _deepWeights!.Count; i++)
        {
            inputs.Add(_deepWeights[i]);
            inputs.Add(_deepBiases![i]);
        }
        return _deepCompiled!(inputs.ToArray());
    }

    // ===== Compilation Overhead Benchmark =====

    [Benchmark(Description = "Compilation time (simple graph)")]
    public Func<Tensor<float>[], Tensor<float>[]> CompilationOverhead()
    {
        // Measure pure compilation time
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 8, 8 })) { OperationType = OperationType.Input };
        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 8, 8 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        // Create new compiler instance to avoid caching
        var jit = new global::AiDotNet.JitCompiler.JitCompiler();
        return jit.Compile(relu, new List<ComputationNode<float>> { input });
    }

    [Benchmark(Description = "Compilation with cache hit")]
    public Func<Tensor<float>[], Tensor<float>[]> CachedCompilation()
    {
        // This should hit the cache from Setup
        return _jit!.Compile(_simpleGraph!, _simpleInputs!);
    }

    // ===== Helper Methods =====

    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2.0 - 1.0);  // Range: [-1, 1]
        }

        return tensor;
    }
}

/// <summary>
/// Benchmark runner helper class.
/// To run benchmarks, use: dotnet run --project tests/AiDotNet.Tests --configuration Release
/// Or use BenchmarkSwitcher in a dedicated benchmark host project.
/// </summary>
public class JitCompilerBenchmarkRunner
{
    // Main method removed to avoid entry point conflicts in test projects
    // Use test runner or dedicated benchmark project to execute benchmarks
}
