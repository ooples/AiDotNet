using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Diffusion.NoisePredictors;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Benchmarks memory allocation patterns: raw allocation vs TensorAllocator vs new Tensor.
/// Measures GC pressure, allocation count, and throughput.
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0)]
[MemoryDiagnoser]
[RankColumn]
public class MemoryBenchmarks
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    // Immutable inputs — never modified by benchmarks
    private Tensor<double> _inputA = null!;
    private Tensor<double> _inputB = null!;

    // Mutable scratch tensors — reset before each in-place benchmark via IterationSetup
    private Tensor<double> _scratchA = null!;
    private Tensor<double> _dest = null!;

    // Pre-allocated kernel for Conv2D benchmark (not allocated per iteration)
    private Tensor<double> _convKernel = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _inputA = new Tensor<double>([1, 256, 16, 16]);
        _inputB = new Tensor<double>([1, 256, 16, 16]);
        _scratchA = new Tensor<double>([1, 256, 16, 16]);
        _dest = new Tensor<double>([1, 256, 16, 16]);
        _convKernel = new Tensor<double>([256, 256, 3, 3]);
        for (int i = 0; i < _inputA.Length; i++)
        {
            _inputA[i] = rng.NextDouble();
            _inputB[i] = rng.NextDouble();
        }
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Reset scratch tensor before each iteration so in-place ops don't accumulate.
        // AsWritableSpan is internal on Tensor<T>, so use TensorAdd with an implicit
        // zero: actually fastest to materialize through AsSpan() + indexer.
        var srcSpan = _inputA.AsSpan();
        for (int i = 0; i < srcSpan.Length; i++) _scratchA[i] = srcSpan[i];
    }

    [Benchmark(Description = "TensorAdd (allocating)")]
    public Tensor<double> Add_Allocating()
        => _engine.TensorAdd(_inputA, _inputB);

    [Benchmark(Description = "TensorAddInPlace (zero-alloc)")]
    public void Add_InPlace()
        => _engine.TensorAddInPlace(_scratchA, _inputB);

    [Benchmark(Description = "TensorAddInto (zero-alloc, pre-alloc dest)")]
    public void Add_Into()
        => _engine.TensorAddInto(_dest, _inputA, _inputB);

    [Benchmark(Description = "new Tensor<double>([1,256,16,16])")]
    public Tensor<double> Alloc_NewTensor()
        => new([1, 256, 16, 16]);

    [Benchmark(Description = "TensorAllocator.Rent ([1,256,16,16])")]
    public Tensor<double> Alloc_Rent()
    {
        // Rent returns a pre-zeroed tensor from the shared pool. No explicit
        // Return — pool reuse happens via GC + finalizer on Tensors v0.38+.
        return TensorAllocator.Rent<double>([1, 256, 16, 16]);
    }

    [Benchmark(Description = "TensorAllocator.RentUninitialized ([1,256,16,16])")]
    public Tensor<double> Alloc_RentUninitialized()
    {
        // RentUninitialized skips zeroing — faster when the caller will
        // overwrite every element anyway.
        return TensorAllocator.RentUninitialized<double>([1, 256, 16, 16]);
    }

    [Benchmark(Description = "Conv2D 256ch 16x16 (allocating)")]
    public Tensor<double> Conv2D_Allocating()
        => _engine.Conv2D(_inputA, _convKernel, stride: 1, padding: 1);

    [Benchmark(Description = "Sigmoid (allocating)")]
    public Tensor<double> Sigmoid_Allocating()
        => _engine.Sigmoid(_inputA);

    [Benchmark(Description = "SigmoidInPlace (zero-alloc)")]
    public void Sigmoid_InPlace()
        => _engine.SigmoidInPlace(_scratchA);

    [Benchmark(Description = "SigmoidInto (zero-alloc)")]
    public void Sigmoid_Into()
        => _engine.SigmoidInto(_dest, _inputA);

    // =====================================================
    // 50-step Predict loop — measures peak allocation, GC pauses, and
    // total allocation bytes across a full denoising trajectory.
    // Per github.com/ooples/AiDotNet#1015 "Memory benchmark" checklist item.
    // =====================================================

    private UNetNoisePredictor<double>? _predictStepUnet;
    private Tensor<double>? _predictStepInput;

    [GlobalSetup(Target = nameof(UNet_50StepPredictLoop))]
    public void SetupPredictLoop()
    {
        // Small UNet scale — 50 sequential Predict calls on production-sized
        // weights would hit 45-min CI budget. The point of this benchmark is
        // to observe steady-state allocation behavior across the loop, not
        // stress-test throughput.
        _predictStepUnet = new UNetNoisePredictor<double>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 8, seed: 42);
        _predictStepInput = new Tensor<double>([1, 4, 8, 8]);
        for (int i = 0; i < _predictStepInput.Length; i++)
            _predictStepInput[i] = (double)i / _predictStepInput.Length;
    }

    /// <summary>
    /// Runs 50 sequential <c>PredictNoise</c> calls — one per denoising step —
    /// matching the typical DDIM sampler count. BenchmarkDotNet's
    /// <c>[MemoryDiagnoser]</c> tracks peak allocation, GC pauses, and
    /// total allocation bytes.
    /// </summary>
    [Benchmark(Description = "UNet 50-step Predict loop (peak alloc, GC pauses)")]
    public Tensor<double>? UNet_50StepPredictLoop()
    {
        if (_predictStepUnet is null || _predictStepInput is null) return null;
        Tensor<double>? output = null;
        // 50 DDIM steps — timesteps descend linearly from 1000→0 in
        // production; we use a simple int pattern to avoid per-call overhead
        // unrelated to the memory question.
        for (int step = 1000; step > 0; step -= 20)
        {
            output = _predictStepUnet.PredictNoise(_predictStepInput, step, null);
        }
        return output;
    }
}
