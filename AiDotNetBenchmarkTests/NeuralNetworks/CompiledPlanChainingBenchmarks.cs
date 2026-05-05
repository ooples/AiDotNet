#if NET8_0_OR_GREATER
using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;
using TorchModule = TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor>;

namespace AiDotNetBenchmarkTests.NeuralNetworks;

// Issue #1157 verification harness — apples-to-apples PyTorch (via
// TorchSharp / libtorch) vs AiDotNet.Tensors comparison for multi-stage
// inference chaining. Establishes the empirical baseline the issue's
// acceptance criteria are measured against.
//
// All benchmarks run on CPU for now (no GPU in CI). GPU equivalents
// belong in a separate file gated on DirectGpuTensorEngine availability.
//
// Methodology:
//   • Same model architecture in both stacks (small dense MLPs as
//     two/three-stage substitutes for VAE encoder→decoder, transfer-
//     learning encoder→classifier, and the 50-step denoising loop).
//   • Same batch size, same input data, same fp32 precision, same CPU.
//   • PyTorch baselines exercise both the eager call-and-return pattern
//     (most permissive) and the nn.Sequential pattern (closest existing
//     analog to "stitching multiple compiled stages into one").
//   • Tensors variants exercise (a) sequential Predict-then-Predict via
//     two CompiledModelHost instances and (b) a hypothetical async-chain
//     counterfactual implemented with Task.Run + ContinueWith — this is
//     the upper bound real ChainAsync could achieve on CPU. Stitched
//     ICompiledPlan.ThenAsync is NOT directly exercised here because
//     CompiledModelHost wraps the cache and doesn't expose the plan;
//     measuring it requires going through the Tensors compilation API
//     directly, which the Tensors-side PR would naturally cover.

[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class CompiledPlanChainingBenchmarks
{
    // Two-stage model dims: a 256-dim "encoder" and a 256→10-class
    // "classifier head". Transfer-learning canonical shape.
    private const int InputDim = 256;
    private const int HiddenDim = 256;
    private const int OutputDim = 10;

    [Params(1, 32, 128)]
    public int BatchSize { get; set; }

    private float[] _inputData = null!;

    // PyTorch (TorchSharp) state
    private TorchTensor _torchInput = null!;
    private TorchModule _torchEncoder = null!;
    private TorchModule _torchClassifier = null!;
    private TorchModule _torchSequential = null!;
    private torch.Device _torchDevice = null!;

    // AiDotNet.Tensors state — raw weights replicated so apples-to-apples.
    private Tensor<float> _aiInput = null!;
    private Tensor<float> _encoderWeights = null!;
    private Tensor<float> _encoderBias = null!;
    private Tensor<float> _classifierWeights = null!;
    private Tensor<float> _classifierBias = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Force CPU on both stacks.
        torch.set_grad_enabled(false);
        _torchDevice = torch.CPU;
        AiDotNetEngine.Current = new CpuEngine();

        // Deterministic input so all variants see identical data.
        _inputData = new float[BatchSize * InputDim];
        var rng = new Random(42);
        for (int i = 0; i < _inputData.Length; i++)
            _inputData[i] = (float)(rng.NextDouble() - 0.5);

        // PyTorch model: Linear(256, 256) -> ReLU -> Linear(256, 10)
        _torchEncoder = torch.nn.Linear(InputDim, HiddenDim, hasBias: true).to(_torchDevice);
        _torchClassifier = torch.nn.Linear(HiddenDim, OutputDim, hasBias: true).to(_torchDevice);
        _torchSequential = torch.nn.Sequential(
            ("encoder", _torchEncoder),
            ("relu", torch.nn.ReLU()),
            ("classifier", _torchClassifier)).to(_torchDevice);
        _torchInput = torch.tensor(_inputData, new long[] { BatchSize, InputDim }, device: _torchDevice);

        // AiDotNet model — same shape, weights initialized to deterministic
        // values (matching PyTorch's default-init isn't load-bearing here
        // because we're measuring throughput, not numerical equivalence).
        _aiInput = new Tensor<float>(_inputData, new[] { BatchSize, InputDim });
        _encoderWeights = MakeRandom(new[] { InputDim, HiddenDim }, seed: 1);
        _encoderBias = MakeRandom(new[] { HiddenDim }, seed: 2);
        _classifierWeights = MakeRandom(new[] { HiddenDim, OutputDim }, seed: 3);
        _classifierBias = MakeRandom(new[] { OutputDim }, seed: 4);
    }

    // --- PyTorch baselines ----------------------------------------------

    /// <summary>
    /// PyTorch reference path: two separate Module calls with eager
    /// dispatch. Each call goes through ATen's full op-dispatch stack
    /// (Python-equivalent overhead even via TorchSharp because the
    /// underlying libtorch dispatch is the same). Closest analog to
    /// AiDotNet's "two CompiledModelHost.Predict calls in sequence".
    /// </summary>
    [Benchmark(Baseline = true, Description = "PyTorch eager: encoder() + classifier()")]
    public TorchTensor PyTorch_TwoStageSequential()
    {
        var hidden = _torchEncoder.forward(_torchInput);
        hidden = torch.nn.functional.relu(hidden);
        return _torchClassifier.forward(hidden);
    }

    /// <summary>
    /// PyTorch nn.Sequential: closest existing analog to stitching multiple
    /// stages into one fused forward pass. Should win over the eager
    /// two-stage path because the dispatch overhead is amortized over
    /// fewer Module-boundary calls. This is the bar Tensors' chained
    /// path must beat.
    /// </summary>
    [Benchmark(Description = "PyTorch nn.Sequential (single forward)")]
    public TorchTensor PyTorch_Sequential()
        => _torchSequential.forward(_torchInput);

    // --- AiDotNet.Tensors variants --------------------------------------

    /// <summary>
    /// AiDotNet sequential — two separate engine ops with intermediate
    /// Tensor materialization. Mirrors what AiDotNet does today when a
    /// model holds two sub-models and calls Predict on each in sequence
    /// (e.g., the SDXL conditioner -> unet path).
    /// </summary>
    [Benchmark(Description = "Tensors sequential: 2 ops, materialized intermediate")]
    public Tensor<float> Tensors_TwoStageSequential()
    {
        // Stage 1: hidden = relu(input @ encoderW + encoderB)
        var preact1 = AiDotNetEngine.Current.TensorMatMul(_aiInput, _encoderWeights);
        var hidden = AiDotNetEngine.Current.TensorBroadcastAdd(preact1, _encoderBias);
        hidden = AiDotNetEngine.Current.ReLU(hidden);

        // Stage 2: out = hidden @ classW + classB
        var preact2 = AiDotNetEngine.Current.TensorMatMul(hidden, _classifierWeights);
        return AiDotNetEngine.Current.TensorBroadcastAdd(preact2, _classifierBias);
    }

    /// <summary>
    /// AiDotNet "async counterfactual" — what real ChainAsync COULD
    /// deliver on CPU. Uses Task.Run to put stage 2 on a different thread
    /// while stage 1's output is still being written. On a single-batch
    /// inference call this can't help (stage 2 must wait for stage 1's
    /// final byte). The win shows up in BATCHED throughput where the
    /// next batch's stage 1 can start while the current batch's stage 2
    /// is still running. This benchmark measures the per-batch baseline;
    /// a separate StreamThroughputBenchmark below measures the overlap.
    /// </summary>
    [Benchmark(Description = "Tensors async-Task counterfactual (single batch)")]
    public async Task<Tensor<float>> Tensors_AsyncCounterfactual()
    {
        // Stage 1 awaited — there's no sense in starting stage 2 before
        // stage 1's output is done; this measures the per-batch latency
        // overhead of going through the Task scheduler.
        var hidden = await Task.Run(() =>
        {
            var preact1 = AiDotNetEngine.Current.TensorMatMul(_aiInput, _encoderWeights);
            var h = AiDotNetEngine.Current.TensorBroadcastAdd(preact1, _encoderBias);
            return AiDotNetEngine.Current.ReLU(h);
        }).ConfigureAwait(false);

        return await Task.Run(() =>
        {
            var preact2 = AiDotNetEngine.Current.TensorMatMul(hidden, _classifierWeights);
            return AiDotNetEngine.Current.TensorBroadcastAdd(preact2, _classifierBias);
        }).ConfigureAwait(false);
    }

    private static Tensor<float> MakeRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        return new Tensor<float>(data, shape);
    }
}

/// <summary>
/// Streaming-throughput benchmark — the workload where REAL async chaining
/// (which AiDotNet.Tensors does NOT have today; ICompiledPlan.ThenAsync is
/// synchronous despite its name) WOULD show its win. Feeds N batches in
/// rapid succession and measures total wall time vs the time-of-N-sync-batches.
/// On a multi-core CPU with disjoint kernel resources, stage 2 of batch K
/// CAN run on a different core while stage 1 of batch K+1 is starting —
/// IF the API is async. With the current sync API the host blocks on stage
/// 2 before queuing stage 1 of K+1, leaving cores idle.
///
/// PyTorch's torch.compile + CUDA streams achieves this overlap on GPU;
/// PyTorch's CPU async via TBB / oneDNN threadpool achieves it on CPU.
/// Tensors needs an async ChainAsync to compete.
/// </summary>
[SimpleJob(RuntimeMoniker.Net80, warmupCount: 3, iterationCount: 5)]
[MemoryDiagnoser]
public class StreamThroughputBenchmark
{
    private const int InputDim = 256;
    private const int HiddenDim = 256;
    private const int OutputDim = 10;

    [Params(8, 32)]
    public int NumBatches { get; set; }

    [Params(32)]
    public int BatchSize { get; set; }

    private TorchTensor[] _torchInputs = null!;
    private TorchModule _torchSequential = null!;
    private torch.Device _torchDevice = null!;

    private Tensor<float>[] _aiInputs = null!;
    private Tensor<float> _encoderWeights = null!;
    private Tensor<float> _encoderBias = null!;
    private Tensor<float> _classifierWeights = null!;
    private Tensor<float> _classifierBias = null!;

    [GlobalSetup]
    public void Setup()
    {
        torch.set_grad_enabled(false);
        _torchDevice = torch.CPU;
        AiDotNetEngine.Current = new CpuEngine();

        var rng = new Random(42);
        _torchInputs = new TorchTensor[NumBatches];
        _aiInputs = new Tensor<float>[NumBatches];
        for (int b = 0; b < NumBatches; b++)
        {
            var data = new float[BatchSize * InputDim];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
            _torchInputs[b] = torch.tensor(data, new long[] { BatchSize, InputDim }, device: _torchDevice);
            _aiInputs[b] = new Tensor<float>((float[])data.Clone(), new[] { BatchSize, InputDim });
        }

        _torchSequential = torch.nn.Sequential(
            ("encoder", torch.nn.Linear(InputDim, HiddenDim)),
            ("relu", torch.nn.ReLU()),
            ("classifier", torch.nn.Linear(HiddenDim, OutputDim))).to(_torchDevice);

        _encoderWeights = MakeRandom(new[] { InputDim, HiddenDim }, 1);
        _encoderBias = MakeRandom(new[] { HiddenDim }, 2);
        _classifierWeights = MakeRandom(new[] { HiddenDim, OutputDim }, 3);
        _classifierBias = MakeRandom(new[] { OutputDim }, 4);
    }

    [Benchmark(Baseline = true, Description = "PyTorch nn.Sequential — N sequential batches")]
    public void PyTorch_BatchSweep_Sequential()
    {
        for (int b = 0; b < NumBatches; b++)
            _ = _torchSequential.forward(_torchInputs[b]);
    }

    [Benchmark(Description = "Tensors sequential — N sequential batches (current)")]
    public void Tensors_BatchSweep_Sequential()
    {
        for (int b = 0; b < NumBatches; b++)
        {
            var preact1 = AiDotNetEngine.Current.TensorMatMul(_aiInputs[b], _encoderWeights);
            var hidden = AiDotNetEngine.Current.TensorBroadcastAdd(preact1, _encoderBias);
            hidden = AiDotNetEngine.Current.ReLU(hidden);
            var preact2 = AiDotNetEngine.Current.TensorMatMul(hidden, _classifierWeights);
            _ = AiDotNetEngine.Current.TensorBroadcastAdd(preact2, _classifierBias);
        }
    }

    /// <summary>
    /// Async-pipelined counterfactual: schedule batch K's stage 2 immediately
    /// after stage 1 finishes WITHOUT waiting on the host thread, so batch
    /// K+1's stage 1 can start in parallel. If any speedup shows up here over
    /// the sequential variant, that's the perf cell ICompiledPlan.ChainAsync
    /// must capture by being actually async (returning ValueTask&lt;Tensor&lt;T&gt;&gt;
    /// instead of synchronous ICompiledPlan&lt;T&gt;).
    /// </summary>
    [Benchmark(Description = "Tensors async-pipelined counterfactual — N batches with overlap")]
    public async Task Tensors_BatchSweep_PipelinedAsync()
    {
        var stage2Tasks = new Task[NumBatches];
        for (int b = 0; b < NumBatches; b++)
        {
            var inputCopy = _aiInputs[b];

            // Stage 1 on a thread, returns hidden.
            var hiddenTask = Task.Run(() =>
            {
                var preact1 = AiDotNetEngine.Current.TensorMatMul(inputCopy, _encoderWeights);
                var h = AiDotNetEngine.Current.TensorBroadcastAdd(preact1, _encoderBias);
                return AiDotNetEngine.Current.ReLU(h);
            });

            // Stage 2 starts as soon as Stage 1 hands off, freeing the
            // host loop to launch the next batch's Stage 1 immediately.
            stage2Tasks[b] = hiddenTask.ContinueWith(t =>
            {
                var hidden = t.Result;
                var preact2 = AiDotNetEngine.Current.TensorMatMul(hidden, _classifierWeights);
                _ = AiDotNetEngine.Current.TensorBroadcastAdd(preact2, _classifierBias);
            }, TaskContinuationOptions.ExecuteSynchronously);
        }
        await Task.WhenAll(stage2Tasks).ConfigureAwait(false);
    }

    private static Tensor<float> MakeRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        return new Tensor<float>(data, shape);
    }
}
#endif
