using AiDotNet.Memory;
using AiDotNet.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Toolchains.InProcess.Emit;

namespace AiDotNetBenchmarkTests.BenchmarkTests;

/// <summary>
/// Benchmarks comparing tensor allocation strategies for RWKV7Block-like forward passes.
/// Measures: new Tensor (baseline) vs ForwardArena vs LayerWorkspace (target).
/// </summary>
[MemoryDiagnoser]
[Config(typeof(InProcessConfig))]
public class ArenaAllocationBenchmarks
{
    private class InProcessConfig : ManualConfig
    {
        public InProcessConfig()
        {
            AddJob(BenchmarkDotNet.Jobs.Job.ShortRun
                .WithToolchain(InProcessEmitToolchain.Instance));
        }
    }

    private const int BatchSize = 1;
    private const int SeqLen = 32;
    private const int ModelDim = 64;
    private const int TimestepsPerForward = 32;
    private const int TensorsPerTimestep = 7;

    // Buffer indices (matching RWKV7Block pattern)
    private const int TsRInput = 0, TsKInput = 1, TsVInput = 2;
    private const int TsAInput = 3, TsBInput = 4, TsWkvOut = 5, TsYt = 6;
    private const int SqAllR = 0, SqAllK = 1, SqAllV = 2, SqAllA = 3;
    private const int SqAllB = 4, SqAllWkv = 5, SqAllWkvPre = 6, SqAllWkvGated = 7;

    private ForwardArena<float> _arena = null!;
    private int[] _timestepShape = null!;
    private int[] _sequenceShape = null!;
    private LayerWorkspace<float> _workspace = null!;

    [GlobalSetup]
    public void Setup()
    {
        _timestepShape = [BatchSize, ModelDim];
        _sequenceShape = [BatchSize, SeqLen, ModelDim];

        _arena = new ForwardArena<float>();
        _arena.EnsureCapacity(_timestepShape, TensorsPerTimestep);
        _arena.EnsureCapacity(_sequenceShape, 8);

        _workspace = new LayerWorkspace<float>(timestepCount: 7, sequenceCount: 8);
        _workspace.DeclareTimestep(TsRInput, ModelDim);
        _workspace.DeclareTimestep(TsKInput, ModelDim);
        _workspace.DeclareTimestep(TsVInput, ModelDim);
        _workspace.DeclareTimestep(TsAInput, ModelDim);
        _workspace.DeclareTimestep(TsBInput, ModelDim);
        _workspace.DeclareTimestep(TsWkvOut, ModelDim);
        _workspace.DeclareTimestep(TsYt, ModelDim);
        _workspace.DeclareSequence(SqAllR, ModelDim);
        _workspace.DeclareSequence(SqAllK, ModelDim);
        _workspace.DeclareSequence(SqAllV, ModelDim);
        _workspace.DeclareSequence(SqAllA, ModelDim);
        _workspace.DeclareSequence(SqAllB, ModelDim);
        _workspace.DeclareSequence(SqAllWkv, ModelDim);
        _workspace.DeclareSequence(SqAllWkvPre, ModelDim);
        _workspace.DeclareSequence(SqAllWkvGated, ModelDim);
        _workspace.BeginForward(BatchSize, SeqLen);
    }

    /// <summary>
    /// Baseline: raw new Tensor allocation (current RWKV7Block pattern).
    /// Creates 8 sequence + 7×128 timestep tensors = 904 allocations per forward pass.
    /// </summary>
    [Benchmark(Baseline = true)]
    public int RawAllocation_RWKV7Pattern()
    {
        int count = 0;
        for (int i = 0; i < 8; i++)
        {
            var t = new Tensor<float>(_sequenceShape);
            count += t.Length;
        }
        for (int step = 0; step < TimestepsPerForward; step++)
        {
            for (int i = 0; i < TensorsPerTimestep; i++)
            {
                var t = new Tensor<float>(_timestepShape);
                count += t.Length;
            }
        }
        return count;
    }

    /// <summary>
    /// Arena: bump-pointer allocation.
    /// </summary>
    [Benchmark]
    public int Arena_RWKV7Pattern()
    {
        int count = 0;
        _arena.Reset();
        for (int i = 0; i < 8; i++)
            count += _arena.Rent(_sequenceShape).Length;
        for (int step = 0; step < TimestepsPerForward; step++)
        {
            _arena.Reset();
            for (int i = 0; i < TensorsPerTimestep; i++)
                count += _arena.Rent(_timestepShape).Length;
        }
        return count;
    }

    /// <summary>
    /// LayerWorkspace: index-based pre-allocated buffers (production target).
    /// Zero allocation — same tensors returned every call.
    /// </summary>
    [Benchmark]
    public int Workspace_RWKV7Pattern()
    {
        int count = 0;
        _workspace.BeginForward(BatchSize, SeqLen); // Include sizing check in measurement
        // Sequence buffers (pre-allocated, same tensor every call)
        count += _workspace.Sequence(SqAllR).Length;
        count += _workspace.Sequence(SqAllK).Length;
        count += _workspace.Sequence(SqAllV).Length;
        count += _workspace.Sequence(SqAllA).Length;
        count += _workspace.Sequence(SqAllB).Length;
        count += _workspace.Sequence(SqAllWkv).Length;
        count += _workspace.Sequence(SqAllWkvPre).Length;
        count += _workspace.Sequence(SqAllWkvGated).Length;

        // Timestep buffers (same tensor reused each iteration)
        for (int step = 0; step < TimestepsPerForward; step++)
        {
            count += _workspace.Timestep(TsRInput).Length;
            count += _workspace.Timestep(TsKInput).Length;
            count += _workspace.Timestep(TsVInput).Length;
            count += _workspace.Timestep(TsAInput).Length;
            count += _workspace.Timestep(TsBInput).Length;
            count += _workspace.Timestep(TsWkvOut).Length;
            count += _workspace.Timestep(TsYt).Length;
        }
        return count;
    }

    /// <summary>
    /// Micro-benchmark: single new Tensor cost.
    /// </summary>
    [Benchmark]
    public Tensor<float> Single_NewTensor()
    {
        return new Tensor<float>(_timestepShape);
    }

    /// <summary>
    /// Micro-benchmark: single arena rent cost.
    /// </summary>
    [Benchmark]
    public Tensor<float> Single_ArenaRent()
    {
        _arena.Reset();
        return _arena.Rent(_timestepShape);
    }

    /// <summary>
    /// Micro-benchmark: single workspace lookup cost.
    /// </summary>
    [Benchmark]
    public Tensor<float> Single_WorkspaceLookup()
    {
        return _workspace.Timestep(TsRInput);
    }
}
