using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.IR.HighLevel;
using AiDotNet.InferenceOptimization.IR.LowLevel;

namespace AiDotNet.InferenceOptimization.IR;

/// <summary>
/// Interface for the two-tier IR compiler pipeline.
/// </summary>
/// <remarks>
/// <para>
/// The IR compiler transforms models through multiple stages:
/// 1. Model → HLIR: Convert model to high-level IR
/// 2. HLIR Optimization: Apply high-level optimizations
/// 3. HLIR → LLIR: Lower to low-level IR
/// 4. LLIR Optimization: Apply low-level optimizations
/// 5. Code Generation: Generate executable code
/// </para>
/// </remarks>
public interface IIRCompiler<T> where T : struct
{
    /// <summary>
    /// Compiles a model to optimized LLIR.
    /// </summary>
    LLIRGraph Compile(HLIRGraph<T> hlir);

    /// <summary>
    /// Gets the compilation options.
    /// </summary>
    IRCompilerOptions Options { get; }

    /// <summary>
    /// Gets compilation statistics from the last compilation.
    /// </summary>
    IRCompilationStats? LastCompilationStats { get; }
}

/// <summary>
/// Options for IR compilation.
/// </summary>
public class IRCompilerOptions
{
    /// <summary>
    /// Target device for execution.
    /// </summary>
    public DeviceType TargetDevice { get; set; } = DeviceType.CPU;

    /// <summary>
    /// Device configuration.
    /// </summary>
    public DeviceConfiguration DeviceConfig { get; set; } = new();

    /// <summary>
    /// Optimization level.
    /// </summary>
    public IROptimizationLevel OptimizationLevel { get; set; } = IROptimizationLevel.O2;

    /// <summary>
    /// Target data type.
    /// </summary>
    public IRDataType TargetDataType { get; set; } = IRDataType.Float32;

    /// <summary>
    /// Whether to enable fusion optimizations.
    /// </summary>
    public bool EnableFusion { get; set; } = true;

    /// <summary>
    /// Whether to enable constant folding.
    /// </summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>
    /// Whether to enable dead code elimination.
    /// </summary>
    public bool EnableDeadCodeElimination { get; set; } = true;

    /// <summary>
    /// Whether to enable memory optimization.
    /// </summary>
    public bool EnableMemoryOptimization { get; set; } = true;

    /// <summary>
    /// Whether to enable auto-scheduling.
    /// </summary>
    public bool EnableAutoScheduling { get; set; } = true;

    /// <summary>
    /// Whether to enable auto-tuning.
    /// </summary>
    public bool EnableAutoTuning { get; set; } = false;

    /// <summary>
    /// Whether to preserve debug information.
    /// </summary>
    public bool PreserveDebugInfo { get; set; } = false;

    /// <summary>
    /// Maximum number of fusion candidates to explore.
    /// </summary>
    public int MaxFusionCandidates { get; set; } = 100;

    /// <summary>
    /// Memory limit for optimization (in bytes).
    /// </summary>
    public long MemoryLimitBytes { get; set; } = long.MaxValue;
}

/// <summary>
/// Optimization level for IR compilation.
/// </summary>
public enum IROptimizationLevel
{
    /// <summary>
    /// No optimization, fastest compilation.
    /// </summary>
    O0,

    /// <summary>
    /// Basic optimizations (constant folding, DCE).
    /// </summary>
    O1,

    /// <summary>
    /// Standard optimizations (+ fusion, algebraic simplification).
    /// </summary>
    O2,

    /// <summary>
    /// Aggressive optimizations (+ memory optimization, auto-scheduling).
    /// </summary>
    O3,

    /// <summary>
    /// Size optimization.
    /// </summary>
    Os
}

/// <summary>
/// Statistics from IR compilation.
/// </summary>
public class IRCompilationStats
{
    public TimeSpan TotalTime { get; set; }
    public TimeSpan HLIROptimizationTime { get; set; }
    public TimeSpan LoweringTime { get; set; }
    public TimeSpan LLIROptimizationTime { get; set; }

    public int OriginalNodeCount { get; set; }
    public int OptimizedHLIRNodeCount { get; set; }
    public int FinalLLIROpCount { get; set; }

    public int FusionsApplied { get; set; }
    public int ConstantsFolded { get; set; }
    public int DeadNodesEliminated { get; set; }

    public long OriginalMemoryEstimate { get; set; }
    public long OptimizedMemoryEstimate { get; set; }
    public double MemoryReductionPercent =>
        OriginalMemoryEstimate > 0
            ? (1 - (double)OptimizedMemoryEstimate / OriginalMemoryEstimate) * 100
            : 0;

    public override string ToString()
    {
        return $"Compiled in {TotalTime.TotalMilliseconds:F1}ms: " +
               $"{OriginalNodeCount} → {FinalLLIROpCount} ops, " +
               $"{FusionsApplied} fusions, {MemoryReductionPercent:F1}% memory reduction";
    }
}
