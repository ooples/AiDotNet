using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a redacted (safe-to-share) summary of an AutoML run.
/// </summary>
/// <remarks>
/// <para>
/// This summary is intended for facade outputs (for example, <c>PredictionModelResult</c>) and avoids exposing
/// hyperparameters or sensitive model details. It provides transparency into the AutoML process while protecting
/// proprietary implementation choices.
/// </para>
/// <para><b>For Beginners:</b> AutoML is an automatic process that tries multiple model attempts ("trials")
/// and keeps the best one. This class tells you how that search went, including:
/// - How many trials ran
/// - The best score found
/// - A per-trial outcome history (without exposing secret settings)
/// </para>
/// </remarks>
public sealed class AutoMLRunSummary
{
    /// <summary>
    /// Gets or sets the search strategy used for the AutoML run, if known.
    /// </summary>
    public AutoMLSearchStrategy? SearchStrategy { get; set; }

    /// <summary>
    /// Gets or sets the time limit used for the AutoML search.
    /// </summary>
    public TimeSpan TimeLimit { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of trials allowed for the AutoML search.
    /// </summary>
    public int TrialLimit { get; set; }

    /// <summary>
    /// Gets or sets the optimization metric used to rank trials.
    /// </summary>
    public MetricType OptimizationMetric { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether higher metric values are better.
    /// </summary>
    public bool MaximizeMetric { get; set; }

    /// <summary>
    /// Gets or sets the best score achieved during the AutoML search.
    /// </summary>
    public double BestScore { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether AutoML selected an ensemble as the final model.
    /// </summary>
    public bool UsedEnsemble { get; set; }

    /// <summary>
    /// Gets or sets the number of models in the selected ensemble, when applicable.
    /// </summary>
    public int? EnsembleSize { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the AutoML search started.
    /// </summary>
    public DateTimeOffset SearchStartedUtc { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the AutoML search ended.
    /// </summary>
    public DateTimeOffset SearchEndedUtc { get; set; }

    /// <summary>
    /// Gets or sets a redacted list of trial summaries.
    /// </summary>
    public List<AutoMLTrialSummary> Trials { get; set; } = new();

    // ============================================================
    // NAS (Neural Architecture Search) specific fields
    // ============================================================

    /// <summary>
    /// Gets or sets NAS-specific result information when a NAS strategy was used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is populated when <see cref="SearchStrategy"/> is one of the NAS strategies:
    /// <see cref="AutoMLSearchStrategy.NeuralArchitectureSearch"/>, <see cref="AutoMLSearchStrategy.DARTS"/>,
    /// <see cref="AutoMLSearchStrategy.GDAS"/>, or <see cref="AutoMLSearchStrategy.OnceForAll"/>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> NAS automatically discovers the best neural network architecture.
    /// This field contains information about the discovered architecture, including its structure
    /// and any hardware-specific optimizations that were applied.
    /// </para>
    /// </remarks>
    public NASResultSummary? NASResult { get; set; }
}

/// <summary>
/// Represents a redacted summary of Neural Architecture Search (NAS) results.
/// </summary>
/// <remarks>
/// <para>
/// This summary captures the discovered architecture and any hardware-aware optimizations
/// without exposing proprietary implementation details.
/// </para>
/// <para>
/// <b>For Beginners:</b> After NAS completes, this tells you what architecture was discovered:
/// - How many layers/nodes were selected
/// - What operations were chosen at each position
/// - Any hardware constraints that were considered
/// </para>
/// </remarks>
public sealed class NASResultSummary
{
    /// <summary>
    /// Gets or sets a human-readable description of the discovered architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Example: "MobileNetV3-like with 12 layers, 3 skip connections, avg pooling"
    /// </para>
    /// </remarks>
    public string? ArchitectureDescription { get; set; }

    /// <summary>
    /// Gets or sets the number of nodes/layers in the discovered architecture.
    /// </summary>
    public int? DiscoveredNodeCount { get; set; }

    /// <summary>
    /// Gets or sets the estimated parameter count of the discovered architecture.
    /// </summary>
    public long? EstimatedParameters { get; set; }

    /// <summary>
    /// Gets or sets the estimated FLOPs (floating-point operations) of the discovered architecture.
    /// </summary>
    public long? EstimatedFLOPs { get; set; }

    /// <summary>
    /// Gets or sets the target hardware platform, if hardware-aware search was used.
    /// </summary>
    public string? TargetPlatform { get; set; }

    /// <summary>
    /// Gets or sets the latency constraint in milliseconds, if specified.
    /// </summary>
    public double? LatencyConstraintMs { get; set; }

    /// <summary>
    /// Gets or sets the memory constraint in megabytes, if specified.
    /// </summary>
    public double? MemoryConstraintMB { get; set; }

    /// <summary>
    /// Gets or sets whether quantization-aware search was enabled.
    /// </summary>
    public bool QuantizationAware { get; set; }

    /// <summary>
    /// Gets or sets a list of operation types selected at each position in the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This provides a high-level view of the discovered architecture without exposing
    /// exact weights or proprietary details.
    /// </para>
    /// <para>
    /// Example: ["conv3x3", "skip", "sep_conv5x5", "max_pool", "conv3x3"]
    /// </para>
    /// </remarks>
    public List<string>? SelectedOperations { get; set; }

    /// <summary>
    /// Gets or sets OnceForAll-specific subnet information when OFA strategy was used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// OnceForAll trains a single "supernet" and then extracts specialized subnets.
    /// This field describes the extracted subnet configuration.
    /// </para>
    /// </remarks>
    public OFASubnetSummary? OFASubnet { get; set; }

    /// <summary>
    /// Gets or sets the number of architecture search epochs/iterations completed.
    /// </summary>
    public int? SearchIterations { get; set; }

    /// <summary>
    /// Gets or sets the final architecture score (validation accuracy or combined metric).
    /// </summary>
    public double? FinalArchitectureScore { get; set; }
}

/// <summary>
/// Represents OnceForAll subnet configuration summary.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OnceForAll (OFA) is a NAS technique that trains one large "supernet"
/// containing many possible architectures. After training, you can extract smaller "subnets"
/// optimized for specific hardware without retraining. This summary describes the extracted subnet.
/// </para>
/// </remarks>
public sealed class OFASubnetSummary
{
    /// <summary>
    /// Gets or sets the depth configuration (number of blocks per stage).
    /// </summary>
    public List<int>? Depths { get; set; }

    /// <summary>
    /// Gets or sets the width multipliers per stage.
    /// </summary>
    public List<double>? Widths { get; set; }

    /// <summary>
    /// Gets or sets the kernel sizes selected per block.
    /// </summary>
    public List<int>? KernelSizes { get; set; }

    /// <summary>
    /// Gets or sets the expansion ratios per block.
    /// </summary>
    public List<double>? ExpansionRatios { get; set; }

    /// <summary>
    /// Gets or sets the estimated latency of the subnet on the target platform.
    /// </summary>
    public double? EstimatedLatencyMs { get; set; }

    /// <summary>
    /// Gets or sets the estimated accuracy of the subnet.
    /// </summary>
    public double? EstimatedAccuracy { get; set; }
}
