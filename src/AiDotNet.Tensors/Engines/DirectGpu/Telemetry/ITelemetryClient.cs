// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.Telemetry;

/// <summary>
/// Interface for GPU telemetry submission and profile retrieval.
/// </summary>
public interface ITelemetryClient : IDisposable
{
    /// <summary>
    /// Gets whether telemetry is enabled.
    /// </summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Submits GPU A/B test results to the telemetry service.
    /// </summary>
    /// <param name="result">The tuning result to submit.</param>
    /// <param name="gpuInfo">GPU information.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SubmitTuningResultAsync(
        TuningResultData result,
        GpuInfoData gpuInfo,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieves the optimal GPU profile for the specified GPU and matrix size.
    /// </summary>
    /// <param name="gpuInfo">GPU information.</param>
    /// <param name="minDimension">Minimum matrix dimension.</param>
    /// <param name="maxDimension">Maximum matrix dimension.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The optimal config, or null if not found.</returns>
    Task<GpuProfileData?> GetProfileAsync(
        GpuInfoData gpuInfo,
        int minDimension,
        int maxDimension,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Submits an exception to the telemetry service.
    /// </summary>
    /// <param name="exception">The exception data to submit.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task SubmitExceptionAsync(
        ExceptionData exception,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Opts out of telemetry for this client.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task OptOutAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Checks if this client has opted out of telemetry.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    Task<bool> IsOptedOutAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// GPU tuning result data for telemetry submission.
/// </summary>
public sealed class TuningResultData
{
    public int MatrixM { get; init; }
    public int MatrixN { get; init; }
    public int MatrixK { get; init; }
    public string ConfigJson { get; init; } = string.Empty;
    public double MeasuredGflops { get; init; }
    public double EfficiencyPercent { get; init; }
}

/// <summary>
/// GPU information for telemetry.
/// </summary>
public sealed class GpuInfoData
{
    public string Vendor { get; init; } = string.Empty;
    public string Model { get; init; } = string.Empty;
    public string? Architecture { get; init; }
    public string? DriverVersion { get; init; }
    public string OsPlatform { get; init; } = string.Empty;
}

/// <summary>
/// GPU profile data retrieved from telemetry service.
/// </summary>
public sealed class GpuProfileData
{
    public string ConfigJson { get; init; } = string.Empty;
    public double MeasuredGflops { get; init; }
    public double EfficiencyPercent { get; init; }
    public int SampleCount { get; init; }
}

/// <summary>
/// Exception data for telemetry submission.
/// </summary>
public sealed class ExceptionData
{
    public string ExceptionType { get; init; } = string.Empty;
    public string? ExceptionMessage { get; init; }
    public string? StackTrace { get; init; }
    public string? InnerExceptionType { get; init; }
    public string? InnerExceptionMessage { get; init; }
    public string Component { get; init; } = string.Empty;
    public string? Operation { get; init; }
    public string AidotnetVersion { get; init; } = string.Empty;
    public string? DotnetVersion { get; init; }
    public string? OsPlatform { get; init; }
    public string? OsVersion { get; init; }
    public string? GpuVendor { get; init; }
    public string? GpuModel { get; init; }
    public string? AdditionalContextJson { get; init; }
}
