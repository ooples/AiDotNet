using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>Which series the decomposition audit actually analyzed.</summary>
public enum DecompositionAnalysisSource
{
    /// <summary>The model's own training target series (re-decomposed with the configured method).</summary>
    TargetSeries,

    /// <summary>The series the configured decomposition instance was originally built on (re-decomposition was not possible).</summary>
    ConfiguredInstance,
}

/// <summary>
/// The time-series decomposition audit produced by a configured decomposition: not just the components, but a
/// grade of how good the decomposition is (trend/seasonal strength, leftover residual structure, additive
/// reconstruction fidelity) and a measure of whether decomposition-based forecasting would actually help.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Most libraries hand back trend/seasonal/residual and stop. This report <b>grades</b> the split: trend and
/// seasonal strength (Wang/Hyndman variance-ratio measures, 0..1), whether the residual still carries
/// autocorrelated structure the decomposition missed, and whether the components add back to the original
/// series. It also forecasts a held-out tail through the decomposition (trend extrapolation + seasonal-naive
/// repeat) versus a random-walk baseline, so the reported skill says whether decomposing the series is worth it.
/// </para>
/// </remarks>
public sealed class TimeSeriesDecompositionReport<T>
{
    /// <summary>The configured decomposition method's type name.</summary>
    public string MethodName { get; init; } = string.Empty;

    /// <summary>The length of the analyzed series.</summary>
    public int SeriesLength { get; init; }

    /// <summary>Which series was analyzed — the model's training target, or the configured instance's own series.</summary>
    public DecompositionAnalysisSource AnalyzedSeries { get; init; }

    /// <summary>The decomposition components that were present and used.</summary>
    public IReadOnlyList<DecompositionComponentType> ComponentsPresent { get; init; } = new List<DecompositionComponentType>();

    /// <summary>
    /// Trend strength F_T in 0..1 (Wang/Hyndman): 1 minus the variance of the residual over the variance of the
    /// detrended-plus-residual. Near 1 means a strong, well-separated trend; near 0 means little trend.
    /// </summary>
    public double TrendStrength { get; init; }

    /// <summary>
    /// Seasonal strength F_S in 0..1 (Wang/Hyndman): 1 minus the variance of the residual over the variance of
    /// the deseasonalized-plus-residual. Near 1 means strong seasonality; near 0 means little or none.
    /// </summary>
    public double SeasonalStrength { get; init; }

    /// <summary>Lag-1 autocorrelation of the residual — near 0 means the residual looks like white noise.</summary>
    public double ResidualAutocorrelation { get; init; }

    /// <summary>
    /// Whether the residual still carries structure the decomposition failed to capture (its lag-1
    /// autocorrelation exceeds the ~2/sqrt(n) white-noise band).
    /// </summary>
    public bool ResidualHasStructure { get; init; }

    /// <summary>Root-mean-square error of the additive reconstruction (trend + seasonal + residual) versus the series.</summary>
    public double ReconstructionError { get; init; }

    /// <summary>Whether an additive reconstruction was computable (trend and residual components were present).</summary>
    public bool ReconstructionAvailable { get; init; }

    /// <summary>The number of tail points held out for the forecasting-value comparison.</summary>
    public int ForecastHorizon { get; init; }

    /// <summary>RMSE of the decomposition-based forecast (trend extrapolation + seasonal-naive) on the held-out tail.</summary>
    public double DecompositionForecastRmse { get; init; }

    /// <summary>RMSE of the random-walk (last-value carry-forward) baseline on the same held-out tail.</summary>
    public double NaiveForecastRmse { get; init; }

    /// <summary>
    /// Forecast skill of decomposition over the naive baseline: 1 minus decomposition RMSE over naive RMSE.
    /// Positive means decomposing the series improves the forecast; negative means it does not.
    /// </summary>
    public double ForecastSkill { get; init; }

    /// <summary>Whether the held-out forecasting comparison was evaluated (enough points and components).</summary>
    public bool ForecastEvaluated { get; init; }
}
