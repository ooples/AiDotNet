namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for influence analysis in regression models.
/// </summary>
/// <remarks>
/// <para>
/// Influence analysis identifies data points that have outsized impact on model predictions
/// or parameter estimates. These influential points may be outliers, errors, or important cases.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some data points have more influence on the model than others:
/// <list type="bullet">
/// <item><b>High leverage:</b> Point has unusual X values (far from center)</item>
/// <item><b>High influence:</b> Removing this point significantly changes the model</item>
/// </list>
/// A point can be high leverage without being influential (if it fits the pattern).
/// Influential points deserve extra scrutiny - they might be errors or key insights.
/// </para>
/// </remarks>
public class InfluenceAnalysisOptions
{
    /// <summary>
    /// Whether to compute Cook's distance. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Cook's distance measures how much all predictions change
    /// when a point is removed. Values > 1 (or > 4/n) are traditionally flagged as influential.</para>
    /// </remarks>
    public bool? ComputeCooksDistance { get; set; }

    /// <summary>
    /// Cook's distance threshold for flagging. Default: null (auto: 4/n or 1).
    /// </summary>
    public double? CooksDistanceThreshold { get; set; }

    /// <summary>
    /// Whether to compute leverage (hat values). Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Leverage measures how unusual a point's X values are.
    /// High leverage means the point could potentially have high influence.</para>
    /// </remarks>
    public bool? ComputeLeverage { get; set; }

    /// <summary>
    /// Leverage threshold for flagging. Default: null (auto: 2*(p+1)/n).
    /// </summary>
    public double? LeverageThreshold { get; set; }

    /// <summary>
    /// Whether to compute DFFITS. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DFFITS measures how much the fitted value for a point
    /// changes when that point is excluded. Large values indicate influence on its own prediction.</para>
    /// </remarks>
    public bool? ComputeDFFITS { get; set; }

    /// <summary>
    /// DFFITS threshold for flagging. Default: null (auto: 2*sqrt((p+1)/n)).
    /// </summary>
    public double? DFFITSThreshold { get; set; }

    /// <summary>
    /// Whether to compute DFBETAS. Default: false (expensive).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DFBETAS measures how much each coefficient changes when
    /// a point is removed. Large values indicate influence on specific coefficients.</para>
    /// </remarks>
    public bool? ComputeDFBETAS { get; set; }

    /// <summary>
    /// DFBETAS threshold for flagging. Default: null (auto: 2/sqrt(n)).
    /// </summary>
    public double? DFBETASThreshold { get; set; }

    /// <summary>
    /// Specific coefficients to compute DFBETAS for. Default: null (all).
    /// </summary>
    public int[]? DFBETASCoefficients { get; set; }

    /// <summary>
    /// Whether to compute covariance ratio. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Covariance ratio measures how much the parameter
    /// covariance matrix changes when a point is removed. Indicates precision impact.</para>
    /// </remarks>
    public bool? ComputeCovarianceRatio { get; set; }

    /// <summary>
    /// Whether to compute Welsch-Kuh distance. Default: false.
    /// </summary>
    public bool? ComputeWelschKuhDistance { get; set; }

    /// <summary>
    /// Whether to compute Hadi's influence measure. Default: false.
    /// </summary>
    public bool? ComputeHadisMeasure { get; set; }

    /// <summary>
    /// Whether to generate influence plot data. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Influence plot typically shows studentized residuals
    /// vs leverage, with bubble size representing Cook's distance.</para>
    /// </remarks>
    public bool? GenerateInfluencePlot { get; set; }

    /// <summary>
    /// Whether to generate added variable plots. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Added variable plots (partial regression plots) show the
    /// relationship between Y and each X after removing the effect of other variables.</para>
    /// </remarks>
    public bool? GenerateAddedVariablePlots { get; set; }

    /// <summary>
    /// Features for added variable plots. Default: null (all).
    /// </summary>
    public int[]? AddedVariableFeatures { get; set; }

    /// <summary>
    /// Whether to generate component-plus-residual plots. Default: false.
    /// </summary>
    public bool? GenerateCPRPlots { get; set; }

    /// <summary>
    /// Whether to identify influential points. Default: true.
    /// </summary>
    public bool? IdentifyInfluentialPoints { get; set; }

    /// <summary>
    /// Maximum number of influential points to report. Default: 20.
    /// </summary>
    public int? MaxInfluentialPointsToReport { get; set; }

    /// <summary>
    /// Whether to report influence on specific predictions. Default: false.
    /// </summary>
    public bool? ReportPredictionInfluence { get; set; }

    /// <summary>
    /// Prediction indices to analyze influence for. Default: null.
    /// </summary>
    public int[]? PredictionIndicesToAnalyze { get; set; }

    /// <summary>
    /// Whether to compute approximate leave-one-out impact. Default: true.
    /// </summary>
    public bool? ComputeApproximateLOOImpact { get; set; }

    /// <summary>
    /// Whether to compute exact leave-one-out (refit for each point). Default: false (expensive).
    /// </summary>
    public bool? ComputeExactLOOImpact { get; set; }

    /// <summary>
    /// Subset of points for exact LOO (indices). Default: null (all influential points).
    /// </summary>
    public int[]? ExactLOOSubset { get; set; }

    /// <summary>
    /// Whether to include recommendations for handling influential points. Default: true.
    /// </summary>
    public bool? IncludeRecommendations { get; set; }
}
