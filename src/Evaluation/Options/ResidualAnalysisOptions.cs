namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for residual analysis in regression models.
/// </summary>
/// <remarks>
/// <para>
/// Residual analysis examines the differences between predicted and actual values to
/// diagnose model problems like heteroscedasticity, non-normality, and autocorrelation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Residuals are errors: Actual - Predicted. Good residuals should:
/// <list type="bullet">
/// <item>Have mean zero (no systematic bias)</item>
/// <item>Be normally distributed (for inference)</item>
/// <item>Have constant variance (homoscedasticity)</item>
/// <item>Be independent (no patterns over time or by predicted value)</item>
/// </list>
/// If these assumptions are violated, your model might be missing something.
/// </para>
/// </remarks>
public class ResidualAnalysisOptions
{
    /// <summary>
    /// Whether to compute standardized residuals. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standardized residuals divide by the standard deviation,
    /// making them comparable across models. Values > 2 or &lt; -2 are potential outliers.</para>
    /// </remarks>
    public bool? ComputeStandardizedResiduals { get; set; }

    /// <summary>
    /// Whether to compute studentized residuals. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like standardized, but accounts for leverage (how unusual
    /// the input is). Better for identifying outliers.</para>
    /// </remarks>
    public bool? ComputeStudentizedResiduals { get; set; }

    /// <summary>
    /// Whether to compute deleted residuals. Default: false (expensive).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> What would the residual be if this point wasn't in the
    /// training data? Useful for detecting influential points.</para>
    /// </remarks>
    public bool? ComputeDeletedResiduals { get; set; }

    /// <summary>
    /// Whether to generate residual vs fitted plot data. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This plot shows residuals on y-axis, fitted values on x-axis.
    /// Should look like random scatter. Patterns indicate problems:
    /// <list type="bullet">
    /// <item>Funnel shape: Heteroscedasticity (non-constant variance)</item>
    /// <item>Curve: Non-linearity (model is missing something)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public bool? GenerateResidualVsFittedPlot { get; set; }

    /// <summary>
    /// Whether to generate Q-Q plot data. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Q-Q plot compares residual distribution to normal.
    /// Points should follow the diagonal line. Deviations indicate non-normality.</para>
    /// </remarks>
    public bool? GenerateQQPlot { get; set; }

    /// <summary>
    /// Whether to generate scale-location plot data. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows sqrt(|standardized residuals|) vs fitted values.
    /// A horizontal band indicates constant variance. Trends indicate heteroscedasticity.</para>
    /// </remarks>
    public bool? GenerateScaleLocationPlot { get; set; }

    /// <summary>
    /// Whether to generate residual vs leverage plot data. Default: true.
    /// </summary>
    public bool? GenerateResidualLeveragePlot { get; set; }

    /// <summary>
    /// Whether to compute residual ACF/PACF for autocorrelation. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ACF (Autocorrelation Function) checks if residuals are
    /// correlated with lagged versions of themselves. Significant ACF indicates patterns
    /// the model missed.</para>
    /// </remarks>
    public bool? ComputeResidualACF { get; set; }

    /// <summary>
    /// Maximum lag for ACF computation. Default: 20.
    /// </summary>
    public int? MaxACFLag { get; set; }

    /// <summary>
    /// Whether to run Shapiro-Wilk normality test. Default: true.
    /// </summary>
    public bool? RunShapiroWilkTest { get; set; }

    /// <summary>
    /// Whether to run Jarque-Bera normality test. Default: true.
    /// </summary>
    public bool? RunJarqueBeraTest { get; set; }

    /// <summary>
    /// Whether to run Kolmogorov-Smirnov test. Default: false.
    /// </summary>
    public bool? RunKolmogorovSmirnovTest { get; set; }

    /// <summary>
    /// Whether to run Durbin-Watson test for autocorrelation. Default: true.
    /// </summary>
    public bool? RunDurbinWatsonTest { get; set; }

    /// <summary>
    /// Whether to run Ljung-Box test for autocorrelation. Default: false.
    /// </summary>
    public bool? RunLjungBoxTest { get; set; }

    /// <summary>
    /// Lag for Ljung-Box test. Default: 10.
    /// </summary>
    public int? LjungBoxLag { get; set; }

    /// <summary>
    /// Whether to run Breusch-Pagan test for heteroscedasticity. Default: true.
    /// </summary>
    public bool? RunBreuschPaganTest { get; set; }

    /// <summary>
    /// Whether to run White test for heteroscedasticity. Default: false.
    /// </summary>
    public bool? RunWhiteTest { get; set; }

    /// <summary>
    /// Whether to run Goldfeld-Quandt test. Default: false.
    /// </summary>
    public bool? RunGoldfeldQuandtTest { get; set; }

    /// <summary>
    /// Whether to run RESET test for functional form. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> RESET tests if the model is missing non-linear terms.
    /// Significant result suggests adding polynomial or interaction terms.</para>
    /// </remarks>
    public bool? RunRESETTest { get; set; }

    /// <summary>
    /// Power terms for RESET test. Default: [2, 3].
    /// </summary>
    public int[]? RESETPowers { get; set; }

    /// <summary>
    /// Significance level for all tests. Default: 0.05.
    /// </summary>
    public double? SignificanceLevel { get; set; }

    /// <summary>
    /// Threshold for outlier detection (standard deviations). Default: 2.5.
    /// </summary>
    public double? OutlierThreshold { get; set; }

    /// <summary>
    /// Whether to identify outliers. Default: true.
    /// </summary>
    public bool? IdentifyOutliers { get; set; }

    /// <summary>
    /// Maximum number of outliers to report. Default: 20.
    /// </summary>
    public int? MaxOutliersToReport { get; set; }

    /// <summary>
    /// Whether to compute partial residual plots. Default: false.
    /// </summary>
    public bool? ComputePartialResidualPlots { get; set; }

    /// <summary>
    /// Features for partial residual plots. Default: null (all).
    /// </summary>
    public int[]? PartialResidualFeatures { get; set; }
}
