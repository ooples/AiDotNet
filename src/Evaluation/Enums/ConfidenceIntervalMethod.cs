namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the method for computing confidence intervals on evaluation metrics.
/// </summary>
/// <remarks>
/// <para>
/// Confidence intervals quantify the uncertainty in metric estimates. Different methods
/// make different assumptions about the underlying distribution and have varying accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you calculate a metric like accuracy = 85%, that's just a point
/// estimate. The "true" accuracy might be anywhere from 82% to 88%. A confidence interval
/// tells you this range. Different methods compute this range differently:
/// <list type="bullet">
/// <item><b>Bootstrap methods</b>: Resample your data many times to see how the metric varies</item>
/// <item><b>Analytical methods</b>: Use mathematical formulas (faster but more assumptions)</item>
/// </list>
/// Bootstrap methods are generally preferred as they make fewer assumptions about your data.
/// </para>
/// </remarks>
public enum ConfidenceIntervalMethod
{
    /// <summary>
    /// Percentile bootstrap: Uses percentiles of the bootstrap distribution directly.
    /// Simple and intuitive but can have coverage issues for small samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Resample your data 1000+ times, compute the metric each time,
    /// then take the 2.5th and 97.5th percentile of those values for a 95% CI.
    /// Simple and easy to understand.</para>
    /// <para><b>When to use:</b> Large samples, symmetric distributions.</para>
    /// <para><b>Limitations:</b> Can have poor coverage for skewed metrics or small samples.</para>
    /// </remarks>
    PercentileBootstrap = 0,

    /// <summary>
    /// Bias-Corrected and Accelerated (BCa) bootstrap: Corrects for bias and skewness.
    /// Generally provides better coverage than percentile bootstrap.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An improved version of percentile bootstrap that adjusts
    /// for bias (when the bootstrap distribution is shifted from the true value) and
    /// acceleration (when the standard error depends on the parameter value).</para>
    /// <para><b>When to use:</b> Most situations - this is the recommended default.</para>
    /// <para><b>Research reference:</b> Efron (1987), widely considered the gold standard.</para>
    /// </remarks>
    BCaBootstrap = 1,

    /// <summary>
    /// Studentized (bootstrap-t) bootstrap: Uses t-statistics for pivoting.
    /// Theoretically optimal but requires variance estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of resampling the metric directly, this resamples
    /// t-statistics (metric minus mean, divided by standard error). More complex but can
    /// provide better coverage when standard errors vary.</para>
    /// <para><b>When to use:</b> When you have good variance estimates.</para>
    /// <para><b>Limitations:</b> Requires nested bootstrapping, computationally expensive.</para>
    /// </remarks>
    StudentizedBootstrap = 2,

    /// <summary>
    /// Basic bootstrap (reverse percentile): Reflects the bootstrap distribution.
    /// Simple alternative to percentile bootstrap.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to percentile bootstrap but uses the reflection
    /// principle: if bootstrap estimates are too high, the true value is probably lower.
    /// Interval is computed by reflecting around the point estimate.</para>
    /// <para><b>When to use:</b> Historical method, usually prefer BCa instead.</para>
    /// </remarks>
    BasicBootstrap = 3,

    /// <summary>
    /// Wilson score interval: Analytical method for proportions/accuracy.
    /// Better coverage than naive normal approximation, especially near 0 or 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A formula-based method specifically designed for
    /// proportions (like accuracy). Unlike the simple ± formula, it works well even
    /// when accuracy is very high (98%) or very low (2%).</para>
    /// <para><b>When to use:</b> Accuracy, precision, recall, and other proportions.</para>
    /// <para><b>Research reference:</b> Wilson (1927), recommended by Agresti & Coull (1998).</para>
    /// </remarks>
    WilsonScore = 4,

    /// <summary>
    /// Clopper-Pearson exact interval: Conservative exact interval for proportions.
    /// Guarantees at least the stated coverage level.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The most conservative method for proportions. It guarantees
    /// that the interval will contain the true value at least 95% of the time (for a 95% CI).
    /// Can be wider than necessary but never too narrow.</para>
    /// <para><b>When to use:</b> When you need guaranteed coverage (regulatory settings).</para>
    /// <para><b>Note:</b> Often overly conservative; Wilson score is usually preferred.</para>
    /// </remarks>
    ClopperPearsonExact = 5,

    /// <summary>
    /// Agresti-Coull adjusted interval: Adds pseudo-observations for better coverage.
    /// Good balance between accuracy and simplicity.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A clever trick: add 2 successes and 2 failures to your data
    /// before computing the normal approximation interval. This simple adjustment dramatically
    /// improves coverage, especially for extreme proportions.</para>
    /// <para><b>When to use:</b> Quick calculation with good coverage for proportions.</para>
    /// <para><b>Research reference:</b> Agresti & Coull (1998) "Approximate is better than exact".</para>
    /// </remarks>
    AgrestiCoull = 6,

    /// <summary>
    /// Normal approximation: Simple z-interval using normal distribution.
    /// Fast but can have poor coverage for small samples or extreme values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The classic ± 1.96 * standard_error formula for 95% CI.
    /// Fast and simple but assumes your metric is normally distributed, which may not
    /// be true for metrics bounded between 0 and 1.</para>
    /// <para><b>When to use:</b> Large samples where central limit theorem applies.</para>
    /// <para><b>Limitations:</b> Poor coverage for small samples, metrics near 0 or 1.</para>
    /// </remarks>
    NormalApproximation = 7,

    /// <summary>
    /// t-distribution interval: Uses t-distribution instead of normal.
    /// Better for small samples when standard error is estimated.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like normal approximation but uses the t-distribution,
    /// which has heavier tails. This accounts for the additional uncertainty from
    /// estimating the standard error from the sample.</para>
    /// <para><b>When to use:</b> Small to moderate samples with estimated standard errors.</para>
    /// </remarks>
    TDistribution = 8,

    /// <summary>
    /// DeLong method: Specialized method for AUROC confidence intervals.
    /// Uses covariance structure of the ROC curve.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A method specifically designed for computing CIs on
    /// the Area Under the ROC Curve. It accounts for the special structure of ROC
    /// curves and typically provides better intervals than general bootstrap.</para>
    /// <para><b>When to use:</b> AUROC confidence intervals.</para>
    /// <para><b>Research reference:</b> DeLong et al. (1988), standard for ROC analysis.</para>
    /// </remarks>
    DeLong = 9,

    /// <summary>
    /// Bayesian credible interval: Uses posterior distribution from Bayesian inference.
    /// Provides probability statements about the parameter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Unlike frequentist CIs that say "95% of intervals computed
    /// this way contain the true value", Bayesian credible intervals say "there's a 95%
    /// probability the true value is in this range" (given your prior beliefs).</para>
    /// <para><b>When to use:</b> When you have prior information or prefer Bayesian interpretation.</para>
    /// </remarks>
    BayesianCredible = 10
}
