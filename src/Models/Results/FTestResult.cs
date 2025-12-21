namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of an F-test, which is used to compare the variances of two populations.
/// </summary>
/// <remarks>
/// <para>
/// The F-test is a statistical test used to determine whether two populations have equal variances. It is based on 
/// the ratio of two sample variances. This class stores the results of such a test, including the F-statistic, 
/// p-value, degrees of freedom, the variances being compared, confidence intervals, and whether the result is 
/// statistically significant. The F-test is commonly used in analysis of variance (ANOVA) and as a preliminary 
/// test before applying other statistical tests that assume equal variances. The class uses generic type parameter 
/// T to support different numeric types for the statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> The F-test helps determine if two groups have similar or different amounts of variability.
/// 
/// For example, you might use this test to answer questions like:
/// - Do men and women have the same variability in test scores?
/// - Is the precision of one measurement method better than another?
/// - Are the variances in two manufacturing processes comparable?
/// 
/// The test works by:
/// 1. Calculating the ratio of the two sample variances
/// 2. Comparing this ratio to what would be expected if the population variances were equal
/// 3. Determining if the difference is statistically significant
/// 
/// This class stores all the information about the test results, helping you interpret whether
/// the observed difference in variability is likely due to chance or represents a real difference.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class FTestResult<T>
{
    /// <summary>
    /// Gets or sets the F-statistic value.
    /// </summary>
    /// <value>The calculated F-statistic.</value>
    /// <remarks>
    /// <para>
    /// This property represents the F-statistic, which is the ratio of the larger sample variance to the smaller 
    /// sample variance. An F-statistic close to 1 suggests that the two populations have similar variances, while 
    /// values much larger than 1 suggest that the variances are different. The F-statistic follows an F-distribution 
    /// with degrees of freedom determined by the sample sizes. This statistic is used in conjunction with the degrees 
    /// of freedom to calculate the p-value, which determines statistical significance.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the ratio of the larger variance to the smaller variance.
    /// 
    /// The F-statistic:
    /// - Is calculated as the ratio of two sample variances
    /// - Is always positive
    /// - A value close to 1 suggests similar variances
    /// - Values much larger than 1 suggest different variances
    /// 
    /// For example, an F-statistic of 1.05 suggests the variances are very similar,
    /// while a value of 4.75 suggests one group has much more variability than the other.
    /// </para>
    /// </remarks>
    public T FStatistic { get; set; }

    /// <summary>
    /// Gets or sets the p-value associated with the F-test.
    /// </summary>
    /// <value>The calculated p-value.</value>
    /// <remarks>
    /// <para>
    /// This property represents the p-value of the F-test, which is the probability of observing an F-statistic as 
    /// extreme as, or more extreme than, the one calculated from the sample data, assuming the null hypothesis is true. 
    /// The null hypothesis typically states that the two populations have equal variances. A small p-value (typically 
    /// = 0.05) suggests that the observed data is unlikely under the null hypothesis, leading to its rejection in favor 
    /// of the alternative hypothesis that the variances are different. The p-value is calculated from the F-statistic 
    /// and the degrees of freedom using the cumulative distribution function of the F-distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how likely your results could occur by random chance.
    /// 
    /// The p-value:
    /// - Ranges from 0 to 1
    /// - Smaller values indicate stronger evidence against the null hypothesis
    /// - Represents the probability of seeing your results (or more extreme) if the variances are actually equal
    /// 
    /// Common interpretation:
    /// - p = 0.05: Results are statistically significant (commonly used threshold)
    /// - p = 0.01: Results are highly significant
    /// - p > 0.05: Results are not statistically significant
    /// 
    /// For example, a p-value of 0.02 means there's only a 2% chance of seeing your results
    /// if the variances are truly equal, suggesting the difference in variability
    /// is likely real and not due to random chance.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Gets or sets the degrees of freedom for the numerator.
    /// </summary>
    /// <value>An integer representing the numerator degrees of freedom.</value>
    /// <remarks>
    /// <para>
    /// This property represents the degrees of freedom for the numerator of the F-statistic, which is typically one 
    /// less than the sample size of the first group (n1 - 1). The numerator degrees of freedom is one of the parameters 
    /// of the F-distribution used to calculate the p-value. It affects the shape of the F-distribution and thus the 
    /// interpretation of the F-statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This value is related to the sample size of the first group.
    /// 
    /// The numerator degrees of freedom:
    /// - Is typically calculated as (n1 - 1), where n1 is the sample size of the first group
    /// - Is a parameter needed to interpret the F-statistic
    /// - Helps determine the shape of the F-distribution used to calculate the p-value
    /// 
    /// For example, if your first sample has 20 observations, the numerator degrees of freedom would be 19.
    /// </para>
    /// </remarks>
    public int NumeratorDegreesOfFreedom { get; set; }

    /// <summary>
    /// Gets or sets the degrees of freedom for the denominator.
    /// </summary>
    /// <value>An integer representing the denominator degrees of freedom.</value>
    /// <remarks>
    /// <para>
    /// This property represents the degrees of freedom for the denominator of the F-statistic, which is typically one 
    /// less than the sample size of the second group (n2 - 1). The denominator degrees of freedom is one of the 
    /// parameters of the F-distribution used to calculate the p-value. It affects the shape of the F-distribution and 
    /// thus the interpretation of the F-statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This value is related to the sample size of the second group.
    /// 
    /// The denominator degrees of freedom:
    /// - Is typically calculated as (n2 - 1), where n2 is the sample size of the second group
    /// - Is a parameter needed to interpret the F-statistic
    /// - Helps determine the shape of the F-distribution used to calculate the p-value
    /// 
    /// For example, if your second sample has 15 observations, the denominator degrees of freedom would be 14.
    /// </para>
    /// </remarks>
    public int DenominatorDegreesOfFreedom { get; set; }

    /// <summary>
    /// Gets or sets the variance of the left (or first) sample.
    /// </summary>
    /// <value>The calculated variance of the left sample.</value>
    /// <remarks>
    /// <para>
    /// This property represents the variance of the left (or first) sample, which is a measure of the spread or 
    /// dispersion of the data points around their mean. The variance is calculated as the average of the squared 
    /// differences from the mean. This is one of the two variances being compared in the F-test. In the context of 
    /// the F-test, the left variance is often the numerator in the F-statistic calculation, especially if it's the 
    /// larger of the two variances.
    /// </para>
    /// <para><b>For Beginners:</b> This value measures how spread out the data is in the first group.
    /// 
    /// The left variance:
    /// - Quantifies the variability or dispersion in the first sample
    /// - Higher values indicate more spread-out data
    /// - Is one of the two variances being compared in the F-test
    /// 
    /// For example, if measuring test scores, a variance of 100 means the scores are more
    /// spread out than a variance of 25.
    /// </para>
    /// </remarks>
    public T LeftVariance { get; set; }

    /// <summary>
    /// Gets or sets the variance of the right (or second) sample.
    /// </summary>
    /// <value>The calculated variance of the right sample.</value>
    /// <remarks>
    /// <para>
    /// This property represents the variance of the right (or second) sample, which is a measure of the spread or 
    /// dispersion of the data points around their mean. The variance is calculated as the average of the squared 
    /// differences from the mean. This is one of the two variances being compared in the F-test. In the context of 
    /// the F-test, the right variance is often the denominator in the F-statistic calculation, especially if it's 
    /// the smaller of the two variances.
    /// </para>
    /// <para><b>For Beginners:</b> This value measures how spread out the data is in the second group.
    /// 
    /// The right variance:
    /// - Quantifies the variability or dispersion in the second sample
    /// - Higher values indicate more spread-out data
    /// - Is one of the two variances being compared in the F-test
    /// 
    /// For example, if measuring reaction times, a variance of 0.5 means the times are more
    /// consistent than a variance of 2.0.
    /// </para>
    /// </remarks>
    public T RightVariance { get; set; }

    /// <summary>
    /// Gets or sets the lower bound of the confidence interval for the ratio of population variances.
    /// </summary>
    /// <value>The lower bound of the confidence interval.</value>
    /// <remarks>
    /// <para>
    /// This property represents the lower bound of the confidence interval for the ratio of the two population variances. 
    /// The confidence interval provides a range of plausible values for the true ratio of population variances, given 
    /// the observed sample data. If this interval includes 1, then the null hypothesis that the population variances 
    /// are equal cannot be rejected at the specified significance level. The width of the confidence interval depends 
    /// on the significance level and the sample sizes.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the lower end of the range where the true variance ratio likely falls.
    /// 
    /// The lower confidence interval:
    /// - Forms the lower bound of a range that likely contains the true ratio of population variances
    /// - Is calculated based on the F-statistic, degrees of freedom, and significance level
    /// - Helps assess the precision of your variance ratio estimate
    /// 
    /// This value is important because:
    /// - If the interval includes 1, the variances might be equal
    /// - If both the lower and upper bounds are above 1, the first population likely has higher variance
    /// - If both bounds are below 1, the second population likely has higher variance
    /// 
    /// For example, a lower bound of 1.2 suggests that the first population's variance is
    /// at least 1.2 times the second population's variance.
    /// </para>
    /// </remarks>
    public T LowerConfidenceInterval { get; set; }

    /// <summary>
    /// Gets or sets the upper bound of the confidence interval for the ratio of population variances.
    /// </summary>
    /// <value>The upper bound of the confidence interval.</value>
    /// <remarks>
    /// <para>
    /// This property represents the upper bound of the confidence interval for the ratio of the two population variances. 
    /// The confidence interval provides a range of plausible values for the true ratio of population variances, given 
    /// the observed sample data. If this interval includes 1, then the null hypothesis that the population variances 
    /// are equal cannot be rejected at the specified significance level. The width of the confidence interval depends 
    /// on the significance level and the sample sizes.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the upper end of the range where the true variance ratio likely falls.
    /// 
    /// The upper confidence interval:
    /// - Forms the upper bound of a range that likely contains the true ratio of population variances
    /// - Is calculated based on the F-statistic, degrees of freedom, and significance level
    /// - Helps assess the precision of your variance ratio estimate
    /// 
    /// This value is important because:
    /// - If the interval includes 1, the variances might be equal
    /// - A very high upper bound suggests uncertainty about how much larger the first variance might be
    /// - Together with the lower bound, it shows the precision of your estimate
    /// 
    /// For example, an upper bound of 3.5 suggests that the first population's variance could
    /// be as much as 3.5 times the second population's variance.
    /// </para>
    /// </remarks>
    public T UpperConfidenceInterval { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the F-test result is statistically significant.
    /// </summary>
    /// <value>True if the result is statistically significant; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the F-test result is statistically significant at the specified significance 
    /// level. A result is considered significant if the p-value is less than or equal to the significance level. 
    /// Statistical significance suggests that the observed difference between the sample variances is unlikely to 
    /// have occurred by chance alone, leading to the rejection of the null hypothesis that the population variances 
    /// are equal. This property provides a convenient boolean indicator of significance without requiring the user 
    /// to compare the p-value to the significance level.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether your result is statistically significant or not.
    /// 
    /// The significance indicator:
    /// - Provides a simple yes/no answer about statistical significance
    /// - True means the variances are significantly different
    /// - False means the difference in variances could reasonably be due to random chance
    /// 
    /// This is determined by:
    /// - Comparing the p-value to the significance level
    /// 
    /// This value makes it easy to interpret your results without having to manually
    /// check if the p-value is below the significance threshold. It's particularly
    /// useful when automating analysis or presenting results to non-statisticians.
    /// </para>
    /// </remarks>
    public bool IsSignificant { get; set; }

    /// <summary>
    /// Gets or sets the significance level used for the test.
    /// </summary>
    /// <value>The significance level, typically 0.05.</value>
    /// <remarks>
    /// <para>
    /// This property represents the significance level used for the F-test, which is the threshold p-value below which 
    /// the result is considered statistically significant. Common values are 0.05 (5%), 0.01 (1%), and 0.001 (0.1%). 
    /// The significance level represents the probability of rejecting the null hypothesis when it is actually true 
    /// (Type I error). A lower significance level reduces the risk of Type I errors but increases the risk of Type II 
    /// errors (failing to reject a false null hypothesis). The significance level also determines the width of the 
    /// confidence interval, with lower significance levels resulting in wider intervals.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the threshold that determines when a result is considered statistically significant.
    /// 
    /// The significance level:
    /// - Is typically set to 0.05 (5%) by convention
    /// - Represents your tolerance for false positives
    /// - Lower values (like 0.01) make the test more conservative
    /// - Higher values (like 0.10) make the test more lenient
    /// 
    /// This value is important because:
    /// - It directly determines whether a result is considered "significant"
    /// - It affects the width of the confidence interval
    /// - It represents the risk you're willing to take of finding a difference when none exists
    /// 
    /// For example, with a significance level of 0.05, you're accepting a 5% chance of
    /// incorrectly concluding the variances are different when they're actually equal.
    /// </para>
    /// </remarks>
    public T SignificanceLevel { get; set; }

    /// <summary>
    /// Initializes a new instance of the FTestResult class with the specified test statistics and parameters.
    /// </summary>
    /// <param name="fStatistic">The F-statistic value.</param>
    /// <param name="pValue">The p-value associated with the F-test.</param>
    /// <param name="numeratorDf">The degrees of freedom for the numerator.</param>
    /// <param name="denominatorDf">The degrees of freedom for the denominator.</param>
    /// <param name="leftVariance">The variance of the left (or first) sample.</param>
    /// <param name="rightVariance">The variance of the right (or second) sample.</param>
    /// <param name="lowerCI">The lower bound of the confidence interval for the ratio of population variances.</param>
    /// <param name="upperCI">The upper bound of the confidence interval for the ratio of population variances.</param>
    /// <param name="significanceLevel">The significance level used for the test.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new FTestResult instance with the specified test statistics and parameters. It initializes 
    /// all properties of the class and calculates the IsSignificant property by comparing the p-value to the significance 
    /// level. The IsSignificant property is set to true if the p-value is less than the significance level, indicating that 
    /// the test result is statistically significant. This constructor provides a convenient way to create a complete 
    /// FTestResult object in a single step after performing an F-test.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with all the F-test statistics and parameters.
    /// 
    /// When a new FTestResult is created with this constructor:
    /// - All the test statistics and parameters are set to the values you provide
    /// - The IsSignificant property is automatically calculated by comparing the p-value to the significance level
    /// 
    /// This constructor is useful because:
    /// - It creates a complete result object in one step
    /// - It automatically determines statistical significance
    /// - It ensures all the related test information is kept together
    /// 
    /// The IsSignificant calculation uses MathHelper to compare the p-value to the significance level,
    /// which works regardless of what numeric type T is (float, double, decimal, etc.).
    /// </para>
    /// </remarks>
    public FTestResult(T fStatistic, T pValue, int numeratorDf, int denominatorDf, T leftVariance, T rightVariance, T lowerCI, T upperCI, T significanceLevel)
    {
        FStatistic = fStatistic;
        PValue = pValue;
        NumeratorDegreesOfFreedom = numeratorDf;
        DenominatorDegreesOfFreedom = denominatorDf;
        LeftVariance = leftVariance;
        RightVariance = rightVariance;
        LowerConfidenceInterval = lowerCI;
        UpperConfidenceInterval = upperCI;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}
