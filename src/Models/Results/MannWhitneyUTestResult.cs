namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a Mann-Whitney U test, which is a non-parametric statistical test used to determine 
/// whether two independent samples come from the same distribution.
/// </summary>
/// <remarks>
/// <para>
/// The Mann-Whitney U test (also known as the Wilcoxon rank-sum test) is a non-parametric alternative to the 
/// independent samples t-test. It's used when the data doesn't meet the assumptions required for the t-test, 
/// particularly when the data isn't normally distributed. This class stores the results of such a test, including 
/// the U statistic, Z-score, p-value, and whether the result is statistically significant. The test is commonly 
/// used to compare the medians of two groups, though it technically tests whether one sample tends to have larger 
/// values than the other. The class uses generic type parameter T to support different numeric types for the 
/// statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> The Mann-Whitney U test helps determine if two groups are different when you can't use a regular t-test.
/// 
/// For example, you might use this test to answer questions like:
/// - Do patients on treatment A have different recovery times than those on treatment B?
/// - Do students using one learning method score differently than those using another method?
/// - Are customer satisfaction ratings different between two product versions?
/// 
/// The test works by:
/// 1. Ranking all values from both groups together
/// 2. Calculating how much the ranks differ between groups
/// 3. Determining if this difference is statistically significant
/// 
/// This test is particularly useful when:
/// - Your data doesn't follow a normal distribution
/// - You have ordinal data (rankings) rather than continuous measurements
/// - You have outliers that might skew the results of a t-test
/// 
/// This class stores all the information about the test results, helping you interpret whether
/// the observed difference between groups is likely due to chance or represents a real difference.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class MannWhitneyUTestResult<T>
{
    /// <summary>
    /// Gets or sets the U statistic value.
    /// </summary>
    /// <value>The calculated U statistic.</value>
    /// <remarks>
    /// <para>
    /// This property represents the Mann-Whitney U statistic, which is calculated by counting the number of times a 
    /// value in the first sample exceeds a value in the second sample, and vice versa. The U statistic is the smaller 
    /// of these two counts. When the null hypothesis is true (the distributions are the same), U tends to be around 
    /// half the product of the two sample sizes. Values of U that deviate significantly from this expected value 
    /// suggest that the samples come from different distributions. The U statistic is used to calculate the Z-score 
    /// and p-value, which determine statistical significance.
    /// </para>
    /// <para><b>For Beginners:</b> This value measures how different the rankings are between your two groups.
    /// 
    /// The U statistic:
    /// - Is calculated based on the rankings of values in both groups
    /// - Measures how much the two groups overlap or differ
    /// - Has a range from 0 to the product of the two sample sizes
    /// 
    /// When the groups are similar:
    /// - U will be close to half the product of the sample sizes
    /// 
    /// When the groups are different:
    /// - U will be closer to either 0 or the maximum possible value
    /// 
    /// For example, with two groups of 10 samples each, U would be around 50 if the groups
    /// are similar, but might be closer to 20 or 80 if they're substantially different.
    /// </para>
    /// </remarks>
    public T UStatistic { get; set; }

    /// <summary>
    /// Gets or sets the Z-score associated with the U statistic.
    /// </summary>
    /// <value>The calculated Z-score.</value>
    /// <remarks>
    /// <para>
    /// This property represents the Z-score, which is a standardized version of the U statistic. The Z-score is 
    /// calculated by subtracting the expected value of U (under the null hypothesis) from the observed U statistic, 
    /// and then dividing by the standard deviation of U. For large sample sizes, the Z-score approximately follows 
    /// a standard normal distribution, which allows for the calculation of the p-value. A Z-score with a large 
    /// absolute value (typically > 1.96 for a two-tailed test at a = 0.05) suggests that the observed U statistic 
    /// is significantly different from what would be expected if the null hypothesis were true.
    /// </para>
    /// <para><b>For Beginners:</b> This value standardizes the U statistic to make it easier to interpret.
    /// 
    /// The Z-score:
    /// - Converts the U statistic to a standardized form
    /// - Follows a standard normal distribution for large samples
    /// - Makes it easier to calculate the p-value
    /// 
    /// Interpretation:
    /// - Z-scores close to 0 suggest the groups are similar
    /// - Z-scores with absolute values greater than 1.96 (for a = 0.05) suggest significant differences
    /// - Negative Z-scores indicate the first group tends to have lower values
    /// - Positive Z-scores indicate the first group tends to have higher values
    /// 
    /// For example, a Z-score of -2.5 suggests that the first group has significantly
    /// lower values than the second group.
    /// </para>
    /// </remarks>
    public T ZScore { get; set; }

    /// <summary>
    /// Gets or sets the p-value associated with the Mann-Whitney U test.
    /// </summary>
    /// <value>The calculated p-value.</value>
    /// <remarks>
    /// <para>
    /// This property represents the p-value of the Mann-Whitney U test, which is the probability of observing a U 
    /// statistic as extreme as, or more extreme than, the one calculated from the sample data, assuming the null 
    /// hypothesis is true. The null hypothesis typically states that the two samples come from the same distribution 
    /// or have the same median. A small p-value (typically = 0.05) suggests that the observed data is unlikely under 
    /// the null hypothesis, leading to its rejection in favor of the alternative hypothesis that the distributions 
    /// differ. The p-value is calculated from the Z-score using the cumulative distribution function of the standard 
    /// normal distribution for large samples, or from tables of critical values for small samples.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how likely your results could occur by random chance.
    /// 
    /// The p-value:
    /// - Ranges from 0 to 1
    /// - Smaller values indicate stronger evidence against the null hypothesis
    /// - Represents the probability of seeing your results (or more extreme) if the groups are actually the same
    /// 
    /// Common interpretation:
    /// - p = 0.05: Results are statistically significant (commonly used threshold)
    /// - p = 0.01: Results are highly significant
    /// - p > 0.05: Results are not statistically significant
    /// 
    /// For example, a p-value of 0.03 means there's only a 3% chance of seeing your results
    /// if the groups truly come from the same distribution, suggesting the difference
    /// is likely real and not due to random chance.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the Mann-Whitney U test result is statistically significant.
    /// </summary>
    /// <value>True if the result is statistically significant; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the Mann-Whitney U test result is statistically significant at the specified 
    /// significance level. A result is considered significant if the p-value is less than or equal to the significance 
    /// level. Statistical significance suggests that the observed difference between the samples is unlikely to have 
    /// occurred by chance alone, leading to the rejection of the null hypothesis that the samples come from the same 
    /// distribution. This property provides a convenient boolean indicator of significance without requiring the user 
    /// to compare the p-value to the significance level.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether your result is statistically significant or not.
    /// 
    /// The significance indicator:
    /// - Provides a simple yes/no answer about statistical significance
    /// - True means the groups are significantly different
    /// - False means the difference between groups could reasonably be due to random chance
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
    /// This property represents the significance level used for the Mann-Whitney U test, which is the threshold p-value 
    /// below which the result is considered statistically significant. Common values are 0.05 (5%), 0.01 (1%), and 
    /// 0.001 (0.1%). The significance level represents the probability of rejecting the null hypothesis when it is 
    /// actually true (Type I error). A lower significance level reduces the risk of Type I errors but increases the 
    /// risk of Type II errors (failing to reject a false null hypothesis).
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
    /// - It represents the risk you're willing to take of finding a difference when none exists
    /// 
    /// For example, with a significance level of 0.05, you're accepting a 5% chance of
    /// incorrectly concluding the groups are different when they're actually the same.
    /// </para>
    /// </remarks>
    public T SignificanceLevel { get; set; }

    /// <summary>
    /// Initializes a new instance of the MannWhitneyUTestResult class with the specified test statistics and parameters.
    /// </summary>
    /// <param name="uStatistic">The U statistic value.</param>
    /// <param name="zScore">The Z-score associated with the U statistic.</param>
    /// <param name="pValue">The p-value associated with the Mann-Whitney U test.</param>
    /// <param name="significanceLevel">The significance level used for the test.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new MannWhitneyUTestResult instance with the specified test statistics and parameters. 
    /// It initializes all properties of the class and calculates the IsSignificant property by comparing the p-value to 
    /// the significance level. The IsSignificant property is set to true if the p-value is less than the significance 
    /// level, indicating that the test result is statistically significant. This constructor provides a convenient way 
    /// to create a complete MannWhitneyUTestResult object in a single step after performing a Mann-Whitney U test.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with all the Mann-Whitney U test statistics and parameters.
    /// 
    /// When a new MannWhitneyUTestResult is created with this constructor:
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
    public MannWhitneyUTestResult(T uStatistic, T zScore, T pValue, T significanceLevel)
    {
        UStatistic = uStatistic;
        ZScore = zScore;
        PValue = pValue;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}
