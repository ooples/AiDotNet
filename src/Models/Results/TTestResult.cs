namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a t-test, which is a statistical hypothesis test used to determine if there is a significant 
/// difference between the means of two groups.
/// </summary>
/// <remarks>
/// <para>
/// The t-test is one of the most commonly used statistical tests for comparing means. This class stores the results of 
/// such a test, including the t-statistic, degrees of freedom, p-value, and whether the result is statistically significant. 
/// The t-test is used when the test statistic follows a t-distribution under the null hypothesis, which typically occurs 
/// when comparing means from normally distributed populations with unknown variances. The class uses generic type parameter 
/// T to support different numeric types for the statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> This class stores the results of a t-test, which helps determine if the difference between two groups is statistically significant.
/// 
/// For example, you might use a t-test to answer questions like:
/// - Is there a significant difference in test scores between two teaching methods?
/// - Does a new medication significantly change blood pressure compared to a placebo?
/// - Are the average sales before and after a marketing campaign significantly different?
/// 
/// The t-test works by:
/// 1. Calculating a t-statistic based on the difference between group means
/// 2. Determining how likely this t-statistic would occur by chance
/// 3. Producing a p-value that represents this probability
/// 
/// This test is particularly useful when:
/// - You're comparing means between two groups
/// - Your data approximately follows a normal distribution
/// - You have relatively small sample sizes
/// 
/// This class stores all the information about the test results, helping you interpret whether
/// the observed difference is statistically significant.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class TTestResult<T>
{
    /// <summary>
    /// Gets or sets the t-statistic value.
    /// </summary>
    /// <value>The calculated t-statistic.</value>
    /// <remarks>
    /// <para>
    /// This property represents the t-statistic, which is a measure of the difference between the groups being compared, 
    /// scaled by the variability in the data. The t-statistic is calculated by dividing the difference between the means 
    /// by the standard error of the difference. A larger absolute value of the t-statistic indicates a greater difference 
    /// between the groups relative to the variability within the groups. The sign of the t-statistic indicates the direction 
    /// of the difference (positive if the first group mean is larger, negative if the second group mean is larger).
    /// </para>
    /// <para><b>For Beginners:</b> This value measures how different your groups are, taking into account the variability in your data.
    /// 
    /// The t-statistic:
    /// - Measures the size of the difference between groups relative to the variation within groups
    /// - Larger absolute values indicate stronger evidence of a real difference
    /// - The sign (+ or -) indicates which group has the higher mean
    /// 
    /// For example, a t-statistic of 2.5 suggests the difference between groups is 2.5 times
    /// larger than what you might expect from random variation alone.
    /// 
    /// This value is important because:
    /// - It's used to calculate the p-value
    /// - It indicates both the direction and magnitude of the effect
    /// - It can be compared to critical values from t-distribution tables
    /// </para>
    /// </remarks>
    public T TStatistic { get; set; }

    /// <summary>
    /// Gets or sets the degrees of freedom for the t-test.
    /// </summary>
    /// <value>An integer representing the degrees of freedom.</value>
    /// <remarks>
    /// <para>
    /// This property represents the degrees of freedom for the t-test, which is a parameter that determines the shape of 
    /// the t-distribution used to calculate the p-value. The degrees of freedom is typically related to the sample sizes 
    /// of the groups being compared. For an independent samples t-test, it is often calculated as the sum of the sample 
    /// sizes minus 2, though more complex formulas may be used for Welch's t-test or other variants. For a paired t-test, 
    /// it is typically the number of pairs minus 1. As the degrees of freedom increases, the t-distribution approaches the 
    /// normal distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This value helps determine the shape of the t-distribution used to calculate the p-value.
    /// 
    /// The degrees of freedom:
    /// - Is related to your sample size
    /// - Affects the critical values used to determine significance
    /// - Influences how conservative the test is
    /// 
    /// For common t-tests:
    /// - Independent samples t-test: df = n1 + n2 - 2 (where n1 and n2 are the sample sizes)
    /// - Paired t-test: df = n - 1 (where n is the number of pairs)
    /// - Welch's t-test: uses a more complex formula that accounts for unequal variances
    /// 
    /// As degrees of freedom increase:
    /// - The t-distribution becomes closer to a normal distribution
    /// - The test becomes more powerful (better at detecting real differences)
    /// </para>
    /// </remarks>
    public int DegreesOfFreedom { get; set; }

    /// <summary>
    /// Gets or sets the p-value associated with the t-test.
    /// </summary>
    /// <value>The calculated p-value.</value>
    /// <remarks>
    /// <para>
    /// This property represents the p-value of the t-test, which is the probability of observing a t-statistic as extreme 
    /// as, or more extreme than, the one calculated from the sample data, assuming the null hypothesis is true. The null 
    /// hypothesis typically states that there is no difference between the means of the groups being compared. A small 
    /// p-value (typically = 0.05) suggests that the observed difference is unlikely under the null hypothesis, leading to 
    /// its rejection in favor of the alternative hypothesis that the means differ. The p-value is calculated from the 
    /// t-statistic and the degrees of freedom using the cumulative distribution function of the t-distribution.
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
    /// For example, a p-value of 0.03 means there's only a 3% chance of seeing a difference
    /// as large as yours if the groups truly have the same mean, suggesting the difference
    /// is likely real and not due to random chance.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the t-test result is statistically significant.
    /// </summary>
    /// <value>True if the result is statistically significant; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the t-test result is statistically significant at the specified significance level. 
    /// A result is considered significant if the p-value is less than the significance level. Statistical significance 
    /// suggests that the observed difference between the groups is unlikely to have occurred by chance alone, leading to 
    /// the rejection of the null hypothesis that the means are equal. This property provides a convenient boolean indicator 
    /// of significance without requiring the user to compare the p-value to the significance level.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether your result is statistically significant or not.
    /// 
    /// The significance indicator:
    /// - Provides a simple yes/no answer about statistical significance
    /// - True means the difference between groups is statistically significant
    /// - False means the difference could reasonably be due to random chance
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
    /// This property represents the significance level used for the t-test, which is the threshold p-value below which the 
    /// result is considered statistically significant. Common values are 0.05 (5%), 0.01 (1%), and 0.001 (0.1%). The 
    /// significance level represents the probability of rejecting the null hypothesis when it is actually true (Type I error). 
    /// A lower significance level reduces the risk of Type I errors but increases the risk of Type II errors (failing to 
    /// reject a false null hypothesis).
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
    /// incorrectly concluding there's a difference when there's actually none.
    /// </para>
    /// </remarks>
    public T SignificanceLevel { get; set; }

    /// <summary>
    /// Initializes a new instance of the TTestResult class with the specified test statistics and parameters.
    /// </summary>
    /// <param name="tStatistic">The t-statistic value.</param>
    /// <param name="degreesOfFreedom">The degrees of freedom for the t-test.</param>
    /// <param name="pValue">The p-value associated with the t-test.</param>
    /// <param name="significanceLevel">The significance level used for the test.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new TTestResult instance with the specified test statistics and parameters. It initializes 
    /// all properties of the class and calculates the IsSignificant property by comparing the p-value to the significance 
    /// level. The IsSignificant property is set to true if the p-value is less than the significance level, indicating that 
    /// the test result is statistically significant. This constructor provides a convenient way to create a complete 
    /// TTestResult object in a single step after performing a t-test.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with all the t-test statistics and parameters.
    /// 
    /// When a new TTestResult is created with this constructor:
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
    public TTestResult(T tStatistic, int degreesOfFreedom, T pValue, T significanceLevel)
    {
        TStatistic = tStatistic;
        DegreesOfFreedom = degreesOfFreedom;
        PValue = pValue;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}
