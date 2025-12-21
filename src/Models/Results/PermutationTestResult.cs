namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a permutation test, which is a non-parametric statistical significance test that determines
/// whether the observed difference between two groups is statistically significant.
/// </summary>
/// <remarks>
/// <para>
/// The permutation test is a resampling method that creates a reference distribution by randomly reassigning observations 
/// to groups and recalculating the test statistic many times. This class stores the results of such a test, including 
/// the observed difference between groups, the p-value, the number of permutations performed, the count of extreme values, 
/// and whether the result is statistically significant. Permutation tests are particularly useful when the assumptions of 
/// parametric tests (like t-tests) are not met, or when dealing with small sample sizes. The class uses generic type 
/// parameter T to support different numeric types for the statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> This class stores the results of a permutation test, which helps determine if a difference between groups is real or could have happened by chance.
/// 
/// For example, you might use this test to answer questions like:
/// - Is the difference in treatment outcomes between two groups statistically significant?
/// - Could the observed difference in test scores between teaching methods have occurred by random chance?
/// - Is the relationship between two variables stronger than would be expected by chance?
/// 
/// The test works by:
/// 1. Calculating the actual difference between your groups
/// 2. Randomly shuffling your data many times and recalculating the difference each time
/// 3. Seeing how often the random shuffles produce a difference as extreme as your actual difference
/// 
/// This approach is particularly useful when:
/// - You have small sample sizes
/// - Your data doesn't follow a normal distribution
/// - You want to avoid making assumptions about the underlying distribution
/// 
/// This class stores all the information about the test results, helping you interpret whether
/// the observed difference is statistically significant.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class PermutationTestResult<T>
{
    /// <summary>
    /// Gets or sets the observed difference between the two groups being compared.
    /// </summary>
    /// <value>The calculated difference between the groups.</value>
    /// <remarks>
    /// <para>
    /// This property represents the actual difference observed between the two groups in the original data. The specific 
    /// meaning of this difference depends on the test statistic used, which could be a difference in means, medians, 
    /// proportions, or any other measure of interest. This observed difference is compared to the distribution of differences 
    /// obtained through permutations to determine statistical significance. The larger the absolute value of the observed 
    /// difference relative to the permutation distribution, the more likely it is to be statistically significant.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the actual difference you observed between your groups.
    /// 
    /// The observed difference:
    /// - Is calculated from your original, unpermuted data
    /// - Represents the effect size you're testing for significance
    /// - Could be a difference in means, medians, proportions, or other statistics
    /// 
    /// For example, if comparing test scores between two teaching methods, this might be
    /// the difference in average scores: Method A (85) - Method B (78) = 7 points.
    /// 
    /// This value is important because:
    /// - It tells you the magnitude of the effect you observed
    /// - It's what you're testing to see if it could have occurred by chance
    /// - It helps with practical interpretation of your results
    /// </para>
    /// </remarks>
    public T ObservedDifference { get; set; }

    /// <summary>
    /// Gets or sets the p-value associated with the permutation test.
    /// </summary>
    /// <value>The calculated p-value.</value>
    /// <remarks>
    /// <para>
    /// This property represents the p-value of the permutation test, which is the probability of observing a difference as 
    /// extreme as, or more extreme than, the one observed in the original data, assuming the null hypothesis is true. The 
    /// null hypothesis typically states that there is no real difference between the groups, and any observed difference is 
    /// due to random chance. The p-value is calculated as the proportion of permutations that resulted in a difference as 
    /// extreme as, or more extreme than, the observed difference. A small p-value (typically = 0.05) suggests that the 
    /// observed difference is unlikely to have occurred by chance alone, leading to the rejection of the null hypothesis.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how likely your results could occur by random chance.
    /// 
    /// The p-value:
    /// - Ranges from 0 to 1
    /// - Smaller values indicate stronger evidence against the null hypothesis
    /// - Is calculated as the proportion of permutations that produced a difference as extreme as yours
    /// 
    /// Common interpretation:
    /// - p = 0.05: Results are statistically significant (commonly used threshold)
    /// - p = 0.01: Results are highly significant
    /// - p > 0.05: Results are not statistically significant
    /// 
    /// For example, a p-value of 0.03 means that only 3% of random permutations produced
    /// a difference as extreme as what you observed, suggesting your result is unlikely
    /// to be due to random chance.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Gets or sets the number of permutations performed during the test.
    /// </summary>
    /// <value>An integer representing the number of permutations.</value>
    /// <remarks>
    /// <para>
    /// This property represents the total number of permutations (random reassignments of observations to groups) performed 
    /// during the test. A higher number of permutations generally provides a more accurate estimate of the p-value, but also 
    /// requires more computational resources. Typical values range from 1,000 to 10,000 permutations, though more may be 
    /// needed for very small p-values. The precision of the p-value is limited by the number of permutations; for example, 
    /// with 1,000 permutations, the smallest possible non-zero p-value is 0.001.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many random reshufflings were performed during the test.
    /// 
    /// The permutations count:
    /// - Indicates how many times the data was randomly reshuffled
    /// - Higher values provide more precise p-values
    /// - Typical values range from 1,000 to 10,000
    /// 
    /// This value is important because:
    /// - It affects the precision of your p-value
    /// - The smallest possible p-value is 1/permutations
    /// - More permutations require more computation time
    /// 
    /// For example, with 1,000 permutations, the smallest possible p-value is 0.001,
    /// while with 10,000 permutations, you can detect p-values as small as 0.0001.
    /// </para>
    /// </remarks>
    public int Permutations { get; set; }

    /// <summary>
    /// Gets or sets the count of permutations that resulted in a difference as extreme as, or more extreme than, the observed difference.
    /// </summary>
    /// <value>An integer representing the count of extreme values.</value>
    /// <remarks>
    /// <para>
    /// This property represents the number of permutations that resulted in a difference as extreme as, or more extreme than, 
    /// the observed difference. This count is used to calculate the p-value, which is equal to CountExtremeValues divided by 
    /// Permutations. Storing this count separately from the p-value provides additional information about the precision of 
    /// the p-value and can be useful for reporting and verification purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This counts how many random reshufflings produced results as extreme as your actual data.
    /// 
    /// The count of extreme values:
    /// - Tells you how many permutations resulted in a difference as large as or larger than yours
    /// - Is used to calculate the p-value (p-value = CountExtremeValues / Permutations)
    /// - Provides information about the precision of your p-value
    /// 
    /// This value is useful because:
    /// - It gives you the raw count behind the p-value calculation
    /// - It helps you understand how rare your observed difference is
    /// - It can be important for reporting and verification
    /// 
    /// For example, if you ran 1,000 permutations and only 12 produced differences as
    /// extreme as yours, the count would be 12 and the p-value would be 0.012.
    /// </para>
    /// </remarks>
    public int CountExtremeValues { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the permutation test result is statistically significant.
    /// </summary>
    /// <value>True if the result is statistically significant; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the permutation test result is statistically significant at the specified significance 
    /// level. A result is considered significant if the p-value is less than or equal to the significance level. Statistical 
    /// significance suggests that the observed difference between the groups is unlikely to have occurred by chance alone, 
    /// leading to the rejection of the null hypothesis that there is no real difference between the groups. This property 
    /// provides a convenient boolean indicator of significance without requiring the user to compare the p-value to the 
    /// significance level.
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
    /// This property represents the significance level used for the permutation test, which is the threshold p-value below 
    /// which the result is considered statistically significant. Common values are 0.05 (5%), 0.01 (1%), and 0.001 (0.1%). 
    /// The significance level represents the probability of rejecting the null hypothesis when it is actually true (Type I 
    /// error). A lower significance level reduces the risk of Type I errors but increases the risk of Type II errors (failing 
    /// to reject a false null hypothesis).
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
    /// Initializes a new instance of the PermutationTestResult class with the specified test statistics and parameters.
    /// </summary>
    /// <param name="observedDifference">The observed difference between the two groups being compared.</param>
    /// <param name="pValue">The p-value associated with the permutation test.</param>
    /// <param name="permutations">The number of permutations performed during the test.</param>
    /// <param name="countExtremeValues">The count of permutations that resulted in a difference as extreme as, or more extreme than, the observed difference.</param>
    /// <param name="significanceLevel">The significance level used for the test.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new PermutationTestResult instance with the specified test statistics and parameters. It 
    /// initializes all properties of the class and calculates the IsSignificant property by comparing the p-value to the 
    /// significance level. The IsSignificant property is set to true if the p-value is less than the significance level, 
    /// indicating that the test result is statistically significant. This constructor provides a convenient way to create 
    /// a complete PermutationTestResult object in a single step after performing a permutation test.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with all the permutation test statistics and parameters.
    /// 
    /// When a new PermutationTestResult is created with this constructor:
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
    public PermutationTestResult(T observedDifference, T pValue, int permutations, int countExtremeValues, T significanceLevel)
    {
        ObservedDifference = observedDifference;
        PValue = pValue;
        Permutations = permutations;
        CountExtremeValues = countExtremeValues;
        SignificanceLevel = significanceLevel;
        IsSignificant = MathHelper.GetNumericOperations<T>().LessThan(PValue, SignificanceLevel);
    }
}
