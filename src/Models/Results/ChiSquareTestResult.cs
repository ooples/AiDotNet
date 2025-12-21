namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a Chi-Square statistical test, which is used to determine whether there is a significant 
/// association between two categorical variables.
/// </summary>
/// <remarks>
/// <para>
/// The Chi-Square test is a statistical hypothesis test that evaluates whether observed frequencies differ significantly 
/// from expected frequencies. It is commonly used to test the independence of two categorical variables or to assess 
/// the goodness of fit between observed data and a theoretical distribution. This class stores the results of such a 
/// test, including the test statistic, p-value, degrees of freedom, observed and expected frequencies, and whether 
/// the result is statistically significant. The class uses generic type parameter T to support different numeric types 
/// for the statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> The Chi-Square test helps determine if there's a meaningful relationship between two categorical variables.
/// 
/// For example, you might use this test to answer questions like:
/// - Is there a relationship between gender and preference for a product?
/// - Does treatment type affect recovery rates?
/// - Are survey responses distributed as expected?
/// 
/// The test works by:
/// 1. Comparing observed frequencies (what you actually counted) with expected frequencies (what you would expect if there's no relationship)
/// 2. Calculating a statistic that measures how different these frequencies are
/// 3. Determining if this difference is statistically significant
/// 
/// This class stores all the information about the test results, helping you interpret whether
/// the observed differences are likely due to chance or represent a real association.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class ChiSquareTestResult<T>
{
    /// <summary>
    /// Gets or sets the Chi-Square test statistic value.
    /// </summary>
    /// <value>The calculated Chi-Square statistic, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents the Chi-Square test statistic, which measures the difference between observed and 
    /// expected frequencies. It is calculated as the sum of (observed - expected)²/expected across all categories. 
    /// Larger values indicate greater differences between observed and expected frequencies, suggesting a stronger 
    /// association between the variables or a poorer fit to the expected distribution. The Chi-Square statistic 
    /// follows a Chi-Square distribution with degrees of freedom determined by the number of categories in the data. 
    /// This statistic is used in conjunction with the degrees of freedom to calculate the p-value, which determines 
    /// statistical significance.
    /// </para>
    /// <para><b>For Beginners:</b> This value measures how different your observed data is from what you expected.
    /// 
    /// The Chi-Square statistic:
    /// - Quantifies the total difference between observed and expected frequencies
    /// - Is always positive (or zero if observed exactly matches expected)
    /// - Larger values indicate greater differences
    /// 
    /// This value is calculated by:
    /// 1. Finding the difference between each observed and expected frequency
    /// 2. Squaring each difference (to make all values positive)
    /// 3. Dividing by the expected frequency (to standardize)
    /// 4. Summing all these values
    /// 
    /// For example, a Chi-Square statistic of 0 means perfect agreement between observed and expected,
    /// while a value of 15.3 indicates substantial differences that may be statistically significant.
    /// </para>
    /// </remarks>
    public T ChiSquareStatistic { get; set; }

    /// <summary>
    /// Gets or sets the p-value associated with the Chi-Square test.
    /// </summary>
    /// <value>The calculated p-value, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents the p-value of the Chi-Square test, which is the probability of observing a test 
    /// statistic as extreme as, or more extreme than, the one calculated from the sample data, assuming the null 
    /// hypothesis is true. The null hypothesis typically states that there is no association between the variables 
    /// or that the data follows the expected distribution. A small p-value (typically = 0.05) suggests that the 
    /// observed data is unlikely under the null hypothesis, leading to its rejection in favor of the alternative 
    /// hypothesis. The p-value is calculated from the Chi-Square statistic and the degrees of freedom using the 
    /// cumulative distribution function of the Chi-Square distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how likely your results could occur by random chance.
    /// 
    /// The p-value:
    /// - Ranges from 0 to 1
    /// - Smaller values indicate stronger evidence against the null hypothesis
    /// - Represents the probability of seeing your results (or more extreme) if there's no real relationship
    /// 
    /// Common interpretation:
    /// - p = 0.05: Results are statistically significant (commonly used threshold)
    /// - p = 0.01: Results are highly significant
    /// - p > 0.05: Results are not statistically significant
    /// 
    /// For example, a p-value of 0.03 means there's only a 3% chance of seeing your results
    /// if there's no real relationship between the variables, suggesting the relationship
    /// is likely real and not due to random chance.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Gets or sets the degrees of freedom for the Chi-Square test.
    /// </summary>
    /// <value>An integer representing the degrees of freedom.</value>
    /// <remarks>
    /// <para>
    /// This property represents the degrees of freedom for the Chi-Square test, which is a parameter of the Chi-Square 
    /// distribution used to calculate the p-value. For a test of independence between two categorical variables, the 
    /// degrees of freedom is calculated as (r-1)×(c-1), where r is the number of rows (categories of the first variable) 
    /// and c is the number of columns (categories of the second variable). For a goodness-of-fit test, the degrees of 
    /// freedom is k-1-m, where k is the number of categories and m is the number of parameters estimated from the data. 
    /// The degrees of freedom affects the shape of the Chi-Square distribution and thus the interpretation of the 
    /// Chi-Square statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This value represents the number of values that are free to vary in your statistical calculation.
    /// 
    /// The degrees of freedom:
    /// - Is a parameter needed to interpret the Chi-Square statistic
    /// - Depends on the number of categories in your data
    /// - For a test of independence: (rows-1) × (columns-1)
    /// - For a goodness-of-fit test: (categories-1)
    /// 
    /// This value is important because:
    /// - The same Chi-Square statistic can have different meanings with different degrees of freedom
    /// - It determines the shape of the Chi-Square distribution used to calculate the p-value
    /// - Higher degrees of freedom generally require larger Chi-Square values to be significant
    /// 
    /// For example, a Chi-Square statistic of 7.5 with 1 degree of freedom is significant at p < 0.01,
    /// but the same statistic with 3 degrees of freedom is not significant at p < 0.05.
    /// </para>
    /// </remarks>
    public int DegreesOfFreedom { get; set; }

    /// <summary>
    /// Gets or sets the observed frequencies for the left variable or category.
    /// </summary>
    /// <value>A vector of observed frequencies, initialized to an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property represents the observed frequencies for the left variable or category in the Chi-Square test. 
    /// Observed frequencies are the actual counts or occurrences of each category in the sample data. In a test of 
    /// independence, this might represent the counts for different categories of one variable, while in a goodness-of-fit 
    /// test, it might represent the observed counts for each category being tested against an expected distribution. 
    /// These observed frequencies are compared with the expected frequencies to calculate the Chi-Square statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the actual counts you observed in your data for the first variable.
    /// 
    /// The left observed frequencies:
    /// - Represent the actual counts from your data collection
    /// - Are compared against expected frequencies to calculate the Chi-Square statistic
    /// - For a contingency table, these would be the counts for one of the variables
    /// 
    /// For example, if testing the relationship between gender and product preference,
    /// this might contain the counts of males and females in your sample.
    /// </para>
    /// </remarks>
    public Vector<T> LeftObserved { get; set; }

    /// <summary>
    /// Gets or sets the observed frequencies for the right variable or category.
    /// </summary>
    /// <value>A vector of observed frequencies, initialized to an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property represents the observed frequencies for the right variable or category in the Chi-Square test. 
    /// Observed frequencies are the actual counts or occurrences of each category in the sample data. In a test of 
    /// independence, this might represent the counts for different categories of the second variable. These observed 
    /// frequencies are compared with the expected frequencies to calculate the Chi-Square statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the actual counts you observed in your data for the second variable.
    /// 
    /// The right observed frequencies:
    /// - Represent the actual counts from your data collection
    /// - Are compared against expected frequencies to calculate the Chi-Square statistic
    /// - For a contingency table, these would be the counts for the other variable
    /// 
    /// For example, if testing the relationship between gender and product preference,
    /// this might contain the counts of people preferring each product option.
    /// </para>
    /// </remarks>
    public Vector<T> RightObserved { get; set; }

    /// <summary>
    /// Gets or sets the expected frequencies for the left variable or category.
    /// </summary>
    /// <value>A vector of expected frequencies, initialized to an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property represents the expected frequencies for the left variable or category in the Chi-Square test. 
    /// Expected frequencies are the counts that would be expected if the null hypothesis were true (i.e., if there 
    /// were no association between the variables or if the data followed the expected distribution). In a test of 
    /// independence, expected frequencies are calculated based on the marginal totals of the contingency table. 
    /// These expected frequencies are compared with the observed frequencies to calculate the Chi-Square statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the counts you would expect for the first variable if there's no relationship.
    /// 
    /// The left expected frequencies:
    /// - Represent what you would expect to see if the null hypothesis is true
    /// - Are calculated based on the marginal totals and sample size
    /// - The difference between these and observed frequencies drives the Chi-Square statistic
    /// 
    /// For example, in a gender and product preference test, this might contain the expected
    /// counts of males and females based on the overall proportions in your sample.
    /// </para>
    /// </remarks>
    public Vector<T> LeftExpected { get; set; }

    /// <summary>
    /// Gets or sets the expected frequencies for the right variable or category.
    /// </summary>
    /// <value>A vector of expected frequencies, initialized to an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property represents the expected frequencies for the right variable or category in the Chi-Square test. 
    /// Expected frequencies are the counts that would be expected if the null hypothesis were true (i.e., if there 
    /// were no association between the variables or if the data followed the expected distribution). In a test of 
    /// independence, expected frequencies are calculated based on the marginal totals of the contingency table. 
    /// These expected frequencies are compared with the observed frequencies to calculate the Chi-Square statistic.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the counts you would expect for the second variable if there's no relationship.
    /// 
    /// The right expected frequencies:
    /// - Represent what you would expect to see if the null hypothesis is true
    /// - Are calculated based on the marginal totals and sample size
    /// - The difference between these and observed frequencies drives the Chi-Square statistic
    /// 
    /// For example, in a gender and product preference test, this might contain the expected
    /// counts for each product option based on the overall proportions in your sample.
    /// </para>
    /// </remarks>
    public Vector<T> RightExpected { get; set; }

    /// <summary>
    /// Gets or sets the critical value for the Chi-Square test at the specified significance level.
    /// </summary>
    /// <value>The critical value from the Chi-Square distribution, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents the critical value from the Chi-Square distribution for the given degrees of freedom 
    /// and significance level (typically 0.05). The critical value is the threshold value of the Chi-Square statistic 
    /// above which the null hypothesis would be rejected at the specified significance level. If the calculated 
    /// Chi-Square statistic exceeds this critical value, the result is considered statistically significant. The 
    /// critical value is determined by the inverse cumulative distribution function of the Chi-Square distribution, 
    /// using the degrees of freedom and the complement of the significance level.
    /// </para>
    /// <para><b>For Beginners:</b> This value is the threshold that determines statistical significance.
    /// 
    /// The critical value:
    /// - Is the cutoff point for the Chi-Square statistic to be considered significant
    /// - Depends on the degrees of freedom and chosen significance level (typically 0.05)
    /// - If the Chi-Square statistic exceeds this value, the result is statistically significant
    /// 
    /// This value is important because:
    /// - It provides a clear threshold for decision-making
    /// - It's directly related to your chosen significance level
    /// - It allows for quick determination of significance without checking p-values
    /// 
    /// For example, with 2 degrees of freedom and a significance level of 0.05,
    /// the critical value is approximately 5.99. Any Chi-Square statistic above
    /// this value would be considered statistically significant.
    /// </para>
    /// </remarks>
    public T CriticalValue { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the Chi-Square test result is statistically significant.
    /// </summary>
    /// <value>True if the result is statistically significant; otherwise, false.</value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the Chi-Square test result is statistically significant at the specified 
    /// significance level (typically 0.05). A result is considered significant if the p-value is less than or equal 
    /// to the significance level, or equivalently, if the Chi-Square statistic is greater than or equal to the 
    /// critical value. Statistical significance suggests that the observed differences between the observed and 
    /// expected frequencies are unlikely to have occurred by chance alone, leading to the rejection of the null 
    /// hypothesis. This property provides a convenient boolean indicator of significance without requiring the 
    /// user to compare the p-value or Chi-Square statistic to thresholds.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether your result is statistically significant or not.
    /// 
    /// The significance indicator:
    /// - Provides a simple yes/no answer about statistical significance
    /// - True means the relationship or difference is statistically significant
    /// - False means the result could reasonably be due to random chance
    /// 
    /// This is determined by:
    /// - Comparing the p-value to the significance level (typically 0.05)
    /// - Or comparing the Chi-Square statistic to the critical value
    /// 
    /// This value makes it easy to interpret your results without having to manually
    /// check if the p-value is below the significance threshold. It's particularly
    /// useful when automating analysis or presenting results to non-statisticians.
    /// </para>
    /// </remarks>
    public bool IsSignificant { get; set; }

    /// <summary>
    /// Initializes a new instance of the ChiSquareTestResult class with all statistical values set to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ChiSquareTestResult instance and initializes all statistical values 
    /// (ChiSquareStatistic, PValue, CriticalValue) to zero and all frequency vectors (LeftObserved, RightObserved, 
    /// LeftExpected, RightExpected) to empty vectors. It uses the MathHelper.GetNumericOperations method to obtain 
    /// the appropriate numeric operations for the generic type T, which allows the class to work with different 
    /// numeric types such as float, double, or decimal. This initialization ensures that the statistical values 
    /// start at a well-defined state before being updated with actual results from the Chi-Square test.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with all values initialized to zero.
    /// 
    /// When a new ChiSquareTestResult is created:
    /// - All statistical values are set to zero
    /// - All frequency vectors are initialized as empty
    /// - The constructor uses MathHelper to handle different numeric types
    /// - This provides a clean starting point before actual results are calculated
    /// 
    /// This initialization is important because:
    /// - It ensures consistent behavior regardless of how the object is created
    /// - It prevents potential issues with uninitialized values
    /// - It makes the code more robust across different numeric types
    /// 
    /// You typically won't need to call this constructor directly, as it will be
    /// used internally by the Chi-Square test implementation.
    /// </para>
    /// </remarks>
    public ChiSquareTestResult()
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        ChiSquareStatistic = numOps.Zero;
        PValue = numOps.Zero;
        CriticalValue = numOps.Zero;
        LeftObserved = Vector<T>.Empty();
        RightObserved = Vector<T>.Empty();
        LeftExpected = Vector<T>.Empty();
        RightExpected = Vector<T>.Empty();
    }
}
