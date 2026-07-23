namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of statistical tests used to evaluate hypotheses and determine significance in data analysis.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Statistical tests help us decide if patterns we see in data are real or just due to chance.
/// 
/// Think of statistical tests like different tools in a toolbox - each one is designed for specific situations:
/// 
/// Imagine you're trying to determine if a coin is fair (50% chance of heads). You could flip it 100 times
/// and count how many heads you get. If you get exactly 50 heads, it seems fair. But what if you get 55 heads?
/// Or 60? At what point do you decide the coin is unfair? Statistical tests give us mathematical ways to
/// make these decisions based on probability rather than just guessing.
/// 
/// Different tests are designed for different types of data and questions, just like you'd use different
/// tools for different home repair jobs.
/// 
/// These tests calculate a "p-value" - the probability that the pattern you observed could happen by random chance.
/// A small p-value (typically &lt; 0.05) suggests the pattern is statistically significant and not just random.
/// </remarks>
public enum TestStatisticType
{
    /// <summary>
    /// A statistical test used to determine if there is a significant association between categorical variables.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Chi-Square test helps us understand if two categorical characteristics are related.
    /// 
    /// Imagine you want to know if ice cream flavor preference (chocolate, vanilla, strawberry) is related
    /// to gender. The Chi-Square test compares the actual distribution of preferences across genders with
    /// what we would expect if there was no relationship.
    /// 
    /// When to use it:
    /// - When your data falls into categories (like yes/no, red/blue/green, etc.)
    /// - When you want to know if one categorical variable is related to another
    /// - When you have counted data (frequencies) rather than measurements
    /// 
    /// Example: Testing if treatment type (medication A, B, or placebo) is related to recovery outcome (recovered/not recovered).
    /// </remarks>
    ChiSquare,

    /// <summary>
    /// A statistical test that compares the variances of two or more groups to determine if they are significantly different.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The F-Test helps us compare the spread or variability between different groups.
    /// 
    /// Imagine comparing the test scores from three different teaching methods. The F-Test can tell us
    /// if one method produces more consistent results (less variability) than others.
    /// 
    /// The F-Test is also the foundation of ANOVA (Analysis of Variance), which compares means across multiple groups.
    /// 
    /// When to use it:
    /// - When comparing more than two groups
    /// - When your data is numerical (like heights, weights, scores)
    /// - When you want to know if groups differ in their average values
    /// 
    /// Example: Testing if three different fertilizers produce different average crop yields.
    /// </remarks>
    FTest,

    /// <summary>
    /// A statistical test used to determine if there is a significant difference between the means of two groups.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The T-Test helps us decide if two groups have different averages.
    /// 
    /// Imagine comparing the heights of men and women. The T-Test tells us if the difference
    /// in average height between the two groups is statistically significant or could have
    /// happened by chance.
    /// 
    /// When to use it:
    /// - When comparing exactly two groups
    /// - When your data is numerical (like heights, weights, scores)
    /// - When your data approximately follows a normal distribution (bell curve)
    /// 
    /// Example: Testing if a new medication affects blood pressure by comparing before and after measurements.
    /// </remarks>
    TTest,

    /// <summary>
    /// A non-parametric test that compares two independent samples without assuming they follow a normal distribution.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Mann-Whitney U test is like a T-Test but works when your data doesn't follow a nice, neat pattern.
    /// 
    /// Imagine comparing customer satisfaction ratings (1-5 stars) between two restaurants. Since ratings
    /// are often skewed (not following a bell curve), the Mann-Whitney U test is more appropriate than a T-Test.
    /// 
    /// When to use it:
    /// - When comparing two independent groups
    /// - When your data doesn't follow a normal distribution
    /// - When your data is ordinal (has a natural order but not equal intervals)
    /// - When you have outliers that might skew results
    /// 
    /// Example: Comparing pain relief scores (on a scale of 1-10) between two different treatments.
    /// </remarks>
    MannWhitneyU,

    /// <summary>
    /// A resampling-based test that repeatedly shuffles observed data to determine if patterns are statistically significant.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The Permutation Test is like shuffling a deck of cards many times to see how likely a particular arrangement is.
    /// 
    /// Imagine you have test scores from students who studied using two different methods. You mix all scores together
    /// and randomly reassign them to the two methods thousands of times. If the original difference between methods
    /// is larger than what you typically see in these random reassignments, it suggests the difference is significant.
    /// 
    /// When to use it:
    /// - When traditional tests' assumptions aren't met
    /// - When you have small sample sizes
    /// - When you want to avoid making assumptions about your data's distribution
    /// - When you need a flexible approach for complex data
    /// 
    /// Example: Testing if gene expression patterns differ between healthy and diseased tissue samples.
    /// 
    /// Note: Permutation tests are computationally intensive but very flexible and powerful.
    /// </remarks>
    PermutationTest
}
