namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Conditional Inference Trees, a statistically-driven approach to decision tree learning.
/// </summary>
/// <remarks>
/// <para>
/// Conditional Inference Trees use statistical tests to select variables and determine split points,
/// which helps reduce selection bias toward variables with many possible split points.
/// This approach often produces more reliable and statistically sound trees compared to traditional methods.
/// </para>
/// <para><b>For Beginners:</b> Conditional Inference Trees are a special type of decision tree that uses 
/// statistics to make better decisions about how to split data. Regular decision trees sometimes favor 
/// certain types of data unfairly (like preferring variables with more possible values). This approach is 
/// like having a referee that makes sure the tree-building process is fair and statistically sound. 
/// The result is often a more reliable model, especially for data where some variables have many possible 
/// values and others have few.</para>
/// </remarks>
public class ConditionalInferenceTreeOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the statistical significance level used for hypothesis testing when selecting split variables.
    /// </summary>
    /// <value>The significance level as a decimal between 0 and 1, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// The significance level determines how strong the evidence must be before the algorithm will split on a variable.
    /// Lower values (e.g., 0.01) require stronger evidence and typically result in simpler trees.
    /// Higher values (e.g., 0.1) accept weaker evidence and may produce more complex trees.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how "picky" the algorithm is when deciding whether to split 
    /// the data. The default value (0.05 or 5%) is a standard threshold in statistics. Think of it like a judge in a 
    /// courtroom - a strict judge (lower value like 0.01) needs very strong evidence before making a decision, resulting 
    /// in fewer splits and a simpler tree. A more lenient judge (higher value like 0.1) will accept weaker evidence, 
    /// potentially creating more splits and a more complex tree. If your tree seems too simple, you might increase this 
    /// value slightly; if it's too complex, you might decrease it.</para>
    /// </remarks>
    public double SignificanceLevel { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the type of statistical test used to evaluate potential splits.
    /// </summary>
    /// <value>The statistical test type, defaulting to TTest.</value>
    /// <remarks>
    /// <para>
    /// Different statistical tests are appropriate for different types of data:
    /// - T-tests are suitable for continuous target variables
    /// - Chi-square tests work well for categorical target variables
    /// - ANOVA is useful when comparing multiple groups
    /// The algorithm will automatically select an appropriate test in many cases, but this option allows manual override.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines what kind of statistical method the algorithm uses to decide 
    /// if a split is meaningful. Think of it like choosing the right tool for a job - you wouldn't use a hammer to tighten 
    /// a screw. The default (TTest) works well for numerical data (like predicting prices or temperatures). If you're 
    /// working with categorical data (like predicting categories or yes/no outcomes), a different test like ChiSquare 
    /// might be more appropriate. In many cases, the algorithm can select the right test automatically, so you only need 
    /// to change this if you have specific knowledge about your data.</para>
    /// </remarks>
    public TestStatisticType StatisticalTest { get; set; } = TestStatisticType.TTest;

    /// <summary>
    /// Gets or sets the maximum number of parallel operations when building the tree.
    /// </summary>
    /// <value>The maximum degree of parallelism, defaulting to the number of processor cores available.</value>
    /// <remarks>
    /// <para>
    /// This setting controls how many CPU cores the algorithm will use when building the tree.
    /// Higher values can speed up training on multi-core systems but may increase memory usage.
    /// The default uses all available processor cores for maximum performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many parts of the tree-building process can happen at the 
    /// same time. By default, it uses all the processing power your computer has available (all CPU cores). Think of it 
    /// like having multiple workers building different branches of the tree simultaneously. This makes the process faster, 
    /// but sometimes uses more memory. If your computer is struggling with memory issues when building large trees, you 
    /// might want to reduce this number (e.g., to half the number of cores). For most users, the default setting works 
    /// well and provides the best performance.</para>
    /// </remarks>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}
