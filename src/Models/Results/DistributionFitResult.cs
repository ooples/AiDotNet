namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the result of fitting a statistical distribution to a dataset, including the distribution type,
/// goodness of fit measure, and estimated parameters.
/// </summary>
/// <remarks>
/// <para>
/// When analyzing data, it's often useful to determine which statistical distribution best describes the data. 
/// This class stores the results of such distribution fitting, including the type of distribution that best fits 
/// the data, a measure of how well the distribution fits (goodness of fit), and the estimated parameters of the 
/// distribution. The goodness of fit measure allows for comparing different distribution types to determine which 
/// one provides the best representation of the data. The class uses generic type parameter T to support different 
/// numeric types for the statistical values, such as float, double, or decimal.
/// </para>
/// <para><b>For Beginners:</b> This class stores information about how well a statistical distribution matches your data.
/// 
/// When analyzing data, you often want to know:
/// - Which standard statistical distribution (like Normal, Exponential, etc.) best describes your data
/// - How well that distribution fits your data
/// - What the specific parameters of that distribution are
/// 
/// This class stores all that information after a distribution fitting process, making it easy to:
/// - Identify the best distribution for your data
/// - Compare how well different distributions fit
/// - Access the parameters needed to use the distribution for further analysis
/// 
/// For example, if your data follows a normal distribution, this class would tell you that,
/// provide a measure of how well it fits, and give you the mean and standard deviation parameters.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for statistical values, typically float or double.</typeparam>
public class DistributionFitResult<T>
{
    private readonly INumericOperations<T> _ops;

    /// <summary>
    /// Gets or sets the type of distribution that best fits the data.
    /// </summary>
    /// <value>A value from the DistributionType enumeration indicating the distribution type.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the type of statistical distribution that was fitted to the data. Common distribution 
    /// types include Normal (Gaussian), Exponential, Weibull, Gamma, Beta, and others. Each distribution type has 
    /// different characteristics and is suitable for modeling different kinds of data. For example, the Normal 
    /// distribution is often used for data that clusters around a mean value, while the Exponential distribution 
    /// is suitable for modeling time between events in a Poisson process. The distribution type is typically 
    /// determined by comparing the goodness of fit measures for different distributions and selecting the one with 
    /// the best fit.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you which statistical distribution best matches your data.
    /// 
    /// The distribution type:
    /// - Identifies which standard probability distribution was found to fit your data
    /// - Each distribution has different characteristics and is suitable for different types of data
    /// 
    /// Common distribution types include:
    /// - Normal (Gaussian): The classic bell curve, good for many natural phenomena
    /// - Exponential: Good for modeling time between random events
    /// - Uniform: Equal probability across a range
    /// - Poisson: Good for counting rare events in fixed time/space
    /// - Weibull: Often used for reliability and lifetime data
    /// 
    /// Knowing the distribution type helps you:
    /// - Understand the underlying patterns in your data
    /// - Make predictions about future observations
    /// - Apply appropriate statistical methods for further analysis
    /// 
    /// For example, if this property is set to DistributionType.Normal, it means your
    /// data approximately follows a bell curve pattern.
    /// </para>
    /// </remarks>
    public DistributionType DistributionType { get; set; }

    /// <summary>
    /// Gets or sets the goodness of fit measure. Lower values indicate better fit.
    /// </summary>
    /// <value>A numeric value representing the goodness of fit, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents a measure of how well the selected distribution fits the data. Common goodness of fit 
    /// measures include the Kolmogorov-Smirnov statistic, Anderson-Darling statistic, or negative log-likelihood. 
    /// Lower values typically indicate a better fit, meaning the distribution more closely matches the observed data. 
    /// This measure can be used to compare different distribution types to determine which one provides the best 
    /// representation of the data. The specific interpretation of the goodness of fit value depends on the measure 
    /// used, but it generally allows for relative comparisons between different distribution fits.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how well the distribution matches your data.
    /// 
    /// The goodness of fit measure:
    /// - Quantifies how closely the distribution matches your actual data
    /// - Lower values indicate a better fit (less discrepancy between the model and data)
    /// - Allows you to compare different distributions to find the best one
    /// 
    /// This value is important because:
    /// - It helps you determine if the distribution is a good model for your data
    /// - It allows you to compare multiple distribution types objectively
    /// - It can indicate when no standard distribution fits your data well
    /// 
    /// Common goodness of fit measures include:
    /// - Kolmogorov-Smirnov statistic: Based on the maximum difference between empirical and theoretical CDFs
    /// - Anderson-Darling statistic: Similar but gives more weight to the tails of the distribution
    /// - Negative log-likelihood: Based on the probability of observing your data given the distribution
    /// 
    /// For example, if you fit both Normal and Exponential distributions to your data,
    /// the one with the lower goodness of fit value would be considered the better match.
    /// </para>
    /// </remarks>
    public T GoodnessOfFit { get; set; }

    /// <summary>
    /// Gets or sets the parameters of the fitted distribution.
    /// The keys are parameter names, and the values are the corresponding parameter values.
    /// </summary>
    /// <value>A dictionary mapping parameter names to their values, initialized as an empty dictionary.</value>
    /// <remarks>
    /// <para>
    /// This property stores the estimated parameters of the fitted distribution as a dictionary, where the keys are 
    /// the parameter names and the values are the corresponding parameter values. The specific parameters depend on 
    /// the distribution type. For example, a Normal distribution has "mean" and "standardDeviation" parameters, an 
    /// Exponential distribution has a "rate" parameter, and a Weibull distribution has "shape" and "scale" parameters. 
    /// These parameters fully define the distribution and can be used for further statistical analysis, such as 
    /// calculating probabilities, generating random samples, or making predictions. The parameter values are estimated 
    /// from the data during the distribution fitting process, typically using methods such as maximum likelihood 
    /// estimation or method of moments.
    /// </para>
    /// <para><b>For Beginners:</b> This dictionary contains the specific values that define the distribution.
    /// 
    /// The distribution parameters:
    /// - Are the specific values that define the shape and characteristics of the distribution
    /// - Vary depending on which distribution type was selected
    /// - Are estimated from your data during the fitting process
    /// 
    /// Common parameters for different distributions:
    /// - Normal distribution: "mean" and "standardDeviation"
    /// - Exponential distribution: "rate" or "lambda"
    /// - Uniform distribution: "minimum" and "maximum"
    /// - Weibull distribution: "shape" and "scale"
    /// 
    /// These parameters are important because:
    /// - They completely define the distribution
    /// - They allow you to calculate probabilities and statistics
    /// - They can be used to generate random samples from the distribution
    /// - They often have meaningful interpretations in your domain
    /// 
    /// For example, if fitting a Normal distribution to height data, the "mean" parameter
    /// would represent the average height, and "standardDeviation" would represent how
    /// spread out the heights are around that average.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Parameters { get; set; }

    /// <summary>
    /// Initializes a new instance of the DistributionFitResult class with default values.
    /// </summary>
    /// <param name="ops">Optional numeric operations provider for type T. If null, default operations will be used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new DistributionFitResult instance and initializes the GoodnessOfFit property to zero 
    /// and the Parameters property to an empty dictionary. It takes an optional INumericOperations&lt;T&gt; parameter 
    /// that provides numeric operations for the generic type T. If this parameter is null, the constructor uses 
    /// MathHelper.GetNumericOperations&lt;T&gt;() to obtain the default numeric operations for type T. This allows the 
    /// class to work with different numeric types such as float, double, or decimal. The constructor provides a clean 
    /// starting point for storing distribution fitting results.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with default values.
    /// 
    /// When a new DistributionFitResult is created:
    /// - The goodness of fit is initialized to zero
    /// - The parameters dictionary is created as empty
    /// - Numeric operations appropriate for type T are set up
    /// 
    /// The ops parameter:
    /// - Provides mathematical operations for the numeric type T
    /// - Can be null, in which case default operations are used
    /// - Allows the class to work with different numeric types (float, double, etc.)
    /// 
    /// This initialization is important because:
    /// - It ensures consistent behavior regardless of how the object is created
    /// - It prevents potential issues with uninitialized values
    /// - It makes the code more robust across different numeric types
    /// 
    /// You typically won't need to call this constructor directly, as it will be
    /// used internally by the distribution fitting process.
    /// </para>
    /// </remarks>
    public DistributionFitResult(INumericOperations<T>? ops = null)
    {
        _ops = ops ?? MathHelper.GetNumericOperations<T>();
        GoodnessOfFit = _ops.Zero;
        Parameters = new Dictionary<string, T>();
    }
}
