namespace AiDotNet.Enums;

/// <summary>
/// Represents different probability distributions used in statistical modeling and machine learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Probability distributions are mathematical functions that describe how likely different outcomes are.
/// 
/// Think of a probability distribution like a recipe for how values are spread out:
/// - Some distributions create values clustered around a central point (like Normal)
/// - Others spread values out differently (some have long tails, some are skewed)
/// - Each has specific mathematical properties that make it useful for different situations
/// 
/// In machine learning, we use these distributions to:
/// - Model uncertainty and randomness
/// - Generate synthetic data
/// - Make predictions with confidence intervals
/// - Understand the underlying patterns in our data
/// 
/// Choosing the right distribution depends on what kind of data you're working with and
/// what assumptions you can reasonably make about how that data is generated.
/// </para>
/// </remarks>
public enum DistributionType
{
    /// <summary>
    /// The bell-shaped distribution that is symmetric around its mean (also known as Gaussian distribution).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Normal distribution is the most commonly used probability distribution, characterized by its 
    /// symmetric bell shape.
    /// 
    /// It's defined by two parameters:
    /// - Mean (µ): The center of the distribution
    /// - Standard deviation (s): How spread out the values are
    /// 
    /// Think of it as a distribution where:
    /// - Most values cluster around the mean
    /// - About 68% of values fall within 1 standard deviation of the mean
    /// - About 95% of values fall within 2 standard deviations
    /// - About 99.7% of values fall within 3 standard deviations
    /// 
    /// Best used for:
    /// - Natural phenomena that result from many small, independent factors
    /// - Measurement errors
    /// - Average values of large samples (due to Central Limit Theorem)
    /// - Many biological measurements (height, weight, etc.)
    /// 
    /// Examples:
    /// - Heights of people in a population
    /// - Measurement errors in scientific experiments
    /// - Test scores in large populations
    /// - Financial returns in efficient markets
    /// </para>
    /// </remarks>
    Normal,

    /// <summary>
    /// A distribution with a sharper peak and heavier tails than the Normal distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Laplace distribution (also called the double exponential distribution) looks similar 
    /// to the Normal distribution but has a sharper peak and heavier tails.
    /// 
    /// Think of it as a more "extreme" version of the Normal distribution:
    /// - It has a pointy center instead of a rounded one
    /// - It has fatter tails, meaning extreme values are more likely
    /// 
    /// It's defined by two parameters:
    /// - Location (µ): The center of the distribution
    /// - Scale (b): Controls how spread out the values are
    /// 
    /// Best used for:
    /// - Data with occasional large deviations
    /// - Error distributions in some financial models
    /// - Modeling sparse signals
    /// - Regularization in machine learning (L1 regularization)
    /// 
    /// Examples:
    /// - Financial returns with occasional large movements
    /// - Differences between paired observations
    /// - Error distributions in some regression problems
    /// - Modeling noise in signal processing
    /// </para>
    /// </remarks>
    Laplace,

    /// <summary>
    /// A family of distributions that resemble the Normal distribution but have heavier tails (also known as t-distribution).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Student's t-distribution (often just called t-distribution) looks similar to the 
    /// Normal distribution but has heavier tails, meaning extreme values are more likely.
    /// 
    /// Think of it as a Normal distribution that's more forgiving of outliers:
    /// - It looks like a bell curve but with thicker tails
    /// - As the degrees of freedom increase, it gets closer to a Normal distribution
    /// 
    /// It's defined by one parameter:
    /// - Degrees of freedom (?): Controls the heaviness of the tails
    ///   - Lower values = heavier tails
    ///   - Higher values = more like a Normal distribution
    /// 
    /// Best used for:
    /// - Small sample statistics
    /// - Robust statistical methods
    /// - Data where outliers are expected
    /// - Confidence intervals when the population standard deviation is unknown
    /// 
    /// Examples:
    /// - Statistical tests with small samples
    /// - Financial returns modeling
    /// - Robust regression
    /// - Bayesian inference
    /// </para>
    /// </remarks>
    Student,

    /// <summary>
    /// A distribution whose logarithm follows a Normal distribution, resulting in a skewed shape.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The LogNormal distribution applies to data whose logarithm follows a Normal distribution.
    /// 
    /// Think of it as what happens when you take normally distributed data and exponentiate it (e^x):
    /// - It's always positive (never zero or negative)
    /// - It's skewed to the right (has a long tail on the right side)
    /// - Most values are clustered on the left side
    /// 
    /// It's defined by two parameters:
    /// - µ: The mean of the logarithm of the data
    /// - s: The standard deviation of the logarithm of the data
    /// 
    /// Best used for:
    /// - Quantities that are the product of many small independent factors
    /// - Data that can't be negative and is right-skewed
    /// - Growth processes
    /// 
    /// Examples:
    /// - Income distributions
    /// - House prices
    /// - Stock prices
    /// - Particle sizes
    /// - Length of time to complete a task
    /// - Many biological measurements
    /// </para>
    /// </remarks>
    LogNormal,

    /// <summary>
    /// A distribution that models the time between events in a process where events occur continuously and independently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Exponential distribution models the time between independent events that occur at a constant average rate.
    /// 
    /// Think of it as describing how long you wait for something to happen when the chance of it happening 
    /// at any moment stays the same:
    /// - It's skewed to the right
    /// - Short waiting times are more likely than long ones
    /// - It has the "memoryless" property: the probability of waiting another hour doesn't depend on how long you've already waited
    /// 
    /// It's defined by one parameter:
    /// - ? (lambda): The rate parameter, which is the average number of events per unit time
    /// 
    /// Best used for:
    /// - Time between events in a Poisson process
    /// - Survival analysis
    /// - Reliability engineering
    /// - Waiting times
    /// 
    /// Examples:
    /// - Time between customer arrivals
    /// - Time until equipment failure
    /// - Length of phone calls
    /// - Time between radioactive decay events
    /// - Time until the next earthquake
    /// </para>
    /// </remarks>
    Exponential,

    /// <summary>
    /// A flexible distribution used to model a wide variety of data, particularly in reliability analysis and lifetime modeling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Weibull distribution is a versatile distribution often used to model lifetimes and failure rates.
    /// 
    /// Think of it as a flexible distribution that can take many different shapes:
    /// - It can look like an Exponential distribution
    /// - It can look similar to a Normal distribution
    /// - It can be skewed in different ways
    /// 
    /// It's defined by two main parameters:
    /// - k (shape): Controls the shape of the distribution
    ///   - k < 1: Failure rate decreases over time
    ///   - k = 1: Constant failure rate (becomes Exponential distribution)
    ///   - k > 1: Failure rate increases over time
    /// - ? (scale): Stretches or compresses the distribution
    /// 
    /// Best used for:
    /// - Lifetime modeling
    /// - Reliability engineering
    /// - Survival analysis
    /// - Wind speed distributions
    /// 
    /// Examples:
    /// - Product lifetimes
    /// - Time-to-failure of components
    /// - Wind speed distributions
    /// - Material strength modeling
    /// - Biological survival times
    /// </para>
    /// </remarks>
    Weibull
}
