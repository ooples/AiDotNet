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
    /// - Mean (μ): The center of the distribution
    /// - Standard deviation (σ): How spread out the values are
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
    /// - Location (μ): The center of the distribution
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
    /// - Degrees of freedom (ν): Controls the heaviness of the tails
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
    /// - μ: The mean of the logarithm of the data
    /// - σ: The standard deviation of the logarithm of the data
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
    /// - λ (lambda): The rate parameter, which is the average number of events per unit time
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
    /// - λ (scale): Stretches or compresses the distribution
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
    Weibull,

    /// <summary>
    /// A distribution that models the number of events occurring in a fixed interval of time or space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Poisson distribution models the number of events that occur in a fixed interval when these events happen at a constant average rate.
    /// 
    /// Think of it as counting how many times something happens in a specific period:
    /// - It applies to discrete events (you can count them as whole numbers)
    /// - Events occur independently of each other
    /// - Events occur at a constant average rate
    /// 
    /// It's defined by one parameter:
    /// - λ (lambda): The average number of events in the interval
    /// 
    /// Best used for:
    /// - Counting rare events in fixed intervals
    /// - Events that occur independently at a constant rate
    /// - Modeling arrival processes
    /// 
    /// Examples:
    /// - Number of calls received by a call center per hour
    /// - Number of defects in a manufactured product
    /// - Number of accidents at an intersection per month
    /// - Number of typos per page in a book
    /// - Number of mutations in a DNA segment
    /// </para>
    /// </remarks>
    Poisson,

    /// <summary>
    /// A distribution with constant probability over a finite range.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Uniform distribution represents equal probability across all values in a range.
    /// 
    /// Think of it as a flat line where every value is equally likely:
    /// - All values within the range have the same probability
    /// - No values outside the range are possible
    /// 
    /// It's defined by two parameters:
    /// - a: The minimum value (lower bound)
    /// - b: The maximum value (upper bound)
    /// 
    /// Best used for:
    /// - Modeling complete uncertainty within bounds
    /// - Random number generation
    /// - Prior distributions in Bayesian analysis when no information is available
    /// 
    /// Examples:
    /// - Random number generators
    /// - Rounding errors in measurements
    /// - Arrival time within a specified window
    /// - Position of a randomly placed point on a line segment
    /// </para>
    /// </remarks>
    Uniform,

    /// <summary>
    /// A distribution that models the number of successes in a fixed number of independent trials.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Binomial distribution models the number of successes in a fixed number of independent yes/no experiments.
    /// 
    /// Think of it as counting how many times you get heads when flipping a coin multiple times:
    /// - Each trial has only two possible outcomes (success/failure)
    /// - All trials have the same probability of success
    /// - Trials are independent of each other
    /// - The number of trials is fixed
    /// 
    /// It's defined by two parameters:
    /// - n: The number of trials
    /// - p: The probability of success on a single trial
    /// 
    /// Best used for:
    /// - Modeling situations with a fixed number of yes/no trials
    /// - Quality control (pass/fail testing)
    /// - Election polling
    /// - A/B testing
    /// 
    /// Examples:
    /// - Number of heads in 10 coin flips
    /// - Number of defective items in a batch of 100
    /// - Number of successful sales calls out of 20 attempts
    /// - Number of patients who recover out of a group of 50
    /// </para>
    /// </remarks>
    Binomial,

    /// <summary>
    /// A distribution that models the waiting time until the first success in a sequence of independent trials.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Geometric distribution models how many trials you need until you get your first success.
    /// 
    /// Think of it as counting how many times you need to flip a coin until you get heads:
    /// - Each trial has only two possible outcomes (success/failure)
    /// - All trials have the same probability of success
    /// - Trials are independent of each other
    /// - You count until the first success occurs
    /// 
    /// It's defined by one parameter:
    /// - p: The probability of success on a single trial
    /// 
    /// Best used for:
    /// - Modeling the number of attempts until first success
    /// - Quality control (number of items inspected until finding a defect)
    /// - Modeling rare events
    /// 
    /// Examples:
    /// - Number of coin flips until getting heads
    /// - Number of job applications until getting an offer
    /// - Number of attempts until making a successful sale
    /// - Number of rolls of a die until getting a six
    /// </para>
    /// </remarks>
    Geometric,

    /// <summary>
    /// A distribution that models the sum of squares of independent standard normal random variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Chi-Square distribution arises when you sum the squares of independent standard normal random variables.
    /// 
    /// Think of it as what happens when you:
    /// - Take several normally distributed variables
    /// - Square each one
    /// - Add them all together
    /// 
    /// It's defined by one parameter:
    /// - k: Degrees of freedom (the number of independent standard normal variables being summed)
    /// 
    /// Best used for:
    /// - Hypothesis testing
    /// - Confidence interval estimation
    /// - Quality control
    /// - Testing goodness of fit
    /// 
    /// Examples:
    /// - Testing if a sample comes from a specific distribution
    /// - Testing independence in contingency tables
    /// - Testing variance in normal populations
    /// - Confidence intervals for population variance
    /// </para>
    /// </remarks>
    ChiSquare,

    /// <summary>
    /// A distribution that models the ratio of two chi-square distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The F-distribution models the ratio of two chi-square distributed variables, each divided by their degrees of freedom.
    /// 
    /// Think of it as comparing the variability between two different groups:
    /// - It's always positive
    /// - It's skewed to the right
    /// - It approaches a normal distribution as the degrees of freedom increase
    /// 
    /// It's defined by two parameters:
    /// - d1: Degrees of freedom for the numerator
    /// - d2: Degrees of freedom for the denominator
    /// 
    /// Best used for:
    /// - Comparing variances of two populations
    /// - Analysis of Variance (ANOVA)
    /// - Testing regression models
    /// 
    /// Examples:
    /// - Testing if two groups have the same variance
    /// - Determining if a regression model fits the data better than a simpler model
    /// - Testing the significance of added variables in multiple regression
    /// </para>
    /// </remarks>
    F,

    /// <summary>
    /// A distribution that models the sum of exponentially distributed random variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gamma distribution models the time until k independent events occur in a Poisson process.
    /// 
    /// Think of it as waiting for multiple events to happen when each follows an exponential distribution:
    /// - It's always positive
    /// - It can take many different shapes depending on its parameters
    /// - It generalizes several other distributions
    /// 
    /// It's defined by two parameters:
    /// - k (shape): Controls the shape of the distribution
    /// - θ (scale): Controls the spread of the distribution
    /// 
    /// Best used for:
    /// - Modeling waiting times for multiple events
    /// - Rainfall amounts
    /// - Insurance claim amounts
    /// - Reliability analysis
    /// 
    /// Examples:
    /// - Total rainfall in a month
    /// - Time until k failures occur in a system
    /// - Size of insurance claims
    /// - Cell lifetimes in biology
    /// </para>
    /// </remarks>
    Gamma,

    /// <summary>
    /// A distribution that models proportions or probabilities, constrained between 0 and 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Beta distribution models random variables that are constrained to lie between 0 and 1, like proportions or probabilities.
    /// 
    /// Think of it as modeling the probability of an event when you have prior information:
    /// - Values are always between 0 and 1
    /// - It can take many different shapes (uniform, U-shaped, J-shaped, bell-shaped)
    /// - It's very flexible for modeling bounded data
    /// 
    /// It's defined by two parameters:
    /// - α (alpha): First shape parameter
    /// - β (beta): Second shape parameter
    /// 
    /// Best used for:
    /// - Modeling proportions or percentages
    /// - Bayesian statistics (as a prior distribution for probabilities)
    /// - Modeling random variables with fixed bounds
    /// 
    /// Examples:
    /// - Proportion of time spent on a website
    /// - Batting averages in baseball
    /// - Conversion rates in marketing
    /// - Success probabilities in Bayesian statistics
    /// </para>
    /// </remarks>
    Beta
}