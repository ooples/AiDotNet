namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the permutation test fit detector, which helps identify overfitting,
/// underfitting, and high variance in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The permutation test is a statistical method used to evaluate model performance by comparing
/// the actual model performance against performance on randomly shuffled (permuted) data. This approach
/// helps determine if the model is truly learning patterns in the data or merely capitalizing on random
/// correlations. By running multiple permutation tests, we can establish statistical confidence in our
/// model's performance and detect common issues like overfitting, underfitting, and high variance.
/// The permutation test is particularly valuable when working with small datasets or when evaluating
/// complex models where traditional validation methods might be insufficient.
/// </para>
/// <para><b>For Beginners:</b> The permutation test fit detector is a tool to check if your AI model is learning properly.
/// 
/// Imagine you're teaching someone to identify birds:
/// - You want them to learn actual bird characteristics (feathers, beak shape, etc.)
/// - Not just memorize the specific pictures you showed them
/// 
/// What this detector does:
/// - It takes your data and randomly shuffles it many times
/// - It checks how your model performs on the shuffled data
/// - It compares this to performance on the real, unshuffled data
/// - If your model only works well on the original arrangement, it's learning real patterns
/// - If it works equally well on random arrangements, it might be "cheating" or guessing
/// 
/// This helps detect three common problems:
/// 1. Overfitting: Your model memorized examples instead of learning general rules
///    (like knowing what a robin looks like only if it's in the exact same pose as your training image)
/// 
/// 2. Underfitting: Your model is too simple to capture the patterns in your data
///    (like only using bird size to identify species, ignoring color, shape, etc.)
/// 
/// 3. High Variance: Your model gives inconsistent results with small data changes
///    (like completely changing its bird identification if the lighting is slightly different)
/// 
/// This class lets you configure different aspects of this testing process.
/// </para>
/// </remarks>
public class PermutationTestFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the number of random permutations to perform during the test.
    /// </summary>
    /// <value>The number of permutations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many times the data will be randomly shuffled to create permuted
    /// datasets. For each permutation, the model performance is measured and compared against the 
    /// performance on the original, unpermuted dataset. A higher number of permutations provides more
    /// statistically robust results, at the cost of increased computation time. The distribution of 
    /// performance metrics across all permutations forms the basis for calculating p-values and 
    /// determining statistical significance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many random shuffles of your data the test will perform.
    /// 
    /// The default value of 1000 means:
    /// - The system will shuffle your data 1000 different ways
    /// - It will test your model on each of these random arrangements
    /// - More shuffles give more reliable results but take longer to run
    /// 
    /// Think of it like shuffling a deck of cards:
    /// - If you only shuffle once or twice, you might not get a truly random arrangement
    /// - With 1000 different shuffles, you can be confident you've tested many possible arrangements
    /// 
    /// You might want more permutations (like 5000 or 10000) if:
    /// - You need very precise statistical results
    /// - You're working on a critical application where accuracy is paramount
    /// - You have the computational resources and time available
    /// 
    /// You might want fewer permutations (like 100 or 500) if:
    /// - You're doing quick preliminary tests
    /// - You have limited computational resources
    /// - You're working with very large datasets where each evaluation is time-consuming
    /// </para>
    /// </remarks>
    public int NumberOfPermutations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the statistical significance level for the permutation test.
    /// </summary>
    /// <value>The significance level, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// The significance level determines the threshold for statistical significance in the permutation test.
    /// It represents the probability of rejecting the null hypothesis when it is actually true (Type I error).
    /// The null hypothesis in this context is that the model is not learning any meaningful patterns from the data.
    /// A common value is 0.05, which means there is a 5% chance of concluding the model is learning significant
    /// patterns when it is not. Lower values (e.g., 0.01) make the test more conservative, requiring stronger
    /// evidence to conclude the model is performing better than random chance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how confident we need to be before concluding that
    /// your model is truly learning patterns rather than getting lucky.
    /// 
    /// The default value of 0.05 means:
    /// - We need to be 95% confident that your model's performance isn't due to random chance
    /// - Only 5% chance that we're wrong when we say "this model is learning real patterns"
    /// 
    /// Think of it like a weather forecast:
    /// - If a meteorologist says there's a 95% chance of rain, you'd bring an umbrella
    /// - Similarly, with 95% confidence that your model is learning, you can trust its predictions
    /// 
    /// You might want a lower significance level (like 0.01 = 99% confidence):
    /// - For critical applications where mistakes are costly
    /// - When you need extra assurance your model is truly learning
    /// 
    /// You might accept a higher significance level (like 0.1 = 90% confidence):
    /// - During early exploration or when consequences of errors are minor
    /// - When you're more concerned about missing potentially useful patterns
    /// </para>
    /// </remarks>
    public double SignificanceLevel { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting.
    /// </summary>
    /// <value>The overfitting threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is classified as overfitting based on the difference between
    /// training and validation performance. If the gap between training and validation metrics exceeds this value,
    /// the model is considered to be overfitting to the training data. Overfitting occurs when a model learns
    /// the training data too well, including its noise and peculiarities, resulting in poor generalization to
    /// new, unseen data. The appropriate threshold depends on the specific application, the complexity of the 
    /// underlying patterns, and the noise level in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered to be "memorizing" rather than "learning."
    /// 
    /// The default value of 0.1 means:
    /// - If your model does more than 10% better on training data than on test data
    /// - We'll flag it as overfitting (memorizing rather than learning general rules)
    /// 
    /// Think of it like a student studying for a test:
    /// - A student who only memorizes exact questions and answers from practice tests
    /// - Might do perfectly on questions they've seen before
    /// - But will struggle with new questions that test the same concepts
    /// - That performance gap indicates memorization rather than understanding
    /// 
    /// You might want a smaller threshold (like 0.05 or 5%):
    /// - For applications requiring very reliable generalization
    /// - When even small drops in performance on new data would be problematic
    /// 
    /// You might accept a larger threshold (like 0.15 or 15%):
    /// - For very complex problems where some performance gap is expected
    /// - When the training and validation data might have inherent differences
    /// - In cases where some overfitting is acceptable for your application
    /// </para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting.
    /// </summary>
    /// <value>The underfitting threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is classified as underfitting based on its overall performance level.
    /// If the model's performance on both training and validation data is below this threshold above random chance,
    /// the model is considered to be underfitting. Underfitting occurs when a model is too simple to capture the 
    /// underlying patterns in the data, resulting in poor performance across all datasets. The appropriate threshold
    /// depends on the specific application, the complexity of the task, and the baseline performance expected from
    /// a minimal model.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered too simple to capture the patterns in your data.
    /// 
    /// The default value of 0.05 means:
    /// - If your model performs less than 5% better than random guessing
    /// - We'll flag it as underfitting (too simple to learn the patterns)
    /// 
    /// Think of it like trying to draw a complex shape:
    /// - If you're limited to using only straight lines
    /// - You'll never accurately represent a circle
    /// - No matter how many straight lines you use, there's a limit to how well you can approximate the curve
    /// - This limitation represents underfitting - your model is too simplistic for the task
    /// 
    /// You might want a smaller threshold (like 0.02 or 2%):
    /// - For problems where even small improvements over random are valuable
    /// - When the relationships in the data are known to be very subtle
    /// - In early exploratory phases where you want to catch any potential signal
    /// 
    /// You might set a larger threshold (like 0.1 or 10%):
    /// - When you expect strong relationships in your data
    /// - For problems where minor improvements aren't practically useful
    /// - When you want to ensure your model has substantial predictive power
    /// </para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance.
    /// </summary>
    /// <value>The high variance threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is classified as having high variance based on the standard deviation
    /// of its performance across multiple runs or cross-validation folds. If the standard deviation exceeds this
    /// threshold, the model is considered to have high variance. High variance indicates that the model is unstable
    /// and sensitive to the specific data points used for training, often resulting in significantly different
    /// performance on different subsets of the data. The appropriate threshold depends on the specific application,
    /// the expected variability in the domain, and the consequences of inconsistent model behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when your model is considered too inconsistent in its predictions.
    /// 
    /// The default value of 0.1 means:
    /// - If your model's performance varies by more than 10% across different data subsets
    /// - We'll flag it as having high variance (too sensitive to the specific training examples)
    /// 
    /// Think of it like asking different people for directions:
    /// - Ideally, everyone would give you roughly the same route
    /// - If each person suggests a completely different path
    /// - You'd be uncertain which way to go
    /// - This inconsistency represents high variance - your model gives very different answers depending on which training examples it sees
    /// 
    /// You might want a smaller threshold (like 0.05 or 5%):
    /// - For applications requiring highly stable predictions
    /// - When consistency across different scenarios is critical
    /// - In regulated environments where predictability is valued
    /// 
    /// You might accept a larger threshold (like 0.15 or 15%):
    /// - For naturally variable domains where some inconsistency is expected
    /// - When exploring complex relationships that might legitimately have different solutions
    /// - During early experimentation phases before focusing on stability
    /// </para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.1;
}
