using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers are algorithms that find the minimum or maximum of a function
/// by following the direction of steepest descent or ascent (the gradient).
/// </para>
/// <para><b>For Beginners:</b> Imagine you're in a hilly landscape and want to find the lowest point.
/// Gradient-based optimization is like always walking downhill in the steepest direction until you can't go any lower.
/// The "gradient" is simply the direction of the steepest slope at your current position.
/// </para>
/// <para>
/// These algorithms are fundamental to training many machine learning models, including neural networks,
/// linear regression, and logistic regression.
/// </para>
/// <para>
/// This class inherits from <see cref="OptimizationAlgorithmOptions"/>, which means it includes all the
/// base configuration options for optimization algorithms plus any additional options specific to
/// gradient-based methods.
/// </para>
/// </remarks>
public class GradientBasedOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the gradient cache to use for storing and retrieving computed gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gradient cache helps avoid redundant gradient calculations by storing previously computed gradients.
    /// This can significantly improve performance, especially when the same model is evaluated multiple times.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as a memory that stores calculations you've already done.
    /// Instead of recalculating the same gradient multiple times, the optimizer can look it up in this cache,
    /// saving computational resources.
    /// </para>
    /// </remarks>
    public IGradientCache<T> GradientCache { get; set; } = new DefaultGradientCache<T>();

    /// <summary>
    /// Gets or sets the loss function to use for evaluating model performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function measures how well the model's predictions match the actual target values.
    /// Different loss functions are appropriate for different types of problems (e.g., regression vs. classification).
    /// </para>
    /// <para><b>For Beginners:</b> The loss function is like a scorecard that tells you how well your model is doing.
    /// A higher loss means worse performance, so the optimizer tries to find model parameters that minimize this loss.
    /// </para>
    /// </remarks>
    public ILossFunction<T> LossFunction { get; set; } = new MeanSquaredErrorLoss<T>();

    /// <summary>
    /// Gets or sets the regularization method to use for preventing overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Regularization adds a penalty for complexity to the loss function, which helps prevent the model
    /// from overfitting to the training data. Common regularization methods include L1 (Lasso) and L2 (Ridge).
    /// </para>
    /// <para><b>For Beginners:</b> Regularization is like adding a rule that says "keep it simple."
    /// It prevents your model from becoming too complex and fitting the training data too perfectly,
    /// which can actually hurt performance on new, unseen data.
    /// </para>
    /// </remarks>
    public IRegularization<T, TInput, TOutput> Regularization { get; set; } = new L2Regularization<T, TInput, TOutput>();

    /// <summary>
    /// Gets or sets the optional data sampler for advanced sampling strategies during batch creation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A data sampler controls how training examples are selected and ordered during batch creation.
    /// This enables advanced sampling strategies like:
    /// - Weighted sampling for class imbalance
    /// - Stratified sampling to maintain class proportions
    /// - Curriculum learning to start with easy examples
    /// - Importance sampling to focus on high-loss examples
    /// - Active learning to prioritize uncertain examples
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as choosing which examples to show the model and in what order.
    /// If you have more examples of cats than dogs, weighted sampling can help the model see dogs more often.
    /// Curriculum learning shows easy examples first, like learning to walk before running.
    ///
    /// **Example:**
    /// <code>
    /// // Balanced sampling for imbalanced classes
    /// options.DataSampler = Samplers.Balanced(labels, numClasses: 2);
    ///
    /// // Curriculum learning (easy to hard)
    /// options.DataSampler = Samplers.Curriculum(difficulties);
    /// </code>
    /// </para>
    /// </remarks>
    public IDataSampler? DataSampler { get; set; }

    /// <summary>
    /// Gets or sets whether to shuffle data at the beginning of each epoch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Shuffling the training data at each epoch helps prevent the model from learning the order
    /// of training examples rather than the underlying patterns. This is ignored if a custom
    /// DataSampler is provided.
    /// </para>
    /// <para><b>For Beginners:</b> Like shuffling a deck of cards before each deal,
    /// this ensures the model sees examples in different orders, which helps it learn better patterns.
    /// </para>
    /// </remarks>
    public bool ShuffleData { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to drop the last incomplete batch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When the training data size is not evenly divisible by the batch size, the last batch
    /// will be smaller. Setting this to true discards that incomplete batch.
    /// </para>
    /// <para><b>For Beginners:</b> If you have 100 examples and a batch size of 32, you'll have
    /// 3 full batches (96 examples) and 1 partial batch (4 examples). Setting DropLastBatch=true
    /// discards that partial batch, which can help with training stability.
    /// </para>
    /// </remarks>
    public bool DropLastBatch { get; set; } = false;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Setting a seed ensures the same random sequence is generated for shuffling and sampling,
    /// making experiments reproducible.
    /// </para>
    /// <para><b>For Beginners:</b> Like a recipe, a seed lets you recreate the exact same training run.
    /// This is useful for debugging and comparing different model configurations.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether gradient clipping is enabled.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Gradient clipping helps prevent exploding gradients during training by limiting the magnitude
    /// of gradients. This is particularly important for deep networks and recurrent neural networks.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes during training, gradients can become extremely large,
    /// causing the model to take huge steps that destabilize learning. Gradient clipping is like
    /// putting a speed limit on these updates to keep training stable.
    /// </para>
    /// </remarks>
    public bool EnableGradientClipping { get; set; } = false;

    /// <summary>
    /// Gets or sets the gradient clipping method to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Two main methods are available:
    /// - <see cref="GradientClippingMethod.ByNorm"/>: Scales the entire gradient vector if its norm exceeds a threshold (recommended)
    /// - <see cref="GradientClippingMethod.ByValue"/>: Clips each gradient element independently to a range
    /// </para>
    /// <para><b>For Beginners:</b> ClipByNorm is generally preferred because it preserves the direction
    /// of the gradient while only reducing its magnitude. ClipByValue is simpler but can change the
    /// gradient direction.
    /// </para>
    /// </remarks>
    public GradientClippingMethod GradientClippingMethod { get; set; } = GradientClippingMethod.ByNorm;

    /// <summary>
    /// Gets or sets the maximum gradient norm for norm-based clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using <see cref="GradientClippingMethod.ByNorm"/>, gradients are scaled down if their
    /// L2 norm exceeds this value. A typical value is 1.0, but this may need to be tuned for your model.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "speed limit" for the total gradient magnitude.
    /// If the gradient vector is longer than this value, it gets scaled down proportionally.
    /// </para>
    /// </remarks>
    public double MaxGradientNorm { get; set; } = GradientClippingHelper.DefaultMaxNorm;

    /// <summary>
    /// Gets or sets the maximum gradient value for value-based clipping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using <see cref="GradientClippingMethod.ByValue"/>, each gradient element is clipped
    /// to the range [-MaxGradientValue, MaxGradientValue].
    /// </para>
    /// <para><b>For Beginners:</b> This is the "speed limit" for each individual gradient component.
    /// Any gradient value larger than this gets capped at this value.
    /// </para>
    /// </remarks>
    public double MaxGradientValue { get; set; } = GradientClippingHelper.DefaultMaxValue;
}

/// <summary>
/// Specifies the method used for gradient clipping.
/// </summary>
public enum GradientClippingMethod
{
    /// <summary>
    /// Clips gradients by scaling the entire gradient vector if its L2 norm exceeds a threshold.
    /// This preserves the gradient direction and is generally the preferred method.
    /// </summary>
    ByNorm,

    /// <summary>
    /// Clips each gradient element independently to a fixed range.
    /// Simpler but may change the gradient direction.
    /// </summary>
    ByValue
}
