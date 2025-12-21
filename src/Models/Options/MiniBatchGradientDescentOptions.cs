namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Mini-Batch Gradient Descent, an optimization algorithm that
/// updates model parameters using the average gradient computed from small random subsets of training data.
/// </summary>
/// <remarks>
/// <para>
/// Mini-Batch Gradient Descent is a variation of the gradient descent optimization algorithm that strikes a
/// balance between the efficiency of stochastic gradient descent and the stability of batch gradient descent.
/// It updates model parameters after processing small randomly-selected subsets (mini-batches) of the training
/// data, rather than processing individual samples (as in stochastic gradient descent) or the entire dataset
/// (as in batch gradient descent). This approach often converges faster than batch methods while providing
/// more stable updates than purely stochastic methods.
/// </para>
/// <para><b>For Beginners:</b> Mini-Batch Gradient Descent is a method for training machine learning models
/// that tries to find the best values for the model's internal settings (parameters).
/// 
/// Imagine you're trying to find the lowest point in a hilly landscape while blindfolded:
/// - Full Batch Gradient Descent: You survey the entire landscape before taking each step
/// - Stochastic Gradient Descent: You take a step based on checking just one random spot
/// - Mini-Batch Gradient Descent: You check a small random sample of spots before each step
/// 
/// This middle-ground approach is popular because:
/// - It's faster than checking the entire landscape each time
/// - It's more stable than making decisions based on just one spot
/// - It works well with modern hardware that can efficiently process small batches
/// 
/// This class allows you to configure how this learning process works: how many examples to look at
/// in each batch, how long to train, and how the algorithm adjusts its step size over time.
/// </para>
/// </remarks>
public class MiniBatchGradientDescentOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the number of training examples used in each mini-batch.
    /// </summary>
    /// <value>The mini-batch size, defaulting to 32.</value>
    /// <remarks>
    /// <para>
    /// The batch size determines how many training examples are processed before updating the model parameters.
    /// Smaller batch sizes lead to more frequent updates and potentially faster convergence, but with more noise
    /// in the gradient estimates. Larger batch sizes provide more stable and accurate gradient estimates but
    /// require more computation per update and may converge more slowly in terms of epochs. The optimal batch
    /// size often depends on the specific problem, available computational resources, and the size and structure
    /// of the training dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many examples the algorithm looks at before
    /// making each adjustment to the model.
    /// 
    /// Think of it like taste-testing a soup:
    /// - BatchSize = 1: You taste just one spoonful before adding seasoning (frequent but potentially misleading feedback)
    /// - BatchSize = 32: You taste 32 spoonfuls and consider the average flavor (more reliable feedback, but less frequent adjustments)
    /// - BatchSize = [entire pot]: You taste the entire pot before making any adjustment (very reliable but very slow)
    /// 
    /// The default value of 32 works well for many problems because:
    /// - It's large enough to provide somewhat stable gradient estimates
    /// - It's small enough to allow for frequent updates and efficient training
    /// - It often fits well in modern hardware memory for parallel processing
    /// 
    /// You might want to increase this value if:
    /// - Your training seems unstable (parameters jumping around too much)
    /// - You have plenty of computational resources
    /// - Your dataset is very noisy
    /// 
    /// You might want to decrease this value if:
    /// - Training seems to be progressing too slowly
    /// - You have limited memory available
    /// - You want to escape local minima more easily
    /// 
    /// Common batch sizes are powers of 2 (16, 32, 64, 128, 256) because they often optimize performance on GPUs and other hardware.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the maximum number of complete passes through the training dataset.
    /// </summary>
    /// <value>The maximum number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// An epoch represents one complete pass through the entire training dataset. This parameter sets the
    /// maximum number of epochs the algorithm will perform during training. The actual training might
    /// terminate earlier based on other stopping criteria, such as convergence or validation performance.
    /// More epochs allow the model more opportunities to learn from the training data but increase the
    /// risk of overfitting and computational cost.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many times the algorithm will work through
    /// your entire training dataset.
    /// 
    /// Imagine you're studying for an exam:
    /// - Each "epoch" is like reading through your entire textbook once
    /// - You might need to read it multiple times to fully understand the material
    /// - But reading it too many times might lead to memorizing specific examples rather than understanding the concepts
    /// 
    /// The default value of 100 means the algorithm will go through your entire dataset up to 100 times.
    /// 
    /// You might want to increase this value if:
    /// - Your model is complex and needs more time to learn
    /// - You're using techniques to prevent overfitting (like regularization)
    /// - Your learning rate is very small, requiring more iterations
    /// 
    /// You might want to decrease this value if:
    /// - Your model is overfitting (performing well on training data but poorly on new data)
    /// - You have a very large dataset and training is taking too long
    /// - You're doing initial experimentation and don't need perfect results
    /// 
    /// In practice, you'll often use early stopping based on validation performance rather than
    /// relying solely on a fixed number of epochs.
    /// </para>
    /// </remarks>
    public int MaxEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate for the optimization process.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines the step size in the parameter space during each update. This parameter
    /// sets an upper limit on how large the learning rate can become, even when using adaptive techniques that
    /// might otherwise increase it further. The 'new' keyword indicates this property overrides a similar
    /// property in the base class, potentially with a different default value or behavior specific to
    /// mini-batch processing.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the maximum size of adjustments the algorithm
    /// can make to your model's parameters.
    /// 
    /// Think of it like adjusting the temperature on a thermostat:
    /// - A higher MaxLearningRate (like 0.5) allows for bigger adjustments
    /// - A lower MaxLearningRate (like 0.01) forces the algorithm to make smaller, more cautious adjustments
    /// 
    /// The default value of 0.1 provides a reasonable balance for many problems:
    /// - High enough to make meaningful progress quickly
    /// - Low enough to avoid wildly overshooting the optimal values
    /// 
    /// You might want to increase this value if:
    /// - Training seems to be progressing too slowly
    /// - You're confident the function being optimized is well-behaved
    /// 
    /// You might want to decrease this value if:
    /// - Training is unstable (loss fluctuating wildly)
    /// - Your model is very sensitive to small parameter changes
    /// - You're working with a complex or ill-conditioned problem
    /// 
    /// Note: This property overrides a similar setting in the parent class, which is why it has the 'new' keyword.
    /// </para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is increased when the loss is improving.
    /// </summary>
    /// <value>The learning rate increase multiplier, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how aggressively the learning rate is increased when the optimization is
    /// making progress (i.e., when the loss is decreasing). After a successful update, the current learning
    /// rate is multiplied by this factor, allowing the algorithm to take larger steps when moving in a
    /// promising direction. This adaptive approach can speed up convergence by taking larger steps when
    /// it's safe to do so, but the rate will never exceed the MaxLearningRate.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm increases its step size
    /// when things are going well.
    /// 
    /// Imagine you're walking downhill trying to reach the lowest point:
    /// - When you're making good progress, you might want to speed up
    /// - This setting determines how much faster you go with each successful step
    /// 
    /// The default value of 1.05 means:
    /// - Each time the model improves, the learning rate increases by 5%
    /// - For example, a learning rate of 0.1 would become 0.105 after a successful update
    /// 
    /// This gradual increase helps the algorithm:
    /// - Speed up when moving in the right direction
    /// - Cover large flat areas more quickly
    /// - Potentially escape shallow local minima
    /// 
    /// You might want to increase this value (like to 1.1) if:
    /// - Training seems too slow
    /// - Your optimization landscape has large flat regions
    /// 
    /// You might want to decrease this value (like to 1.01) if:
    /// - The learning rate becomes unstable too quickly
    /// - You want more conservative adaptation
    /// 
    /// The learning rate will never exceed the MaxLearningRate value, regardless of how many
    /// successful updates occur.
    /// </para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which the learning rate is decreased when the loss is getting worse.
    /// </summary>
    /// <value>The learning rate decrease multiplier, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the learning rate is reduced when the optimization encounters
    /// difficulties (i.e., when the loss increases). After an unsuccessful update, the current learning
    /// rate is multiplied by this factor, forcing the algorithm to take smaller, more cautious steps. This
    /// adaptive approach helps the algorithm recover from overshooting and navigate complex loss landscapes
    /// by automatically adjusting the step size based on the observed performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm decreases its step size
    /// when it makes a mistake.
    /// 
    /// Continuing the walking downhill analogy:
    /// - If you take a step and end up higher than before, you've gone in the wrong direction
    /// - You'd want to be more careful with your next step
    /// - This setting determines how much more cautious you become
    /// 
    /// The default value of 0.95 means:
    /// - Each time the model gets worse, the learning rate decreases by 5%
    /// - For example, a learning rate of 0.1 would become 0.095 after an unsuccessful update
    /// 
    /// This adjustment helps the algorithm:
    /// - Recover from overshooting the optimal values
    /// - Navigate tricky, curved areas of the loss landscape
    /// - Eventually settle into a minimum
    /// 
    /// You might want to decrease this value (like to 0.8) if:
    /// - Training seems unstable
    /// - You want the algorithm to become more cautious more quickly
    /// 
    /// You might want to increase this value (like to 0.99) if:
    /// - You want to be more persistent with the current learning rate
    /// - The loss function has many local minima you want to try to escape
    /// 
    /// Finding the right balance between increasing and decreasing the learning rate is important
    /// for efficient training.
    /// </para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;
}
