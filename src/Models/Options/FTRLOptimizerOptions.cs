namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Follow-The-Regularized-Leader (FTRL) optimizer, an advanced gradient-based
/// optimization algorithm particularly effective for sparse datasets and online learning.
/// </summary>
/// <remarks>
/// <para>
/// FTRL (Follow-The-Regularized-Leader) is an optimization algorithm developed by Google that combines
/// the benefits of Adaptive Gradient (AdaGrad) for sparse data and Regularized Dual Averaging (RDA) for
/// better regularization. It's particularly effective for large-scale linear models and online learning
/// scenarios where data arrives sequentially.
/// </para>
/// <para><b>For Beginners:</b> Think of FTRL as a smart learning algorithm that adjusts how quickly your
/// model learns based on past experience. Unlike simpler optimizers that use the same learning approach
/// for all features, FTRL can learn different features at different rates. This makes it especially good
/// for data where many inputs might be zero or missing (called "sparse data"), like text analysis where
/// most words don't appear in most documents. FTRL was developed by Google and has been particularly
/// successful for online advertising and recommendation systems where models need to update continuously
/// as new data arrives.</para>
/// </remarks>
public class FTRLOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the batch size for mini-batch gradient descent.
    /// </summary>
    /// <value>A positive integer, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The batch size controls how many examples the optimizer looks at
    /// before making an update to the model. FTRL is designed for online learning (batch size 1), but
    /// can also work with mini-batches for better hardware utilization. Use batch size 1 for true
    /// online learning, or larger values (32-128) for mini-batch training.</para>
    /// </remarks>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Gets or sets the alpha parameter, which controls the learning rate.
    /// </summary>
    /// <value>The alpha value, defaulting to 0.005.</value>
    /// <remarks>
    /// <para>
    /// Alpha is the main learning rate parameter in FTRL. It controls how aggressively the model updates
    /// its parameters in response to new data. Smaller values lead to more conservative updates and
    /// potentially slower convergence but more stability.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big of steps the model takes when learning from data.
    /// With the default value of 0.005, the model takes small, cautious steps. Think of it like learning
    /// a new skill - if you try to change too many things at once (high alpha), you might make fast progress
    /// but risk developing bad habits; with smaller steps (low alpha), you learn more slowly but often more
    /// thoroughly. If your model is learning too slowly, you might increase this value, but if it's behaving
    /// erratically, you might want to decrease it.</para>
    /// </remarks>
    public double Alpha { get; set; } = 0.005;

    /// <summary>
    /// Gets or sets the beta parameter, which helps prevent too large updates for infrequent features.
    /// </summary>
    /// <value>The beta value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Beta is a smoothing parameter that helps stabilize learning, especially for features that appear
    /// infrequently in the data. It prevents the algorithm from making overly aggressive updates based
    /// on limited information.
    /// </para>
    /// <para><b>For Beginners:</b> This parameter helps the model be cautious about features it doesn't
    /// see very often. With the default value of 1.0, the model applies a standard level of caution.
    /// Imagine you're trying to judge someone's skill at a game - if you've only seen them play once,
    /// you'd be less confident in your assessment than if you'd watched them play 20 times. Beta works
    /// similarly, helping the model be appropriately cautious about features it hasn't encountered much.
    /// Higher values make the model more cautious about rare features.</para>
    /// </remarks>
    public double Beta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the L1 regularization strength, which encourages sparsity in the model.
    /// </summary>
    /// <value>The L1 regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Lambda1 controls the strength of L1 regularization, which penalizes the absolute magnitude of model
    /// parameters. L1 regularization encourages sparsity by pushing less important parameters exactly to zero,
    /// effectively performing feature selection.
    /// </para>
    /// <para><b>For Beginners:</b> This parameter helps your model focus on only the most important features
    /// and ignore the rest. With the default value of 1.0, the model applies a standard level of "feature
    /// selection pressure." Think of it like decluttering your home - L1 regularization is like deciding to
    /// completely remove items you rarely use (setting their importance to exactly zero). Higher values will
    /// make your model more aggressive about eliminating features, resulting in a simpler model that uses
    /// fewer inputs. This can help prevent overfitting and make your model faster and more interpretable.</para>
    /// </remarks>
    public double Lambda1 { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the L2 regularization strength, which prevents any single feature from having too much influence.
    /// </summary>
    /// <value>The L2 regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Lambda2 controls the strength of L2 regularization, which penalizes the squared magnitude of model
    /// parameters. L2 regularization discourages large parameter values and helps prevent overfitting by
    /// keeping the model weights small and more evenly distributed.
    /// </para>
    /// <para><b>For Beginners:</b> This parameter prevents any single feature from becoming too influential
    /// in your model. With the default value of 1.0, the model applies a standard level of restraint on feature
    /// importance. Unlike L1 regularization which completely removes features, L2 is like putting all your
    /// features on a diet - it makes their influence smaller but rarely zero. Think of it like ensuring no
    /// single player dominates a team sport - everyone's contribution is kept in check. Higher values create
    /// more balanced models where features have more similar levels of influence, which often helps the model
    /// generalize better to new data.</para>
    /// </remarks>
    public double Lambda2 { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum learning rate allowed during training.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter caps how large the effective learning rate can become, even if the adaptive rate
    /// calculation would suggest a larger value. It helps prevent unstable behavior that can occur with
    /// excessively large learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> This sets an upper limit on how big of learning steps the model can take,
    /// regardless of what the adaptive algorithm suggests. With the default value of 0.1, the model will never
    /// take extremely large steps, even if it thinks it should. Think of it like setting a speed limit - even
    /// if the algorithm thinks it can go faster, this parameter ensures it doesn't exceed a safe speed. This
    /// helps prevent the model from "overshooting" good solutions or becoming unstable during training. If your
    /// model is training well but occasionally has dramatic shifts in performance, you might want to lower this
    /// value.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when progress is good.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the model is making good progress (error is decreasing), the learning rate can be increased
    /// by this factor to accelerate learning. Values greater than 1.0 allow the learning rate to increase,
    /// with larger values leading to more aggressive acceleration.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the model speeds up its learning when things are
    /// going well. With the default value of 1.05, the learning rate increases by 5% whenever the model sees
    /// it's making good progress. It's like noticing you're doing well at learning a new skill and deciding
    /// to challenge yourself a bit more. If this value is too high, the model might speed up too quickly and
    /// become unstable; if it's too close to 1.0, the model might not take advantage of opportunities to learn
    /// faster when conditions are favorable.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when progress stalls or errors increase.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// When the model's progress stalls or errors increase, the learning rate can be decreased by this factor
    /// to take more careful steps. Values less than 1.0 allow the learning rate to decrease, with smaller
    /// values leading to more aggressive deceleration.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the model slows down its learning when it starts
    /// making mistakes or stops improving. With the default value of 0.95, the learning rate decreases by 5%
    /// whenever the model detects problems. It's like noticing you're making errors when learning a new skill
    /// and deciding to slow down and be more careful. If this value is too low, the model might slow down too
    /// drastically and get stuck; if it's too close to 1.0, the model might not slow down enough when it needs
    /// to be more cautious.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum number of iterations (passes through the training data) allowed during training.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits how many times the algorithm will process the training data, preventing
    /// excessively long training times. The algorithm may stop earlier if it converges to a solution
    /// before reaching this limit.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a limit on how many times the model will go through your training
    /// data. With the default value of 1000, the model will stop after 1000 complete passes through your data,
    /// even if it hasn't fully learned yet. Think of it like setting a time limit on a study session - at some
    /// point, you need to stop even if you haven't mastered everything. For simple problems, the model might
    /// need far fewer iterations (like 50-100), while complex problems might benefit from more iterations.
    /// This parameter prevents your model from training forever if it's struggling to find a good solution.</para>
    /// </remarks>
    public new int MaxIterations { get; set; } = 1000;
}
