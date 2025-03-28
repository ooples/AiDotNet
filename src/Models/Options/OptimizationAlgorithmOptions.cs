namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for optimization algorithms used in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Optimization algorithms are methods used to find the best parameters for a machine learning model
/// by minimizing or maximizing an objective function.
/// </para>
/// <para><b>For Beginners:</b> Think of optimization as the process of "learning" in machine learning.
/// It's like adjusting the knobs on a radio until you get the clearest signal. These settings control
/// how quickly and accurately the algorithm learns from your data.
/// </para>
/// </remarks>
public class OptimizationAlgorithmOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations (epochs) for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many times the algorithm will go through your entire dataset.
    /// Each complete pass is called an "epoch". More iterations give the algorithm more chances to learn,
    /// but too many can lead to overfitting (memorizing the data instead of learning patterns).</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to use early stopping to prevent overfitting.
    /// </summary>
    /// <value>True to use early stopping (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Early stopping is like knowing when to stop studying - if your performance
    /// isn't improving anymore, it's time to stop. This prevents the model from memorizing the training data
    /// (overfitting) instead of learning general patterns.</para>
    /// </remarks>
    public bool UseEarlyStopping { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of iterations to wait before stopping if no improvement is observed.
    /// </summary>
    /// <value>The number of iterations to wait, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how patient the algorithm is. If it hasn't seen any improvement
    /// for this many iterations, it will stop training. Higher values make the algorithm more patient,
    /// giving it more chances to find improvements.</para>
    /// </remarks>
    public int EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of iterations to wait before adjusting parameters when the model is performing poorly.
    /// </summary>
    /// <value>The number of iterations to wait, defaulting to 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your model is performing poorly, this setting determines how long to wait
    /// before making adjustments. It's like giving the model a few more chances before changing your approach.</para>
    /// </remarks>
    public int BadFitPatience { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum number of features to consider in the model.
    /// </summary>
    /// <value>The minimum number of features.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features are the characteristics or attributes in your data (like height, weight, color).
    /// This setting defines the minimum number of these characteristics the model should use.</para>
    /// </remarks>
    public int MinimumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of features to consider in the model.
    /// </summary>
    /// <value>The maximum number of features.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how many characteristics (features) from your data the model can use.
    /// Setting a maximum can help prevent the model from becoming too complex.</para>
    /// </remarks>
    public int MaximumFeatures { get; set; }

    /// <summary>
    /// Gets or sets whether to use expression trees for optimization.
    /// </summary>
    /// <value>True to use expression trees, false otherwise (default).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Expression trees are a way to represent code as data, which can make
    /// certain operations faster. This is an advanced feature that most beginners won't need to change.</para>
    /// </remarks>
    public bool UseExpressionTrees { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial learning rate for the optimization algorithm.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big of steps the algorithm takes when learning.
    /// A higher rate means bigger steps (faster learning but might overshoot), while a lower rate means smaller steps
    /// (more precise but slower learning). Think of it like adjusting the speed of learning.</para>
    /// </remarks>
    public double InitialLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to automatically adjust the learning rate during training.
    /// </summary>
    /// <value>True to use adaptive learning rate (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the algorithm will automatically adjust how fast it learns
    /// based on its progress. It's like slowing down when you're getting close to your destination.</para>
    /// </remarks>
    public bool UseAdaptiveLearningRate { get; set; } = true;

    /// <summary>
    /// Gets or sets the rate at which the learning rate decreases over time.
    /// </summary>
    /// <value>The learning rate decay factor, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how quickly the learning rate decreases. A value of 0.99 means
    /// the learning rate becomes 99% of its previous value after each iteration, gradually slowing down the learning process.</para>
    /// </remarks>
    public double LearningRateDecay { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the learning rate from becoming too small.
    /// Even if the rate keeps decreasing, it won't go below this value.</para>
    /// </remarks>
    public double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the learning rate from becoming too large,
    /// which could cause the algorithm to make wild guesses instead of learning properly.</para>
    /// </remarks>
    public double MaxLearningRate { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets whether to automatically adjust the momentum during training.
    /// </summary>
    /// <value>True to use adaptive momentum (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum helps the algorithm push through flat areas and avoid getting stuck in local minima.
    /// It's like a ball rolling downhill that can roll through small bumps. When adaptive momentum is enabled,
    /// the algorithm will adjust this effect based on its progress.</para>
    /// </remarks>
    public bool UseAdaptiveMomentum { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial momentum value.
    /// </summary>
    /// <value>The initial momentum, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum determines how much the previous update influences the current one.
    /// A value of 0.9 means 90% of the previous update is carried forward. Higher values make the algorithm
    /// less likely to get stuck but might make it harder to settle on the best solution.</para>
    /// </remarks>
    public double InitialMomentum { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the factor by which momentum increases when performance improves.
    /// </summary>
    /// <value>The momentum increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is improving, momentum will be increased by this factor.
    /// A value of 1.05 means momentum becomes 105% of its previous value, helping the model move faster
    /// in promising directions.</para>
    /// </remarks>
    public double MomentumIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which momentum decreases when performance worsens.
    /// </summary>
    /// <value>The momentum decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is getting worse, momentum will be decreased by this factor.
    /// A value of 0.95 means momentum becomes 95% of its previous value, helping the model slow down
    /// when it might be heading in the wrong direction.</para>
    /// </remarks>
    public double MomentumDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed momentum value.
    /// </summary>
    /// <value>The minimum momentum, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the momentum from becoming too small.
    /// Even if the momentum keeps decreasing, it won't go below this value.</para>
    /// </remarks>
    public double MinMomentum { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum allowed momentum value.
    /// </summary>
    /// <value>The maximum momentum, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the momentum from becoming too large,
    /// which could cause the algorithm to overshoot good solutions.</para>
    /// </remarks>
    public double MaxMomentum { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the exploration rate for reinforcement learning and some optimization algorithms.
    /// </summary>
    /// <value>The exploration rate, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This balances between trying new things (exploration) and using what's already known to work (exploitation).
    /// A value of 0.5 means the algorithm spends equal time exploring new possibilities and exploiting known good solutions.</para>
    /// </remarks>
    public double ExplorationRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum allowed exploration rate.
    /// </summary>
    /// <value>The minimum exploration rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures the algorithm always spends at least some time exploring new
    /// possibilities (at least 10% with the default value), even if it's found a good solution. This helps
    /// prevent the algorithm from getting stuck in a suboptimal solution.</para>
    /// </remarks>
    public double MinExplorationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed exploration rate.
    /// </summary>
    /// <value>The maximum exploration rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how much time the algorithm spends exploring new possibilities
    /// (at most 90% with the default value). This ensures the algorithm always spends some time using what
    /// it already knows works well.</para>
    /// </remarks>
    public double MaxExplorationRate { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// Tolerance defines how close the algorithm needs to get to the optimal solution before considering
    /// the optimization process complete. When the change in the objective function between iterations
    /// becomes smaller than this value, the algorithm will stop, assuming it has converged to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> Think of tolerance as the precision level of your answer. A smaller value
    /// (like 0.000001) means the algorithm will try to find a more precise answer, but might take longer.
    /// It's like deciding whether you need to measure something to the nearest millimeter or just the nearest
    /// centimeter. If the improvement between steps becomes smaller than this value, the algorithm decides
    /// it's close enough to the best answer and stops searching.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;
}