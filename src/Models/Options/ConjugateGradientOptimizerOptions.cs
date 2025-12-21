namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Conjugate Gradient optimization algorithm, which is used to train machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Conjugate Gradient method is an advanced optimization algorithm that often converges faster than
/// standard gradient descent. It uses information from previous iterations to determine more efficient search directions,
/// making it particularly effective for problems with many parameters.
/// </para>
/// <para><b>For Beginners:</b> Think of training a machine learning model like finding the lowest point in a valley.
/// The Conjugate Gradient method is like a smart hiker who remembers previous paths they've taken and uses that
/// information to find shortcuts to the bottom. This is often faster than basic methods that only look at the
/// current slope. This class lets you control how this "smart hiker" behaves - how big steps they take, when they
/// decide they're close enough to the bottom, and how many attempts they'll make before giving up.</para>
/// </remarks>
public class ConjugateGradientOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the initial learning rate, which controls the size of the first optimization steps.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines how large of a step to take in the direction of the gradient.
    /// A larger value can speed up convergence but risks overshooting the optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting how big the first steps are when searching for the lowest point.
    /// A value of 0.1 means taking moderate-sized steps at the beginning. If set too high (like 1.0), you might step
    /// right over the lowest point; if too low (like 0.001), it will take a very long time to reach the bottom.
    /// This value overrides the one from the parent class to provide a more appropriate default for the conjugate gradient method.</para>
    /// </remarks>
    public new double InitialLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum learning rate, which prevents the steps from becoming too small.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// If the learning rate becomes smaller than this value during adaptation, it will be reset to this minimum value.
    /// This prevents the algorithm from taking steps that are too small to make meaningful progress.
    /// </para>
    /// <para><b>For Beginners:</b> This sets a lower limit on how small the steps can become during training.
    /// The default (0.000001) ensures that even if the algorithm starts taking tiny steps, they won't become
    /// so small that progress effectively stops. This value overrides the one from the parent class to provide
    /// a more appropriate default for the conjugate gradient method.</para>
    /// </remarks>
    public new double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum learning rate, which prevents the steps from becoming too large.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// If the learning rate becomes larger than this value during adaptation, it will be capped at this maximum value.
    /// This prevents the algorithm from taking steps that are too large, which could cause instability.
    /// </para>
    /// <para><b>For Beginners:</b> This sets an upper limit on how big the steps can become during training.
    /// The default (1.0) ensures that even if the algorithm starts taking larger steps to speed up progress,
    /// they won't become so large that they cause the training to become unstable or miss the optimal solution.
    /// This value overrides the one from the parent class to provide a more appropriate default for the conjugate gradient method.</para>
    /// </remarks>
    public new double MaxLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the factor by which to increase the learning rate when progress is good.
    /// </summary>
    /// <value>The learning rate increase factor, defaulting to 1.05 (5% increase).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm is making good progress (error is decreasing), the learning rate can be increased
    /// by this factor to speed up convergence. A value greater than 1.0 means the learning rate will increase.
    /// </para>
    /// <para><b>For Beginners:</b> If the algorithm is making good progress toward finding the lowest point,
    /// it can try taking slightly bigger steps to get there faster. This setting controls how much bigger those
    /// steps become. The default (1.05) means each step can be up to 5% larger than the previous one when things
    /// are going well. A higher value like 1.1 would make the steps increase more quickly, potentially speeding up
    /// training but risking overshooting the optimal solution.</para>
    /// </remarks>
    public double LearningRateIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which to decrease the learning rate when progress stalls or errors increase.
    /// </summary>
    /// <value>The learning rate decrease factor, defaulting to 0.95 (5% decrease).</value>
    /// <remarks>
    /// <para>
    /// When the algorithm encounters difficulties (error increases or plateaus), the learning rate can be decreased
    /// by this factor to take more careful steps. A value less than 1.0 means the learning rate will decrease.
    /// </para>
    /// <para><b>For Beginners:</b> If the algorithm starts making mistakes or isn't improving, it will take
    /// smaller steps to be more careful. This setting controls how much smaller those steps become. The default (0.95)
    /// means each step will be 5% smaller than the previous one when progress stalls. A lower value like 0.8 would
    /// make the steps decrease more quickly, helping to recover from overshooting but potentially slowing down training.</para>
    /// </remarks>
    public double LearningRateDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the maximum number of iterations the algorithm will perform before stopping.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This is a safety limit to ensure the algorithm terminates even if convergence is not achieved.
    /// The algorithm will stop after this many iterations regardless of whether the tolerance criterion has been met.
    /// </para>
    /// <para><b>For Beginners:</b> This is a safety setting that prevents the algorithm from running forever.
    /// Even if it hasn't found the perfect solution, it will stop after this many attempts. The default (1000)
    /// is usually enough for moderately complex problems. For simpler problems, you might reduce this to save time,
    /// while for very complex problems, you might need to increase it to allow more time for convergence.
    /// This value overrides the one from the parent class to provide a more appropriate default for the conjugate gradient method.</para>
    /// </remarks>
    public new int MaxIterations { get; set; } = 1000;
}
