namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a simple meta-learning algorithm that repeatedly:
/// 1. Samples a task
/// 2. Trains on it using SGD
/// 3. Moves the initialization towards the trained weights
/// </para>
/// <para>
/// <b>For Beginners:</b> Reptile is a simpler alternative to MAML that achieves
/// similar results but is easier to implement and understand.
///
/// Think of it like this:
/// - You start with some initial skills (parameters)
/// - You practice a specific task and get better at it
/// - Instead of only keeping those task-specific skills, you move your initial
///   skills slightly toward where you ended up
/// - Over time, your initial skills become a good starting point for any task
///
/// Reptile is faster than MAML because it doesn't need to compute second-order gradients.
/// </para>
/// </remarks>
public class ReptileAlgorithmOptions<T, TInput, TOutput> : MetaLearningAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the interpolation coefficient for meta-updates.
    /// </summary>
    /// <value>The interpolation coefficient, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much to move toward the adapted parameters.
    /// - 1.0 means fully replace with adapted parameters (fastest learning)
    /// - 0.5 means move halfway toward adapted parameters
    /// - Smaller values give more stable but slower learning
    ///
    /// This is similar to the outer learning rate but uses interpolation instead of gradient descent.
    /// </para>
    /// </remarks>
    public double Interpolation { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of inner batches per task.
    /// </summary>
    /// <value>The number of inner batches, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many times to sample and train on data
    /// from the same task before moving to the next task. More inner batches mean
    /// the model adapts more thoroughly to each task.
    /// </para>
    /// </remarks>
    public int InnerBatches { get; set; } = 5;
}
