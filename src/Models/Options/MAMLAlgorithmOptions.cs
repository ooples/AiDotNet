namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// MAML is a meta-learning algorithm that learns initial model parameters that can be quickly
/// adapted to new tasks with a few gradient steps. It is "model-agnostic" because it can be
/// applied to any model trained with gradient descent.
/// </para>
/// <para>
/// <b>For Beginners:</b> MAML finds the best starting point for your model's parameters.
/// Think of it like finding the center of a city from which you can quickly reach
/// any neighborhood - MAML finds the "center" in parameter space from which you can
/// quickly adapt to any task.
/// </para>
/// </remarks>
public class MAMLAlgorithmOptions<T, TInput, TOutput> : MetaLearningAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets whether to allow unused gradients in the computation graph.
    /// </summary>
    /// <value>True to allow unused gradients, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In some cases, not all parts of the model contribute to the
    /// final output. This setting determines whether that's okay or should raise an error.
    /// Typically, you want this to be false to catch potential bugs.
    /// </para>
    /// </remarks>
    public bool AllowUnusedGradients { get; set; } = false;
}
