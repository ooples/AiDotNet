namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for optimization algorithms that use gradients to find the best parameters for a model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix<T> for regression, Tensor<T> for neural networks).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector<T> for regression, Tensor<T> for neural networks).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface defines methods for algorithms that help machine learning models learn efficiently.
/// 
/// Imagine you're trying to find the lowest point in a hilly landscape while blindfolded:
/// - You can feel which way is downhill from your current position (this is the "gradient")
/// - You take steps in the downhill direction to try to reach the lowest point
/// - Sometimes you might take bigger steps, sometimes smaller steps
/// - You need to decide how far to step each time
/// 
/// In machine learning:
/// - The "hills" represent how wrong your model's predictions are
/// - The "lowest point" is where your model makes the fewest mistakes
/// - The "gradient" tells you which direction to adjust your model's parameters
/// - The "optimizer" decides how big of an adjustment to make
/// 
/// Gradient-based optimizers are special because they use this directional information (the gradient)
/// to make smarter adjustments to your model, helping it learn faster and better than if you were
/// just making random changes.
/// 
/// Common examples include:
/// - Gradient Descent: Takes simple steps in the downhill direction
/// - Adam: A more sophisticated approach that adapts the step size for each parameter
/// - RMSProp: Adjusts step sizes based on recent gradient history
/// </remarks>
public interface IGradientBasedOptimizer<T, TInput, TOutput> : IOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Updates matrix parameters based on their gradients.
    /// </summary>
    /// <param name="parameters">The current matrix parameter values.</param>
    /// <param name="gradient">The gradient matrix indicating the direction of steepest increase in error.</param>
    /// <returns>The updated matrix parameter values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method adjusts a grid of numbers (matrix parameters) to make your model better.
    /// 
    /// The parameters:
    /// - parameters: The current matrix settings of your model (like weights in a neural network layer)
    /// - gradient: Information about which direction to change each matrix element to reduce errors
    /// 
    /// What this method does:
    /// 1. Takes your current matrix parameters
    /// 2. Looks at the gradient matrix to see which direction would reduce errors for each element
    /// 3. Decides how big of a step to take in that direction for each element
    /// 4. Returns a new, improved matrix of parameter values
    /// 
    /// Think of it like adjusting multiple rows and columns of knobs on a complex control panel:
    /// - The parameters matrix represents the current positions of all these knobs
    /// - The gradient matrix tells you which knobs to turn up or down and by how much
    /// - This method returns the new positions for all the knobs, organized in the same grid
    /// 
    /// This matrix-specific version avoids the need to flatten matrices into vectors and then
    /// reshape them back, making the code more efficient and easier to understand when working
    /// with matrix parameters like neural network weights.
    /// </remarks>
    Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient);

    /// <summary>
    /// Updates parameters based on their gradients.
    /// </summary>
    /// <param name="parameters">The current parameter values.</param>
    /// <param name="gradient">The gradient indicating the direction of steepest increase in error.</param>
    /// <returns>The updated parameter values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method adjusts a set of numbers (parameters) to make your model better.
    /// 
    /// The parameters:
    /// - parameters: The current settings of your model (like weights in a neural network)
    /// - gradient: Information about which direction to change each parameter to reduce errors
    /// 
    /// What this method does:
    /// 1. Takes your current model parameters
    /// 2. Looks at the gradient to see which direction would reduce errors
    /// 3. Decides how big of a step to take in that direction
    /// 4. Returns new, improved parameter values
    /// 
    /// Think of it like adjusting the volume, bass, and treble knobs on a stereo:
    /// - The parameters are the current knob positions
    /// - The gradient tells you which knobs to turn up or down
    /// - This method returns the new positions for all the knobs
    /// 
    /// Different optimizers (like Adam, SGD, or RMSProp) will make different decisions
    /// about how far to turn each knob based on the gradient information.
    /// 
    /// This method is flexible enough to handle different data structures (e.g., vectors, matrices, or tensors)
    /// depending on the type of model and the specific implementation of the optimizer.
    /// </remarks>
    Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient);

    /// <summary>
    /// Updates the parameters of all layers in a model based on their calculated gradients.
    /// </summary>
    /// <param name="layers">A list of layers in the model whose parameters need to be updated.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method adjusts the settings (parameters) of each part (layer) of your model to make it better at its task.
    /// 
    /// What this method does:
    /// 1. Goes through each layer of your model
    /// 2. For each layer that can be trained:
    ///    - Looks at how the layer's current settings are contributing to errors (the gradients)
    ///    - Decides how much to change each setting to reduce errors
    ///    - Updates the layer's settings with these new values
    /// 
    /// Think of it like tuning a complex machine with many knobs:
    /// - Each layer is a set of knobs
    /// - The gradients tell you which way to turn each knob
    /// - This method goes through and adjusts all the knobs to make the machine work better
    /// 
    /// This method is crucial in the training process because:
    /// - It applies the learning from the backward pass to actually improve the model
    /// - It handles the intricacies of updating different types of layers (e.g., convolutional, recurrent)
    /// - It ensures that all trainable parts of your model are updated consistently
    /// 
    /// Different optimizers may implement this method differently, using various strategies to determine
    /// the best way to update parameters based on the gradients and potentially other factors like
    /// momentum or adaptive learning rates.
    /// </remarks>
    void UpdateParameters(List<ILayer<T>> layers);

    /// <summary>
    /// Gets the gradients computed during the last optimization step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the gradients (partial derivatives) calculated
    /// during the most recent optimization. Essential for distributed training, gradient clipping,
    /// and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> Gradients are "directions" showing how to adjust each parameter
    /// to improve the model. This property lets you see those directions after optimization runs.
    /// </para>
    /// <para><b>Industry Standard:</b>
    /// PyTorch, TensorFlow, and JAX all expose gradients for features like gradient clipping,
    /// true Distributed Data Parallel (DDP), and gradient compression.
    /// </para>
    /// </remarks>
    /// <value>
    /// Vector of gradients for each parameter. Returns empty vector if no optimization performed yet.
    /// </value>
    Vector<T> LastComputedGradients { get; }

    /// <summary>
    /// Applies pre-computed gradients to a model's parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Allows applying externally-computed or modified gradients (averaged, compressed, clipped, etc.)
    /// to update model parameters. Essential for production distributed training.
    /// </para>
    /// <para><b>For Beginners:</b> This takes pre-calculated "directions" (gradients) and uses them
    /// to update the model. Like having a GPS tell you which way to go, this method moves you there.
    /// </para>
    /// <para><b>Production Use Cases:</b>
    /// - **True DDP**: Average gradients across GPUs, then apply
    /// - **Gradient Compression**: Compress, sync, decompress, then apply
    /// - **Federated Learning**: Average gradients from clients before applying
    /// - **Gradient Clipping**: Clip gradients to prevent exploding, then apply
    /// </para>
    /// </remarks>
    /// <param name="gradients">Gradients to apply (must match model parameter count)</param>
    /// <param name="model">Model whose parameters should be updated</param>
    /// <returns>Model with updated parameters</returns>
    /// <exception cref="ArgumentNullException">If gradients or model is null</exception>
    /// <exception cref="ArgumentException">If gradient size doesn't match parameters</exception>
    IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> gradients, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Applies pre-computed gradients to explicit original parameters (double-step safe).
    /// </summary>
    /// <remarks>
    /// <para><b>⚠️ RECOMMENDED for Distributed Training:</b>
    /// This overload accepts originalParameters explicitly, making it impossible to accidentally
    /// apply gradients twice. Use this in distributed optimizers where you need explicit control
    /// over which parameter state to start from.
    /// </para>
    /// <para>
    /// Prevents double-stepping bug:
    /// - WRONG: ApplyGradients(g_avg, modelWithLocalUpdate) → double step!
    /// - RIGHT: ApplyGradients(originalParams, g_avg, modelTemplate) → single step!
    /// </para>
    /// <para><b>Distributed Pattern:</b>
    /// 1. Save originalParams before local optimization
    /// 2. Run local optimization → get localGradients
    /// 3. Synchronize gradients → get avgGradients
    /// 4. Call ApplyGradients(originalParams, avgGradients, model) → correct result!
    /// </para>
    /// </remarks>
    /// <param name="originalParameters">Pre-update parameters to start from</param>
    /// <param name="gradients">Gradients to apply</param>
    /// <param name="model">Model template (only used for structure, parameters ignored)</param>
    /// <returns>New model with updated parameters</returns>
    IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> originalParameters, Vector<T> gradients, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Reverses a gradient update to recover original parameters before the update was applied.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method computes the original parameters given updated parameters and the gradients
    /// that were applied. Each optimizer implements this differently based on its update rule.
    /// </para>
    /// <para><b>For Beginners:</b> This is like "undo" for a gradient update. Given where you are now
    /// (updated parameters) and the directions you took (gradients), it calculates where you started.
    /// </para>
    /// <para><b>Optimizer-Specific Behavior:</b>
    /// - **SGD**: params_old = params_new + learning_rate * gradients
    /// - **Adam**: Requires reversing momentum and adaptive learning rate adjustments
    /// - **RMSprop**: Requires reversing adaptive learning rate based on gradient history
    /// </para>
    /// <para><b>Production Use Cases:</b>
    /// - **Distributed Training**: Reverse local updates before applying synchronized gradients
    /// - **Checkpointing**: Recover previous parameter states
    /// - **Debugging**: Validate gradient application correctness
    /// </para>
    /// </remarks>
    /// <param name="updatedParameters">Parameters after gradient application</param>
    /// <param name="appliedGradients">The gradients that were applied to produce updated parameters</param>
    /// <returns>Original parameters before the gradient update</returns>
    /// <exception cref="ArgumentNullException">If parameters or gradients are null</exception>
    /// <exception cref="ArgumentException">If parameter and gradient sizes don't match</exception>
    Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients);
}
