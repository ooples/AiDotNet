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
}