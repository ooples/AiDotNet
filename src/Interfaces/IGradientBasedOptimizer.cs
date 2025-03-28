namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for optimization algorithms that use gradients to find the best parameters for a model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
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
public interface IGradientBasedOptimizer<T> : IOptimizer<T>
{
    /// <summary>
    /// Updates a vector of parameters based on their gradients.
    /// </summary>
    /// <param name="parameters">The current parameter values as a vector.</param>
    /// <param name="gradient">The gradient vector indicating the direction of steepest increase in error.</param>
    /// <returns>A new vector containing the updated parameter values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method adjusts a list of numbers (parameters) to make your model better.
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
    /// </remarks>
    Vector<T> UpdateVector(Vector<T> parameters, Vector<T> gradient);

    /// <summary>
    /// Updates a matrix of parameters based on their gradients.
    /// </summary>
    /// <param name="parameters">The current parameter values as a matrix.</param>
    /// <param name="gradient">The gradient matrix indicating the direction of steepest increase in error.</param>
    /// <returns>A new matrix containing the updated parameter values.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method is similar to UpdateVector, but works with a table of numbers instead of a list.
    /// 
    /// The parameters:
    /// - parameters: The current settings of your model organized in a table (rows and columns)
    /// - gradient: Information about which direction to change each parameter to reduce errors
    /// 
    /// What this method does:
    /// 1. Takes your current model parameters arranged in a table
    /// 2. Looks at the gradient to see which direction would reduce errors
    /// 3. Decides how big of a step to take in that direction
    /// 4. Returns new, improved parameter values
    /// 
    /// This is commonly used for:
    /// - Weight matrices in neural networks
    /// - Transformation matrices in computer vision
    /// - Any model parameters that are naturally organized in a grid
    /// 
    /// The matrix version is necessary because many machine learning models organize their
    /// parameters in multi-dimensional structures rather than simple lists.
    /// </remarks>
    Matrix<T> UpdateMatrix(Matrix<T> parameters, Matrix<T> gradient);

    /// <summary>
    /// Resets the internal state of the optimizer.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method clears the optimizer's memory and starts fresh.
    /// 
    /// Many advanced optimizers (like Adam or RMSProp) keep track of information from previous
    /// updates to make better decisions about future updates. This is like having a "memory"
    /// of how parameters have been changing.
    /// 
    /// When you call Reset():
    /// - All this accumulated memory is cleared
    /// - The optimizer starts fresh, as if it had just been created
    /// - Any adaptive behavior starts over from scratch
    /// 
    /// You might want to reset an optimizer when:
    /// - Starting to train a new model
    /// - Making a significant change to your training approach
    /// - The optimizer seems to be stuck and you want it to try a fresh start
    /// - You're reusing the same optimizer instance for multiple training runs
    /// </remarks>
    void Reset();
}