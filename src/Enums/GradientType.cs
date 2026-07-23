namespace AiDotNet.Enums;

/// <summary>
/// Specifies different types of gradient descent optimization algorithms used in machine learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gradient descent is how AI models learn from data. Think of it like finding the 
/// lowest point in a valley by taking steps downhill. Different gradient types represent different 
/// strategies for how to take these steps - some are faster but riskier, others are slower but more reliable.
/// </para>
/// </remarks>
public enum GradientType
{
    /// <summary>
    /// Represents the standard gradient descent algorithm with momentum and adaptive learning rates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Omega gradient combines the best features of several optimization methods.
    /// 
    /// Think of it like a smart car that:
    /// - Remembers which direction was working well (momentum)
    /// - Automatically adjusts its speed based on the terrain (adaptive learning)
    /// - Can handle both steep and gentle slopes efficiently
    /// 
    /// This is often a good default choice when you're not sure which gradient type to use,
    /// as it balances speed and stability for most common machine learning problems.
    /// </para>
    /// </remarks>
    Omega,

    /// <summary>
    /// Represents a gradient descent algorithm that prioritizes speed of convergence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha gradient is designed for speed, taking larger steps to reach the solution faster.
    /// 
    /// Think of it like a sports car that:
    /// - Moves quickly toward the destination
    /// - Can reach the solution in fewer iterations
    /// - Works well when the "landscape" is smooth
    /// - Might overshoot or become unstable on tricky problems
    /// 
    /// Alpha gradient is good when:
    /// - You need quick results
    /// - Your data is well-behaved (not too noisy)
    /// - You're willing to risk some instability for speed
    /// </para>
    /// </remarks>
    Alpha,

    /// <summary>
    /// Represents a gradient descent algorithm that prioritizes stability and reliability over speed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Beta gradient is designed for reliability, taking careful steps to ensure stable learning.
    /// 
    /// Think of it like an off-road vehicle that:
    /// - Moves more cautiously but consistently
    /// - Is less likely to get stuck or go off-track
    /// - Handles difficult or noisy data better
    /// - Takes more time but is more dependable
    /// 
    /// Beta gradient is good when:
    /// - Your data is noisy or complex
    /// - You value consistent, reliable results
    /// - You're willing to wait longer for a solution
    /// - You're dealing with a challenging problem where other methods might fail
    /// </para>
    /// </remarks>
    Beta
}
