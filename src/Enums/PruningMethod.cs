namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines the available pruning methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different pruning methods use different criteria to determine which weights
    /// to prune.
    /// </para>
    /// <para><b>For Beginners:</b> This defines different ways to decide which connections to remove.
    /// 
    /// The choice of method affects:
    /// - Which weights get pruned
    /// - How well accuracy is preserved
    /// - How the pruning process is performed
    /// </para>
    /// </remarks>
    public enum PruningMethod
    {
        /// <summary>
        /// Prunes weights based on their absolute magnitude.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Magnitude-based pruning removes the weights with the smallest absolute values.
        /// </para>
        /// <para><b>For Beginners:</b> This removes the smallest weights first.
        /// 
        /// Magnitude pruning:
        /// - Removes weights closest to zero
        /// - Assumes smaller weights are less important
        /// - Is simple and effective for many models
        /// - Is the most commonly used method
        /// </para>
        /// </remarks>
        Magnitude = 0,
        
        /// <summary>
        /// Prunes weights randomly.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Random pruning removes weights randomly without considering their values.
        /// </para>
        /// <para><b>For Beginners:</b> This removes connections at random.
        /// 
        /// Random pruning:
        /// - Removes weights regardless of their value
        /// - Is simple to implement
        /// - Often used as a baseline for comparison
        /// - Generally performs worse than other methods
        /// </para>
        /// </remarks>
        Random = 1,
        
        /// <summary>
        /// Prunes weights based on their importance to the loss function.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Importance-based pruning considers the impact of weights on the loss function,
        /// typically by examining gradients or Hessian information.
        /// </para>
        /// <para><b>For Beginners:</b> This removes weights that have the least impact on performance.
        /// 
        /// Importance pruning:
        /// - Estimates each weight's contribution to model performance
        /// - Removes weights that affect the loss function the least
        /// - Is more computationally expensive than magnitude pruning
        /// - Often gives better results, especially at high sparsity levels
        /// </para>
        /// </remarks>
        Importance = 2
    }
}