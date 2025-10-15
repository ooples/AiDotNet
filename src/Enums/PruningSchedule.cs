namespace AiDotNet.Enums
{
    /// <summary>
    /// Defines the available pruning schedules.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Pruning schedules determine how pruning is applied over iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This defines how gradually pruning is applied.
    /// 
    /// Different schedules have different approaches:
    /// - Removing weights all at once
    /// - Gradually removing weights over time
    /// - Various patterns for increasing sparsity
    /// </para>
    /// </remarks>
    public enum PruningSchedule
    {
        /// <summary>
        /// Prunes all weights at once.
        /// </summary>
        /// <remarks>
        /// <para>
        /// One-shot pruning applies all pruning in a single step.
        /// </para>
        /// <para><b>For Beginners:</b> This removes all selected connections at once.
        /// 
        /// One-shot pruning:
        /// - Is simple and fast
        /// - Doesn't require iterative training
        /// - Often results in more accuracy loss than gradual pruning
        /// - Works well for lower sparsity levels
        /// </para>
        /// </remarks>
        OneShot = 0,
        
        /// <summary>
        /// Gradually increases sparsity over time.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Gradual pruning increases sparsity gradually over multiple iterations,
        /// allowing the model to adapt to each level of sparsity.
        /// </para>
        /// <para><b>For Beginners:</b> This slowly increases the number of pruned connections.
        /// 
        /// Gradual pruning:
        /// - Starts with low sparsity and increases over time
        /// - Allows the model to recover from pruning at each step
        /// - Generally preserves accuracy better than one-shot pruning
        /// - Takes more time due to multiple pruning/fine-tuning cycles
        /// </para>
        /// </remarks>
        Gradual = 1,
        
        /// <summary>
        /// Applies pruning in cycles of pruning and recovery.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Iterative pruning applies pruning in cycles, where each cycle involves
        /// pruning followed by a period of recovery training.
        /// </para>
        /// <para><b>For Beginners:</b> This alternates between pruning and recovery training.
        /// 
        /// Iterative pruning:
        /// - Prunes some weights
        /// - Allows the model to recover through training
        /// - Repeats the process until the target sparsity is reached
        /// - Often yields the best results, especially for high sparsity levels
        /// </para>
        /// </remarks>
        Iterative = 2
    }
}