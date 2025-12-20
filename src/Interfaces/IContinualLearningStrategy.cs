namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a strategy for continual learning that helps neural networks learn multiple tasks
/// sequentially without forgetting previously learned knowledge.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Continual learning addresses a fundamental challenge in neural networks
/// called "catastrophic forgetting." When a neural network learns a new task, it often forgets
/// how to perform previous tasks. This happens because the network's weights are modified
/// to optimize for the new task, overwriting the knowledge from earlier tasks.</para>
///
/// <para>Continual learning strategies help networks learn multiple tasks sequentially while
/// preserving knowledge from previous tasks. Common approaches include:</para>
///
/// <list type="bullet">
/// <item><description><b>Elastic Weight Consolidation (EWC):</b> Identifies important weights from
/// previous tasks using Fisher Information and penalizes changes to those weights.</description></item>
/// <item><description><b>Gradient Episodic Memory (GEM):</b> Stores examples from previous tasks
/// and ensures gradients don't interfere with performance on those examples.</description></item>
/// <item><description><b>Learning without Forgetting (LwF):</b> Uses knowledge distillation to
/// preserve the network's predictions on new inputs for previous tasks.</description></item>
/// </list>
///
/// <para><b>Typical Usage Flow:</b></para>
/// <code>
/// // Before learning task 1
/// strategy.BeforeTask(network, taskId: 0);
/// // ... train on task 1 ...
/// strategy.AfterTask(network, taskData, taskId: 0);
///
/// // Before learning task 2
/// strategy.BeforeTask(network, taskId: 1);
/// // ... train on task 2 with regularization ...
/// var loss = baseLoss + strategy.ComputeLoss(network);
/// strategy.AfterTask(network, taskData, taskId: 1);
/// </code>
/// </remarks>
public interface IContinualLearningStrategy<T>
{
    /// <summary>
    /// Prepares the strategy before starting to learn a new task.
    /// </summary>
    /// <param name="network">The neural network that will be trained.</param>
    /// <param name="taskId">The identifier for the upcoming task (0-indexed).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is called before you start training on a new task.
    /// It allows the strategy to capture the network's current state or prepare any necessary
    /// data structures for protecting knowledge from previous tasks.</para>
    ///
    /// <para>For example, in Learning without Forgetting (LwF), this might store the network's
    /// predictions on the new task's inputs before training begins, so we can later encourage
    /// the network to maintain similar predictions.</para>
    /// </remarks>
    void BeforeTask(INeuralNetwork<T> network, int taskId);

    /// <summary>
    /// Processes information after completing training on a task.
    /// </summary>
    /// <param name="network">The neural network that was trained.</param>
    /// <param name="taskData">Data from the completed task for computing importance measures.</param>
    /// <param name="taskId">The identifier for the completed task.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is called after you finish training on a task.
    /// It allows the strategy to compute and store information about what the network learned,
    /// which will be used to protect this knowledge when learning future tasks.</para>
    ///
    /// <para>For example, in Elastic Weight Consolidation (EWC), this computes the Fisher
    /// Information Matrix to identify which weights are most important for the completed task.</para>
    /// </remarks>
    void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId);

    /// <summary>
    /// Computes the regularization loss to prevent forgetting previous tasks.
    /// </summary>
    /// <param name="network">The neural network being trained.</param>
    /// <returns>The regularization loss value that should be added to the task loss.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates an additional loss term that penalizes
    /// the network for deviating from its learned knowledge of previous tasks. You add this
    /// to your regular task loss during training:</para>
    ///
    /// <code>
    /// var totalLoss = taskLoss + strategy.ComputeLoss(network);
    /// </code>
    ///
    /// <para>For example, in EWC, this returns a penalty proportional to how much important
    /// weights have changed from their optimal values for previous tasks. Larger changes to
    /// important weights result in higher loss, discouraging the network from forgetting.</para>
    /// </remarks>
    T ComputeLoss(INeuralNetwork<T> network);

    /// <summary>
    /// Modifies the gradient to prevent catastrophic forgetting.
    /// </summary>
    /// <param name="network">The neural network being trained.</param>
    /// <param name="gradients">The gradients from the current task loss.</param>
    /// <returns>Modified gradients that protect previous task knowledge.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some continual learning strategies work by modifying the
    /// gradients (the update directions for weights) rather than adding a loss term.
    /// This method takes the gradients computed from the current task and modifies them
    /// to avoid interfering with previously learned tasks.</para>
    ///
    /// <para>For example, in Gradient Episodic Memory (GEM), if a gradient would hurt
    /// performance on stored examples from previous tasks, it's projected to the closest
    /// gradient that doesn't interfere with those examples.</para>
    ///
    /// <para>If a strategy doesn't use gradient modification, this should return the
    /// gradients unchanged.</para>
    /// </remarks>
    Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients);

    /// <summary>
    /// Gets the regularization strength parameter (lambda) for loss-based continual learning.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lambda controls how strongly the strategy prevents forgetting.
    /// A higher lambda means the network is more conservative about changing weights important
    /// for previous tasks, but this might make it harder to learn new tasks effectively.</para>
    ///
    /// <para>Typical values range from 100 to 10000, depending on the complexity of tasks
    /// and how important it is to preserve old knowledge versus learning new knowledge.</para>
    /// </remarks>
    double Lambda { get; set; }

    /// <summary>
    /// Resets the strategy, clearing all stored task information.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears all the information the strategy has
    /// accumulated about previous tasks. After calling this, the network will be free to
    /// learn new tasks without any constraints from previously learned tasks.</para>
    ///
    /// <para>Use this when you want to start fresh or when you're done with a sequence
    /// of tasks and want to begin a new independent sequence.</para>
    /// </remarks>
    void Reset();
}
