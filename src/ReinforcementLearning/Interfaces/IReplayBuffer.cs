using AiDotNet.NeuralNetworks;

namespace AiDotNet.ReinforcementLearning.Interfaces;

/// <summary>
/// Represents a replay buffer for storing and sampling experiences in reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// A replay buffer is a critical component in off-policy reinforcement learning algorithms like DQN.
/// It stores past experiences (state, action, reward, next state, done) and allows random sampling
/// to break temporal correlations and improve learning stability.
/// </para>
/// <para><b>For Beginners:</b> Think of a replay buffer as the agent's memory bank.
///
/// Key concepts:
/// - The agent stores memories of what it did and what happened
/// - Later, it randomly reviews these memories to learn from them
/// - This is better than only learning from recent experiences
/// - It's like studying from random flashcards instead of just the last page you read
///
/// Why random sampling helps:
/// - Consecutive experiences are often very similar (correlated)
/// - Learning from correlated data can lead to unstable training
/// - Random sampling breaks these correlations
/// - This leads to more stable and efficient learning
/// </para>
/// </remarks>
public interface IReplayBuffer<T>
{
    /// <summary>
    /// Gets the current number of experiences stored in the buffer.
    /// </summary>
    /// <value>The number of experiences currently in the buffer.</value>
    /// <remarks>
    /// <para>
    /// This property tracks how many experiences have been added to the buffer. Once the buffer
    /// reaches its maximum capacity, old experiences are removed to make room for new ones.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many memories are currently stored.
    ///
    /// The count:
    /// - Starts at 0 when the buffer is created
    /// - Increases as experiences are added
    /// - Stops growing once the buffer is full
    /// - After that, old memories are replaced by new ones
    /// </para>
    /// </remarks>
    int Count { get; }

    /// <summary>
    /// Gets the maximum capacity of the replay buffer.
    /// </summary>
    /// <value>The maximum number of experiences that can be stored.</value>
    /// <remarks>
    /// <para>
    /// The capacity defines the maximum number of experiences the buffer can hold. Once this limit
    /// is reached, the oldest experiences are removed when new ones are added. Larger buffers can
    /// store more diverse experiences but require more memory.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum number of memories the buffer can hold.
    ///
    /// Buffer capacity trade-offs:
    /// - Larger buffers: More diverse memories, but use more computer memory
    /// - Smaller buffers: Less memory usage, but less diversity in experiences
    /// - Typical values range from 10,000 to 1,000,000 experiences
    ///
    /// Think of it like a notebook with a fixed number of pages. Once it's full,
    /// you have to erase old pages to write new ones.
    /// </para>
    /// </remarks>
    int Capacity { get; }

    /// <summary>
    /// Adds a new experience to the replay buffer.
    /// </summary>
    /// <param name="experience">The experience to add to the buffer.</param>
    /// <remarks>
    /// <para>
    /// This method adds a new experience to the buffer. If the buffer has reached its maximum
    /// capacity, the oldest experience is removed to make room for the new one. This implements
    /// a first-in-first-out (FIFO) policy.
    /// </para>
    /// <para><b>For Beginners:</b> This stores a new memory in the buffer.
    ///
    /// When you add an experience:
    /// - If there's space, it's simply added to the buffer
    /// - If the buffer is full, the oldest memory is deleted first
    /// - The new experience is always added
    ///
    /// Each experience contains:
    /// - The situation before (state)
    /// - What action was taken
    /// - What reward was received
    /// - The situation after (next state)
    /// - Whether the episode ended (done)
    /// </para>
    /// </remarks>
    void Add(Experience<T> experience);

    /// <summary>
    /// Samples a batch of random experiences from the replay buffer.
    /// </summary>
    /// <param name="batchSize">The number of experiences to sample.</param>
    /// <returns>An array of randomly sampled experiences.</returns>
    /// <remarks>
    /// <para>
    /// This method randomly samples experiences from the buffer for training. Random sampling
    /// breaks temporal correlations between consecutive experiences, which is crucial for stable
    /// learning in algorithms like DQN. The sampling is performed with replacement, meaning the
    /// same experience could theoretically be sampled multiple times (though this is unlikely
    /// with large buffers).
    /// </para>
    /// <para><b>For Beginners:</b> This picks random memories for the agent to learn from.
    ///
    /// Why sample randomly:
    /// - Recent experiences are often very similar to each other
    /// - Learning only from similar experiences can confuse the agent
    /// - Random sampling gives a diverse mix of situations
    /// - This helps the agent learn more general patterns
    ///
    /// The batch size is typically:
    /// - Between 32 and 256 experiences
    /// - Larger batches are more stable but slower to process
    /// - Smaller batches are faster but more noisy
    ///
    /// Think of it like reviewing random pages from your notebook instead of
    /// just the most recent page over and over.
    /// </para>
    /// </remarks>
    Experience<T>[] Sample(int batchSize);

    /// <summary>
    /// Clears all experiences from the replay buffer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method removes all stored experiences from the buffer, resetting it to an empty state.
    /// This is useful when starting a new training phase or when you want to discard old experiences
    /// that may no longer be relevant.
    /// </para>
    /// <para><b>For Beginners:</b> This erases all memories from the buffer.
    ///
    /// You might want to clear the buffer when:
    /// - Starting a completely new training phase
    /// - The agent's behavior has changed dramatically
    /// - You want to free up memory
    /// - You're switching to a different task
    ///
    /// Think of it like erasing your entire notebook to start fresh.
    /// </para>
    /// </remarks>
    void Clear();

    /// <summary>
    /// Gets whether the buffer has enough experiences to sample a batch of the specified size.
    /// </summary>
    /// <param name="batchSize">The desired batch size.</param>
    /// <returns>True if the buffer contains at least batchSize experiences; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the buffer contains enough experiences to fill a batch of the requested
    /// size. It's useful to call this before attempting to sample to avoid errors when the buffer
    /// is still being filled at the start of training.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if there are enough memories to learn from.
    ///
    /// Why this matters:
    /// - At the start of training, the buffer is empty or has few experiences
    /// - You need enough experiences to fill a batch before you can train
    /// - This method prevents errors by checking first
    ///
    /// Typical usage:
    /// - Collect experiences until the buffer has enough
    /// - Only then start training
    /// - Usually you want at least a few thousand experiences before training begins
    ///
    /// It's like making sure you've taken enough notes before studying for a test.
    /// </para>
    /// </remarks>
    bool CanSample(int batchSize);
}
