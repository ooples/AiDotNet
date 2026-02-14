namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Represents a dataset that can sample episodic tasks for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An episodic dataset is a special kind of dataset used in meta-learning.
/// Instead of just giving you examples one at a time, it can create complete "tasks" or "episodes"
/// that each have their own training and test sets.
///
/// For example, if you have a dataset of animal images, an episodic dataset can:
/// 1. Randomly pick 5 animal types (e.g., cat, dog, bird, fish, rabbit) - this is "5-way"
/// 2. Give you 1 image of each animal for training - this is "1-shot"
/// 3. Give you more images of those same animals for testing
///
/// This allows the model to practice learning new tasks quickly, which is the core idea of meta-learning.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MetaLearningEpisodicDataset")]
public interface IEpisodicDataset<T, TInput, TOutput>
{
    /// <summary>
    /// Samples a batch of N-way K-shot tasks from the dataset.
    /// </summary>
    /// <param name="numTasks">The number of tasks to sample.</param>
    /// <param name="numWays">The number of classes per task (N in N-way K-shot).</param>
    /// <param name="numShots">The number of support examples per class (K in N-way K-shot).</param>
    /// <param name="numQueryPerClass">The number of query examples per class.</param>
    /// <returns>An array of sampled tasks.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates multiple learning tasks from your dataset.
    /// Each task will have N classes, K examples per class for training (support set),
    /// and additional examples for testing (query set).
    /// </para>
    /// </remarks>
    IMetaLearningTask<T, TInput, TOutput>[] SampleTasks(int numTasks, int numWays, int numShots, int numQueryPerClass);

    /// <summary>
    /// Gets the total number of classes available in the dataset.
    /// </summary>
    /// <value>The total number of classes.</value>
    int NumClasses { get; }

    /// <summary>
    /// Gets the number of examples per class in the dataset.
    /// </summary>
    /// <value>A dictionary mapping class indices to their example counts.</value>
    Dictionary<int, int> ClassCounts { get; }

    /// <summary>
    /// Gets the split type of this dataset (train, validation, or test).
    /// </summary>
    /// <value>The split type.</value>
    DatasetSplit Split { get; }

    /// <summary>
    /// Sets the random seed for reproducible task sampling.
    /// </summary>
    /// <param name="seed">The random seed value.</param>
    void SetRandomSeed(int seed);
}

/// <summary>
/// Represents the type of dataset split.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In machine learning, we typically split our data into three parts:
/// - Train: Used during meta-training to learn how to learn
/// - Validation: Used to tune hyperparameters without overfitting
/// - Test: Used for final evaluation to see how well the model generalizes
///
/// In meta-learning, each split contains different classes to ensure the model learns to
/// generalize to completely new tasks, not just new examples of seen classes.
/// </para>
/// </remarks>
public enum DatasetSplit
{
    /// <summary>
    /// Training split used for meta-training.
    /// </summary>
    Train,

    /// <summary>
    /// Validation split used for hyperparameter tuning and early stopping.
    /// </summary>
    Validation,

    /// <summary>
    /// Test split used for final evaluation.
    /// </summary>
    Test
}
