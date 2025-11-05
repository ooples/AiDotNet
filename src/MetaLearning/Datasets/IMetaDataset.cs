using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Datasets;

/// <summary>
/// Interface for meta-learning datasets that provide episodic access to data.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// For Beginners:
/// Meta-learning datasets are organized by classes rather than individual examples.
/// Each class contains multiple examples, allowing us to sample N-way K-shot tasks.
/// For example, Omniglot has 1623 character classes, each with 20 examples.
/// </remarks>
public interface IMetaDataset<T> where T : struct
{
    /// <summary>
    /// Gets the total number of classes in the dataset.
    /// </summary>
    int NumClasses { get; }

    /// <summary>
    /// Gets the shape of a single data sample (e.g., [28, 28] for images).
    /// </summary>
    int[] DataShape { get; }

    /// <summary>
    /// Gets the dataset split (train, validation, test).
    /// </summary>
    DatasetSplit Split { get; }

    /// <summary>
    /// Gets all examples for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index</param>
    /// <returns>Tensor containing all examples for the class</returns>
    Tensor<T> GetClassData(int classIndex);

    /// <summary>
    /// Gets the number of examples available for a specific class.
    /// </summary>
    /// <param name="classIndex">The class index</param>
    /// <returns>Number of examples</returns>
    int GetClassSize(int classIndex);

    /// <summary>
    /// Gets the list of all available class indices.
    /// </summary>
    /// <returns>Array of class indices</returns>
    int[] GetClassIndices();

    /// <summary>
    /// Gets a random sample from a specific class.
    /// </summary>
    /// <param name="classIndex">The class index</param>
    /// <param name="count">Number of samples to retrieve</param>
    /// <param name="random">Random number generator for reproducibility</param>
    /// <returns>Tensor containing the sampled examples</returns>
    Tensor<T> SampleFromClass(int classIndex, int count, Random random);
}

/// <summary>
/// Represents the split of a meta-learning dataset.
/// </summary>
public enum DatasetSplit
{
    /// <summary>
    /// Training split for meta-training.
    /// </summary>
    Train,

    /// <summary>
    /// Validation split for hyperparameter tuning.
    /// </summary>
    Validation,

    /// <summary>
    /// Test split for final evaluation.
    /// </summary>
    Test
}
