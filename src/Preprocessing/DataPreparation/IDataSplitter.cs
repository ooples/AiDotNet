using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Interface for data splitting strategies that divide datasets into train/validation/test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>What is Data Splitting?</b>
/// Data splitting divides your dataset into separate subsets for different purposes:
/// - <b>Training set:</b> Data the model learns from (typically 60-80%)
/// - <b>Validation set:</b> Data used to tune hyperparameters (typically 10-20%)
/// - <b>Test set:</b> Data for final evaluation (typically 10-20%)
/// </para>
/// <para>
/// <b>Why Split Data?</b>
/// If you train and test on the same data, you can't tell if your model actually learned
/// generalizable patterns or just memorized the training examples. Splitting ensures
/// you evaluate on data the model has never seen.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of it like studying for an exam:
/// - Training set = Your study materials
/// - Validation set = Practice tests you use to check understanding
/// - Test set = The actual exam that determines your grade
/// </para>
/// <para>
/// <b>Data Splitting vs Data Preprocessing</b>
/// - <b>Data Splitting</b> (this interface): Changes the NUMBER of rows by dividing data into subsets
/// - <b>Data Preprocessing</b>: Transforms values (scaling, encoding) WITHOUT changing row count
///
/// Data splitting is part of Data Preparation (along with outlier removal and augmentation)
/// and only happens during training, never during prediction.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface IDataSplitter<T>
{
    /// <summary>
    /// Performs a single train/test (and optionally validation) split on Matrix data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes your full dataset and divides it into
    /// separate pieces for training and testing (and possibly validation).
    /// </para>
    /// </remarks>
    /// <param name="X">The feature matrix where rows are samples and columns are features.</param>
    /// <param name="y">Optional target vector. Some splitters (like stratified) need this to preserve class distribution.</param>
    /// <returns>A result containing the split datasets and their indices.</returns>
    DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null);

    /// <summary>
    /// Performs a single train/test (and optionally validation) split on Tensor data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tensors are multi-dimensional arrays used for complex data like images.
    /// This method works the same as the Matrix version but supports higher-dimensional data.
    /// </para>
    /// </remarks>
    /// <param name="X">The feature tensor where the first dimension represents samples.</param>
    /// <param name="y">Optional target tensor.</param>
    /// <returns>A result containing the split datasets and their indices.</returns>
    TensorSplitResult<T> SplitTensor(Tensor<T> X, Tensor<T>? y = null);

    /// <summary>
    /// Generates multiple train/test splits for cross-validation methods.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>What is Cross-Validation?</b>
    /// Instead of a single split, cross-validation creates multiple splits so every data point
    /// gets used for both training and testing. This gives more reliable performance estimates.
    /// </para>
    /// <para>
    /// <b>Example - 5-Fold Cross-Validation:</b>
    /// <code>
    /// Fold 1: [Test][Train][Train][Train][Train]
    /// Fold 2: [Train][Test][Train][Train][Train]
    /// Fold 3: [Train][Train][Test][Train][Train]
    /// Fold 4: [Train][Train][Train][Test][Train]
    /// Fold 5: [Train][Train][Train][Train][Test]
    /// </code>
    /// Each fold, a different portion is used for testing.
    /// </para>
    /// <para>
    /// <b>For simple train/test splitters:</b> This returns a single split (same as calling Split()).
    /// <b>For k-fold splitters:</b> This returns k different splits.
    /// </para>
    /// </remarks>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">Optional target vector.</param>
    /// <returns>An enumerable of split results, one for each fold/iteration.</returns>
    IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null);

    /// <summary>
    /// Generates multiple train/test splits for cross-validation on Tensor data.
    /// </summary>
    /// <param name="X">The feature tensor.</param>
    /// <param name="y">Optional target tensor.</param>
    /// <returns>An enumerable of split results for tensor data.</returns>
    IEnumerable<TensorSplitResult<T>> GetTensorSplits(Tensor<T> X, Tensor<T>? y = null);

    /// <summary>
    /// Gets the number of splits this splitter generates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Examples:</b>
    /// - TrainTestSplitter: NumSplits = 1
    /// - KFoldSplitter(k=5): NumSplits = 5
    /// - RepeatedKFoldSplitter(k=5, repeats=3): NumSplits = 15
    /// </para>
    /// </remarks>
    int NumSplits { get; }

    /// <summary>
    /// Gets whether this splitter requires target labels (y) to function.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Examples:</b>
    /// - RandomSplitter: RequiresLabels = false (splits randomly)
    /// - StratifiedSplitter: RequiresLabels = true (needs labels to preserve class distribution)
    /// </para>
    /// </remarks>
    bool RequiresLabels { get; }

    /// <summary>
    /// Gets whether this splitter supports providing a validation set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most splitters only create train/test splits. Some splitters (like TrainValTestSplitter)
    /// can create three-way splits with a validation set.
    /// </para>
    /// </remarks>
    bool SupportsValidation { get; }

    /// <summary>
    /// Gets a human-readable description of the splitting strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Example:</b> "5-fold cross-validation with stratification"
    /// </para>
    /// </remarks>
    string Description { get; }
}
