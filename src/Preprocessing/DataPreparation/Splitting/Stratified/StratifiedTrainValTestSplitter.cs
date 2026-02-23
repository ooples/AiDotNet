using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;

/// <summary>
/// Stratified three-way split that preserves class distribution in all sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This creates train/validation/test splits while ensuring
/// each set has the same class proportions as the original data.
/// </para>
/// <para>
/// <b>Example:</b>
/// If original data has 60% Class A, 30% Class B, 10% Class C:
/// - Training set: ~60% A, ~30% B, ~10% C
/// - Validation set: ~60% A, ~30% B, ~10% C
/// - Test set: ~60% A, ~30% B, ~10% C
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class StratifiedTrainValTestSplitter<T> : DataSplitterBase<T>
{
    private readonly double _trainSize;
    private readonly double _validationSize;

    /// <summary>
    /// Creates a new stratified three-way splitter.
    /// </summary>
    /// <param name="trainSize">Proportion for training. Default is 0.7 (70%).</param>
    /// <param name="validationSize">Proportion for validation. Default is 0.15 (15%).</param>
    /// <param name="shuffle">Whether to shuffle within classes. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public StratifiedTrainValTestSplitter(
        double trainSize = 0.7,
        double validationSize = 0.15,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (trainSize <= 0 || trainSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainSize), "Train size must be between 0 and 1.");
        }

        if (validationSize <= 0 || validationSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(validationSize), "Validation size must be between 0 and 1.");
        }

        if (trainSize + validationSize >= 1)
        {
            throw new ArgumentException("Train + validation size must be less than 1.");
        }

        _trainSize = trainSize;
        _validationSize = validationSize;
    }

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override bool SupportsValidation => true;

    /// <inheritdoc/>
    public override string Description
    {
        get
        {
            double testSize = 1 - _trainSize - _validationSize;
            return $"Stratified Train/Val/Test ({_trainSize * 100:F0}%/{_validationSize * 100:F0}%/{testSize * 100:F0}%)";
        }
    }

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y), "Stratified split requires target labels (y).");
        }

        var labelGroups = GroupByLabel(y);
        var trainIndices = new List<int>();
        var validationIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var group in labelGroups)
        {
            var classIndices = group.Value.ToArray();

            if (_shuffle)
            {
                ShuffleIndices(classIndices);
            }

            int total = classIndices.Length;
            int classTrain = Math.Max(1, (int)(total * _trainSize));
            int classVal = Math.Max(1, (int)(total * _validationSize));
            int classTest = total - classTrain - classVal;

            // Ensure at least 1 in each if possible
            if (classTest < 1 && total >= 3)
            {
                classTest = 1;
                classVal = Math.Max(1, classVal - 1);
            }

            int idx = 0;
            for (int i = 0; i < classTrain && idx < total; i++, idx++)
            {
                trainIndices.Add(classIndices[idx]);
            }

            for (int i = 0; i < classVal && idx < total; i++, idx++)
            {
                validationIndices.Add(classIndices[idx]);
            }

            while (idx < total)
            {
                testIndices.Add(classIndices[idx++]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(), validationIndices.ToArray());
    }
}
