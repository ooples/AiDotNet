namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a Group K-Fold cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// This class provides a Group K-Fold cross-validation implementation, where the data is split into k folds
/// based on a group identifier. This ensures that all samples from the same group are in the same fold.
/// </para>
/// <para><b>For Beginners:</b> Group K-Fold cross-validation is useful when your data has natural groupings.
///
/// What this class does:
/// - Splits your data into k parts (folds) based on group identifiers
/// - Ensures that all data points from the same group stay together
/// - Uses each part once for testing and the rest for training
/// - Repeats this process k times, so each part gets a chance to be the test set
/// - Calculates how well your model performs on average across all these tests
///
/// This is particularly useful when:
/// - Your data has natural groups (e.g., multiple measurements from the same person)
/// - You want to ensure that related data points are not split between training and testing sets
/// </para>
/// </remarks>
public class GroupKFoldCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The group identifiers for each sample in the dataset.
    /// </summary>
    private readonly int[] _groups;

    /// <summary>
    /// Initializes a new instance of the GroupKFoldCrossValidator class.
    /// </summary>
    /// <param name="groups">The group identifiers for each sample in the dataset.</param>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the GroupKFoldCrossValidator with the provided group identifiers and options.
    /// It sets up the cross-validator to perform group k-fold cross-validation based on the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the group k-fold cross-validator with your chosen settings and group information.
    /// 
    /// What it does:
    /// - Takes in the group identifiers for your data points
    /// - Takes in your preferences for how to do the cross-validation (or uses default settings if you don't specify any)
    /// - Prepares the cross-validator to split your data into groups
    /// - Gets everything ready to start the cross-validation process
    /// 
    /// It's like setting up a series of tests that respect the groupings in your data.
    /// </para>
    /// </remarks>
    public GroupKFoldCrossValidator(int[] groups, CrossValidationOptions? options = null) : base(options ?? new())
    {
        _groups = groups;
    }

    /// <summary>
    /// Performs the group k-fold cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core group k-fold cross-validation logic. It creates the folds using the CreateFolds method,
    /// respecting the group structure of the data, then performs the cross-validation using these folds and the provided optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual group k-fold cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Creates group-based folds using the CreateFolds method and the group identifiers provided in the constructor
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains your model using the optimizer multiple times, each time using different groups for testing
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across all folds.
    ///
    /// It's like putting your model through a series of tests that respect the natural groupings in your data,
    /// using a standardized training procedure.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var folds = CreateFolds(X, y, _groups);
        return PerformCrossValidation(model, X, y, folds, optimizer);
    }

    /// <summary>
    /// Creates the folds for group k-fold cross-validation based on the provided group identifiers.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <param name="groups">The group identifiers for each sample.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each fold of the group k-fold cross-validation process. 
    /// It ensures that all samples with the same group identifier are placed in the same fold.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your grouped data for cross-validation.
    /// 
    /// What it does:
    /// - Identifies all unique groups in your data
    /// - Splits these groups into k subsets
    /// - For each fold:
    ///   - Uses one subset of groups for validation
    ///   - Uses all other groups for training
    /// - Returns these group-based splits so the main method can use them
    /// 
    /// It's like dividing a class into study groups, then using each group's results to test
    /// how well the teaching method works for the whole class.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y, int[] groups)
    {
        var uniqueGroups = groups.Distinct().ToArray();
        var groupIndices = uniqueGroups.Select(g => groups.Select((v, i) => (v, i)).Where(t => t.v == g).Select(t => t.i).ToArray()).ToArray();

        int numberOfFolds = Options.NumberOfFolds;
        var folds = new List<int[]>();

        for (int i = 0; i < numberOfFolds; i++)
        {
            var testGroups = groupIndices.Where((_, index) => index % numberOfFolds == i).SelectMany(x => x).ToArray();
            var trainGroups = groupIndices.Where((_, index) => index % numberOfFolds != i).SelectMany(x => x).ToArray();

            yield return (trainGroups, testGroups);
        }
    }
}
