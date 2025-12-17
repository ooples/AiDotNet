namespace AiDotNet.CrossValidators;

/// <summary>
/// Implements a stratified k-fold cross-validation strategy for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <typeparam name="TMetadata">The type of class labels or metadata used for stratification.</typeparam>
/// <remarks>
/// <para>
/// This class provides a stratified k-fold cross-validation implementation, where the data is split into k folds
/// while maintaining the proportion of samples for each class.
/// </para>
/// <para><b>For Beginners:</b> Stratified k-fold cross-validation is like k-fold, but it ensures that each fold
/// has roughly the same proportion of different types of data as the whole dataset.
///
/// What this class does:
/// - Splits your data into k parts (folds), maintaining the balance of different classes in each fold
/// - Uses each part once for testing and the rest for training
/// - Repeats this process k times, so each part gets a chance to be the test set
/// - Calculates how well your model performs on average across all these tests
///
/// This is particularly useful when:
/// - Your data has imbalanced classes (some types of data are much more common than others)
/// - You want to ensure each fold is representative of the overall dataset
/// </para>
/// </remarks>
public class StratifiedKFoldCrossValidator<T, TInput, TOutput, TMetadata> : CrossValidatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the StratifiedKFoldCrossValidator class.
    /// </summary>
    /// <param name="options">The options for cross-validation. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes the StratifiedKFoldCrossValidator with the provided options or default options if none are specified.
    /// It sets up the cross-validator to perform stratified k-fold cross-validation based on the specified parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the stratified k-fold cross-validator with your chosen settings.
    /// 
    /// What it does:
    /// - Takes in your preferences for how to do the cross-validation (or uses default settings if you don't specify any)
    /// - Prepares the cross-validator to split your data into the number of parts you specified, while keeping the balance of different data types
    /// - Gets everything ready to start the cross-validation process
    /// 
    /// It's like setting up a series of balanced tests for your model based on your instructions.
    /// </para>
    /// </remarks>
    public StratifiedKFoldCrossValidator(CrossValidationOptions? options = null) : base(options ?? new())
    {
    }

    /// <summary>
    /// Performs the stratified k-fold cross-validation process on the given model using the provided data and optimizer.
    /// </summary>
    /// <param name="model">The machine learning model to validate.</param>
    /// <param name="X">The feature matrix containing the input data.</param>
    /// <param name="y">The target vector containing the output data.</param>
    /// <param name="optimizer">The optimizer to use for training the model on each fold.</param>
    /// <returns>A CrossValidationResult containing the results of the validation process.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core stratified k-fold cross-validation logic. It creates the stratified folds using the CreateFolds method,
    /// then performs the cross-validation using these folds and the provided optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual stratified k-fold cross-validation happens.
    ///
    /// What it does:
    /// - Takes your model, your data (X and y), and an optimizer for training
    /// - Splits your data into k balanced parts (folds) using the CreateFolds method
    /// - Runs the PerformCrossValidation method, which:
    ///   - Trains your model using the optimizer k times, each time using a different balanced part as the test set
    ///   - Collects and summarizes the results of all these tests
    ///
    /// The optimizer ensures consistent training across all folds.
    ///
    /// It's like putting your model through a series of k balanced tests using a standardized training procedure
    /// and then giving you a report card that shows how well it performed overall.
    /// </para>
    /// </remarks>
    public override CrossValidationResult<T, TInput, TOutput> Validate(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
    {
        var folds = CreateFolds(X, y);
        return PerformCrossValidation(model, X, y, folds, optimizer);
    }

    /// <summary>
    /// Creates the stratified folds for k-fold cross-validation based on the provided options.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>An enumerable of tuples containing the train and validation indices for each stratified fold.</returns>
    /// <remarks>
    /// <para>
    /// This method generates the indices for the training and validation sets for each fold of the stratified k-fold cross-validation process. 
    /// It maintains the proportion of samples for each class in each fold, supports data shuffling, and uses the specified number of folds from the options.
    /// </para>
    /// <para><b>For Beginners:</b> This method decides how to split your data for stratified k-fold cross-validation.
    /// 
    /// What it does:
    /// - Groups your data by class (category)
    /// - For each class:
    ///   - Splits the data into k parts, maintaining the original proportion
    /// - Combines these parts to create k balanced folds
    /// - For each fold:
    ///   - Uses that fold as the validation data
    ///   - Uses all other folds as the training data
    /// - Returns these balanced splits so the main method can use them
    /// 
    /// It's like dealing a deck of cards into k piles, but making sure each pile has the same proportion of each suit as the whole deck.
    /// </para>
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X, TOutput y)
    {
        var yVector = ConversionsHelper.ConvertToVector<T, TOutput>(y);
        var classes = yVector.Distinct().Where(c => c != null).ToArray();
        var classIndices = new Dictionary<int, List<int>>();

        for (int classIndex = 0; classIndex < classes.Length; classIndex++)
        {
            var currentClass = classes[classIndex];
            classIndices[classIndex] = [.. yVector.Select((v, i) => (v, i))
                                         .Where(vi => vi.v != null && vi.v.Equals(currentClass))
                                         .Select(vi => vi.i)];
        }

        if (Options.ShuffleData)
        {
            foreach (var indices in classIndices.Values)
            {
                indices.Sort((a, b) => Random.Next(-1, 2));
            }
        }

        for (int i = 0; i < Options.NumberOfFolds; i++)
        {
            var validationIndices = new List<int>();
            var trainIndices = new List<int>();

            foreach (var classGroup in classIndices)
            {
                int foldSize = classGroup.Value.Count / Options.NumberOfFolds;
                validationIndices.AddRange(classGroup.Value.Skip(i * foldSize).Take(foldSize));
                trainIndices.AddRange(classGroup.Value.Except(validationIndices));
            }

            yield return ([.. trainIndices], [.. validationIndices]);
        }
    }
}
