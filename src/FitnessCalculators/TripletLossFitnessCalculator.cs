namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Triplet Loss to evaluate model performance, particularly for similarity learning and embedding tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps train models that learn to recognize similarities between items.
/// 
/// Triplet Loss works with three examples at a time:
/// - An "anchor" (the reference item)
/// - A "positive" (an item similar to the anchor)
/// - A "negative" (an item different from the anchor)
/// 
/// The goal is to teach the model to:
/// - Push the anchor and positive close together in the feature space
/// - Push the anchor and negative far apart in the feature space
/// 
/// Think of it like organizing a bookshelf:
/// - You have a fantasy novel (anchor)
/// - You want to place other fantasy novels (positives) close to it
/// - You want to place non-fantasy books (negatives) far away from it
/// 
/// Triplet Loss is especially useful for:
/// - Face recognition (same person = positive, different person = negative)
/// - Product recommendations (similar products = positive, different products = negative)
/// - Image search (visually similar images = positive, different images = negative)
/// - Document similarity (documents on same topic = positive, different topics = negative)
/// 
/// The "margin" parameter controls how far apart the negative examples should be from the anchor
/// compared to the positive examples.
/// </para>
/// </remarks>
public class TripletLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The margin value that determines how far negative examples should be from anchor examples compared to positive examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The margin is like a minimum required distance between different classes.
    /// 
    /// Think of it as a safety buffer:
    /// - If margin = 1.0 (default), the model will try to ensure that negative examples are at least 1.0 units
    ///   further away from the anchor than positive examples are.
    /// - A larger margin (e.g., 2.0) forces the model to create more separation between different classes.
    /// - A smaller margin (e.g., 0.5) is less strict about separation.
    /// 
    /// Adjusting the margin can help with:
    /// - Preventing overfitting (if margin is too small)
    /// - Ensuring better separation between classes (if margin is larger)
    /// </para>
    /// </remarks>
    private readonly T _margin;

    /// <summary>
    /// Initializes a new instance of the TripletLossFitnessCalculator class.
    /// </summary>
    /// <param name="margin">The margin value that determines the minimum distance between anchor-negative pairs compared to anchor-positive pairs. Default is 1.0.</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Triplet Loss
    /// to evaluate your model's performance on similarity learning tasks.
    /// 
    /// The parameters:
    /// - margin: How much further away different items should be compared to similar items (default is 1.0)
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Triplet Loss,
    /// lower values indicate better performance (0 would be a perfect model).
    /// </para>
    /// </remarks>
    public TripletLossFitnessCalculator(T? margin = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _margin = margin ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Triplet Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing features and actual values.</param>
    /// <returns>The calculated Triplet Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Triplet Loss.
    /// 
    /// It works by:
    /// 1. Preparing triplets of data points (anchor, positive, negative)
    /// 2. Calculating how close positive examples are to their anchors
    /// 3. Calculating how far negative examples are from their anchors
    /// 4. Ensuring the difference meets or exceeds the margin
    /// 5. Returning the average loss across all triplets
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// This method first prepares the triplet data by finding appropriate anchor, positive, and negative examples,
    /// then uses the NeuralNetworkHelper to calculate the actual Triplet Loss.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        var (anchor, positive, negative) = PrepareTripletData(ConversionsHelper.ConvertToMatrix<T, TInput>(dataSet.Features),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
        return new TripletLoss<T>(Convert.ToDouble(_margin)).CalculateLoss(anchor, positive, negative);
    }

    /// <summary>
    /// Prepares triplet data (anchor, positive, negative) from the input features and labels.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <returns>A tuple containing matrices for anchor, positive, and negative examples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method organizes your data into triplets needed for Triplet Loss calculation.
    /// 
    /// For each data point (anchor):
    /// 1. It finds another example of the same class (positive)
    /// 2. It finds an example from a different class (negative)
    /// 3. It groups these together to form a triplet
    /// 
    /// Think of it like creating learning sets:
    /// - "This is a dog (anchor)"
    /// - "This is also a dog (positive)"
    /// - "This is a cat, not a dog (negative)"
    /// 
    /// The model learns from these triplets to recognize that the anchor should be more similar to the positive
    /// than to the negative.
    /// 
    /// Technical details:
    /// - The method creates lists for anchors, positives, and negatives
    /// - It randomly selects appropriate positive and negative examples for each anchor
    /// - It skips cases where suitable positive or negative examples can't be found
    /// - It returns matrices containing all the valid triplets found
    /// </para>
    /// </remarks>
    private (Matrix<T> Anchor, Matrix<T> Positive, Matrix<T> Negative) PrepareTripletData(Matrix<T> X, Vector<T> y)
    {
        var classes = y.Distinct().ToList();
        var anchorList = new List<Vector<T>>();
        var positiveList = new List<Vector<T>>();
        var negativeList = new List<Vector<T>>();

        for (int i = 0; i < X.Rows; i++)
        {
            var anchor = X.GetRow(i);
            var anchorClass = y[i];

            // Find a positive example (same class as anchor)
            var positiveIndices = Enumerable.Range(0, y.Length)
                .Where(j => j != i && _numOps.Equals(y[j], anchorClass))
                .ToList();

            if (positiveIndices.Count == 0) continue; // Skip if no positive example found

            var positiveIndex = positiveIndices[RandomHelper.CreateSecureRandom().Next(positiveIndices.Count)];
            var positive = X.GetRow(positiveIndex);

            // Find a negative example (different class from anchor)
            var negativeClass = classes.Where(c => !_numOps.Equals(c, anchorClass)).RandomElement();
            var negativeIndices = Enumerable.Range(0, y.Length)
                .Where(j => _numOps.Equals(y[j], negativeClass))
                .ToList();

            if (negativeIndices.Count == 0) continue; // Skip if no negative example found

            var negativeIndex = negativeIndices[RandomHelper.CreateSecureRandom().Next(negativeIndices.Count)];
            var negative = X.GetRow(negativeIndex);

            anchorList.Add(anchor);
            positiveList.Add(positive);
            negativeList.Add(negative);
        }

        return (
            new Matrix<T>([.. anchorList]),
            new Matrix<T>([.. positiveList]),
            new Matrix<T>([.. negativeList])
        );
    }
}
