namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Contrastive Loss to evaluate model performance for similarity learning tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is learning to determine if two items are similar or different.
/// 
/// Contrastive Loss is used in "similarity learning" - a type of machine learning where the goal is to learn 
/// which items are similar and which are different. Some common applications include:
/// 
/// - Face recognition (are these two photos of the same person?)
/// - Signature verification (did the same person sign both documents?)
/// - Product recommendations (finding similar products)
/// - Duplicate detection (finding duplicate documents or images)
/// 
/// The loss function works by:
/// - Pulling similar items closer together in the feature space
/// - Pushing dissimilar items farther apart, up to a certain distance (called the "margin")
/// 
/// Think of it like organizing a classroom: you want students working on the same project to sit close together,
/// while students working on different projects should sit at least a certain distance apart.
/// </para>
/// </remarks>
public class ContrastiveLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The margin value that defines the minimum distance between dissimilar pairs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The margin is like a "safe distance" that we want to maintain between items that are different.
    /// 
    /// - If two items are similar, we want their distance to be close to 0
    /// - If two items are different, we want their distance to be at least as large as this margin value
    /// 
    /// A typical default value is 1.0, but you can adjust it based on your specific problem.
    /// </para>
    /// </remarks>
    private readonly T _margin;

    /// <summary>
    /// Initializes a new instance of the ContrastiveLossFitnessCalculator class.
    /// </summary>
    /// <param name="margin">The margin value that defines the minimum distance between dissimilar pairs (default is 1.0).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Contrastive Loss
    /// to evaluate your model's performance on similarity learning tasks.
    /// 
    /// The "margin" parameter sets the minimum distance we want between dissimilar items.
    /// If you don't specify a value, it defaults to 1.0, which works well for many problems.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Contrastive Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public ContrastiveLossFitnessCalculator(T? margin = default, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _margin = margin ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Contrastive Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The average Contrastive Loss value across all pairs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is learning to determine similarity between pairs of items.
    /// 
    /// The method works by:
    /// 1. Splitting the input data into pairs (first half and second half)
    /// 2. For each pair, determining if they should be similar or different
    /// 3. Calculating how well the model's predictions match this similarity information
    /// 4. Averaging the loss across all pairs
    /// 
    /// Lower values mean your model is doing a better job at learning which items are similar and which are different.
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        var (output1, output2) = SplitOutputs(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted));
        var (actual1, actual2) = SplitOutputs(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
        T totalLoss = _numOps.Zero;

        for (int i = 0; i < output1.Length; i++)
        {
            T similarityLabel = CalculateSimilarityLabel(actual1[i], actual2[i]);
            T pairLoss = new ContrastiveLoss<T>(Convert.ToDouble(_margin)).CalculateLoss(output1, output2, similarityLabel);
            totalLoss = _numOps.Add(totalLoss, pairLoss);
        }

        // Return average loss
        return _numOps.Divide(totalLoss, _numOps.FromDouble(output1.Length));
    }

    /// <summary>
    /// Splits a vector into two equal parts, representing the first and second items in each pair.
    /// </summary>
    /// <param name="predicted">The vector to split.</param>
    /// <returns>A tuple containing two vectors, each representing half of the original vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method takes a long list of values and splits it into two equal parts.
    /// 
    /// In similarity learning, we work with pairs of items. This method helps organize the data by:
    /// - Taking the first half of the values as the first item in each pair
    /// - Taking the second half of the values as the second item in each pair
    /// 
    /// For example, if your input is [A,B,C,D,E,F], this method will split it into:
    /// - First half: [A,B,C]
    /// - Second half: [D,E,F]
    /// 
    /// This allows us to compare corresponding items (A with D, B with E, C with F).
    /// </para>
    /// </remarks>
    private static (Vector<T> Output1, Vector<T> Output2) SplitOutputs(Vector<T> predicted)
    {
        int halfLength = predicted.Length / 2;
        var output1 = new Vector<T>(halfLength);
        var output2 = new Vector<T>(halfLength);

        for (int i = 0; i < halfLength; i++)
        {
            output1[i] = predicted[i];
            output2[i] = predicted[i + halfLength];
        }

        return (output1, output2);
    }

    /// <summary>
    /// Determines if two samples should be considered similar or different.
    /// </summary>
    /// <param name="sample1">The first sample.</param>
    /// <param name="sample2">The second sample.</param>
    /// <returns>1 if the samples are the same, 0 if they are different.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This helper method looks at two items and decides if they should be considered similar or different.
    /// 
    /// It returns:
    /// - 1 (one) if the items are the same or similar
    /// - 0 (zero) if the items are different
    /// 
    /// This binary label (0 or 1) tells the contrastive loss function how to treat this pair:
    /// - For similar pairs (label=1): The loss will try to make their representations closer
    /// - For different pairs (label=0): The loss will try to push their representations apart
    /// 
    /// The method simply checks if the two values are exactly equal, but in more complex applications,
    /// you might customize this to use a different definition of similarity.
    /// </para>
    /// </remarks>
    private T CalculateSimilarityLabel(T sample1, T sample2)
    {
        // Return 1 if the samples are the same, 0 otherwise
        return _numOps.Equals(sample1, sample2) ? _numOps.One : _numOps.Zero;
    }
}
