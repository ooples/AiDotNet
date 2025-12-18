namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Ordinal Regression Loss to evaluate model performance, particularly for ordinal classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps evaluate how well your model is performing on ordinal classification tasks,
/// which are a special type of classification where the categories have a meaningful order or rank.
/// 
/// Ordinal Regression Loss is designed specifically for problems where:
/// - You're predicting categories that have a natural order
/// - The distance between categories matters
/// - You want to penalize predictions that are further from the true category more heavily
/// 
/// Examples of ordinal classification problems:
/// - Rating predictions (1-5 stars)
/// - Education levels (elementary, middle, high school, college)
/// - Customer satisfaction levels (very dissatisfied, dissatisfied, neutral, satisfied, very satisfied)
/// - Disease severity (mild, moderate, severe)
/// 
/// How Ordinal Regression Loss works:
/// - It recognizes that predicting a 4 when the true value is 5 is better than predicting a 1
/// - It penalizes predictions based on how far they are from the true category
/// - It takes into account the ordered nature of your categories
/// 
/// Think of it like this:
/// Imagine you're predicting movie ratings (1-5 stars):
/// - Predicting 4 stars when the actual rating is 5 stars is a small error
/// - Predicting 1 star when the actual rating is 5 stars is a large error
/// - Ordinal Regression Loss will penalize the second error much more heavily
/// 
/// Key characteristics:
/// - It's specifically designed for ordered categories
/// - It penalizes errors based on the distance between predicted and actual categories
/// - Lower values are better (0 would be perfect predictions)
/// - It can automatically detect if your problem is suitable for ordinal regression
/// 
/// This calculator is smart enough to:
/// - Use ordinal regression loss when appropriate
/// - Fall back to other loss functions when your data doesn't fit the ordinal pattern
/// </para>
/// </remarks>
public class OrdinalRegressionLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The number of classes or categories in the ordinal classification problem.
    /// </summary>
    /// <remarks>
    /// This is stored as nullable (int?) because the number of classes might not be known in advance
    /// and can be automatically determined from the data.
    /// </remarks>
    private readonly int? _numClasses;

    /// <summary>
    /// Initializes a new instance of the OrdinalRegressionLossFitnessCalculator class.
    /// </summary>
    /// <param name="numberOfClassifications">The number of distinct classes or categories in your ordinal data (optional).</param>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Ordinal Regression Loss
    /// to evaluate your model's performance on problems where your categories have a meaningful order.
    /// 
    /// Parameters:
    /// - numberOfClassifications: How many different categories or classes exist in your data
    ///   * For example, if you're predicting ratings from 1-5 stars, this would be 5
    ///   * If you leave this blank (null), the calculator will try to figure it out automatically
    /// 
    /// - dataSetType: Which data to evaluate (default is Validation)
    ///   * Training: The data used to train the model (not recommended for evaluation)
    ///   * Validation: A separate set of data used to tune the model (recommended)
    ///   * Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" in the base constructor because in Ordinal Regression Loss,
    /// lower values indicate better performance (0 would be perfect).
    /// 
    /// When to use this calculator:
    /// - When your categories have a natural order (like ratings, grades, or severity levels)
    /// - When you want to penalize predictions that are further from the true category more heavily
    /// - When the distance between categories matters in your problem
    /// </para>
    /// </remarks>
    public OrdinalRegressionLossFitnessCalculator(int? numberOfClassifications = null, DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
        _numClasses = numberOfClassifications;
    }

    /// <summary>
    /// Calculates the Ordinal Regression Loss fitness score for the given dataset.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated Ordinal Regression Loss score.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model is performing using Ordinal Regression Loss.
    /// 
    /// It works by:
    /// 1. Checking if you provided the number of classes when creating the calculator
    /// 2. If yes, it uses that number to calculate the ordinal regression loss
    /// 3. If not, it tries to automatically determine the best approach based on your data
    /// 
    /// A lower score means better performance (0 would be perfect).
    /// 
    /// The ordinal regression loss takes into account that:
    /// - Predictions that are further from the true category should be penalized more
    /// - The categories have a meaningful order
    /// - The distance between categories matters
    /// 
    /// This method is smart enough to handle different scenarios:
    /// - If you specified the number of classes, it uses that information directly
    /// - If not, it analyzes your data to determine the best approach
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        if (_numClasses.HasValue)
        {
            return new OrdinalRegressionLoss<T>(_numClasses.Value).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
                ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
        }
        else
        {
            // Default process when numClasses is not provided
            // Potential way to use this loss function with other problems like regression
            return DefaultLossCalculation(dataSet);
        }
    }

    /// <summary>
    /// Calculates the appropriate loss when the number of classes is not explicitly provided.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The calculated loss score using the most appropriate method.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is called when you didn't specify how many categories your data has.
    /// It tries to figure out the best way to evaluate your model based on the data itself.
    /// 
    /// It works by:
    /// 1. Analyzing your data to determine if it looks like a classification problem
    /// 2. If it does look like classification, it counts the unique values and uses ordinal regression loss
    /// 3. If it doesn't look like classification, it falls back to mean squared error (a common regression metric)
    /// 
    /// This automatic detection helps make the calculator more flexible and user-friendly,
    /// as you don't need to know in advance exactly what type of problem you're solving.
    /// </para>
    /// </remarks>
    private T DefaultLossCalculation(DataSetStats<T, TInput, TOutput> dataSet)
    {
        // Determine the type of problem based on the data
        if (IsClassificationProblem(dataSet))
        {
            // For classification, use the number of unique values in Actual as numClasses
            int numClasses = ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual).Distinct().Count();
            return new OrdinalRegressionLoss<T>(numClasses).CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
                ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
        }
        else
        {
            // For regression or other problems, use a different loss calculation
            return new MeanSquaredErrorLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
                ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
        }
    }

    /// <summary>
    /// Determines whether the dataset represents a classification problem based on its characteristics.
    /// </summary>
    /// <param name="dataSet">The dataset to analyze.</param>
    /// <returns>True if the data appears to represent a classification problem; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at your data and tries to determine if you're working on
    /// a classification problem (predicting categories) or a regression problem (predicting continuous values).
    /// 
    /// It uses several clues to make this determination:
    /// 
    /// 1. Are all your values whole numbers (integers)?
    ///    - Classification often involves discrete categories (1, 2, 3, 4, 5)
    ///    - Regression often involves continuous values (1.2, 3.7, 4.5)
    /// 
    /// 2. How many unique values are in your data?
    ///    - Classification typically has a small number of unique values
    ///    - Regression typically has many different values
    /// 
    /// 3. Are your values evenly spaced?
    ///    - Ordinal categories are often evenly spaced (like 1, 2, 3, 4, 5)
    ///    - This checks if the difference between consecutive values is constant
    /// 
    /// 4. Are your values within a specific range (like 0 to 1)?
    ///    - Some classification problems use probability values in this range
    /// 
    /// By combining these checks, the method makes an educated guess about whether
    /// your data represents categories (classification) or continuous values (regression).
    /// This helps the calculator choose the most appropriate loss function.
    /// </para>
    /// </remarks>
    private bool IsClassificationProblem(DataSetStats<T, TInput, TOutput> dataSet)
    {
        // Get unique values
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual);
        var uniqueValues = actual.Distinct().ToList();
        int uniqueCount = uniqueValues.Count;
        int totalCount = actual.Length;

        // Check if all values are integers (or can be parsed as integers)
        bool allIntegers = uniqueValues.All(v => int.TryParse(v?.ToString(), out _));

        // Check if the number of unique values is small relative to the total number of samples
        bool fewUniqueValues = uniqueCount <= Math.Min(10, Math.Sqrt(totalCount));

        // Check if values are evenly spaced (for ordinal data)
        bool evenlySpaced = false;
        if (allIntegers && uniqueCount > 1)
        {
            var sortedValues = uniqueValues.Select(v => Convert.ToInt32(v)).OrderBy(v => v).ToList();
            int commonDifference = sortedValues[1] - sortedValues[0];
            evenlySpaced = sortedValues.Zip(sortedValues.Skip(1), (a, b) => b - a)
                                       .All(diff => diff == commonDifference);
        }

        // Check if values are within a specific range (e.g., 0 to 1 for probabilities)
        bool withinProbabilityRange = uniqueValues.All(v => _numOps.GreaterThanOrEquals(v, _numOps.Zero) && _numOps.LessThanOrEquals(v, _numOps.One));

        // Combine all checks
        return (allIntegers && fewUniqueValues) || evenlySpaced || withinProbabilityRange;
    }
}
