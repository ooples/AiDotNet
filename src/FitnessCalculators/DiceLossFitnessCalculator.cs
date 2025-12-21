namespace AiDotNet.FitnessCalculators;

/// <summary>
/// A fitness calculator that uses Dice Loss to evaluate model performance for image segmentation and other tasks where overlap between predictions and actual values is important.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This calculator helps you evaluate how well your model is performing on tasks where you need to 
/// identify specific regions or areas in data, especially in images.
/// 
/// Dice Loss (named after Lee R. Dice who created the Dice coefficient) measures the overlap between 
/// two sets - in this case, between your model's predictions and the actual correct regions.
/// 
/// Think of it like comparing two traced outlines:
/// - If both outlines perfectly match, the Dice score is 1 (and the loss is 0)
/// - If they partially overlap, the score is between 0 and 1 (and the loss is between 1 and 0)
/// - If they don't overlap at all, the score is 0 (and the loss is 1)
/// 
/// Some common applications include:
/// - Medical image segmentation (identifying organs, tumors, or other structures in medical scans)
/// - Satellite image analysis (identifying buildings, roads, or forests in aerial images)
/// - Object detection in photos (identifying the exact boundaries of objects)
/// - Document layout analysis (identifying paragraphs, images, or tables in documents)
/// 
/// Dice Loss is particularly useful when dealing with imbalanced data, such as when the region you're 
/// trying to identify is much smaller than the background (like finding a small tumor in a large scan).
/// </para>
/// </remarks>
public class DiceLossFitnessCalculator<T, TInput, TOutput> : FitnessCalculatorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the DiceLossFitnessCalculator class.
    /// </summary>
    /// <param name="dataSetType">The type of dataset to use for fitness calculation (default is Validation).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calculator that will use Dice Loss
    /// to evaluate your model's performance on segmentation tasks.
    /// 
    /// The "dataSetType" parameter lets you choose which data to evaluate:
    /// - Training: The data used to train the model (not recommended for evaluation)
    /// - Validation: A separate set of data used to tune the model (default and recommended)
    /// - Test: A completely separate set of data used for final evaluation
    /// 
    /// We set the first parameter to "false" because in Dice Loss, lower values indicate
    /// better performance (0 would be a perfect model). This tells the system that smaller numbers are better.
    /// </para>
    /// </remarks>
    public DiceLossFitnessCalculator(DataSetType dataSetType = DataSetType.Validation)
        : base(false, dataSetType)
    {
    }

    /// <summary>
    /// Calculates the Dice Loss between predicted and actual values.
    /// </summary>
    /// <param name="dataSet">The dataset containing predicted and actual values.</param>
    /// <returns>The Dice Loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how well your model's predicted regions match the actual regions.
    /// 
    /// The method works by:
    /// 1. Taking the predicted regions from your model (e.g., "these pixels are part of a tumor")
    /// 2. Comparing these with the actual correct regions (e.g., "these are the actual tumor pixels")
    /// 3. Calculating how much they overlap using the formula: 2 * (overlap) / (total predicted + total actual)
    /// 4. Converting this to a loss value (1 - Dice coefficient)
    /// 
    /// For example:
    /// - If your model perfectly identifies all the correct regions, the loss is 0 (perfect)
    /// - If your model identifies some correct regions but misses others, the loss is between 0 and 1
    /// - If your model completely misses all regions, the loss is 1 (worst case)
    /// 
    /// The method is called internally when you evaluate your model's fitness, so you
    /// typically won't need to call it directly.
    /// </para>
    /// </remarks>
    protected override T GetFitnessScore(DataSetStats<T, TInput, TOutput> dataSet)
    {
        return new DiceLoss<T>().CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Predicted),
            ConversionsHelper.ConvertToVector<T, TOutput>(dataSet.Actual));
    }
}
