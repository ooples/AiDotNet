namespace AiDotNet.FitDetectors;

/// <summary>
/// A specialized detector for evaluating the fit quality of neural network models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps you understand if your neural network is performing well or not.
/// It analyzes how your model performs on different data sets and gives you recommendations
/// on how to improve it.
/// 
/// Think of it like a health check for your neural network that tells you:
/// - If your model is working well (good fit)
/// - If it's memorizing the training data instead of learning patterns (overfitting)
/// - If it's not complex enough to learn the patterns in your data (underfitting)
/// - What steps you can take to improve your model
/// </para>
/// </remarks>
public class NeuralNetworkFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the neural network fit detector.
    /// </summary>
    private readonly NeuralNetworkFitDetectorOptions _options;

    /// <summary>
    /// The error measurement on the training dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how well your model performs on the data it was trained with.
    /// Lower values mean better performance.
    /// </para>
    /// </remarks>
    private double _trainingLoss { get; set; }

    /// <summary>
    /// The error measurement on the validation dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how well your model performs on data it hasn't seen during training
    /// but is used to check progress during training. Lower values mean better performance.
    /// </para>
    /// </remarks>
    private double _validationLoss { get; set; }

    /// <summary>
    /// The error measurement on the test dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how well your model performs on completely new data
    /// that wasn't used during training or validation. Lower values mean better performance.
    /// </para>
    /// </remarks>
    private double _testLoss { get; set; }

    /// <summary>
    /// A measure of how much the model is overfitting to the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This score tells you if your model is "memorizing" the training data
    /// instead of learning general patterns. A higher score means more overfitting,
    /// which is usually a problem you want to fix.
    /// </para>
    /// </remarks>
    private double _overfittingScore { get; set; }

    /// <summary>
    /// Creates a new instance of the neural network fit detector.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a new tool that will analyze your neural network's performance.
    /// You can provide custom settings or use the default ones.
    /// </para>
    /// </remarks>
    public NeuralNetworkFitDetector(NeuralNetworkFitDetectorOptions? options = null)
    {
        _options = options ?? new NeuralNetworkFitDetectorOptions();
    }

    /// <summary>
    /// Analyzes the model's performance data and determines the quality of fit.
    /// </summary>
    /// <param name="evaluationData">Performance metrics from the model's training and evaluation.</param>
    /// <returns>A detailed result containing the fit assessment and recommendations.</returns>
    /// <exception cref="ArgumentNullException">Thrown when evaluationData is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method that examines how well your neural network is performing.
    /// It looks at the errors on different datasets, calculates an overfitting score,
    /// determines the type of fit, and provides recommendations for improvement.
    /// 
    /// The method returns a comprehensive report that includes:
    /// - The quality of fit (good, moderate, poor, etc.)
    /// - How confident the detector is in its assessment
    /// - Specific recommendations to improve your model
    /// - Additional information like loss values and overfitting score
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        _trainingLoss = Convert.ToDouble(evaluationData.TrainingSet.ErrorStats.MSE);
        _validationLoss = Convert.ToDouble(evaluationData.ValidationSet.ErrorStats.MSE);
        _testLoss = Convert.ToDouble(evaluationData.TestSet.ErrorStats.MSE);
        _overfittingScore = CalculateOverfittingScore(evaluationData);

        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "TrainingLoss", _trainingLoss },
                { "ValidationLoss", _validationLoss },
                { "TestLoss", _testLoss },
                { "OverfittingScore", _overfittingScore }
            }
        };
    }

    /// <summary>
    /// Determines the type of fit based on validation loss and overfitting score.
    /// </summary>
    /// <param name="evaluationData">Performance metrics from the model's training and evaluation.</param>
    /// <returns>The assessed fit type of the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method decides how well your neural network fits the data by checking:
    /// 
    /// 1. Validation Loss: How well your model performs on data it hasn't seen during training
    /// 2. Overfitting Score: Whether your model is memorizing training data instead of learning patterns
    /// 
    /// Based on these checks, it categorizes your model's fit as:
    /// - Good Fit: Low validation loss and low overfitting score
    /// - Moderate: Acceptable validation loss and moderate overfitting
    /// - Poor Fit: High validation loss or significant overfitting
    /// - Very Poor Fit: Very high validation loss and severe overfitting
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        if (_validationLoss <= _options.GoodFitThreshold && _overfittingScore <= _options.OverfittingThreshold)
            return FitType.GoodFit;
        else if (_validationLoss <= _options.ModerateFitThreshold && _overfittingScore <= _options.OverfittingThreshold * 1.5)
            return FitType.Moderate;
        else if (_validationLoss <= _options.PoorFitThreshold || _overfittingScore <= _options.OverfittingThreshold * 2)
            return FitType.PoorFit;
        else
            return FitType.VeryPoorFit;
    }

    /// <summary>
    /// Calculates how confident the detector is in its fit assessment.
    /// </summary>
    /// <param name="evaluationData">Performance metrics from the model's training and evaluation.</param>
    /// <returns>A confidence value between 0 and 1, where higher values indicate greater confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how sure we are about our assessment of your model.
    /// 
    /// The confidence is based on two factors:
    /// 1. Loss Confidence: How far your validation loss is from what we consider "poor"
    /// 2. Overfitting Confidence: How far your overfitting score is from severe overfitting
    /// 
    /// The final confidence is the average of these two values:
    /// - Values closer to 1 mean we're very confident in our assessment
    /// - Values closer to 0 mean we're less confident
    /// 
    /// Lower confidence might mean your model's behavior is unusual or on the borderline
    /// between different fit categories.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var lossConfidence = Math.Max(0, 1 - (_validationLoss / _options.PoorFitThreshold));
        var overfittingConfidence = Math.Max(0, 1 - (_overfittingScore / (_options.OverfittingThreshold * 2)));

        var overallConfidence = (lossConfidence + overfittingConfidence) / 2;
        return NumOps.FromDouble(overallConfidence);
    }

    /// <summary>
    /// Generates specific recommendations for improving the model based on its fit type.
    /// </summary>
    /// <param name="fitType">The assessed fit type of the model.</param>
    /// <param name="evaluationData">Performance metrics from the model's training and evaluation.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical advice on how to improve your neural network
    /// based on the problems we've detected.
    /// 
    /// The recommendations vary depending on your model's fit type:
    /// 
    /// - For Good Fit models: Minor fine-tuning suggestions
    /// - For Moderate Fit models: Suggestions to reduce overfitting and experiment with different architectures
    /// - For Poor/Very Poor Fit models: More extensive recommendations like:
    ///   * Increasing model complexity if the model is underfitting
    ///   * Adding regularization techniques if the model is overfitting
    ///   * Reviewing and improving your input data
    ///   * Trying different neural network architectures
    /// 
    /// These recommendations are practical steps you can take to improve your model's performance.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();

        if (fitType == FitType.GoodFit)
        {
            recommendations.Add("The neural network shows good fit. Consider fine-tuning for potential improvements.");
        }
        else if (fitType == FitType.Moderate)
        {
            recommendations.Add("The neural network shows moderate performance. Consider the following:");
            if (_overfittingScore > _options.OverfittingThreshold)
                recommendations.Add("- Implement regularization techniques to reduce overfitting.");
            recommendations.Add("- Experiment with different network architectures or hyperparameters.");
        }
        else if (fitType == FitType.PoorFit || fitType == FitType.VeryPoorFit)
        {
            recommendations.Add("The neural network shows poor fit. Consider the following:");
            if (_trainingLoss > _options.PoorFitThreshold)
                recommendations.Add("- Increase model capacity by adding more layers or neurons.");
            if (_overfittingScore > _options.OverfittingThreshold * 1.5)
                recommendations.Add("- Implement strong regularization techniques (e.g., dropout, L1/L2 regularization).");
            recommendations.Add("- Review and preprocess the input data for potential issues.");
            recommendations.Add("- Consider using a different type of neural network architecture.");
        }

        return recommendations;
    }

    /// <summary>
    /// Calculates a score that measures how much the model is overfitting to the training data.
    /// </summary>
    /// <param name="evaluationData">Performance metrics from the model's training and evaluation.</param>
    /// <returns>A score representing the degree of overfitting, where higher values indicate more severe overfitting.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how much your model might be "memorizing" the training data
    /// instead of learning general patterns that work on new data.
    /// 
    /// The overfitting score is calculated by comparing:
    /// - How well your model performs on training data (data it has seen)
    /// - How well your model performs on validation data (data it hasn't seen)
    /// 
    /// The formula is: (Validation Loss - Training Loss) / Training Loss
    /// 
    /// What this means:
    /// - A score of 0 means no overfitting (model performs equally well on both datasets)
    /// - Higher scores mean more overfitting (model performs much better on training data than validation data)
    /// - We never allow negative scores (which would mean the model performs better on validation data)
    /// 
    /// If your overfitting score is high, you might need to use techniques like regularization,
    /// dropout, or early stopping to help your model generalize better.
    /// </para>
    /// </remarks>
    private double CalculateOverfittingScore(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        return Math.Max(0, (_validationLoss - _trainingLoss) / _trainingLoss);
    }
}
