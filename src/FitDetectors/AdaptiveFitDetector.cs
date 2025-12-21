namespace AiDotNet.FitDetectors;

/// <summary>
/// An adaptive fit detector that dynamically selects the most appropriate detection method based on data characteristics.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A fit detector helps determine whether a machine learning model is underfitting 
/// (too simple to capture the patterns in the data) or overfitting (too complex, capturing noise instead 
/// of true patterns).
/// </para>
/// <para>
/// This adaptive detector analyzes your data and model performance to automatically choose the most 
/// appropriate detection method. Think of it like a doctor who selects different diagnostic tools 
/// based on your symptoms - it uses the right approach for your specific situation.
/// </para>
/// </remarks>
public class AdaptiveFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// A detector that analyzes model residuals (prediction errors) to determine fit.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This detector looks at the pattern of errors your model makes to determine 
    /// if it's underfitting or overfitting.
    /// </remarks>
    private readonly ResidualAnalysisFitDetector<T, TInput, TOutput> _residualAnalyzer;

    /// <summary>
    /// A detector that analyzes learning curves to determine fit.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This detector examines how your model's performance changes as it sees 
    /// more training data to determine if it's underfitting or overfitting.
    /// </remarks>
    private readonly LearningCurveFitDetector<T, TInput, TOutput> _learningCurveDetector;

    /// <summary>
    /// A detector that combines multiple detection methods.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This detector uses multiple approaches together to get a more comprehensive 
    /// assessment of your model's fit.
    /// </remarks>
    private readonly HybridFitDetector<T, TInput, TOutput> _hybridDetector;

    /// <summary>
    /// Configuration options for the adaptive fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector behaves, including thresholds 
    /// for determining data complexity and model performance.
    /// </remarks>
    private readonly AdaptiveFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the AdaptiveFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new adaptive fit detector with either custom options 
    /// or default settings. It also initializes the specialized detectors that will be used internally.
    /// </para>
    /// <para>
    /// The adaptive detector contains three specialized detectors:
    /// <list type="bullet">
    /// <item><description>Residual Analysis Detector: Used for simple data with good model performance</description></item>
    /// <item><description>Learning Curve Detector: Used for moderately complex data or moderate model performance</description></item>
    /// <item><description>Hybrid Detector: Used for complex data with poor model performance</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public AdaptiveFitDetector(AdaptiveFitDetectorOptions? options = null)
    {
        _options = options ?? new AdaptiveFitDetectorOptions();
        _residualAnalyzer = new ResidualAnalysisFitDetector<T, TInput, TOutput>(_options.ResidualAnalysisOptions);
        _learningCurveDetector = new LearningCurveFitDetector<T, TInput, TOutput>(_options.LearningCurveOptions);
        _hybridDetector = new HybridFitDetector<T, TInput, TOutput>(_residualAnalyzer, _learningCurveDetector, _options.HybridOptions);
    }

    /// <summary>
    /// Detects the fit type of a model based on evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values for training, validation, and test sets.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your model's performance to determine if it's underfitting, 
    /// overfitting, or has a good fit. It automatically selects the most appropriate detection method based 
    /// on your data characteristics and model performance.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model is underfitting, overfitting, or has a good fit</description></item>
    /// <item><description>ConfidenceLevel: How confident the detector is in its assessment</description></item>
    /// <item><description>Recommendations: Suggestions for improving the model based on the detected fit type</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);

        var confidenceLevel = CalculateConfidenceLevel(evaluationData);

        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    /// <summary>
    /// Determines the fit type of a model based on evaluation data.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type (Underfitting, Overfitting, or GoodFit).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data and model performance to choose the most 
    /// appropriate detection method, then uses that method to determine if your model is underfitting, 
    /// overfitting, or has a good fit.
    /// </para>
    /// <para>
    /// The selection of the detection method is based on:
    /// <list type="bullet">
    /// <item><description>Data Complexity: How complex the patterns in your data are</description></item>
    /// <item><description>Model Performance: How well your model is performing overall</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var dataComplexity = AssessDataComplexity(evaluationData);
        var modelPerformance = AssessModelPerformance(evaluationData);

        FitDetectorResult<T> result;

        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
        {
            result = _residualAnalyzer.DetectFit(evaluationData);
        }
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
        {
            result = _learningCurveDetector.DetectFit(evaluationData);
        }
        else
        {
            result = _hybridDetector.DetectFit(evaluationData);
        }

        return result.FitType;
    }

    /// <summary>
    /// Calculates the confidence level of the fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. A higher value means more confidence in the result.
    /// </para>
    /// <para>
    /// Like the fit type determination, the confidence calculation uses the most appropriate detection 
    /// method based on your data complexity and model performance.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var dataComplexity = AssessDataComplexity(evaluationData);
        var modelPerformance = AssessModelPerformance(evaluationData);

        FitDetectorResult<T> result;

        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
        {
            result = _residualAnalyzer.DetectFit(evaluationData);
        }
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
        {
            result = _learningCurveDetector.DetectFit(evaluationData);
        }
        else
        {
            result = _hybridDetector.DetectFit(evaluationData);
        }

        return result.ConfidenceLevel ?? NumOps.Zero;
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type and evaluation data.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for improving your model based 
    /// on the detected fit type and the characteristics of your data.
    /// </para>
    /// <para>
    /// The recommendations are tailored to your specific situation, taking into account both the 
    /// complexity of your data and the current performance of your model.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var recommendations = new List<string>();
        var dataComplexity = AssessDataComplexity(evaluationData);
        var modelPerformance = AssessModelPerformance(evaluationData);

        recommendations.Add(GetAdaptiveRecommendation(dataComplexity, modelPerformance));

        return recommendations;
    }

    /// <summary>
    /// Assesses the complexity of the data based on variance.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The assessed data complexity (Simple, Moderate, or Complex).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method analyzes how much variation exists in your data to 
    /// determine its complexity. Data with more variation is generally more complex and harder to model.
    /// </para>
    /// <para>
    /// The method calculates the overall variance across training, validation, and test sets, then 
    /// compares it to thresholds to categorize the complexity as:
    /// <list type="bullet">
    /// <item><description>Simple: Low variance, relatively easy to model</description></item>
    /// <item><description>Moderate: Medium variance, moderately difficult to model</description></item>
    /// <item><description>Complex: High variance, challenging to model accurately</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private DataComplexity AssessDataComplexity(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var overallVariance = NumOps.Add(NumOps.Add(evaluationData.TrainingSet.ActualBasicStats.Variance, evaluationData.ValidationSet.ActualBasicStats.Variance),
            evaluationData.TestSet.ActualBasicStats.Variance);
        var threshold = NumOps.FromDouble(_options.ComplexityThreshold);

        if (NumOps.LessThan(overallVariance, threshold))
            return DataComplexity.Simple;
        else if (NumOps.LessThan(overallVariance, NumOps.Multiply(threshold, NumOps.FromDouble(2))))
            return DataComplexity.Moderate;
        else
            return DataComplexity.Complex;
    }

    /// <summary>
    /// Assesses the performance of the model based on R² values.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The assessed model performance (Good, Moderate, or Poor).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method evaluates how well your model is performing by looking 
    /// at its R² (R-squared) values. R² measures how well your model explains the variation in the data, 
    /// with values closer to 1 indicating better performance.
    /// </para>
    /// <para>
    /// The method calculates the average R² across training, validation, and test sets, then 
    /// compares it to thresholds to categorize the performance as:
    /// <list type="bullet">
    /// <item><description>Good: High R², model explains most of the variation in the data</description></item>
    /// <item><description>Moderate: Medium R², model explains some of the variation</description></item>
    /// <item><description>Poor: Low R², model explains little of the variation</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private ModelPerformance AssessModelPerformance(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var averageR2 = NumOps.Divide(
            NumOps.Add(NumOps.Add(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2), evaluationData.TestSet.PredictionStats.R2),
            NumOps.FromDouble(3)
        );

        var threshold = NumOps.FromDouble(_options.PerformanceThreshold);

        if (NumOps.GreaterThan(averageR2, threshold))
            return ModelPerformance.Good;
        else if (NumOps.GreaterThan(averageR2, NumOps.Multiply(threshold, NumOps.FromDouble(0.5))))
            return ModelPerformance.Moderate;
        else
            return ModelPerformance.Poor;
    }

    /// <summary>
    /// Gets an adaptive recommendation based on data complexity and model performance.
    /// </summary>
    /// <param name="dataComplexity">The assessed data complexity.</param>
    /// <param name="modelPerformance">The assessed model performance.</param>
    /// <returns>A recommendation string tailored to the specific situation.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method creates a recommendation that explains which detection 
    /// method was used and provides specific advice based on your data complexity and model performance.
    /// </remarks>
    private string GetAdaptiveRecommendation(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        return $"Based on data complexity ({dataComplexity}) and model performance ({modelPerformance}), " +
               $"the adaptive fit detector used the {GetUsedDetectorName(dataComplexity, modelPerformance)} for analysis. " +
               $"Consider {GetAdditionalRecommendation(dataComplexity, modelPerformance)}";
    }

    /// <summary>
    /// Gets the name of the detector used based on data complexity and model performance.
    /// </summary>
    /// <param name="dataComplexity">The assessed data complexity.</param>
    /// <param name="modelPerformance">The assessed model performance.</param>
    /// <returns>The name of the detector that was used for analysis.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method determines which specialized detector was used 
    /// based on your data complexity and model performance. The adaptive detector selects different 
    /// approaches for different situations.
    /// </remarks>
    private static string GetUsedDetectorName(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
            return "Residual Analysis Detector";
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
            return "Learning Curve Detector";
        else
            return "Hybrid Detector";
    }

    /// <summary>
    /// Gets additional recommendations based on data complexity and model performance.
    /// </summary>
    /// <param name="dataComplexity">The assessed data complexity.</param>
    /// <param name="modelPerformance">The assessed model performance.</param>
    /// <returns>A specific recommendation tailored to the data complexity and model performance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method provides specific advice based on your unique situation. 
    /// Different combinations of data complexity and model performance require different approaches to improvement.
    /// </para>
    /// <para>
    /// For example:
    /// <list type="bullet">
    /// <item><description>Complex data with poor performance might need more advanced modeling techniques</description></item>
    /// <item><description>Complex data with good performance might benefit from additional feature engineering</description></item>
    /// <item><description>Simple data with poor performance might need different model architectures</description></item>
    /// <item><description>Simple data with good performance might just need fine-tuning</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private static string GetAdditionalRecommendation(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        if (dataComplexity == DataComplexity.Complex && modelPerformance == ModelPerformance.Poor)
            return "using more advanced modeling techniques or feature engineering to handle the complex data and improve performance.";
        else if (dataComplexity == DataComplexity.Complex)
            return "exploring additional feature engineering techniques to better capture the complexity of the data.";
        else if (modelPerformance == ModelPerformance.Poor)
            return "trying different model architectures or hyperparameter tuning to improve performance.";
        else
            return "fine-tuning the model and monitoring its performance on new data.";
    }
}
