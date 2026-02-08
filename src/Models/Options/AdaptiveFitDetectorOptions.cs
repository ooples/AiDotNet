namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Adaptive Fit Detector, which automatically selects the most appropriate method
/// to detect overfitting and underfitting in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Adaptive Fit Detector combines multiple strategies to determine if a model is properly fitted to the data,
/// overfitted (too complex, memorizing training data), or underfitted (too simple, missing patterns).
/// It dynamically selects the most appropriate detection method based on the model's complexity and performance.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a smart diagnostic tool that checks if your AI model is learning properly.
/// Just like a doctor might use different tests depending on your symptoms, this detector chooses the right method to check
/// if your model is learning too much detail from your data (overfitting) or not learning enough (underfitting).
/// It automatically picks the best testing approach based on how complex your model is and how well it's performing.</para>
/// </remarks>
public class AdaptiveFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the configuration options for the Residual Analysis method of fit detection.
    /// </summary>
    /// <value>The residual analysis options, initialized with default values.</value>
    /// <remarks>
    /// <para>
    /// Residual analysis examines the differences between predicted and actual values to detect patterns
    /// that indicate overfitting or underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method looks at the errors your model makes (the difference between
    /// what it predicts and the actual correct answers). If these errors show patterns instead of being random,
    /// it suggests your model isn't learning correctly. Think of it like checking if a student's test mistakes
    /// are random or if they consistently make the same type of error, which would indicate a misunderstanding.</para>
    /// </remarks>
    public ResidualAnalysisFitDetectorOptions ResidualAnalysisOptions { get; set; } = new ResidualAnalysisFitDetectorOptions();

    /// <summary>
    /// Gets or sets the configuration options for the Learning Curve method of fit detection.
    /// </summary>
    /// <value>The learning curve options, initialized with default values.</value>
    /// <remarks>
    /// <para>
    /// Learning curve analysis compares performance on training and validation data across different training set sizes
    /// to identify overfitting or underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method tracks how well your model learns as you give it more training examples.
    /// If your model does great with the training data but poorly with new data, it's probably overfitting (like memorizing
    /// answers instead of understanding concepts). If it performs poorly on both training and new data, it's likely underfitting
    /// (like using too simple a formula to solve a complex problem).</para>
    /// </remarks>
    public LearningCurveFitDetectorOptions LearningCurveOptions { get; set; } = new LearningCurveFitDetectorOptions();

    /// <summary>
    /// Gets or sets the configuration options for the Hybrid method of fit detection.
    /// </summary>
    /// <value>The hybrid options, initialized with default values.</value>
    /// <remarks>
    /// <para>
    /// The hybrid approach combines multiple detection methods and metrics to provide a more robust assessment
    /// of model fit quality.
    /// </para>
    /// <para><b>For Beginners:</b> This is like getting a second opinion by combining multiple testing methods.
    /// Rather than relying on just one way to check if your model is learning properly, the hybrid approach uses
    /// several methods together to get a more complete picture. It's similar to how doctors might use both an X-ray
    /// and blood tests to make a more accurate diagnosis.</para>
    /// </remarks>
    public HybridFitDetectorOptions HybridOptions { get; set; } = new HybridFitDetectorOptions();

    /// <summary>
    /// Gets or sets the threshold that determines when to switch between different fit detection methods based on model complexity.
    /// </summary>
    /// <value>The complexity threshold, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Models with complexity scores above this threshold will be analyzed using methods better suited for complex models,
    /// while simpler models will use methods optimized for low-complexity scenarios.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a dividing line that helps decide which testing method to use.
    /// If your model is more complex than this threshold (has more parameters or layers), the detector will use
    /// certain testing methods. If it's simpler, it will use different methods. Think of it like choosing different
    /// tools depending on whether you're working on a bicycle or a car - the complexity determines which tools are most appropriate.</para>
    /// </remarks>
    public double ComplexityThreshold { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the threshold that determines when to switch between different fit detection methods based on model performance.
    /// </summary>
    /// <value>The performance threshold, defaulting to 0.8.</value>
    /// <remarks>
    /// <para>
    /// Models with performance scores above this threshold will be analyzed using methods better suited for high-performing models,
    /// while lower-performing models will use methods that can better diagnose fundamental learning issues.
    /// </para>
    /// <para><b>For Beginners:</b> This is another dividing line, but based on how well your model is performing.
    /// If your model's accuracy (or other performance measure) is above this threshold (e.g., 80% correct), the detector
    /// will use certain testing methods. If it's performing worse, it will use different methods that are better at
    /// finding basic problems. It's like how a coach might focus on advanced techniques for a skilled athlete,
    /// but work on fundamentals with a beginner.</para>
    /// </remarks>
    public double PerformanceThreshold { get; set; } = 0.8;
}
