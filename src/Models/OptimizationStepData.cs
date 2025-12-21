

namespace AiDotNet.Models;

/// <summary>
/// Represents comprehensive data about a single step in an optimization process, including the current solution,
/// selected features, data subsets, fitness scores, and evaluation results.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the relevant data for a single step in an optimization or model selection process. 
/// It stores the current solution (model), the selected features, subsets of the data used for training, validation, 
/// and testing, the fitness score of the solution, overfitting detection results, and comprehensive evaluation data. 
/// This information is useful for tracking the optimization process, analyzing the performance of different solutions, 
/// and making decisions about which solution to select as the final model.
/// </para>
/// <para><b>For Beginners:</b> This class stores all the important information about one step in a model optimization process.
/// 
/// When optimizing machine learning models:
/// - You try different model configurations and feature sets
/// - You need to track how each configuration performs
/// - You want to store the data used to evaluate each configuration
/// 
/// This class stores all that information for a single optimization step, including:
/// - The current model (solution)
/// - Which features were selected for this model
/// - Subsets of data used for training, validation, and testing
/// - How well the model performed (fitness score)
/// - Whether overfitting was detected
/// - Detailed evaluation results across all datasets
/// 
/// This comprehensive information helps you analyze each step in the optimization process
/// and make informed decisions about which model is best.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix<T> for regression, Tensor<T> for neural networks).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector<T> for regression, Tensor<T> for neural networks).</typeparam>
public class OptimizationStepData<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the current solution (model) for this optimization step.
    /// </summary>
    public IFullModel<T, TInput, TOutput> Solution { get; set; }

    /// <summary>
    /// Gets or sets the list of selected feature vectors for this optimization step.
    /// </summary>
    /// <value>A list of Vector&lt;T&gt; objects representing the selected features.</value>
    /// <remarks>
    /// <para>
    /// This property contains a list of feature vectors that were selected for use in this optimization step. Feature 
    /// selection is a common technique in machine learning to improve model performance by focusing on the most relevant 
    /// features and reducing dimensionality. Each vector in the list represents a selected feature, and the collection 
    /// as a whole represents the subset of features used to train and evaluate the solution in this optimization step.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the specific input variables (features) that were used for this model.
    /// 
    /// The selected features:
    /// - Are the input variables chosen for this model
    /// - Are stored as a list of vectors, where each vector represents one feature
    /// - May be a subset of all available features
    /// 
    /// Feature selection is important because:
    /// - Using only relevant features can improve model performance
    /// - It reduces model complexity and training time
    /// - It can help prevent overfitting
    /// 
    /// For example, if predicting house prices, this might contain selected features
    /// like square footage and location, but might exclude less relevant features.
    /// </para>
    /// </remarks>
    public List<Vector<T>> SelectedFeatures { get; set; } = [];

    /// <summary>
    /// Gets or sets the subset of the training data used for this optimization step.
    /// </summary>
    /// <value>A TInput object containing the subset of training data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the training data used for this optimization step. The training data is used to 
    /// train the model and adjust its parameters. The structure of this data depends on the TInput type, which could be a 
    /// Matrix<T> for traditional machine learning models or a Tensor<T> for neural networks. This subset might be created 
    /// through techniques like random sampling, stratified sampling, or other data selection methods to focus on specific 
    /// aspects of the data or to reduce computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used to train the model in this optimization step.
    /// 
    /// The training subset:
    /// - Contains the data used to train the model
    /// - Its structure depends on the type of model you're using (e.g., matrix for simple models, tensor for neural networks)
    /// - May be a subset of the full training data
    /// 
    /// This data is important because:
    /// - It's what the model learns from
    /// - The model adjusts its parameters based on this data
    /// - Different subsets might lead to different model behavior
    /// 
    /// For example, if you have 10,000 house price records, this might contain
    /// a subset of 8,000 records used to train this specific model.
    /// </para>
    /// </remarks>
    public TInput XTrainSubset { get; set; }

    /// <summary>
    /// Gets or sets the subset of the validation data used for this optimization step.
    /// </summary>
    /// <value>A TInput object containing the subset of validation data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the validation data used for this optimization step. The validation data is used to 
    /// evaluate the model during training and make decisions about hyperparameters or early stopping. The structure of this 
    /// data depends on the TInput type, which could be a Matrix<T> for traditional machine learning models or a Tensor<T> for 
    /// neural networks. This subset might be created through techniques like random sampling, stratified sampling, or other 
    /// data selection methods to focus on specific aspects of the data or to reduce computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used to validate the model during optimization.
    /// 
    /// The validation subset:
    /// - Contains data used to evaluate the model during training
    /// - Its structure depends on the type of model you're using (e.g., matrix for simple models, tensor for neural networks)
    /// - Is separate from the training data to provide an unbiased evaluation
    /// 
    /// This data is important because:
    /// - It helps detect overfitting
    /// - It guides decisions about when to stop training
    /// - It helps select the best model configuration
    /// 
    /// For example, if you have 10,000 house price records, this might contain
    /// a subset of 1,000 records used to validate this specific model.
    /// </para>
    /// </remarks>
    public TInput XValSubset { get; set; }

    /// <summary>
    /// Gets or sets the subset of the test data used for this optimization step.
    /// </summary>
    /// <value>A TInput object containing the subset of test data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the test data used for this optimization step. The test data is used to provide a 
    /// final, unbiased evaluation of the model after training and validation are complete. The structure of this data depends 
    /// on the TInput type, which could be a Matrix<T> for traditional machine learning models or a Tensor<T> for neural networks. 
    /// This subset might be created through techniques like random sampling, stratified sampling, or other data selection methods 
    /// to focus on specific aspects of the data or to reduce computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used for the final evaluation of the model.
    /// 
    /// The test subset:
    /// - Contains data used for the final evaluation of the model
    /// - Its structure depends on the type of model you're using (e.g., matrix for simple models, tensor for neural networks)
    /// - Is completely separate from training and validation data
    /// 
    /// This data is important because:
    /// - It provides an unbiased estimate of real-world performance
    /// - It's only used once after all model selection is complete
    /// - It helps set realistic expectations for deployment
    /// 
    /// For example, if you have 10,000 house price records, this might contain
    /// a subset of 1,000 records used for the final test of this specific model.
    /// </para>
    /// </remarks>
    public TInput XTestSubset { get; set; }

    /// <summary>
    /// Gets or sets the fitness score for this optimization step.
    /// </summary>
    /// <value>A value of type T representing the fitness or objective function value.</value>
    /// <remarks>
    /// <para>
    /// This property contains the fitness score for the solution in this optimization step. The fitness score is a measure of 
    /// how good the solution is according to some objective function. In optimization, higher fitness scores typically indicate 
    /// better solutions for maximization problems, while lower scores indicate better solutions for minimization problems. The 
    /// fitness score is a key metric for comparing different solutions and guiding the optimization process toward better solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This contains a single number that represents how good the model is.
    /// 
    /// The fitness score:
    /// - Measures how well the model performs
    /// - Is what the optimization process is trying to improve
    /// - Can be something to maximize (like accuracy) or minimize (like error)
    /// 
    /// This score is important because:
    /// - It allows direct comparison between different models
    /// - It guides the optimization process toward better solutions
    /// - It helps determine when the optimization is complete
    /// 
    /// For example, if optimizing for prediction accuracy, a fitness score of 0.95
    /// would indicate the model correctly predicts 95% of cases.
    /// </para>
    /// </remarks>
    public T FitnessScore { get; set; }

    /// <summary>
    /// Gets or sets the result of overfitting detection for this optimization step.
    /// </summary>
    /// <value>A FitDetectorResult&lt;T&gt; object containing information about potential overfitting.</value>
    /// <remarks>
    /// <para>
    /// This property contains the result of overfitting detection analysis for this optimization step. Overfitting occurs when 
    /// a model performs well on training data but poorly on new, unseen data. The FitDetectorResult object typically includes 
    /// information about whether overfitting has been detected, the severity of the overfitting, and possibly recommendations 
    /// for addressing it. This information is crucial for selecting models that will generalize well to new data rather than 
    /// just memorizing the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about whether the model is overfitting to the training data.
    /// 
    /// The fit detection result:
    /// - Indicates whether overfitting has been detected
    /// - May include measures of how severe the overfitting is
    /// - Helps identify models that won't perform well on new data
    /// 
    /// Overfitting happens when a model learns the training data too well,
    /// including its noise and peculiarities, making it perform poorly on new data.
    /// 
    /// This information is important because:
    /// - It helps avoid selecting models that won't generalize well
    /// - It provides insight into the model's learning behavior
    /// - It can guide decisions about regularization or model complexity
    /// </para>
    /// </remarks>
    public FitDetectorResult<T> FitDetectionResult { get; set; } = new();

    /// <summary>
    /// Gets or sets the comprehensive evaluation data for this optimization step.
    /// </summary>
    /// <value>A ModelEvaluationData&lt;T&gt; object containing detailed evaluation results.</value>
    /// <remarks>
    /// <para>
    /// This property contains comprehensive evaluation data for the solution in this optimization step. The ModelEvaluationData 
    /// object includes detailed statistics and performance metrics for the model on the training, validation, and test datasets. 
    /// This includes error statistics, prediction quality metrics, and other measures that provide a complete picture of the 
    /// model's performance. This detailed evaluation data is valuable for in-depth analysis of the model's strengths and 
    /// weaknesses and for making informed decisions about model selection.
    /// </para>
    /// <para><b>For Beginners:</b> This contains detailed performance information about the model across all datasets.
    /// 
    /// The evaluation data:
    /// - Contains comprehensive statistics about model performance
    /// - Includes results for training, validation, and test datasets
    /// - Provides multiple metrics beyond just the fitness score
    /// 
    /// This detailed information is valuable because:
    /// - It allows for deeper analysis of model behavior
    /// - It helps identify specific strengths and weaknesses
    /// - It provides context for the fitness score
    /// 
    /// For example, it might show that a model has low error on average but
    /// performs poorly on certain types of cases, revealing areas for improvement.
    /// </para>
    /// </remarks>
    public ModelEvaluationData<T, TInput, TOutput> EvaluationData { get; set; } = new();

    /// <summary>
    /// Initializes a new instance of the OptimizationStepData class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new OptimizationStepData instance with default values. It initializes the FitnessScore to 
    /// zero and creates an appropriate model based on the generic type parameters. For Vector models, it uses an empty vector.
    /// For Neural Network models, it creates a minimal network architecture. This constructor is useful when creating a new 
    /// optimization step data object before the actual optimization step is performed.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new optimization step data object with default values.
    /// 
    /// When creating a new optimization step data:
    /// - The fitness score is set to zero
    /// - An appropriate empty model is created based on your data types
    /// - Other properties are set to empty collections or default objects
    /// 
    /// This constructor intelligently chooses the right model type based on what
    /// kind of input and output data you're working with.
    /// </para>
    /// </remarks>
    public OptimizationStepData()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        FitnessScore = numOps.Zero;

        // Create default model data
        var (x, y, _) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();

        // Initialize properties with default values
        XTrainSubset = x;
        XValSubset = x;
        XTestSubset = x;
        Solution = ModelHelper<T, TInput, TOutput>.CreateDefaultModel();
    }
}
