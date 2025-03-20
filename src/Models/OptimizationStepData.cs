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
public class OptimizationStepData<T>
{
    /// <summary>
    /// Gets or sets the current solution (model) for this optimization step.
    /// </summary>
    /// <value>An ISymbolicModel&lt;T&gt; representing the current solution.</value>
    /// <remarks>
    /// <para>
    /// This property contains the current solution or model for this optimization step. The solution is an implementation 
    /// of the ISymbolicModel interface, which represents a model that can make predictions based on input features. 
    /// Different optimization steps might have different solutions with varying structures, complexities, and performance 
    /// characteristics. The solution is the central object of interest in the optimization process, as the goal is typically 
    /// to find the solution with the best performance according to some criteria.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the actual model that was evaluated in this optimization step.
    /// 
    /// The solution:
    /// - Is the machine learning model being evaluated
    /// - Implements the ISymbolicModel interface
    /// - Can make predictions based on input features
    /// 
    /// For example, it might be a linear regression model, a decision tree,
    /// or another type of predictive model.
    /// 
    /// This property is important because:
    /// - It's the main object being optimized
    /// - It contains the structure and parameters of the model
    /// - It can be used to make predictions on new data
    /// </para>
    /// </remarks>
    public ISymbolicModel<T> Solution { get; set; }
    
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
    /// <value>A Matrix&lt;T&gt; containing the subset of training data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the training data used for this optimization step. The training data is used to 
    /// train the model and adjust its parameters. The matrix is organized with each row representing an observation (data point) 
    /// and each column representing a feature (input variable). This subset might be created through techniques like random 
    /// sampling, stratified sampling, or other data selection methods to focus on specific aspects of the data or to reduce 
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used to train the model in this optimization step.
    /// 
    /// The training subset:
    /// - Contains the data used to train the model
    /// - Is organized as a matrix where each row is a data point and each column is a feature
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
    public Matrix<T> XTrainSubset { get; set; } = Matrix<T>.Empty();
    
    /// <summary>
    /// Gets or sets the subset of the validation data used for this optimization step.
    /// </summary>
    /// <value>A Matrix&lt;T&gt; containing the subset of validation data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the validation data used for this optimization step. The validation data is used to 
    /// evaluate the model during training and make decisions about hyperparameters or early stopping. The matrix is organized 
    /// with each row representing an observation (data point) and each column representing a feature (input variable). This 
    /// subset might be created through techniques like random sampling, stratified sampling, or other data selection methods 
    /// to focus on specific aspects of the data or to reduce computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used to validate the model during optimization.
    /// 
    /// The validation subset:
    /// - Contains data used to evaluate the model during training
    /// - Is organized as a matrix where each row is a data point and each column is a feature
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
    public Matrix<T> XValSubset { get; set; } = Matrix<T>.Empty();
    
    /// <summary>
    /// Gets or sets the subset of the test data used for this optimization step.
    /// </summary>
    /// <value>A Matrix&lt;T&gt; containing the subset of test data.</value>
    /// <remarks>
    /// <para>
    /// This property contains a subset of the test data used for this optimization step. The test data is used to provide a 
    /// final, unbiased evaluation of the model after training and validation are complete. The matrix is organized with each 
    /// row representing an observation (data point) and each column representing a feature (input variable). This subset might 
    /// be created through techniques like random sampling, stratified sampling, or other data selection methods to focus on 
    /// specific aspects of the data or to reduce computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the data used for the final evaluation of the model.
    /// 
    /// The test subset:
    /// - Contains data used for the final evaluation of the model
    /// - Is organized as a matrix where each row is a data point and each column is a feature
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
    public Matrix<T> XTestSubset { get; set; } = Matrix<T>.Empty();
    
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
    public ModelEvaluationData<T> EvaluationData { get; set; } = new();

    /// <summary>
    /// Initializes a new instance of the OptimizationStepData class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new OptimizationStepData instance with default values. It initializes the FitnessScore to 
    /// zero using the appropriate numeric operations for type T and sets the Solution to a new VectorModel with an empty vector. 
    /// Other properties are initialized to their default values as specified in their declarations. This constructor is useful 
    /// when creating a new optimization step data object before the actual optimization step is performed.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new optimization step data object with default values.
    /// 
    /// When using this constructor:
    /// - The fitness score is set to zero
    /// - The solution is initialized as an empty vector model
    /// - Other properties are set to empty collections or default objects
    /// 
    /// This constructor is typically used when:
    /// - Creating a new step data object before performing the optimization
    /// - Initializing a collection of step data objects
    /// - Setting up the framework for recording optimization results
    /// 
    /// After creating the object, you would typically populate its properties
    /// with actual data from an optimization step.
    /// </para>
    /// </remarks>
    public OptimizationStepData()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        FitnessScore = numOps.Zero;
        Solution = new VectorModel<T>(Vector<T>.Empty());
    }
}