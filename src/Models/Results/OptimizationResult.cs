namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the comprehensive results of an optimization process for a symbolic model, including the best solution found,
/// performance metrics, feature selection results, and detailed statistics for different datasets.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the information produced during the optimization of a symbolic model. It includes the best 
/// model found, its fitness score, the number of iterations performed, the history of fitness scores during optimization, 
/// the features selected for the model, detailed results for training, validation, and test datasets, fit detection 
/// analysis, and coefficient bounds. This comprehensive collection of information allows for thorough analysis of the 
/// optimization process and the resulting model.
/// </para>
/// <para><b>For Beginners:</b> This class stores everything about an optimization process and its results.
/// 
/// When optimizing a model:
/// - You start with an initial model and try to improve it
/// - You track how the model performs as it evolves
/// - You need to know which features were important
/// - You want to see how well it performs on different datasets
/// 
/// This class stores all that information, including:
/// - The best model found during optimization
/// - How good that model is (fitness score)
/// - How many iterations the optimization process ran
/// - How the model improved over time
/// - Which input features were selected
/// - Detailed performance metrics on training, validation, and test data
/// - Analysis of potential issues like overfitting
/// 
/// Having all this information in one place makes it easier to understand,
/// evaluate, and document your optimization results.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class OptimizationResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the best model found during optimization.
    /// </summary>
    /// <value>An implementation of IFullModel&lt;T, TInput, TOutput&gt; representing the best solution.</value>
    /// <remarks>
    /// <para>
    /// This property represents the best model found during the optimization process. The model can be either a symbolic 
    /// model (such as a mathematical expression) or a more complex structure like a neural network, depending on the 
    /// optimization approach used. For symbolic models, it encapsulates a mathematical expression or algorithm that can 
    /// be evaluated with different inputs to produce predictions. For neural networks or other complex models, it 
    /// represents the optimized network structure and parameters. The model captures the relationship discovered between 
    /// the input features and the target variable during the optimization process.
    /// </para>
    /// <para><b>For Beginners:</b> This is the best model found during optimization.
    /// 
    /// The best solution:
    /// - Contains the actual model structure, whether it's a mathematical formula, a neural network, or another type of model
    /// - Represents the best model found during the optimization process
    /// - Can be used to make predictions with new data
    /// - May be human-readable (for symbolic models) or more complex (for neural networks)
    /// 
    /// For example:
    /// - In symbolic regression, this might be an equation like: y = 3.2x1² + 1.7x2 - 0.5
    /// - For a neural network, it would contain the optimized network structure and weights
    /// 
    /// This property is important because:
    /// - It's the primary output of the optimization process
    /// - It can be used to understand the relationships in your data (especially for symbolic models)
    /// - It can be deployed to make predictions on new data
    /// - It allows for flexibility in the types of models that can be optimized
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? BestSolution { get; set; }

    /// <summary>
    /// Gets or sets the intercept term of the best solution.
    /// </summary>
    /// <value>A numeric value representing the intercept term, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents the intercept term (also known as the bias or constant term) of the best solution. In 
    /// many mathematical models, particularly linear and polynomial models, the intercept is the value of the dependent 
    /// variable when all independent variables are zero. It represents the baseline value of the target variable before 
    /// considering the effects of the input features. The intercept is an important parameter of the model and is often 
    /// optimized along with the coefficients of the input features.
    /// </para>
    /// <para><b>For Beginners:</b> This is the constant term in your model equation.
    /// 
    /// The intercept:
    /// - Is the value your model predicts when all input variables are zero
    /// - Represents the baseline or starting point for predictions
    /// - Is often written as the constant term in an equation
    /// 
    /// For example, in the equation y = 3x + 2, the intercept is 2.
    /// 
    /// This value is important because:
    /// - It establishes the baseline prediction
    /// - It can have meaningful interpretation in some domains
    /// - It's often optimized along with other model parameters
    /// </para>
    /// </remarks>
    public T BestIntercept { get; set; }

    /// <summary>
    /// Gets or sets the fitness score of the best solution.
    /// </summary>
    /// <value>A numeric value representing the best fitness score, initialized to zero.</value>
    /// <remarks>
    /// <para>
    /// This property represents the fitness or performance score of the best solution found during optimization. The 
    /// fitness score is a single numeric value that quantifies how well the model performs, typically on the training 
    /// data. Higher values usually indicate better performance, though the exact interpretation depends on the specific 
    /// fitness function used. Common fitness metrics include R-squared (coefficient of determination), negative mean 
    /// squared error, or accuracy. The fitness score is often used as the primary criterion for comparing and selecting 
    /// models during the optimization process.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how well the best model performs.
    /// 
    /// The fitness score:
    /// - Measures how well the model fits the data
    /// - Is typically a single number that summarizes performance
    /// - Higher values usually indicate better performance
    /// 
    /// Common fitness metrics include:
    /// - R² (R-squared): Measures the proportion of variance explained (higher is better)
    /// - Negative MSE (Mean Squared Error): Measures prediction error (closer to zero is better)
    /// - Accuracy: For classification problems, the percentage of correct predictions
    /// 
    /// This value is important because:
    /// - It tells you how good your best model is
    /// - It can be compared with other models' fitness scores
    /// - It's often the main criterion used during optimization
    /// </para>
    /// </remarks>
    public T BestFitnessScore { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations performed during optimization.
    /// </summary>
    /// <value>An integer representing the number of iterations.</value>
    /// <remarks>
    /// <para>
    /// This property represents the number of iterations or generations performed during the optimization process. Each 
    /// iteration typically involves evaluating and potentially modifying the model to improve its performance. The number 
    /// of iterations can provide insight into the computational effort expended during optimization and can be useful for 
    /// comparing different optimization runs. It may also indicate whether the optimization process converged naturally 
    /// or was terminated due to reaching a maximum iteration limit.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many steps the optimization process took.
    /// 
    /// The iterations count:
    /// - Represents how many times the optimization algorithm tried to improve the model
    /// - Each iteration typically evaluates and potentially modifies the model
    /// - Higher values usually mean more computational effort was expended
    /// 
    /// This value is useful because:
    /// - It helps you understand how long the optimization process ran
    /// - It can indicate whether the process converged naturally or hit a limit
    /// - It allows you to compare the computational effort of different optimization runs
    /// 
    /// For example, an iterations value of 1000 means the optimization algorithm
    /// performed 1000 steps trying to find the best model.
    /// </para>
    /// </remarks>
    public int Iterations { get; set; }

    /// <summary>
    /// Gets or sets the history of fitness scores during optimization.
    /// </summary>
    /// <value>A vector of fitness scores, initialized as an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property contains the history of fitness scores recorded during the optimization process. Each element in 
    /// the vector represents the fitness score of the best model at a particular iteration or checkpoint. This history 
    /// can be used to analyze the convergence behavior of the optimization algorithm, identify plateaus or jumps in 
    /// performance, and visualize the improvement of the model over time. It can also help in diagnosing issues such 
    /// as premature convergence or oscillation.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how the model's performance improved over time.
    /// 
    /// The fitness history:
    /// - Contains the fitness score at each iteration or checkpoint
    /// - Shows how the model improved during optimization
    /// - Can be plotted to visualize the optimization progress
    /// 
    /// This information is valuable because:
    /// - It helps you understand how quickly the model improved
    /// - It can show if the optimization got stuck or plateaued
    /// - It can reveal if more iterations might have helped
    /// 
    /// For example, a steadily increasing fitness history suggests the optimization
    /// was effective, while a flat line might indicate the algorithm got stuck.
    /// </para>
    /// </remarks>
    public Vector<T> FitnessHistory { get; set; }

    /// <summary>
    /// Gets or sets the list of feature vectors selected for the model.
    /// </summary>
    /// <value>A list of Vector&lt;T&gt; objects representing the selected features, initialized as an empty list.</value>
    /// <remarks>
    /// <para>
    /// This property contains the list of feature vectors that were selected for use in the best model. Feature selection 
    /// is the process of identifying and selecting the most relevant features (input variables) for the model, which can 
    /// improve model performance, reduce overfitting, and enhance interpretability. The selected features are represented 
    /// as vectors, where each vector corresponds to a feature and contains the values of that feature across all 
    /// observations in the dataset. This information is useful for understanding which input variables the model considers 
    /// important and for reproducing the model with new data.
    /// </para>
    /// <para><b>For Beginners:</b> This list shows which input variables were used in the best model.
    /// 
    /// The selected features:
    /// - Represent the input variables that were chosen for the model
    /// - Are stored as vectors (arrays) of values
    /// - May be a subset of all available features if feature selection was performed
    /// 
    /// Feature selection is important because:
    /// - Using too many features can lead to overfitting
    /// - Some features may be irrelevant or redundant
    /// - Models with fewer features are often more interpretable
    /// 
    /// For example, if you started with 20 potential input variables but the model
    /// only uses 5 of them, this list would contain those 5 selected features.
    /// </para>
    /// </remarks>
    public List<Vector<T>> SelectedFeatures { get; set; }

    /// <summary>
    /// Gets or sets the detailed results for the training dataset.
    /// </summary>
    /// <value>A DatasetResult object containing detailed statistics for the training dataset.</value>
    /// <remarks>
    /// <para>
    /// This property contains detailed results and statistics for the training dataset, which is the portion of data used 
    /// to fit or train the model. The training result includes the input features (X), target values (Y), model predictions, 
    /// error statistics, prediction statistics, and basic statistics for both actual and predicted values. This information 
    /// is crucial for understanding how well the model fits the training data and for diagnosing potential issues such as 
    /// underfitting or overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on the training data.
    /// 
    /// The training result:
    /// - Shows how well the model fits the data it was trained on
    /// - Contains the actual data, predictions, and various statistics
    /// - Helps you understand the model's basic performance
    /// 
    /// This information is important because:
    /// - It's the baseline for evaluating your model
    /// - It helps identify if the model is learning the patterns in the data
    /// - It can be compared with validation and test results to detect overfitting
    /// 
    /// For example, good performance on training data but poor performance on test data
    /// would suggest the model is overfitting to the training data.
    /// </para>
    /// </remarks>
    public DatasetResult TrainingResult { get; set; }

    /// <summary>
    /// Gets or sets the detailed results for the validation dataset.
    /// </summary>
    /// <value>A DatasetResult object containing detailed statistics for the validation dataset.</value>
    /// <remarks>
    /// <para>
    /// This property contains detailed results and statistics for the validation dataset, which is the portion of data used 
    /// to tune hyperparameters and evaluate the model during training. The validation result includes the input features (X), 
    /// target values (Y), model predictions, error statistics, prediction statistics, and basic statistics for both actual 
    /// and predicted values. This information is crucial for understanding how well the model generalizes to unseen data and 
    /// for making decisions about model selection and hyperparameter tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on the validation data.
    /// 
    /// The validation result:
    /// - Shows how well the model performs on data not used in training
    /// - Helps detect overfitting (when a model performs well on training data but poorly on new data)
    /// - Is used to guide model selection and hyperparameter tuning
    /// 
    /// This information is important because:
    /// - It provides feedback during the model development process
    /// - It helps you choose between different model configurations
    /// - It gives an early indication of how well the model might perform in production
    /// 
    /// For example, if performance is significantly worse on validation data than training data,
    /// you might need to simplify your model to reduce overfitting.
    /// </para>
    /// </remarks>
    public DatasetResult ValidationResult { get; set; }

    /// <summary>
    /// Gets or sets the detailed results for the test dataset.
    /// </summary>
    /// <value>A DatasetResult object containing detailed statistics for the test dataset.</value>
    /// <remarks>
    /// <para>
    /// This property contains detailed results and statistics for the test dataset, which is the portion of data set aside 
    /// and not used during model training or validation. The test result includes the input features (X), target values (Y), 
    /// model predictions, error statistics, prediction statistics, and basic statistics for both actual and predicted values. 
    /// This information provides the most realistic estimate of how well the model will perform on completely new, unseen data 
    /// and is the final evaluation metric used to assess the model's generalization ability.
    /// </para>
    /// <para><b>For Beginners:</b> This contains detailed information about how the model performs on completely new data.
    /// 
    /// The test result:
    /// - Shows how well the model performs on data it has never seen before
    /// - Provides the most realistic estimate of real-world performance
    /// - Is the final evaluation metric for your model
    /// 
    /// This information is important because:
    /// - It represents how your model will likely perform in production
    /// - It's the most honest assessment of your model's capabilities
    /// - It helps determine if your model is ready for deployment
    /// 
    /// For example, good performance on test data suggests your model will
    /// generalize well to new, unseen data in real-world applications.
    /// </para>
    /// </remarks>
    public DatasetResult TestResult { get; set; }

    /// <summary>
    /// Gets or sets the results of fit detection analysis.
    /// </summary>
    /// <value>A FitDetectorResult&lt;T&gt; object containing detailed fit analysis.</value>
    /// <remarks>
    /// <para>
    /// This property contains the results of fit detection analysis, which evaluates how well the model fits the data and 
    /// identifies potential issues such as underfitting or overfitting. Underfitting occurs when the model is too simple to 
    /// capture the underlying patterns in the data, resulting in poor performance on both training and test data. Overfitting 
    /// occurs when the model is too complex and captures noise in the training data, resulting in good performance on training 
    /// data but poor performance on test data. The FitDetectorResult includes the type of fit detected, a confidence level for 
    /// that assessment, and recommendations for improving the model.
    /// </para>
    /// <para><b>For Beginners:</b> This contains an analysis of how well your model fits the data.
    /// 
    /// The fit detection result:
    /// - Identifies if your model is underfitting, overfitting, or has a good fit
    /// - Provides a confidence level for this assessment
    /// - Offers specific recommendations to improve your model
    /// 
    /// This analysis helps you understand:
    /// - If your model is too simple (underfitting)
    /// - If your model is too complex (overfitting)
    /// - What steps you should take to improve it
    /// 
    /// For example, if your model is overfitting, the fit detection might recommend
    /// adding regularization or reducing model complexity.
    /// </para>
    /// </remarks>
    public FitDetectorResult<T> FitDetectionResult { get; set; }

    /// <summary>
    /// Gets or sets the lower bounds for model coefficients.
    /// </summary>
    /// <value>A vector of lower bound values, initialized as an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property contains the lower bounds for the coefficients in the model. Coefficient bounds are constraints that 
    /// limit the range of values that the model coefficients can take during optimization. Lower bounds specify the minimum 
    /// allowed values for each coefficient. These bounds can be used to incorporate domain knowledge into the model, ensure 
    /// physical or logical constraints are respected, or improve the stability of the optimization process. For example, in 
    /// some applications, coefficients might need to be non-negative or above a certain threshold to be physically meaningful.
    /// </para>
    /// <para><b>For Beginners:</b> This specifies the minimum allowed values for each coefficient in your model.
    /// 
    /// The coefficient lower bounds:
    /// - Set the minimum values that each coefficient can have
    /// - Help constrain the model to realistic or meaningful solutions
    /// - Can incorporate domain knowledge into the optimization process
    /// 
    /// These bounds are useful because:
    /// - They can prevent the model from finding mathematically valid but physically impossible solutions
    /// - They can improve optimization stability
    /// - They allow you to enforce known constraints (like non-negativity)
    /// 
    /// For example, if you know a certain effect cannot be negative in your domain,
    /// you might set the lower bound for that coefficient to zero.
    /// </para>
    /// </remarks>
    public Vector<T> CoefficientLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bounds for model coefficients.
    /// </summary>
    /// <value>A vector of upper bound values, initialized as an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property contains the upper bounds for the coefficients in the model. Coefficient bounds are constraints that 
    /// limit the range of values that the model coefficients can take during optimization. Upper bounds specify the maximum 
    /// allowed values for each coefficient. These bounds can be used to incorporate domain knowledge into the model, ensure 
    /// physical or logical constraints are respected, or improve the stability of the optimization process. For example, in 
    /// some applications, coefficients might need to be below a certain threshold to be physically meaningful or to prevent 
    /// numerical issues.
    /// </para>
    /// <para><b>For Beginners:</b> This specifies the maximum allowed values for each coefficient in your model.
    /// 
    /// The coefficient upper bounds:
    /// - Set the maximum values that each coefficient can have
    /// - Help constrain the model to realistic or meaningful solutions
    /// - Can incorporate domain knowledge into the optimization process
    /// 
    /// These bounds are useful because:
    /// - They can prevent the model from finding mathematically valid but physically impossible solutions
    /// - They can improve optimization stability
    /// - They allow you to enforce known constraints
    /// 
    /// For example, if you know a certain effect cannot exceed a specific value in your domain,
    /// you might set the upper bound for that coefficient accordingly.
    /// </para>
    /// </remarks>
    public Vector<T> CoefficientUpperBounds { get; set; }

    /// <summary>
    /// Provides numeric operations for the generic type T.
    /// </summary>
    /// <remarks>
    /// This field stores an instance of INumericOperations&lt;T&gt; that provides mathematical operations for the generic 
    /// type T. It allows the class to perform numeric operations regardless of the specific numeric type used.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the OptimizationResult class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new OptimizationResult instance and initializes all properties to their default values. 
    /// It obtains the appropriate numeric operations for the generic type T, creates a default symbolic model, and initializes 
    /// all vectors, lists, and nested objects. This provides a clean starting point for storing optimization results. The 
    /// default symbolic model is a vector model, which is a simple linear model that can be used as a placeholder until a 
    /// better model is found during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new result object with default values.
    /// 
    /// When a new OptimizationResult is created:
    /// - All numeric values are set to zero
    /// - All collections (vectors, lists) are initialized as empty
    /// - A simple default model is created as a placeholder
    /// - All nested result objects are initialized with their own default values
    /// 
    /// This initialization is important because:
    /// - It ensures consistent behavior regardless of how the object is created
    /// - It prevents potential issues with uninitialized values
    /// - It provides a clean slate for storing optimization results
    /// 
    /// You typically won't need to call this constructor directly, as it will be
    /// used internally by the optimization process.
    /// </para>
    /// </remarks>
    public OptimizationResult()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        FitnessHistory = Vector<T>.Empty();
        SelectedFeatures = new List<Vector<T>>();
        TrainingResult = new DatasetResult();
        ValidationResult = new DatasetResult();
        TestResult = new DatasetResult();
        FitDetectionResult = new FitDetectorResult<T>();
        CoefficientLowerBounds = Vector<T>.Empty();
        CoefficientUpperBounds = Vector<T>.Empty();
        BestIntercept = _numOps.Zero;
        BestFitnessScore = _numOps.Zero;
    }

    /// <summary>
    /// Creates a deep copy of this OptimizationResult instance.
    /// </summary>
    /// <returns>A new OptimizationResult with copied values.</returns>
    public OptimizationResult<T, TInput, TOutput> DeepCopy()
    {
        return new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = BestSolution?.DeepCopy(),
            BestIntercept = BestIntercept,
            BestFitnessScore = BestFitnessScore,
            Iterations = Iterations,
            FitnessHistory = new Vector<T>(FitnessHistory.ToArray()),
            SelectedFeatures = SelectedFeatures.Select(v => new Vector<T>(v.ToArray())).ToList(),
            TrainingResult = TrainingResult,
            ValidationResult = ValidationResult,
            TestResult = TestResult,
            FitDetectionResult = FitDetectionResult,
            CoefficientLowerBounds = new Vector<T>(CoefficientLowerBounds.ToArray()),
            CoefficientUpperBounds = new Vector<T>(CoefficientUpperBounds.ToArray())
        };
    }

    /// <summary>
    /// Creates a new OptimizationResult instance with the best solution updated to use the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to apply to the best solution.</param>
    /// <returns>A new OptimizationResult with the updated model.</returns>
    public OptimizationResult<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new OptimizationResult<T, TInput, TOutput>
        {
            BestSolution = BestSolution?.WithParameters(parameters),
            BestIntercept = BestIntercept,
            BestFitnessScore = BestFitnessScore,
            Iterations = Iterations,
            FitnessHistory = FitnessHistory,
            SelectedFeatures = SelectedFeatures,
            TrainingResult = TrainingResult,
            ValidationResult = ValidationResult,
            TestResult = TestResult,
            FitDetectionResult = FitDetectionResult,
            CoefficientLowerBounds = CoefficientLowerBounds,
            CoefficientUpperBounds = CoefficientUpperBounds
        };
    }

    /// <summary>
    /// Represents detailed results and statistics for a specific dataset (training, validation, or test).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This nested class encapsulates all the data and statistics related to model performance on a specific dataset. 
    /// It includes the input features (X), target values (Y), model predictions, and various statistical measures that 
    /// quantify different aspects of model performance. These statistics include error metrics (such as mean squared error, 
    /// mean absolute error), prediction quality metrics (such as R-squared, correlation), and basic descriptive statistics 
    /// for both the actual and predicted values. This comprehensive collection of information allows for thorough analysis 
    /// of model performance on the dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This class stores all the details about how the model performs on a specific dataset.
    /// 
    /// For each dataset (training, validation, or test), this stores:
    /// - The actual input data (X) and target values (Y)
    /// - The model's predictions
    /// - Various error measurements (how far predictions are from actual values)
    /// - Statistics about prediction quality (how well the model captures patterns)
    /// - Basic statistics about both actual values and predictions
    /// 
    /// This detailed information helps you:
    /// - Understand exactly how well your model is performing
    /// - Identify specific strengths and weaknesses
    /// - Compare performance across different datasets
    /// - Diagnose issues like overfitting or underfitting
    /// </para>
    /// </remarks>
    public class DatasetResult
    {
        /// <summary>
        /// Gets or sets the input feature matrix for the dataset.
        /// </summary>
        /// <value>A matrix where each row represents an observation and each column represents a feature.</value>
        /// <remarks>
        /// <para>
        /// This property contains the input feature matrix for the dataset, where each row represents an observation (data point) 
        /// and each column represents a feature (input variable). This matrix contains the independent variables used to make 
        /// predictions. It is stored to allow for further analysis, visualization, or reuse of the model with the same data.
        /// </para>
        /// <para><b>For Beginners:</b> This contains the input data used for making predictions.
        /// 
        /// The feature matrix:
        /// - Contains all the input variables (features) for each data point
        /// - Is organized with each row representing one observation
        /// - Each column represents a different input variable
        /// 
        /// For example, if predicting house prices, this might contain features like
        /// square footage, number of bedrooms, location, etc. for each house.
        /// </para>
        /// </remarks>
        public TInput X { get; set; }

        /// <summary>
        /// Gets or sets the target values for the dataset.
        /// </summary>
        /// <value>A vector of target values, one for each observation.</value>
        /// <remarks>
        /// <para>
        /// This property contains the target values (dependent variable) for the dataset. Each element in the vector corresponds 
        /// to an observation (row) in the input feature matrix X. These are the actual values that the model attempts to predict. 
        /// They are stored to allow for comparison with the model's predictions and calculation of various performance metrics.
        /// </para>
        /// <para><b>For Beginners:</b> This contains the actual values you're trying to predict.
        /// 
        /// The target values:
        /// - Are what your model is trying to predict
        /// - Each value corresponds to one row in the feature matrix
        /// - Are used to calculate how accurate your predictions are
        /// 
        /// For example, if predicting house prices, this would contain the
        /// actual sale price for each house in your dataset.
        /// </para>
        /// </remarks>
        public TOutput Y { get; set; }

        /// <summary>
        /// Gets or sets the model's predictions for the dataset.
        /// </summary>
        /// <value>A vector of predicted values, one for each observation.</value>
        /// <remarks>
        /// <para>
        /// This property contains the model's predictions for the dataset. Each element in the vector corresponds to an 
        /// observation (row) in the input feature matrix X and represents the model's prediction for that observation. These 
        /// predictions are compared with the actual target values (Y) to calculate various performance metrics and assess the 
        /// model's accuracy.
        /// </para>
        /// <para><b>For Beginners:</b> This contains what your model predicted for each data point.
        /// 
        /// The predictions:
        /// - Are the values your model output for each data point
        /// - Each prediction corresponds to one row in the feature matrix
        /// - Are compared to the actual values to measure performance
        /// 
        /// For example, if predicting house prices, this would contain your model's
        /// estimated price for each house in the dataset.
        /// </para>
        /// </remarks>
        public TOutput Predictions { get; set; }

        /// <summary>
        /// Gets or sets the error statistics for the model's predictions.
        /// </summary>
        /// <value>An ErrorStats&lt;T&gt; object containing various error metrics.</value>
        /// <remarks>
        /// <para>
        /// This property contains various error statistics that quantify the differences between the model's predictions and the 
        /// actual target values. These statistics include measures such as mean squared error (MSE), root mean squared error (RMSE), 
        /// mean absolute error (MAE), and others. Error statistics focus on the magnitude of prediction errors and provide different 
        /// perspectives on the model's accuracy. Lower values for these metrics generally indicate better model performance.
        /// </para>
        /// <para><b>For Beginners:</b> This contains measurements of how far off your predictions are from the actual values.
        /// 
        /// The error statistics:
        /// - Measure the size of prediction errors in different ways
        /// - Lower values indicate better performance
        /// - Different metrics emphasize different aspects of error
        /// 
        /// Common error metrics include:
        /// - MSE (Mean Squared Error): Average of squared differences between predictions and actual values
        /// - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as the target variable
        /// - MAE (Mean Absolute Error): Average of absolute differences between predictions and actual values
        /// 
        /// These metrics help you understand how accurate your model is and
        /// can guide you in improving it.
        /// </para>
        /// </remarks>
        public ErrorStats<T> ErrorStats { get; set; }

        /// <summary>
        /// Gets or sets the prediction quality statistics for the model.
        /// </summary>
        /// <value>A PredictionStats&lt;T&gt; object containing various prediction quality metrics.</value>
        /// <remarks>
        /// <para>
        /// This property contains various statistics that assess the quality of the model's predictions beyond simple error 
        /// measurements. These statistics include measures such as R-squared (coefficient of determination), adjusted R-squared, 
        /// correlation between predictions and actual values, and others. Prediction quality statistics focus on how well the 
        /// model captures the patterns in the data and explains the variance in the target variable. Higher values for these 
        /// metrics generally indicate better model performance.
        /// </para>
        /// <para><b>For Beginners:</b> This contains measurements of how well your model captures patterns in the data.
        /// 
        /// The prediction statistics:
        /// - Assess how well your model explains the patterns in the data
        /// - Higher values typically indicate better performance
        /// - Focus on the relationship between predictions and actual values
        /// 
        /// Common prediction metrics include:
        /// - R² (R-squared): Proportion of variance explained by the model (0-1, higher is better)
        /// - Adjusted R²: R-squared adjusted for the number of predictors
        /// - Correlation: How strongly predictions and actual values are related
        /// 
        /// These metrics help you understand how well your model captures
        /// the underlying patterns rather than just measuring error size.
        /// </para>
        /// </remarks>
        public PredictionStats<T> PredictionStats { get; set; }

        /// <summary>
        /// Gets or sets the basic descriptive statistics for the actual target values.
        /// </summary>
        /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the actual values.</value>
        /// <remarks>
        /// <para>
        /// This property contains basic descriptive statistics for the actual target values (Y) in the dataset. These statistics 
        /// include measures such as mean, median, standard deviation, minimum, maximum, and others. They provide a summary of the 
        /// distribution of the target variable and can be useful for understanding the data and interpreting the model's performance.
        /// </para>
        /// <para><b>For Beginners:</b> This contains summary statistics about the actual values in your dataset.
        /// 
        /// The actual basic statistics:
        /// - Summarize the distribution of the actual target values
        /// - Help you understand the data you're working with
        /// - Provide context for interpreting model performance
        /// 
        /// Common statistics include:
        /// - Mean: The average value
        /// - Median: The middle value when sorted
        /// - Standard Deviation: How spread out the values are
        /// - Min/Max: The smallest and largest values
        /// 
        /// These statistics help you understand what you're trying to predict
        /// and provide context for evaluating your model's predictions.
        /// </para>
        /// </remarks>
        public BasicStats<T> ActualBasicStats { get; set; }

        /// <summary>
        /// Gets or sets the basic descriptive statistics for the predicted values.
        /// </summary>
        /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the predicted values.</value>
        /// <remarks>
        /// <para>
        /// This property contains basic descriptive statistics for the model's predictions. These statistics include measures such 
        /// as mean, median, standard deviation, minimum, maximum, and others. They provide a summary of the distribution of the 
        /// predicted values and can be compared with the statistics for the actual values to assess how well the model captures 
        /// the overall distribution of the target variable.
        /// </para>
        /// <para><b>For Beginners:</b> This contains summary statistics about your model's predictions.
        /// 
        /// The predicted basic statistics:
        /// - Summarize the distribution of the values your model predicted
        /// - Can be compared with the actual statistics to see if your model captures the overall distribution
        /// - Help identify systematic biases in your predictions
        /// 
        /// Common statistics include:
        /// - Mean: The average predicted value
        /// - Median: The middle predicted value when sorted
        /// - Standard Deviation: How spread out the predictions are
        /// - Min/Max: The smallest and largest predictions
        /// 
        /// Comparing these with the actual statistics can reveal issues like:
        /// - Predictions that are systematically too high or too low (comparing means)
        /// - Predictions that don't capture the full range of variation (comparing standard deviations)
        /// - Predictions that don't reach the extremes of the actual data (comparing min/max)
        /// </para>
        /// </remarks>
        public BasicStats<T> PredictedBasicStats { get; set; }

        /// <summary>
        /// Initializes a new instance of the DatasetResult class with empty data structures.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This constructor creates a new DatasetResult instance and initializes all properties to empty data structures. It creates 
        /// empty matrices and vectors for the data and predictions, and empty statistics objects for the various statistical measures. 
        /// This provides a clean starting point for storing dataset results. The empty data structures will be populated with actual 
        /// data and statistics during the evaluation of the model on the dataset.
        /// </para>
        /// <para><b>For Beginners:</b> This constructor creates a new dataset result object with empty placeholders.
        /// 
        /// When a new DatasetResult is created:
        /// - All data structures (matrices, vectors) are initialized as empty
        /// - All statistics objects are initialized as empty
        /// 
        /// This initialization is important because:
        /// - It ensures consistent behavior regardless of how the object is created
        /// - It prevents potential issues with uninitialized values
        /// - It provides a clean slate for storing dataset results
        /// 
        /// You typically won't need to call this constructor directly, as it will be
        /// used internally when creating the OptimizationResult.
        /// </para>
        /// </remarks>
        public DatasetResult()
        {
            (X, Y, Predictions) = ModelHelper<T, TInput, TOutput>.CreateDefaultModelData();
            ErrorStats = ErrorStats<T>.Empty();
            PredictionStats = PredictionStats<T>.Empty();
            ActualBasicStats = BasicStats<T>.Empty();
            PredictedBasicStats = BasicStats<T>.Empty();
        }
    }
}
