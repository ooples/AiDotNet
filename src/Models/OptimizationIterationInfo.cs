namespace AiDotNet.Models;

/// <summary>
/// Represents information about a single iteration in an optimization process, including fitness and overfitting detection results.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates information about a specific iteration in an optimization or training process. It stores the 
/// iteration number, the fitness value achieved at that iteration, and results from overfitting detection. This information 
/// is useful for tracking the progress of optimization algorithms, analyzing convergence behavior, and detecting when 
/// training should be stopped to prevent overfitting.
/// </para>
/// <para><b>For Beginners:</b> This class stores information about one step in a model training or optimization process.
/// 
/// When training machine learning models or optimizing algorithms:
/// - The process typically runs through many iterations (steps)
/// - You want to track how well the model is performing at each step
/// - You need to detect when to stop training to avoid overfitting
/// 
/// This class stores information about a single iteration, including:
/// - Which iteration number it is
/// - How good the solution is at this step (fitness)
/// - Whether overfitting has been detected
/// 
/// This information helps you monitor the optimization process,
/// decide when to stop, and analyze how the solution improved over time.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class OptimizationIterationInfo<T>
{
    /// <summary>
    /// Gets or sets the iteration number.
    /// </summary>
    /// <value>An integer representing the current iteration in the optimization process.</value>
    /// <remarks>
    /// <para>
    /// This property represents the iteration number in the optimization process. Iterations typically start at 0 or 1 
    /// and increment with each step of the algorithm. The iteration number is useful for tracking the progress of the 
    /// optimization, correlating changes in fitness with specific steps in the process, and implementing stopping 
    /// criteria based on the number of iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which step in the optimization process this information is from.
    /// 
    /// The iteration number:
    /// - Counts which step in the process this represents
    /// - Typically starts at 0 or 1 and increases by 1 each step
    /// - Helps you track progress through the optimization
    /// 
    /// For example, if you're training a model for 100 iterations,
    /// this value would range from 0 to 99 (or 1 to 100).
    /// 
    /// This property is useful for:
    /// - Plotting how fitness changes over iterations
    /// - Implementing stopping conditions (e.g., stop after 1000 iterations)
    /// - Identifying when specific events occurred during training
    /// </para>
    /// </remarks>
    public int Iteration { get; set; }

    /// <summary>
    /// The backing field for the Fitness property.
    /// </summary>
    /// <remarks>
    /// This private field stores the fitness value and is accessed through the Fitness property.
    /// </remarks>
    private T _fitness;

    /// <summary>
    /// Gets or sets the fitness value at this iteration.
    /// </summary>
    /// <value>A value of type T representing the fitness or objective function value.</value>
    /// <remarks>
    /// <para>
    /// This property represents the fitness or objective function value achieved at this iteration of the optimization 
    /// process. In optimization, the fitness typically measures how good a solution is, with higher values usually 
    /// indicating better solutions for maximization problems and lower values indicating better solutions for minimization 
    /// problems. The fitness value is a key metric for tracking the progress of the optimization and determining when the 
    /// algorithm has converged to a satisfactory solution.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how good the solution is at this step in the process.
    /// 
    /// The fitness value:
    /// - Measures how good the current solution is
    /// - Is what the optimization process is trying to improve
    /// - Can be something to maximize (like accuracy) or minimize (like error)
    /// 
    /// For example, if training a model to predict house prices,
    /// the fitness might be the negative mean squared error (higher is better).
    /// 
    /// This property is important because:
    /// - It's the main measure of progress in optimization
    /// - It helps determine when the process has found a good solution
    /// - It can be used to compare different optimization runs
    /// </para>
    /// </remarks>
    public T Fitness
    {
        get { return _fitness; }
        set { _fitness = value; }
    }

    /// <summary>
    /// Gets or sets the result of overfitting detection for this iteration.
    /// </summary>
    /// <value>A FitDetectorResult&lt;T&gt; object containing information about potential overfitting.</value>
    /// <remarks>
    /// <para>
    /// This property contains the result of overfitting detection analysis for this iteration. Overfitting occurs when a 
    /// model performs well on training data but poorly on new, unseen data. The FitDetectorResult object typically includes 
    /// information about whether overfitting has been detected, the severity of the overfitting, and possibly recommendations 
    /// for addressing it. This information is crucial for implementing early stopping and other techniques to prevent 
    /// overfitting during the optimization process.
    /// </para>
    /// <para><b>For Beginners:</b> This contains information about whether the model is starting to overfit at this step.
    /// 
    /// The fit detection result:
    /// - Indicates whether overfitting has been detected
    /// - May include measures of how severe the overfitting is
    /// - Helps determine when to stop training
    /// 
    /// Overfitting happens when a model learns the training data too well,
    /// including its noise and peculiarities, making it perform poorly on new data.
    /// 
    /// This property is important because:
    /// - It helps implement "early stopping" to prevent overfitting
    /// - It provides insight into the model's generalization ability
    /// - It can guide decisions about regularization or model complexity
    /// </para>
    /// </remarks>
    public FitDetectorResult<T> FitDetectionResult { get; set; }

    /// <summary>
    /// Initializes a new instance of the OptimizationIterationInfo class with default values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor creates a new OptimizationIterationInfo instance with default values. It initializes the 
    /// FitDetectionResult to a new instance and sets the fitness to zero using the appropriate numeric operations for 
    /// type T. This constructor is useful when creating a new iteration info object before the actual fitness value is known.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new iteration info object with default values.
    /// 
    /// When using this constructor:
    /// - The iteration number starts at 0
    /// - The fitness is initialized to zero
    /// - A new, empty fit detection result is created
    /// 
    /// This constructor is typically used when:
    /// - Creating a new iteration info before values are known
    /// - Initializing a collection of iteration infos
    /// - Setting up the first iteration of an optimization process
    /// </para>
    /// </remarks>
    public OptimizationIterationInfo()
    {
        FitDetectionResult = new FitDetectorResult<T>();
        _fitness = MathHelper.GetNumericOperations<T>().Zero;
    }

    /// <summary>
    /// Initializes a new instance of the OptimizationIterationInfo class with a specified fitness value.
    /// </summary>
    /// <param name="fitness">The initial fitness value.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new OptimizationIterationInfo instance with a specified fitness value. It calls the 
    /// default constructor to initialize other properties and then sets the fitness to the provided value. This constructor 
    /// is useful when creating a new iteration info object when the fitness value is already known.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new iteration info object with a specific fitness value.
    /// 
    /// When using this constructor:
    /// - The iteration number starts at 0
    /// - The fitness is set to the value you provide
    /// - A new, empty fit detection result is created
    /// 
    /// This constructor is typically used when:
    /// - Creating an iteration info after calculating fitness
    /// - Recording the results of an optimization step
    /// - Initializing with a known starting fitness
    /// 
    /// For example: new OptimizationIterationInfo<double>(0.85) creates an
    /// iteration info with a fitness of 0.85.
    /// </para>
    /// </remarks>
    public OptimizationIterationInfo(T fitness) : this()
    {
        Fitness = fitness;
    }
}
