using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Logistic Regression, a statistical method used for binary
/// classification problems in machine learning.
/// </summary>
/// <remarks>
/// <para>
/// Logistic Regression is a supervised learning algorithm that predicts the probability of an
/// observation belonging to a certain class. Despite its name, it's used for classification rather
/// than regression. The algorithm applies the logistic function to a linear combination of features
/// to transform the output to a probability value between 0 and 1. The model parameters are typically
/// learned through an iterative optimization process like gradient descent, which aims to maximize
/// the likelihood of the observed data.
/// </para>
/// <para><b>For Beginners:</b> Logistic Regression is one of the most fundamental classification
/// algorithms in machine learning, used when you want to predict categories (like "yes/no" or "spam/not spam").
/// 
/// Despite having "regression" in its name, it's actually used for classification problems:
/// - It calculates the probability that something belongs to a particular category
/// - If the probability is above 50%, it predicts one category; otherwise, it predicts the other
/// 
/// Think of it like determining whether a student will pass or fail an exam:
/// - You gather information about study hours, attendance, and previous grades
/// - Logistic Regression learns how these factors contribute to passing probability
/// - When a new student comes along, you can predict their outcome based on these factors
/// 
/// This class allows you to configure how the algorithm learns these relationships from your data.
/// </para>
/// </remarks>
public class LogisticRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when fitting the model. The algorithm may terminate earlier if convergence is achieved based on
    /// the tolerance value. Setting this too low might prevent the algorithm from converging to an optimal
    /// solution, while setting it too high might waste computational resources if the algorithm has already
    /// effectively converged.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how long the algorithm will try to improve
    /// its predictions before giving up.
    /// 
    /// Think of it like practicing a skill:
    /// - Each "iteration" is like a practice session
    /// - The algorithm keeps practicing until it stops getting significantly better
    /// - This parameter sets a maximum number of practice sessions
    /// 
    /// The default value of 1000 is sufficient for many problems. You might need to increase it if:
    /// - You have a very large or complex dataset
    /// - Your model isn't reaching good performance and the training logs show it was stopped early
    /// 
    /// You might want to decrease it if:
    /// - Training is taking too long and you want faster results
    /// - You're doing initial experimentation and don't need perfect results
    /// 
    /// Note that the algorithm might stop before reaching this maximum if it determines that
    /// additional iterations won't significantly improve the model (based on the Tolerance setting).
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the learning rate that controls the step size in each iteration of the optimization.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate determines how large of a step the optimization algorithm takes in the direction
    /// of the gradient during each iteration. A higher learning rate means larger steps, which can lead
    /// to faster convergence but risks overshooting the optimal solution or causing instability. A lower
    /// learning rate means smaller steps, which provides more stability but may require more iterations
    /// to converge and risks getting stuck in local optima.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big of an adjustment the algorithm
    /// makes with each practice session.
    /// 
    /// Imagine you're turning a dial to tune a radio station:
    /// - A high learning rate (like 0.1) means making big turns of the dial 
    /// - A low learning rate (like 0.001) means making tiny, precise turns
    /// 
    /// The default value of 0.01 works well for many problems, providing a balance between:
    /// - Speed: Higher values help the model learn faster
    /// - Stability: Lower values help prevent the model from "overshooting" the best answer
    /// 
    /// If your model's performance is fluctuating wildly during training, try decreasing this value.
    /// If your model is learning too slowly, try increasing it.
    /// 
    /// Finding the right learning rate is often a matter of experimentation - it's like finding the
    /// right temperature for cooking: too high and things burn, too low and nothing cooks.
    /// </para>
    /// </remarks>
    /// <value>
    /// The step size of the solver <see cref="AiDotNet.Regression.LogisticRegression{T}.CreateGradientDescentSolver"/>
    /// builds. The default L-BFGS solver chooses its own step by line search, so this has no effect unless that solver
    /// is selected.
    /// </value>
    [OptionBehaviourExempt("Only the gradient descent solver LogisticRegression.CreateGradientDescentSolver builds uses it; the default L-BFGS solver chooses its step by line search.")]
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the optimization algorithm should stop.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.0001 (1e-4).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged.
    /// The algorithm will stop when the improvement in the objective function between consecutive
    /// iterations falls below this tolerance value. A smaller tolerance requires more precision in
    /// the solution, potentially leading to better model performance but requiring more iterations,
    /// while a larger tolerance allows for earlier termination but might result in a less optimal model.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much improvement is considered
    /// "good enough" to keep training.
    /// 
    /// Imagine you're trying to hit a target:
    /// - Each iteration gets you closer to the bullseye
    /// - Eventually, your improvements become very small
    /// - This setting decides when those improvements are small enough to stop
    /// 
    /// The default value of 0.0001 means:
    /// - If an iteration improves the model by less than 0.01% of its current performance
    /// - The algorithm will decide it's "good enough" and stop
    /// 
    /// You might want to decrease this value (like to 0.00001) if:
    /// - You need very precise results
    /// - You have enough computing resources for longer training
    /// 
    /// You might want to increase this value (like to 0.001) if:
    /// - You're more concerned with training speed than perfect accuracy
    /// - You're doing preliminary testing or exploration
    /// 
    /// This is like deciding when a drawing is "finished" - perfectionists will keep making tiny
    /// improvements, while pragmatists will stop once it looks good enough.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the optimizer that minimizes the logistic loss, or <c>null</c> for L-BFGS.
    /// </summary>
    /// <value>
    /// The solver, or <c>null</c> (the default) for the library's <c>LBFGSOptimizer</c>. JSON serialization excludes
    /// this runtime object; a deserialized model keeps its fitted coefficients either way.
    /// </value>
    /// <remarks>
    /// <para>
    /// Every solver minimizes the same objective: the mean log loss over the training rows plus the configured
    /// regularization penalty on the coefficients (the intercept is not penalized), on standardized features. Any
    /// <see cref="IFunctionOptimizer{T}"/> fits, so a caller can pass any of the library's gradient-based optimizers
    /// created for function minimization. <see cref="MaxIterations"/> and <see cref="Tolerance"/> (a bound on the
    /// largest gradient component) are handed to whichever solver runs.
    /// </para>
    /// <para>
    /// L-BFGS is the default because it is what scikit-learn's <c>LogisticRegression</c> uses: it converges in tens of
    /// iterations where fixed-step gradient descent needs thousands, to the same minimum. The model used to be fitted
    /// by gradient descent alone, at <see cref="LearningRate"/>;
    /// <see cref="AiDotNet.Regression.LogisticRegression{T}.CreateGradientDescentSolver"/> rebuilds that solver for a
    /// caller who wants it.
    /// </para>
    /// <para><b>For Beginners:</b> This is the method used to find the best coefficients. Leave it unset unless you
    /// have a reason to change it; the default is fast and reliable.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    public IFunctionOptimizer<T>? Solver { get; set; }
}
