namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for nonlinear regression models, which capture complex, nonlinear relationships
/// between input features and output variables using kernel functions and iterative optimization.
/// </summary>
/// <remarks>
/// <para>
/// Nonlinear regression extends beyond the capabilities of linear models by allowing for curved or complex
/// relationships between variables. This class encapsulates the parameters that control how nonlinear
/// regression models are fitted to data, including convergence criteria, kernel selection, and kernel
/// hyperparameters. These models are particularly valuable when data exhibits patterns that cannot be
/// adequately represented by straight lines or hyperplanes, such as exponential growth, sinusoidal cycles,
/// or interactions between variables. The kernel approach transforms the input space to a higher-dimensional
/// feature space where complex relationships become more manageable.
/// </para>
/// <para><b>For Beginners:</b> Nonlinear regression is a technique for finding patterns in data that don't follow straight lines.
/// 
/// Imagine trying to predict house prices:
/// - Linear regression might assume each additional square foot adds a fixed amount to the price
/// - Nonlinear regression can capture more realistic patterns, like:
///   - Diminishing returns (each extra square foot adds less value as houses get very large)
///   - Threshold effects (prices jump significantly once houses reach certain sizes)
///   - Interactions (extra bathrooms add more value in larger houses than in smaller ones)
/// 
/// This class provides settings that control:
/// - How precisely the model should fit the data
/// - How many attempts it should make to find the best fit
/// - What mathematical approach (kernel) to use for modeling curved relationships
/// 
/// Nonlinear regression is powerful because it can discover complex patterns that simpler
/// models miss, but it requires more careful configuration to work effectively. These options
/// help you control that balance between flexibility and reliability.
/// </para>
/// </remarks>
public class NonLinearRegressionOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations allowed for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter sets an upper limit on how many iterations the optimization algorithm will perform
    /// when fitting the nonlinear regression model. Each iteration adjusts the model parameters to better
    /// fit the training data. The algorithm may terminate earlier if convergence is achieved based on
    /// the tolerance value. The appropriate number of iterations depends on the complexity of the nonlinear
    /// relationship, the amount of data, and the initial parameter values. Very complex models or noisy
    /// data may require more iterations to reach convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many attempts the algorithm
    /// makes to improve its solution before giving up.
    /// 
    /// The default value of 1000 means:
    /// - The algorithm will try up to 1000 times to find the best solution
    /// - It might stop earlier if it finds a good enough answer (based on the Tolerance setting)
    /// 
    /// Think of it like trying to find the bottom of a valley while blindfolded:
    /// - Each iteration is taking a step based on how steep the ground feels
    /// - You need enough steps to reach the bottom, but not so many that you waste effort
    /// 
    /// You might want more iterations (like 5000) if:
    /// - Your data has complex patterns that are hard to fit
    /// - You notice the model is still improving significantly at 1000 iterations
    /// - You have more computing resources and want the best possible result
    /// 
    /// You might want fewer iterations (like 500) if:
    /// - You need faster training times
    /// - Your data has relatively simple nonlinear patterns
    /// - The model converges quickly anyway
    /// 
    /// In practice, it's good to start with the default and adjust based on whether the model
    /// has converged properly by the end of training.
    /// </para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the optimization algorithm should stop.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.001 (1e-3).</value>
    /// <remarks>
    /// <para>
    /// This parameter defines the threshold for determining when the optimization has converged. The algorithm
    /// will stop when the improvement in the objective function (typically the error or loss) between consecutive
    /// iterations falls below this tolerance value. A smaller tolerance requires more precision in the parameter
    /// estimates, potentially leading to better model performance but requiring more iterations. A larger tolerance
    /// allows for earlier termination but might result in less optimal parameter estimates. The appropriate value
    /// depends on the scale of your target variable and the required precision of your predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much improvement between
    /// steps is considered "good enough" to stop the algorithm.
    /// 
    /// The default value of 0.001 (one-thousandth) means:
    /// - If an iteration improves the solution by less than 0.001
    /// - The algorithm will decide it's "close enough" and stop
    /// 
    /// Continuing with our valley analogy:
    /// - If each step only gets you a tiny bit lower (less than the tolerance)
    /// - You decide you're basically at the bottom and stop walking
    /// 
    /// You might want a smaller value (like 0.0001) if:
    /// - You need very precise predictions
    /// - You have plenty of computing resources
    /// - Small differences in accuracy matter in your application
    /// 
    /// You might want a larger value (like 0.01) if:
    /// - You want faster training
    /// - You're doing exploratory analysis
    /// - Ultra-precise parameters aren't necessary for your needs
    /// 
    /// Finding the right tolerance is about balancing precision with computational efficiency -
    /// too strict and training takes forever, too loose and you might get suboptimal results.
    /// </para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the type of kernel function to use for transforming the input space.
    /// </summary>
    /// <value>The kernel type, defaulting to RBF (Radial Basis Function).</value>
    /// <remarks>
    /// <para>
    /// The kernel function is central to nonlinear regression as it implicitly maps input data into a
    /// higher-dimensional space where complex, nonlinear relationships can be modeled more effectively.
    /// Different kernel types have distinct properties and are suitable for different types of data patterns.
    /// The Radial Basis Function (RBF) kernel is the default as it works well for many problems by measuring
    /// similarity based on distance in the feature space. Other options include the linear kernel (for nearly
    /// linear relationships), polynomial kernel (for polynomial trends), and sigmoid kernel (for certain
    /// classification-like regression tasks).
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical approach
    /// used to model curved and complex relationships in your data.
    /// 
    /// The default RBF (Radial Basis Function) kernel:
    /// - Works well for many types of smooth, nonlinear patterns
    /// - Measures how similar data points are based on their distance
    /// - Is versatile but requires tuning the Gamma parameter
    /// 
    /// Think of kernels like different lenses for viewing your data:
    /// - RBF: General-purpose zoom lens that works for most scenes
    /// - Linear: Simple lens for straight-line relationships
    /// - Polynomial: Special lens for capturing wave-like or curved patterns
    /// - Sigmoid: Lens that creates S-curved patterns
    /// 
    /// You might want to try different kernels if:
    /// - RBF isn't capturing your data patterns well
    /// - You have prior knowledge about the shape of relationships in your data
    /// - You want to experiment with different approaches
    /// 
    /// For example:
    /// - For data with polynomial trends (like cubic growth), try the Polynomial kernel
    /// - For data with nearly linear relationships, try the Linear kernel
    /// - For data with complex, localized patterns, stick with RBF
    /// 
    /// The kernel choice is one of the most important decisions for nonlinear regression
    /// as it fundamentally determines what kinds of patterns your model can learn.
    /// </para>
    /// </remarks>
    public KernelType KernelType { get; set; } = KernelType.RBF;

    /// <summary>
    /// Gets or sets the gamma parameter that controls the influence range in RBF, Polynomial, and Sigmoid kernels.
    /// </summary>
    /// <value>The gamma parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The gamma parameter is a critical hyperparameter that controls the "reach" or influence of each training
    /// example in kernel-based models. For the RBF kernel, gamma determines the width of the Gaussian function,
    /// with smaller values creating a broader influence (smoother decision boundary) and larger values creating
    /// a narrower influence (more complex, potentially overfitted boundary). In polynomial and sigmoid kernels,
    /// gamma acts as a scaling factor for the dot product of the input vectors. The optimal value depends on
    /// the scale of your features and the complexity of the underlying relationship.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how "local" or "global" the influence
    /// of each data point should be in your model.
    /// 
    /// The default value of 1.0 provides a moderate influence range:
    /// - For RBF kernel: How quickly similarity decreases with distance
    /// - For Polynomial kernel: How strongly to scale the dot product
    /// - For Sigmoid kernel: Affects the steepness of the S-curve
    /// 
    /// Think of gamma like the focus setting on a camera:
    /// - Low gamma (0.1): Wide focus, sees general trends but might miss details
    /// - High gamma (10): Narrow focus, captures local details but might miss the big picture
    /// 
    /// You might want a lower gamma value if:
    /// - Your model seems to be overfitting (works well on training data but poorly on new data)
    /// - You want to capture broader, smoother trends
    /// - Your data points are sparsely distributed
    /// 
    /// You might want a higher gamma value if:
    /// - Your model seems to be underfitting (not capturing the complexity of your data)
    /// - You need to model very localized patterns
    /// - You have lots of training data to support a more complex model
    /// 
    /// Finding the right gamma often requires experimentation, possibly using techniques
    /// like cross-validation to identify the value that generalizes best to new data.
    /// </para>
    /// </remarks>
    public double Gamma { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the coef0 parameter used in Polynomial and Sigmoid kernels.
    /// </summary>
    /// <value>The coef0 parameter, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// The coef0 parameter acts as an intercept term in Polynomial and Sigmoid kernels, allowing for more
    /// flexible modeling of certain nonlinear relationships. In the polynomial kernel, it influences the
    /// balance between higher-order and lower-order terms in the polynomial expansion. For the sigmoid kernel,
    /// it affects the horizontal shift of the sigmoid function. This parameter has no effect when using the
    /// RBF or linear kernels. The appropriate value depends on the specific characteristics of your data
    /// and the type of nonlinearity you aim to capture.
    /// </para>
    /// <para><b>For Beginners:</b> This setting provides an additional tuning parameter
    /// for Polynomial and Sigmoid kernels, affecting their shape and behavior.
    /// 
    /// The default value of 0.0:
    /// - For Polynomial kernel: Determines the mix of different polynomial degrees
    /// - For Sigmoid kernel: Controls the horizontal shift of the S-curve
    /// - Has no effect when using RBF or Linear kernels
    /// 
    /// Think of coef0 like an adjustment dial that:
    /// - Changes the balance between simpler and more complex terms in polynomial models
    /// - Shifts where the steepest part of the S-curve occurs in sigmoid models
    /// 
    /// You might want to change this value if:
    /// - You're specifically using Polynomial or Sigmoid kernels
    /// - The default value isn't capturing the patterns in your data well
    /// - You have prior knowledge about the polynomial terms or sigmoid shift that would work best
    /// 
    /// For example:
    /// - In polynomial kernels, a higher coef0 gives more weight to higher-order terms
    /// - In sigmoid kernels, it shifts where the transition from low to high values occurs
    /// 
    /// This is an advanced parameter that often requires experimentation alongside other
    /// kernel parameters to find the optimal configuration for your data.
    /// </para>
    /// </remarks>
    public double Coef0 { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the degree of the polynomial when using the Polynomial kernel type.
    /// </summary>
    /// <value>The polynomial degree, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the highest power used in the polynomial kernel function, directly controlling
    /// the complexity and flexibility of the model. A higher degree allows the model to capture more complex,
    /// higher-order relationships but increases the risk of overfitting and computational complexity. This
    /// parameter is only relevant when KernelType is set to Polynomial. A degree of 1 makes the polynomial
    /// kernel equivalent to a linear kernel, while degrees of 2 or 3 are common choices for capturing
    /// quadratic or cubic relationships, respectively.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the highest power used when
    /// creating polynomial features, but only applies when using the Polynomial kernel.
    /// 
    /// The default value of 3 means:
    /// - The polynomial kernel will include terms up to x³ (cubic)
    /// - This can model S-shaped curves and more complex patterns than linear models
    /// 
    /// Think of the polynomial degree like choosing how flexible your curve can be:
    /// - Degree 1: Can only fit straight lines (equivalent to linear regression)
    /// - Degree 2: Can fit parabolas (U-shaped curves)
    /// - Degree 3: Can fit cubic functions (S-shaped curves)
    /// - Higher degrees: Can fit increasingly wiggly, complex curves
    /// 
    /// You might want a higher degree (like 4 or 5) if:
    /// - Your data has very complex, oscillating patterns
    /// - You have enough data to support a more complex model
    /// - Lower degrees aren't capturing important trends
    /// 
    /// You might want a lower degree (like 2) if:
    /// - You want a simpler, more interpretable model
    /// - You're concerned about overfitting
    /// - You have limited training data
    /// 
    /// Remember: This setting has no effect unless you're using the Polynomial kernel type.
    /// Higher degrees give more flexibility but increase the risk of overfitting and
    /// computational complexity.
    /// </para>
    /// </remarks>
    public int PolynomialDegree { get; set; } = 3;
}
