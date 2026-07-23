namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Bayesian regression algorithms.
/// </summary>
/// <typeparam name="T">The data type used by the regression model.</typeparam>
/// <remarks>
/// <para>
/// Bayesian regression is a statistical approach that applies Bayes' theorem to regression analysis.
/// Unlike traditional regression which produces point estimates, Bayesian regression provides probability
/// distributions for the model parameters, allowing for uncertainty quantification in predictions.
/// </para>
/// <para><b>For Beginners:</b> Bayesian regression is like traditional regression (finding relationships between 
/// variables) but with an added layer of confidence information. Instead of just saying "we think y = 2x + 3," 
/// Bayesian regression says "we think y = 2x + 3, and we're 90% confident that the 2 is between 1.8 and 2.2." 
/// This approach is especially useful when you have limited data or want to incorporate prior knowledge about 
/// what the relationship might be. It helps you understand not just what the relationship is, but also how certain 
/// you can be about that relationship.</para>
/// </remarks>
public class BayesianRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the alpha parameter, which controls the precision of the prior distribution.
    /// </summary>
    /// <value>The alpha value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// In Bayesian statistics, alpha is a hyperparameter for the prior distribution that represents
    /// your initial belief about the model parameters before seeing any data.
    /// </para>
    /// <para><b>For Beginners:</b> Think of alpha as how strongly you believe in your initial guess before 
    /// seeing any data. A higher value (greater than 1.0) means you're more confident in your prior beliefs, 
    /// while a lower value means you're more willing to let the data speak for itself. The default value of 1.0 
    /// represents a balanced approach that doesn't strongly favor either prior beliefs or the observed data. If you're 
    /// unsure, it's best to start with this default value.</para>
    /// </remarks>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the beta parameter, which controls the precision of the likelihood function.
    /// </summary>
    /// <value>The beta value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Beta represents the precision (inverse of variance) of the noise in the observed data.
    /// Higher values indicate lower noise levels and more confidence in the observed data.
    /// </para>
    /// <para><b>For Beginners:</b> Beta represents how much you trust your data measurements. A higher beta value 
    /// (greater than 1.0) means you believe your data has less noise or random error, so the algorithm will fit more 
    /// closely to the observed points. A lower value suggests your data might be noisy, so the algorithm will create 
    /// a smoother fit that doesn't necessarily pass through every data point. The default value of 1.0 provides a 
    /// balanced approach for most datasets.</para>
    /// </remarks>
    public double Beta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the type of kernel function to use in the regression model.
    /// </summary>
    /// <value>The kernel type, defaulting to Linear.</value>
    /// <remarks>
    /// <para>
    /// The kernel function determines how the algorithm measures similarity between data points,
    /// which affects how it generalizes from observed data points to make predictions.
    /// </para>
    /// <para><b>For Beginners:</b> The kernel type determines what kind of relationship the model can find between 
    /// your variables. The default Linear kernel looks for straight-line relationships (like y = mx + b). Other options include:
    /// <list type="bullet">
    ///   <item>RBF (Radial Basis Function): Can find complex, curvy relationships</item>
    ///   <item>Polynomial: Can find relationships with curves like y = x² + 2x + 1</item>
    ///   <item>Sigmoid: Useful for S-shaped relationships</item>
    /// </list>
    /// If you're not sure which to use, start with Linear for simplicity, then try RBF if you suspect the relationship 
    /// might be more complex than a straight line.</para>
    /// </remarks>
    public KernelType KernelType { get; set; } = KernelType.Linear;

    /// <summary>
    /// Gets or sets the gamma parameter used in RBF, Polynomial, and Sigmoid kernels.
    /// </summary>
    /// <value>The gamma value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Gamma defines how far the influence of a single training example reaches, with low values meaning 'far'
    /// and high values meaning 'close'. It can be seen as the inverse of the radius of influence of samples
    /// selected by the model as support vectors.
    /// </para>
    /// <para><b>For Beginners:</b> Gamma controls how much influence each data point has on its surroundings. 
    /// Think of it like the radius of influence around each point. A high gamma value means each point only influences 
    /// predictions very close to it (creating a more complex, potentially wiggly model). A low gamma means each point 
    /// influences a wider area (creating a smoother model). This parameter only matters when using non-linear kernels 
    /// like RBF, Polynomial, or Sigmoid. The default value of 1.0 is a good starting point, but you might need to 
    /// adjust it based on your specific data.</para>
    /// </remarks>
    public double Gamma { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the independent term (coef0) in Polynomial and Sigmoid kernels.
    /// </summary>
    /// <value>The coef0 value, defaulting to 0.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter is only significant for Polynomial and Sigmoid kernels. For the Polynomial kernel,
    /// it controls the degree of homogeneity. For the Sigmoid kernel, it defines the vertical shift.
    /// </para>
    /// <para><b>For Beginners:</b> This is an additional parameter that affects how Polynomial and Sigmoid kernels work. 
    /// For Polynomial kernels, it adds a constant term to the equation (like the "+c" in "y = x² + x + c"). 
    /// For Sigmoid kernels, it shifts the S-curve up or down. The default value of 0.0 works well in most cases. 
    /// You can safely ignore this parameter if you're using the Linear or RBF kernel types.</para>
    /// </remarks>
    public double Coef0 { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the degree of the Polynomial kernel.
    /// </summary>
    /// <value>The polynomial degree, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// This parameter is only used when the KernelType is set to Polynomial. It determines the highest
    /// power in the polynomial equation.
    /// </para>
    /// <para><b>For Beginners:</b> If you're using the Polynomial kernel, this sets the highest power in your equation. 
    /// For example, a value of 3 (the default) allows the model to find relationships up to cubic terms (like y = ax³ + bx² + cx + d). 
    /// A higher degree can capture more complex relationships but risks overfitting to your training data. A value of 1 would 
    /// be equivalent to a linear model, while 2 would allow quadratic relationships. This parameter is ignored if you're not 
    /// using the Polynomial kernel type.</para>
    /// </remarks>
    public int PolynomialDegree { get; set; } = 3;

    /// <summary>
    /// Gets or sets the matrix decomposition method used for solving linear systems.
    /// </summary>
    /// <value>The decomposition type, defaulting to Lu.</value>
    /// <remarks>
    /// <para>
    /// Matrix decomposition is a numerical method used to solve the linear algebra problems that arise
    /// during Bayesian regression. Different decomposition methods offer trade-offs between
    /// numerical stability, computational efficiency, and applicability to different types of matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This is a technical setting about how the math is solved behind the scenes. 
    /// The default LU decomposition works well for most problems. Other options include:
    /// <list type="bullet">
    ///   <item>Cholesky: Faster but requires positive definite matrices</item>
    ///   <item>QR: More stable for ill-conditioned problems</item>
    ///   <item>SVD: Most stable but slowest option</item>
    /// </list>
    /// Unless you're experiencing numerical issues or working with very large datasets, you can leave this at the default value.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;

    /// <summary>
    /// Gets or sets the gamma parameter for the Laplacian kernel.
    /// </summary>
    /// <value>The Laplacian gamma value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter is only used when the KernelType is set to Laplacian. It controls the
    /// width of the exponential function in the Laplacian kernel.
    /// </para>
    /// <para><b>For Beginners:</b> This is similar to the regular Gamma parameter, but specifically for the Laplacian kernel 
    /// (if you choose to use it). The Laplacian kernel is less commonly used than RBF but can be better for certain types of data. 
    /// A higher value creates a more complex model that fits training data more closely, while a lower value creates a smoother model. 
    /// The default value of 1.0 is a good starting point if you decide to use the Laplacian kernel. This parameter is ignored if 
    /// you're using any other kernel type.</para>
    /// </remarks>
    public double LaplacianGamma { get; set; } = 1.0;
}
