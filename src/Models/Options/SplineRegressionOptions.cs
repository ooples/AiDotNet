namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for spline regression models, which fit piecewise polynomial functions
/// to data for flexible nonlinear modeling.
/// </summary>
/// <remarks>
/// <para>
/// Spline regression is a flexible nonlinear regression technique that fits piecewise polynomial functions 
/// (splines) to data. Unlike simple polynomial regression, which uses a single polynomial for the entire 
/// data range, spline regression divides the data into segments and fits separate polynomials to each segment, 
/// with constraints ensuring smoothness at the connection points (knots). This approach provides greater 
/// flexibility in modeling complex nonlinear relationships while avoiding the oscillatory behavior often 
/// seen with high-degree polynomials. This class provides configuration options for spline regression, 
/// including the number of knots (which determines the number of segments), the degree of the polynomial 
/// functions, and the matrix decomposition method used for solving the regression equations.
/// </para>
/// <para><b>For Beginners:</b> Spline regression helps model complex relationships in your data using connected smooth curves.
/// 
/// When simple linear regression isn't flexible enough:
/// - Linear regression forces a straight line through your data
/// - Polynomial regression uses a single curved line, but can behave poorly
/// - Spline regression uses multiple connected curves for more flexibility
/// 
/// Think of spline regression like drawing with multiple connected curve segments:
/// - Each segment is a polynomial curve (like a parabola)
/// - The segments connect smoothly at points called "knots"
/// - This creates a flexible curve that can adapt to different patterns in different regions
/// 
/// Benefits of spline regression:
/// - More flexible than simple lines or polynomials
/// - Avoids the wild oscillations that can happen with high-degree polynomials
/// - Can capture complex relationships while still being relatively simple to interpret
/// 
/// This class lets you configure exactly how the spline regression model will be constructed.
/// </para>
/// </remarks>
public class SplineRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the number of knots (internal breakpoints) in the spline function.
    /// </summary>
    /// <value>A positive integer, defaulting to 3.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the number of knots or breakpoints used in the spline function. Knots are the points 
    /// where the different polynomial segments connect. The total number of segments in the spline will be one more 
    /// than the number of knots. More knots allow the spline to capture more complex patterns in the data but increase 
    /// the risk of overfitting and the computational complexity. The default value of 3 provides a moderate level of 
    /// flexibility suitable for many applications. For simpler relationships, fewer knots might be appropriate, while 
    /// for more complex relationships, more knots might be needed. The optimal number of knots can be determined 
    /// through cross-validation or by comparing model fit statistics like AIC (Akaike Information Criterion) or 
    /// BIC (Bayesian Information Criterion).
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many segments the spline curve will have.
    /// 
    /// The number of knots determines:
    /// - How many places the curve can change its behavior
    /// - How flexible the overall fit will be
    /// 
    /// The default value of 3 means:
    /// - The data range will be divided into 4 segments (number of knots + 1)
    /// - Each segment will have its own polynomial curve
    /// - The curves will connect smoothly at the knot points
    /// 
    /// Think of it like this:
    /// - More knots (e.g., 5 or 10): More flexible curve that can capture complex patterns
    /// - Fewer knots (e.g., 1 or 2): Simpler curve that's less likely to overfit
    /// 
    /// When to adjust this value:
    /// - Increase it when your data shows complex patterns that change multiple times
    /// - Decrease it when you want a simpler model or have limited data
    /// 
    /// For example, if modeling temperature throughout a day, 3 knots might allow the curve
    /// to capture morning warming, midday plateau, and evening cooling patterns.
    /// </para>
    /// </remarks>
    public int NumberOfKnots { get; set; } = 3;

    /// <summary>
    /// Gets or sets the degree of the polynomial functions used in each segment of the spline.
    /// </summary>
    /// <value>A positive integer, defaulting to 3 (cubic splines).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the degree of the polynomial functions used in each segment of the spline. The degree 
    /// determines the maximum power of the independent variable in the polynomial. A degree of 1 creates linear 
    /// segments, a degree of 2 creates quadratic segments, and a degree of 3 (the default) creates cubic segments. 
    /// Higher-degree polynomials can capture more complex patterns within each segment but increase the risk of 
    /// overfitting and numerical instability. Cubic splines (degree 3) are the most commonly used in practice 
    /// because they provide a good balance between flexibility and stability, and they ensure continuity of the 
    /// first and second derivatives at the knots, resulting in visually smooth curves. For most applications, 
    /// cubic splines are sufficient, and increasing the number of knots is generally preferred over increasing 
    /// the polynomial degree when more flexibility is needed.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how curved each segment of the spline can be.
    /// 
    /// The degree determines:
    /// - The complexity of each individual segment between knots
    /// - How many times the curve can change direction within a segment
    /// 
    /// The default value of 3 means:
    /// - Cubic splines (3rd-degree polynomials) are used
    /// - Each segment can have up to two changes in direction (like an S-curve)
    /// - The connections between segments will be smooth, with matching slopes and curvatures
    /// 
    /// Common degree values:
    /// - 1: Linear splines (straight line segments, only smooth at knots)
    /// - 2: Quadratic splines (parabolic segments, smooth with matching slopes at knots)
    /// - 3: Cubic splines (the most common, smooth with matching slopes and curvatures)
    /// 
    /// When to adjust this value:
    /// - Decrease to 1 or 2 for simpler, more interpretable models
    /// - Keep at 3 (default) for most applications
    /// - Rarely increase above 3, as adding more knots is usually better than increasing degree
    /// 
    /// For example, cubic splines (degree=3) are commonly used in computer graphics for
    /// smooth curve drawing and in data analysis for flexible trend fitting.
    /// </para>
    /// </remarks>
    public int Degree { get; set; } = 3;

    /// <summary>
    /// Gets or sets the matrix decomposition method used to solve the regression equations.
    /// </summary>
    /// <value>A value from the MatrixDecompositionType enumeration, defaulting to MatrixDecompositionType.Cholesky.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the matrix decomposition method used to solve the system of linear equations that 
    /// arises when fitting the spline regression model. Different decomposition methods have different characteristics 
    /// in terms of numerical stability, computational efficiency, and applicability to specific types of matrices. 
    /// The Cholesky decomposition (the default) is efficient and numerically stable for the positive definite matrices 
    /// that typically arise in regression problems. Other options might include QR decomposition (more stable for 
    /// ill-conditioned matrices but less efficient), SVD (Singular Value Decomposition, most stable but least efficient), 
    /// or LU decomposition (efficient but less stable). The choice of decomposition method can affect the accuracy and 
    /// efficiency of the regression, particularly for large datasets or ill-conditioned problems.
    /// </para>
    /// <para><b>For Beginners:</b> This setting selects the mathematical method used to solve the regression equations.
    /// 
    /// Fitting a spline regression involves solving a system of equations:
    /// - Different mathematical methods (decompositions) can solve these equations
    /// - Each method has trade-offs in terms of speed, accuracy, and numerical stability
    /// 
    /// The default Cholesky decomposition:
    /// - Is very efficient for regression problems
    /// - Works well when the data is well-behaved
    /// - Is the standard choice for most regression applications
    /// 
    /// Other common decomposition types include:
    /// - QR: More stable for problematic data, but slower
    /// - SVD: Most stable for difficult cases, but significantly slower
    /// - LU: Fast but less stable for regression problems
    /// 
    /// When to adjust this value:
    /// - Keep the default (Cholesky) for most applications
    /// - Consider QR or SVD if you encounter numerical stability issues
    ///   (like errors about singular or ill-conditioned matrices)
    /// 
    /// This is an advanced setting that most users won't need to change unless they
    /// encounter specific numerical problems with the default method.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;
}
