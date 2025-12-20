namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Locally Weighted Regression, a non-parametric method
/// that creates a model by fitting simple models to localized subsets of data.
/// </summary>
/// <remarks>
/// <para>
/// Locally Weighted Regression (LWR) is a memory-based technique that performs a regression
/// around a point of interest using only training data that are "local" to that point.
/// Unlike global methods that fit a single model to the entire dataset, LWR fits a separate
/// simple model for each query point by using nearby training examples weighted by their distance.
/// This allows the method to capture complex, non-linear relationships in the data without
/// specifying a global functional form.
/// </para>
/// <para><b>For Beginners:</b> Locally Weighted Regression is like asking your neighbors for advice,
/// but giving more importance to those who live closest to you.
/// 
/// Imagine you're trying to estimate house prices:
/// - Traditional regression creates one formula for the entire city
/// - Locally Weighted Regression works differently - when estimating the price of a specific house,
///   it looks primarily at similar nearby houses
/// - Houses very similar to yours get a high "weight" (strong influence)
/// - Houses quite different from yours get a low "weight" (weak influence)
/// - This helps capture neighborhood-specific patterns that might get lost in a city-wide formula
/// 
/// This approach is particularly useful when different regions of your data behave differently,
/// as it can adapt to local patterns rather than forcing a single model on everything.
/// This class allows you to configure how the "neighborhood" is defined and how the local
/// models are calculated.
/// </para>
/// </remarks>
public class LocallyWeightedRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the bandwidth parameter that controls the size of the "neighborhood" 
    /// used in locally weighted regression.
    /// </summary>
    /// <value>The bandwidth value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The bandwidth parameter determines how quickly the influence of training examples
    /// decreases as their distance from the query point increases. It acts as a smoothing parameter:
    /// larger values create smoother functions that average over more training examples, while
    /// smaller values create more flexible functions that can fit local variations but may be more
    /// sensitive to noise. The optimal value depends on the specific dataset and problem.
    /// </para>
    /// <para><b>For Beginners:</b> The bandwidth controls how large your "neighborhood" is
    /// when making predictions.
    /// 
    /// Think of it like defining your "local community":
    /// - A small bandwidth (like 0.1) means only very similar examples strongly influence the prediction,
    ///   creating a "tight-knit community" effect
    /// - A large bandwidth (like 10.0) means even somewhat different examples have influence,
    ///   creating a "broad community" effect
    /// 
    /// The default value of 1.0 provides a reasonable balance for many problems:
    /// - Too small: Your model might become too "choppy" and sensitive to noise
    /// - Too large: Your model might become too "smoothed out" and miss important local patterns
    /// 
    /// If your predictions seem too sensitive to small changes in input, try increasing this value.
    /// If your predictions seem too generalized and miss obvious patterns, try decreasing it.
    /// </para>
    /// </remarks>
    public double Bandwidth { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the matrix decomposition method used to solve the weighted least squares
    /// problem at each query point.
    /// </summary>
    /// <value>The matrix decomposition type, defaulting to MatrixDecompositionType.Cholesky.</value>
    /// <remarks>
    /// <para>
    /// When fitting a local model at each query point, the algorithm must solve a weighted
    /// least squares problem, which involves matrix operations. Different decomposition methods
    /// offer trade-offs between numerical stability, computational efficiency, and accuracy. 
    /// Cholesky decomposition is the default as it is generally efficient for positive definite
    /// matrices, but SVD (Singular Value Decomposition) might be more stable for ill-conditioned
    /// problems, while QR decomposition offers a balance between the two.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the mathematical method used to
    /// solve the equations when fitting local models.
    /// 
    /// Think of it like choosing a tool for a job:
    /// - Cholesky (the default): Like using a standard screwdriver - efficient for most regular tasks
    /// - SVD: Like using a specialized tool - more reliable for difficult situations but takes longer
    /// - QR: Like a middle-ground option - more versatile than Cholesky but faster than SVD
    /// 
    /// For most problems, the default Cholesky method works well and is efficient. You might
    /// consider changing to SVD if:
    /// - Your model produces warnings about numerical stability
    /// - Your data has highly correlated features (multicollinearity)
    /// - You're working with very small bandwidth values
    /// 
    /// This is a somewhat advanced setting that many beginners won't need to adjust.
    /// </para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Cholesky;

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) mode for JIT compilation support.
    /// </summary>
    /// <value><c>true</c> to enable soft mode; <c>false</c> (default) for traditional LWR behavior.</value>
    /// <remarks>
    /// <para>
    /// When enabled, LocallyWeightedRegression uses a differentiable approximation that embeds
    /// all training data as constants in the computation graph and computes attention-weighted
    /// predictions using the softmax of negative squared distances.
    /// </para>
    /// <para>
    /// Formula: weights = softmax(-||input - xTrain[i]||² / bandwidth)
    ///          output = Σ weights[i] * yTrain[i]
    /// </para>
    /// <para><b>For Beginners:</b> Soft mode allows this model to be JIT compiled for faster inference.
    ///
    /// Traditional LWR solves a new weighted least squares problem for each prediction, which
    /// cannot be represented as a static computation graph. Soft mode uses a simplified approach:
    /// - Compute distances from the query point to all training examples
    /// - Convert distances to weights using softmax (similar to attention mechanisms)
    /// - Return the weighted average of training targets
    ///
    /// This approximation:
    /// - Enables JIT compilation for faster predictions
    /// - Gives similar results for smooth data
    /// - May be less accurate than traditional LWR for complex local patterns
    /// </para>
    /// </remarks>
    public bool UseSoftMode { get; set; } = false;
}
