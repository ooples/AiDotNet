namespace AiDotNet.Enums;

/// <summary>
/// Specifies different kernel functions used in machine learning algorithms like Support Vector Machines (SVMs).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A kernel is a special mathematical function that helps machine learning algorithms 
/// work with complex data. Think of kernels as "similarity measures" between data points.
/// 
/// Imagine you have data that can't be easily separated by a straight line. Kernels help by 
/// transforming your data into a form where patterns become more obvious - like lifting a 
/// 2D drawing into 3D space where you can see separations more clearly.
/// 
/// Kernels are commonly used in:
/// - Support Vector Machines (SVMs)
/// - Kernel regression
/// - Principal Component Analysis (PCA)
/// - Clustering algorithms
/// 
/// Different kernels work better for different types of data and problems. Choosing the right 
/// kernel can significantly improve your model's performance.
/// </para>
/// </remarks>
public enum KernelType
{
    /// <summary>
    /// The simplest kernel function that represents the standard dot product in the input space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Linear kernel is the simplest kernel - it doesn't actually transform your data.
    /// It works by measuring how similar two data points are by multiplying their features together.
    /// 
    /// Think of it as:
    /// - The most basic and fastest kernel
    /// - Like measuring similarity by how much two vectors point in the same direction
    /// - No extra parameters to tune
    /// - Works in the original data space without transformations
    /// 
    /// Best used when:
    /// - Your data is already linearly separable (can be divided by a straight line/plane)
    /// - You have a large number of features compared to samples
    /// - You want a simple, interpretable model
    /// - You need fast computation and low memory usage
    /// - You're working with text classification or high-dimensional data
    /// </para>
    /// </remarks>
    Linear,

    /// <summary>
    /// Radial Basis Function kernel that measures similarity based on distance in a high-dimensional space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The RBF (Radial Basis Function) kernel, also known as the Gaussian kernel, 
    /// transforms data based on how far apart points are from each other.
    /// 
    /// Think of it as:
    /// - Creating a "bubble of influence" around each data point
    /// - Points that are close together are considered very similar
    /// - Points far apart have little influence on each other
    /// - Maps your data into an infinite-dimensional space
    /// 
    /// Best used when:
    /// - Your data isn't linearly separable
    /// - You don't have prior knowledge about the data structure
    /// - You need a powerful, flexible kernel
    /// - You have a reasonable amount of training data
    /// - You're willing to tune parameters (particularly gamma)
    /// 
    /// The RBF kernel is often the first kernel to try when you're not sure which to use.
    /// </para>
    /// </remarks>
    RBF,

    /// <summary>
    /// A flexible kernel that raises the dot product of features to a specified power.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Polynomial kernel measures similarity by raising the dot product 
    /// of two data points to a certain power (degree).
    /// 
    /// Think of it as:
    /// - Looking for more complex patterns than just straight lines
    /// - Able to find curved boundaries between classes
    /// - More flexible than Linear but less complex than RBF
    /// - Creating new features that are combinations of original features
    /// 
    /// Best used when:
    /// - Your data has non-linear relationships
    /// - You want to capture interactions between features
    /// - Working with image processing tasks
    /// - You need a balance between the simplicity of Linear and flexibility of RBF
    /// - You can tune the degree parameter effectively
    /// 
    /// The degree parameter controls the flexibility - higher degrees can model more complex boundaries
    /// but risk overfitting.
    /// </para>
    /// </remarks>
    Polynomial,

    /// <summary>
    /// A kernel function based on the hyperbolic tangent, similar to neural network activation functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Sigmoid kernel is inspired by neural networks and uses the same 
    /// S-shaped function (hyperbolic tangent) that's common in neural network layers.
    /// 
    /// Think of it as:
    /// - Similar to having a simple neural network
    /// - Creates an S-shaped decision boundary
    /// - Can model some non-linear relationships
    /// - Has connections to neural network theory
    /// 
    /// Best used when:
    /// - You're working with problems that might be suited to neural networks
    /// - Your data has specific types of non-linear relationships
    /// - You want to try an alternative to RBF and Polynomial
    /// - You're experimenting with different kernel types
    /// 
    /// The Sigmoid kernel is less commonly used than RBF or Linear but can be effective for 
    /// specific types of problems.
    /// </para>
    /// </remarks>
    Sigmoid,

    /// <summary>
    /// A kernel function that uses the negative exponential of the L1 distance between points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Laplacian kernel is similar to RBF but uses a different way to 
    /// measure distance between points (Manhattan distance instead of Euclidean distance).
    /// 
    /// Think of it as:
    /// - Like RBF but more tolerant of outliers
    /// - Measuring distance as if you can only travel along grid lines
    /// - Less sensitive to changes in parameters
    /// - Good for certain types of sparse data
    /// 
    /// Best used when:
    /// - Your data contains outliers
    /// - You're working with features that naturally use Manhattan distance
    ///   (like city block distances or certain types of molecular data)
    /// - RBF isn't giving good results
    /// - You're working with histogram features or sparse data
    /// 
    /// The Laplacian kernel can be particularly effective for computer vision and text analysis tasks.
    /// </para>
    /// </remarks>
    Laplacian,

    /// <summary>
    /// Indicates that the kernel matrix is precomputed rather than being computed on-the-fly.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you use a Precomputed kernel, you calculate the similarity between
    /// all pairs of data points yourself before training the model.
    ///
    /// Think of it as:
    /// - You provide a matrix of pre-calculated similarities
    /// - Useful when you have a custom similarity measure
    /// - Allows using domain-specific kernels not built into the library
    /// - Must compute kernel values between test and training data at prediction time
    ///
    /// Best used when:
    /// - You need a custom kernel not available in the library
    /// - You have pre-computed kernel values from another source
    /// - You want to use specialized domain kernels (string kernels, graph kernels, etc.)
    /// - You're working with structured data that requires special similarity measures
    /// </para>
    /// </remarks>
    Precomputed
}
