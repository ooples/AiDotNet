namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for kernel functions that measure similarity between data points in machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface defines a method that measures how similar two pieces of data are to each other.
/// 
/// Imagine you have two photographs and you want to determine how similar they are:
/// - You could compare them pixel by pixel, but that's not always the best approach
/// - Instead, you might want to compare certain features or patterns in the images
/// 
/// A kernel function is like a special measuring tool that:
/// - Takes two data points (like images, text documents, or sensor readings)
/// - Transforms them into a special mathematical space where similarities are easier to detect
/// - Returns a single number that represents how similar they are
/// 
/// The higher the number returned, the more similar the two data points are considered to be.
/// 
/// Common kernel functions include:
/// - Linear: Measures simple dot product similarity
/// - Polynomial: Good for capturing interactions between features
/// - Radial Basis Function (RBF): Measures how close points are in space
/// - Sigmoid: Inspired by neural networks
/// 
/// Kernel functions are especially important in:
/// - Support Vector Machines (SVMs)
/// - Kernel regression
/// - Clustering algorithms
/// - Any algorithm that needs to measure similarity between complex data points
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("KernelFunction")]
public interface IKernelFunction<T>
{
    /// <summary>
    /// Calculates the similarity between two vectors using this kernel function.
    /// </summary>
    /// <param name="x1">The first vector to compare.</param>
    /// <param name="x2">The second vector to compare.</param>
    /// <returns>A scalar value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method measures how similar two pieces of data are to each other.
    /// 
    /// The parameters:
    /// - x1: The first data point (represented as a list of numbers)
    /// - x2: The second data point (represented as a list of numbers)
    /// 
    /// What this method does:
    /// 1. Takes your two data points (x1 and x2)
    /// 2. Applies a mathematical formula to measure their similarity
    /// 3. Returns a single number representing that similarity
    /// 
    /// For example:
    /// - A result of 1.0 might mean "exactly the same"
    /// - A result of 0.0 might mean "completely different"
    /// - Values in between indicate partial similarity
    /// 
    /// Different kernel functions will measure similarity in different ways:
    /// - Some focus on the direction of the vectors
    /// - Some focus on the distance between points
    /// - Some apply complex transformations before measuring similarity
    /// 
    /// The beauty of kernel functions is that they let algorithms work with complex data
    /// (like images or text) by focusing only on how similar items are to each other,
    /// rather than having to understand all the details of the data.
    /// </remarks>
    T Calculate(Vector<T> x1, Vector<T> x2);
}
