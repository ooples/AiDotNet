namespace AiDotNet.Interfaces;

/// <summary>
/// Defines functionality for creating vector objects from arrays of values.
/// </summary>
/// <remarks>
/// A vector in machine learning is a one-dimensional array of values that represents
/// a collection of features or a single data point.
/// 
/// For Beginners: Think of a vector as a simple list of numbers arranged in a single column.
/// In machine learning, vectors are used to represent:
/// 
/// - A single data point with multiple features (e.g., a person's age, height, and weight)
/// - A collection of values for a single feature across multiple samples (e.g., ages of 100 different people)
/// - The output of a prediction (e.g., predicted house prices for 50 different houses)
/// 
/// Vectors are fundamental building blocks in machine learning because they allow us to
/// work with multiple values at once using mathematical operations.
/// </remarks>
/// <typeparam name="T">The type of elements stored in the vector. Must be a reference type.</typeparam>
public interface IVector<T> where T : class
{
    /// <summary>
    /// Creates a column vector from an array of values.
    /// </summary>
    /// <remarks>
    /// This method converts a standard array into a specialized column vector object
    /// that can be used in machine learning operations.
    /// 
    /// For Beginners: This method takes your regular list of values (an array) and transforms
    /// it into a special "column vector" format that the machine learning algorithms can work with.
    /// 
    /// For example, if you have an array of test scores [85, 90, 78, 92], this method will
    /// convert it into a column vector that looks like:
    /// 
    /// | 85 |
    /// | 90 |
    /// | 78 |
    /// | 92 |
    /// 
    /// This format makes it easier to perform mathematical operations needed for machine learning,
    /// such as calculating averages, finding patterns, or making predictions.
    /// </remarks>
    /// <param name="values">The array of values to convert into a column vector.</param>
    /// <returns>A new column vector containing the provided values.</returns>
    public ColumnVector<T> BuildVector(T[] values);
}