namespace AiDotNet.Enums;

/// <summary>
/// Specifies the dimensionality of input data for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Dimensionality refers to how many separate values are used to represent each data point.
/// Think of dimensions like coordinates - a 1D point needs just one number (like a position on a line),
/// a 2D point needs two numbers (like a position on a map), and a 3D point needs three numbers
/// (like a position in a room).
/// </para>
/// </remarks>
public enum InputType
{
    /// <summary>
    /// Represents input data with a single value per data point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> One-dimensional data means each data point is represented by a single number.
    /// 
    /// Examples:
    /// - A list of temperatures (each day has one temperature value)
    /// - A list of prices (each item has one price)
    /// - A list of ages (each person has one age)
    /// 
    /// Visualize it as points on a line or a simple list of numbers.
    /// In code, this is typically represented as a simple array or list: [1, 2, 3, 4, 5]
    /// </para>
    /// </remarks>
    OneDimensional,

    /// <summary>
    /// Represents input data with two values per data point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Two-dimensional data means each data point has two values associated with it.
    /// 
    /// Examples:
    /// - Coordinates on a map (latitude and longitude)
    /// - Height and weight measurements
    /// - Price and square footage of houses
    /// 
    /// Visualize it as points on a plane or a table with two columns.
    /// In code, this might be represented as pairs of values: [(1,2), (3,4), (5,6)]
    /// or as two parallel arrays: [1,3,5] and [2,4,6]
    /// </para>
    /// </remarks>
    TwoDimensional,

    /// <summary>
    /// Represents input data with three values per data point.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Three-dimensional data means each data point has three values associated with it.
    /// 
    /// Examples:
    /// - 3D coordinates in space (x, y, z)
    /// - RGB color values (red, green, blue)
    /// - Height, weight, and age of a person
    /// 
    /// Visualize it as points in 3D space or a table with three columns.
    /// In code, this might be represented as triplets: [(1,2,3), (4,5,6)]
    /// or as three parallel arrays: [1,4], [2,5], and [3,6]
    /// </para>
    /// </remarks>
    ThreeDimensional
}
