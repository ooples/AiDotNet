namespace AiDotNet.Models.Inputs;

/// <summary>
/// Represents the input data required for calculating basic statistics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This internal class encapsulates the data needed to calculate basic statistical measures.
/// It provides a container for the vector of values that will be analyzed by the BasicStats class.
/// </para>
/// <para><b>For Beginners:</b> This class is like a data container that holds the numbers
/// you want to analyze. It ensures your data is properly formatted before statistical
/// calculations are performed on it.
/// </para>
/// </remarks>
internal class BasicStatsInputs<T>
{
    /// <summary>
    /// Gets or sets the vector of values to be analyzed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property contains the actual data points that will be used for statistical calculations.
    /// It defaults to an empty vector if no values are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This is the collection of numbers that your statistics
    /// will be calculated from. For example, if you want to find the average height of students,
    /// this would be the list of all student heights.
    /// </para>
    /// </remarks>
    public Vector<T> Values { get; set; } = Vector<T>.Empty();

    /// <summary>
    /// Initializes a new instance of the BasicStatsInputs class with an empty vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This default constructor creates an instance with an empty vector of values.
    /// It's useful when you need to create an instance before you have the actual data.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an empty container that you can fill with
    /// numbers later. It's like preparing an empty basket before you go shopping.
    /// </para>
    /// </remarks>
    public BasicStatsInputs()
    {
        // Default constructor creates an empty vector
    }

    /// <summary>
    /// Initializes a new instance of the BasicStatsInputs class with the specified values.
    /// </summary>
    /// <param name="values">The vector of values to analyze.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an instance with the provided vector of values.
    /// If null is provided, it defaults to an empty vector to prevent null reference exceptions.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a container pre-filled with your numbers.
    /// It also has a safety check to prevent errors if you accidentally provide no data.
    /// </para>
    /// </remarks>
    public BasicStatsInputs(Vector<T> values)
    {
        Values = values ?? Vector<T>.Empty();
    }

    /// <summary>
    /// Gets a value indicating whether this instance contains any values to analyze.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property returns true if the Values vector is empty or has zero length,
    /// indicating that there is no data to analyze.
    /// </para>
    /// <para><b>For Beginners:</b> This is a quick way to check if your data container
    /// has any numbers in it. It's like checking if your shopping basket is empty
    /// before proceeding to checkout.
    /// </para>
    /// </remarks>
    public bool IsEmpty => Values.IsEmpty || Values.Length == 0;
}