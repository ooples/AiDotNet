namespace AiDotNet.Statistics;

/// <summary>
/// Computes and stores the quartiles (Q1, Q2, Q3) of a numeric dataset.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Quartile class calculates the three standard quartiles of a dataset: the first quartile (Q1, 25th percentile),
/// the second quartile (Q2, 50th percentile or median), and the third quartile (Q3, 75th percentile).
/// These quartiles divide the dataset into four equal parts and provide insights into the distribution of the data.
/// </para>
/// <para><b>For Beginners:</b> Quartiles divide your data into four equal parts, giving you a quick way to understand
/// the distribution of your values.
/// 
/// Think of quartiles like dividing a line of people by height into four equal groups:
/// - Q1 (First Quartile): The height where 25% of people are shorter and 75% are taller
/// - Q2 (Second Quartile): The height where 50% are shorter and 50% are taller (this is also called the median)
/// - Q3 (Third Quartile): The height where 75% are shorter and 25% are taller
/// 
/// Quartiles help you understand:
/// - Where the "middle half" of your data lies (between Q1 and Q3)
/// - If your data is skewed (if the distance from Q1 to Q2 is different from Q2 to Q3)
/// - What values might be considered outliers (typically those below Q1-1.5×IQR or above Q3+1.5×IQR)
/// 
/// For example, if test scores have Q1=70, Q2=80, and Q3=90, you know half the scores are between 70 and 90,
/// and the median score is 80.
/// </para>
/// </remarks>
public class Quartile<T>
{
    /// <summary>
    /// The numeric operations appropriate for the generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static field holds a reference to the appropriate numeric operations implementation for the
    /// generic type T, allowing the quartile calculations to perform mathematical operations
    /// regardless of whether T is float, double, or another numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This is a helper that allows the code to work with different number types.
    /// 
    /// Since this class uses a generic type T (which could be float, double, etc.):
    /// - We need a way to perform math operations (+, -, *, /) on these values
    /// - NumOps provides the right methods for whatever numeric type is being used
    /// 
    /// This is a technical detail that ensures the calculations work correctly regardless of what type of
    /// numbers you're analyzing.
    /// </para>
    /// </remarks>
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The sorted vector of data values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds a copy of the input data, sorted in ascending order. Sorting the data is a prerequisite
    /// for calculating quartiles, as quartiles are defined based on the position of values in the ordered dataset.
    /// </para>
    /// <para><b>For Beginners:</b> This stores your data arranged from smallest to largest.
    /// 
    /// Before calculating quartiles:
    /// - The data must be put in order from smallest to largest value
    /// - This sorted copy is stored here for use in the calculations
    /// 
    /// For example, if your original data is [7, 3, 9, 5], it would be stored as [3, 5, 7, 9].
    /// </para>
    /// </remarks>
    private readonly Vector<T> _sortedData;

    /// <summary>
    /// Gets the first quartile (Q1, 25th percentile) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The first quartile (Q1) is the value below which 25% of the observations in the dataset fall.
    /// It represents the median of the lower half of the data. Various methods exist for calculating Q1,
    /// and this implementation uses a standard interpolation method.
    /// </para>
    /// <para><b>For Beginners:</b> Q1 is the value where 25% of your data falls below it.
    /// 
    /// If you divide your sorted data into four equal parts, Q1 is the value at the boundary
    /// of the first and second parts.
    /// 
    /// For example, if you have test scores [60, 70, 75, 80, 85, 90, 95]:
    /// - Q1 would be approximately 70
    /// - This means about 25% of scores are 70 or lower
    /// 
    /// Q1 helps you identify the lower portion of your data range.
    /// </para>
    /// </remarks>
    public T Q1 { get; }

    /// <summary>
    /// Gets the second quartile (Q2, 50th percentile, median) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The second quartile (Q2) is the median of the dataset - the value that divides the dataset into two equal halves.
    /// For datasets with an odd number of elements, it is the middle value; for datasets with an even number of elements,
    /// it is the average of the two middle values.
    /// </para>
    /// <para><b>For Beginners:</b> Q2 is the middle value of your data (also called the median).
    /// 
    /// It's the value where:
    /// - 50% of data points are below it
    /// - 50% of data points are above it
    /// 
    /// For example, if you have test scores [60, 70, 75, 80, 85, 90, 95]:
    /// - Q2 would be 80
    /// - This means half the scores are 80 or lower, and half are 80 or higher
    /// 
    /// The median (Q2) is often a better measure of "central tendency" than the average (mean) when your data
    /// contains outliers.
    /// </para>
    /// </remarks>
    public T Q2 { get; }

    /// <summary>
    /// Gets the third quartile (Q3, 75th percentile) of the dataset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The third quartile (Q3) is the value below which 75% of the observations in the dataset fall.
    /// It represents the median of the upper half of the data. Various methods exist for calculating Q3,
    /// and this implementation uses a standard interpolation method.
    /// </para>
    /// <para><b>For Beginners:</b> Q3 is the value where 75% of your data falls below it.
    /// 
    /// If you divide your sorted data into four equal parts, Q3 is the value at the boundary
    /// of the third and fourth parts.
    /// 
    /// For example, if you have test scores [60, 70, 75, 80, 85, 90, 95]:
    /// - Q3 would be approximately 90
    /// - This means about 75% of scores are 90 or lower
    /// 
    /// Q3 helps you identify the upper portion of your data range.
    /// </para>
    /// </remarks>
    public T Q3 { get; }

    /// <summary>
    /// Initializes a new instance of the Quartile class with the provided dataset.
    /// </summary>
    /// <param name="data">The vector of data values to analyze.</param>
    /// <remarks>
    /// <para>
    /// This constructor calculates all three quartiles (Q1, Q2, Q3) for the provided dataset.
    /// It first sorts the data in ascending order, then uses the StatisticsHelper class to calculate
    /// each quartile as the 25th, 50th, and 75th percentiles respectively.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new quartile object with your data.
    /// 
    /// When you create a Quartile object:
    /// - It takes your set of numbers as input
    /// - It sorts them from smallest to largest
    /// - It calculates all three quartiles (Q1, Q2, Q3)
    /// - It stores these results for you to access later
    /// 
    /// For example, if you provide [85, 60, 95, 70, 80, 75, 90], it will:
    /// 1. Sort them to [60, 70, 75, 80, 85, 90, 95]
    /// 2. Calculate Q1 ˜ 70 (25th percentile)
    /// 3. Calculate Q2 = 80 (50th percentile)
    /// 4. Calculate Q3 ˜ 90 (75th percentile)
    /// </para>
    /// </remarks>
    public Quartile(Vector<T> data)
    {
        _sortedData = new Vector<T>([.. data.OrderBy(x => x)]);
        Q1 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.25));
        Q2 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.5));
        Q3 = StatisticsHelper<T>.CalculateQuantile(_sortedData, NumOps.FromDouble(0.75));
    }
}
