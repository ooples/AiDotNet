namespace AiDotNet.Enums;

/// <summary>
/// Specifies how data is organized in matrices when working with arrays of data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A matrix is simply a rectangular grid of numbers arranged in rows and columns.
/// When working with data in programming, we need to specify how our data is organized.
/// 
/// Think of a spreadsheet:
/// - You can organize your data by putting similar features in columns (ColumnArrays)
/// - Or you can organize your data by putting each data point across a row (RowArrays)
/// 
/// For example, if you have data about people (height, weight, age):
/// 
/// ColumnArrays would look like:
/// - First array: [height1, height2, height3, ...]
/// - Second array: [weight1, weight2, weight3, ...]
/// - Third array: [age1, age2, age3, ...]
/// 
/// RowArrays would look like:
/// - First array: [height1, weight1, age1]
/// - Second array: [height2, weight2, age2]
/// - Third array: [height3, weight3, age3]
/// 
/// The choice of layout affects how you access and process your data, and different
/// algorithms may expect data in different layouts.
/// </para>
/// </remarks>
public enum MatrixLayout
{
    /// <summary>
    /// Data is organized by columns, where each array represents a feature or variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In ColumnArrays layout, each array contains all values for a single feature.
    /// 
    /// Example: If measuring temperature in different cities over days:
    /// - First array: [NYC_Day1, NYC_Day2, NYC_Day3, ...]
    /// - Second array: [LA_Day1, LA_Day2, LA_Day3, ...]
    /// - Third array: [Chicago_Day1, Chicago_Day2, Chicago_Day3, ...]
    /// 
    /// This layout is often preferred when:
    /// - You need to analyze one feature across all data points
    /// - You're adding or removing features (columns) frequently
    /// - Working with certain statistical operations that operate on columns
    /// </para>
    /// </remarks>
    ColumnArrays,

    /// <summary>
    /// Data is organized by rows, where each array represents a single data point with multiple features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In RowArrays layout, each array contains all features for a single data point.
    /// 
    /// Example: If measuring temperature in different cities over days:
    /// - First array: [NYC_Day1, LA_Day1, Chicago_Day1]
    /// - Second array: [NYC_Day2, LA_Day2, Chicago_Day2]
    /// - Third array: [NYC_Day3, LA_Day3, Chicago_Day3]
    /// 
    /// This layout is often preferred when:
    /// - You need to process one data point at a time
    /// - You're adding or removing data points (rows) frequently
    /// - Working with algorithms that process one sample at a time
    /// - Dealing with streaming data where each new data point has multiple features
    /// </para>
    /// </remarks>
    RowArrays
}
