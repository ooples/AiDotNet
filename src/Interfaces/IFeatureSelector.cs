namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a fitted transformer that keeps a subset of its input's columns: a feature selector.
/// </summary>
/// <remarks>
/// <para>
/// A feature selector learns, during <c>Fit</c>, which input columns to keep, and its <c>Transform</c> returns only
/// those columns. This interface exposes that choice so it can be inspected or reused, in the same way as
/// scikit-learn's <c>SelectorMixin.get_support</c>.
/// </para>
/// <para><b>For Beginners:</b> If your data has 100 columns (features) and only 10 of them help predict the target,
/// a feature selector finds those 10. <see cref="SelectedIndices"/> tells you which column numbers it kept, and
/// <see cref="GetSupportMask"/> gives you one true/false flag per original column.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public interface IFeatureSelector<T>
{
    /// <summary>
    /// Gets the zero-based indices of the input columns the selector kept, or <c>null</c> before it has been fitted.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After fitting, this lists the column numbers that survive the selection, so
    /// <c>[0, 4, 7]</c> means the transformed data holds the original columns 0, 4 and 7.
    /// </para>
    /// </remarks>
    int[]? SelectedIndices { get; }

    /// <summary>
    /// Returns one flag per input column of the data the selector was fitted on: <c>true</c> where the column is kept.
    /// </summary>
    /// <returns>A new array, as long as the fitted input's column count, that the caller may modify.</returns>
    /// <exception cref="System.InvalidOperationException">The selector has not been fitted.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> For data with 5 columns where columns 1 and 3 were kept, this returns
    /// <c>[false, true, false, true, false]</c>.
    /// </para>
    /// </remarks>
    bool[] GetSupportMask();
}
