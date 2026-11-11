using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// The fitted encoding of one categorical column: its distinct category values (sorted ascending) and the
/// encoded value of each category.
/// </summary>
/// <typeparam name="T">The numeric type of the data.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Category <c>Categories[i]</c> is replaced by <c>Values[i]</c> when the column is
/// transformed. For example a count encoder stores how often each category appeared, and a target encoder
/// stores each category's (smoothed) target mean.</para>
/// </remarks>
public sealed class CategoryValueMap<T>
{
    /// <summary>Creates a map from sorted distinct categories and their encoded values.</summary>
    /// <param name="categories">The distinct category values, sorted ascending.</param>
    /// <param name="values">The encoded value of each category (same length as <paramref name="categories"/>).</param>
    public CategoryValueMap(Vector<T> categories, Vector<T> values)
    {
        if (categories.Length != values.Length)
            throw new ArgumentException("Each category needs exactly one encoded value.", nameof(values));

        Categories = categories;
        Values = values;
        SortedCategories = Tensor<T>.FromVector(categories);
        ValueTensor = Tensor<T>.FromVector(values);
    }

    internal CategoryValueMap(Tensor<T> sortedCategories, Tensor<T> values)
    {
        SortedCategories = sortedCategories;
        ValueTensor = values;
        Categories = sortedCategories.ToVector();
        Values = values.ToVector();
    }

    /// <summary>Gets the distinct category values, sorted ascending.</summary>
    public Vector<T> Categories { get; }

    /// <summary>Gets the encoded value of each category.</summary>
    public Vector<T> Values { get; }

    /// <summary>Gets the number of categories.</summary>
    public int Count => Categories.Length;

    internal Tensor<T> SortedCategories { get; }

    internal Tensor<T> ValueTensor { get; }

    /// <summary>
    /// Encodes <paramref name="column"/>: each known category becomes its value and every other entry takes
    /// <paramref name="fallback"/>; <paramref name="found"/> is a 0/1 mask of the known entries.
    /// </summary>
    internal Tensor<T> Map(Tensor<T> column, Tensor<T> fallback, out Tensor<T> found)
    {
        var position = PreprocessingTensorOps.LookupSorted(SortedCategories, column, out found);
        if (SortedCategories.Length == 0 || column.Length == 0)
            return fallback;

        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        return engine.TensorWhere(found, engine.TensorIndexSelect(ValueTensor, position, 0), fallback);
    }

    /// <summary>
    /// Encodes the given columns of <paramref name="data"/> through their maps (other columns pass through).
    /// Unknown categories, and every entry of a column without a map, take <paramref name="unknownValue"/> for
    /// that column, or throw <see cref="ArgumentException"/> when <paramref name="throwOnUnknown"/> is set.
    /// </summary>
    internal static Matrix<T> EncodeColumns(
        Matrix<T> data,
        ICollection<int> columnsToEncode,
        IReadOnlyDictionary<int, CategoryValueMap<T>> maps,
        Func<int, T> unknownValue,
        bool throwOnUnknown)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var raw = PreprocessingTensorOps.Column(data, j);
            if (!columnsToEncode.Contains(j))
            {
                columns.Add(raw);
                continue;
            }

            var fallback = PreprocessingTensorOps.Filled(new[] { numRows }, unknownValue(j));
            Tensor<T> found;
            Tensor<T> encoded;
            if (maps.TryGetValue(j, out var map))
            {
                encoded = map.Map(raw, fallback, out found);
            }
            else
            {
                found = new Tensor<T>(new[] { numRows });
                encoded = fallback;
            }

            if (throwOnUnknown)
            {
                var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                if (missing.Length > 0)
                    throw new ArgumentException($"Unknown category value: {raw.GetFlat(missing[0])} in column {j}");
            }

            columns.Add(encoded);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Fits a map for one column: the sorted distinct categories with, per category, the row count and the
    /// sum of <paramref name="target"/>; <paramref name="encode"/> turns those (counts, sums) into the encoded values.
    /// </summary>
    internal static CategoryValueMap<T> FromTargetStatistics(
        Tensor<T> column, Tensor<T> target, Func<Tensor<T>, Tensor<T>, Tensor<T>> encode)
    {
        var categories = PreprocessingTensorOps.SortedUnique(column);
        var position = PreprocessingTensorOps.LookupSorted(categories, column, out _);
        var counts = PreprocessingTensorOps.CategoryCounts<T>(position, categories.Length);
        var sums = PreprocessingTensorOps.CategorySums(position, target, categories.Length);
        return new CategoryValueMap<T>(categories, encode(counts, sums));
    }
}
