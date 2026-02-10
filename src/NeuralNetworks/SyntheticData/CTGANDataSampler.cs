using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Handles conditional vector generation and training-by-sampling for CTGAN.
/// Implements the sampling strategy from the CTGAN paper that ensures all categories
/// are represented equally during training.
/// </summary>
/// <remarks>
/// <para>
/// The CTGAN paper introduces "training-by-sampling" to handle imbalanced categorical columns:
/// 1. Randomly pick a discrete/categorical column
/// 2. Randomly pick a category value from that column (with equal probability)
/// 3. Construct a conditional vector indicating the selected category
/// 4. Sample a real row that has the selected category value
///
/// This ensures that rare categories are sampled proportionally during training,
/// preventing the generator from ignoring minority classes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Real data often has imbalanced categories. For example, in a
/// "car color" column, 80% might be "white" and only 2% "yellow". Without special handling,
/// the generator would mostly learn to produce white cars.
///
/// The data sampler fixes this by:
/// 1. Picking a random categorical column
/// 2. Picking a random category (so "yellow" gets equal chance as "white")
/// 3. Finding a real row with that category to use as training data
/// 4. Creating a "conditional vector" that tells the generator what category to produce
///
/// This ensures the generator learns to produce all categories well.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CTGANDataSampler<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Random _random;

    // Per discrete/categorical column: list of row indices per category
    private readonly List<DiscreteColumnInfo> _discreteColumnInfos = new();

    // Indices of discrete/categorical columns in the original data
    private readonly List<int> _discreteColumnIndices = new();

    // Total width of the conditional vector
    private int _condVectorWidth;

    // Total number of rows in the training data (for random row selection when no categorical columns)
    private int _totalRows;

    /// <summary>
    /// Gets the width of the conditional vector.
    /// </summary>
    public int ConditionalVectorWidth => _condVectorWidth;

    /// <summary>
    /// Gets the number of discrete/categorical columns used for conditioning.
    /// </summary>
    public int NumDiscreteColumns => _discreteColumnIndices.Count;

    /// <summary>
    /// Initializes a new <see cref="CTGANDataSampler{T}"/>.
    /// </summary>
    /// <param name="random">Random number generator.</param>
    public CTGANDataSampler(Random random)
    {
        _random = random;
    }

    /// <summary>
    /// Builds the category-to-row-index tables from the training data.
    /// </summary>
    /// <param name="data">The original (untransformed) data matrix.</param>
    /// <param name="columns">Column metadata.</param>
    public void Fit(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _discreteColumnInfos.Clear();
        _discreteColumnIndices.Clear();
        _condVectorWidth = 0;
        _totalRows = data.Rows;

        for (int col = 0; col < columns.Count; col++)
        {
            var meta = columns[col];
            if (!meta.IsCategorical) continue;

            _discreteColumnIndices.Add(col);

            // Build row-index lists for each category
            int numCats = meta.NumCategories;
            var rowsByCategory = new List<int>[numCats];
            for (int c = 0; c < numCats; c++)
            {
                rowsByCategory[c] = new List<int>();
            }

            for (int row = 0; row < data.Rows; row++)
            {
                double val = NumOps.ToDouble(data[row, col]);
                int catIdx = (int)Math.Round(val);
                if (catIdx >= 0 && catIdx < numCats)
                {
                    rowsByCategory[catIdx].Add(row);
                }
            }

            _discreteColumnInfos.Add(new DiscreteColumnInfo(numCats, rowsByCategory));
            _condVectorWidth += numCats;
        }
    }

    /// <summary>
    /// Samples a conditional vector and a corresponding real row index for training.
    /// </summary>
    /// <returns>
    /// A tuple of (conditionalVector, rowIndex) where conditionalVector is the one-hot
    /// conditional vector and rowIndex is the index of a real row matching the condition.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method:
    /// 1. Picks a random categorical column
    /// 2. Picks a random category from that column (equally likely)
    /// 3. Finds a real data row that has that category
    /// 4. Returns the conditional vector (telling the generator what to produce)
    ///    and the row index (for training the discriminator with a matching real row)
    /// </para>
    /// </remarks>
    public (Vector<T> CondVector, int RowIndex) SampleConditionAndRow()
    {
        var condVector = new Vector<T>(_condVectorWidth);

        if (_discreteColumnInfos.Count == 0)
        {
            // No categorical columns; return zero vector and random index from whole dataset
            int randomRow = _totalRows > 0 ? _random.Next(_totalRows) : 0;
            return (condVector, randomRow);
        }

        // Step 1: Pick a random discrete column
        int colIdx = _random.Next(_discreteColumnInfos.Count);
        var colInfo = _discreteColumnInfos[colIdx];

        // Step 2: Pick a random category from that column
        int catIdx = _random.Next(colInfo.NumCategories);

        // If no rows have this category, pick a random one that does
        while (colInfo.RowsByCategory[catIdx].Count == 0 && colInfo.NumCategories > 1)
        {
            catIdx = _random.Next(colInfo.NumCategories);
        }

        // Step 3: Set the conditional vector
        int condOffset = 0;
        for (int c = 0; c < colIdx; c++)
        {
            condOffset += _discreteColumnInfos[c].NumCategories;
        }
        condVector[condOffset + catIdx] = NumOps.One;

        // Step 4: Sample a real row with this category value
        var rows = colInfo.RowsByCategory[catIdx];
        int rowIdx = rows.Count > 0 ? rows[_random.Next(rows.Count)] : 0;

        return (condVector, rowIdx);
    }

    /// <summary>
    /// Generates a conditional vector for a specific category in generation mode.
    /// </summary>
    /// <param name="discreteColIndex">Index into the discrete columns list (0-based).</param>
    /// <param name="categoryIndex">The category index to condition on.</param>
    /// <returns>A conditional vector with the specified condition set.</returns>
    public Vector<T> CreateConditionVector(int discreteColIndex, int categoryIndex)
    {
        var condVector = new Vector<T>(_condVectorWidth);

        if (discreteColIndex >= 0 && discreteColIndex < _discreteColumnInfos.Count)
        {
            int condOffset = 0;
            for (int c = 0; c < discreteColIndex; c++)
            {
                condOffset += _discreteColumnInfos[c].NumCategories;
            }

            if (categoryIndex >= 0 && categoryIndex < _discreteColumnInfos[discreteColIndex].NumCategories)
            {
                condVector[condOffset + categoryIndex] = NumOps.One;
            }
        }

        return condVector;
    }

    /// <summary>
    /// Generates a random conditional vector for unconditional generation (picks random category).
    /// </summary>
    /// <returns>A random conditional vector.</returns>
    public Vector<T> SampleRandomConditionVector()
    {
        var condVector = new Vector<T>(_condVectorWidth);

        if (_discreteColumnInfos.Count == 0)
        {
            return condVector;
        }

        // Pick random column and random category
        int colIdx = _random.Next(_discreteColumnInfos.Count);
        int catIdx = _random.Next(_discreteColumnInfos[colIdx].NumCategories);

        int condOffset = 0;
        for (int c = 0; c < colIdx; c++)
        {
            condOffset += _discreteColumnInfos[c].NumCategories;
        }
        condVector[condOffset + catIdx] = NumOps.One;

        return condVector;
    }

    #region Internal Types

    private sealed class DiscreteColumnInfo
    {
        public int NumCategories { get; }
        public List<int>[] RowsByCategory { get; }

        public DiscreteColumnInfo(int numCategories, List<int>[] rowsByCategory)
        {
            NumCategories = numCategories;
            RowsByCategory = rowsByCategory;
        }
    }

    #endregion
}
