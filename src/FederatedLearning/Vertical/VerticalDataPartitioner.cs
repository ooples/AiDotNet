using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Partitions features (columns) across parties for vertical federated learning simulation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In a real VFL deployment, each party already holds its own
/// features (the bank has financial data, the hospital has medical data). But for testing
/// and simulation, we need to take a single dataset and split its columns across simulated
/// parties. This class does that splitting.</para>
///
/// <para><b>Partitioning strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Sequential:</b> First N columns to party 1, next M columns to party 2, etc.</description></item>
/// <item><description><b>Interleaved:</b> Alternating columns (1,3,5... to party 1; 2,4,6... to party 2).</description></item>
/// <item><description><b>Random:</b> Randomly assign columns to parties.</description></item>
/// <item><description><b>Custom:</b> User specifies exactly which columns go to which party.</description></item>
/// </list>
///
/// <para><b>Reference:</b> VertiBench (ICLR 2024) recommends diverse feature distribution testing.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalDataPartitioner<T> : FederatedLearningComponentBase<T>
{
    /// <summary>
    /// Partitions features sequentially across the specified number of parties.
    /// </summary>
    /// <param name="totalFeatures">Total number of features (columns) in the dataset.</param>
    /// <param name="numberOfParties">Number of parties to split across.</param>
    /// <returns>A dictionary mapping party index to the list of column indices assigned to it.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have 10 columns and 2 parties, party 0 gets columns 0-4
    /// and party 1 gets columns 5-9.</para>
    /// </remarks>
    public static IReadOnlyDictionary<int, IReadOnlyList<int>> PartitionSequential(
        int totalFeatures, int numberOfParties)
    {
        if (totalFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalFeatures), "Total features must be positive.");
        }

        if (numberOfParties <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Number of parties must be positive.");
        }

        var partitions = new Dictionary<int, IReadOnlyList<int>>();
        int baseSize = totalFeatures / numberOfParties;
        int remainder = totalFeatures % numberOfParties;
        int offset = 0;

        for (int p = 0; p < numberOfParties; p++)
        {
            int size = baseSize + (p < remainder ? 1 : 0);
            var columns = new List<int>(size);
            for (int i = 0; i < size; i++)
            {
                columns.Add(offset + i);
            }

            partitions[p] = columns;
            offset += size;
        }

        return partitions;
    }

    /// <summary>
    /// Partitions features in an interleaved (round-robin) pattern across parties.
    /// </summary>
    /// <param name="totalFeatures">Total number of features.</param>
    /// <param name="numberOfParties">Number of parties.</param>
    /// <returns>A dictionary mapping party index to column indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have 10 columns and 2 parties, party 0 gets columns 0,2,4,6,8
    /// and party 1 gets columns 1,3,5,7,9. This ensures each party gets a mix of features.</para>
    /// </remarks>
    public static IReadOnlyDictionary<int, IReadOnlyList<int>> PartitionInterleaved(
        int totalFeatures, int numberOfParties)
    {
        if (totalFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalFeatures), "Total features must be positive.");
        }

        if (numberOfParties <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Number of parties must be positive.");
        }

        var partitions = new Dictionary<int, List<int>>();
        for (int p = 0; p < numberOfParties; p++)
        {
            partitions[p] = new List<int>();
        }

        for (int col = 0; col < totalFeatures; col++)
        {
            partitions[col % numberOfParties].Add(col);
        }

        var result = new Dictionary<int, IReadOnlyList<int>>();
        foreach (var kvp in partitions)
        {
            result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    /// <summary>
    /// Partitions features randomly across parties.
    /// </summary>
    /// <param name="totalFeatures">Total number of features.</param>
    /// <param name="numberOfParties">Number of parties.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>A dictionary mapping party index to column indices.</returns>
    public static IReadOnlyDictionary<int, IReadOnlyList<int>> PartitionRandom(
        int totalFeatures, int numberOfParties, int? seed = null)
    {
        if (totalFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalFeatures), "Total features must be positive.");
        }

        if (numberOfParties <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Number of parties must be positive.");
        }

        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Fisher-Yates shuffle of column indices
        var indices = new int[totalFeatures];
        for (int i = 0; i < totalFeatures; i++)
        {
            indices[i] = i;
        }

        for (int i = totalFeatures - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // Distribute shuffled indices to parties
        var partitions = new Dictionary<int, List<int>>();
        for (int p = 0; p < numberOfParties; p++)
        {
            partitions[p] = new List<int>();
        }

        for (int i = 0; i < totalFeatures; i++)
        {
            partitions[i % numberOfParties].Add(indices[i]);
        }

        // Sort each party's columns for consistent ordering
        var result = new Dictionary<int, IReadOnlyList<int>>();
        foreach (var kvp in partitions)
        {
            kvp.Value.Sort();
            result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    /// <summary>
    /// Creates a vertical partition from explicit column assignments.
    /// </summary>
    /// <param name="partyColumns">A dictionary mapping party index to their assigned column indices.</param>
    /// <returns>A validated partition dictionary.</returns>
    public static IReadOnlyDictionary<int, IReadOnlyList<int>> PartitionCustom(
        IDictionary<int, IReadOnlyList<int>> partyColumns)
    {
        if (partyColumns is null)
        {
            throw new ArgumentNullException(nameof(partyColumns));
        }

        if (partyColumns.Count == 0)
        {
            throw new ArgumentException("At least one party must be specified.", nameof(partyColumns));
        }

        var result = new Dictionary<int, IReadOnlyList<int>>();
        foreach (var kvp in partyColumns)
        {
            result[kvp.Key] = kvp.Value;
        }

        return result;
    }

    /// <summary>
    /// Extracts a party's feature columns from a full data tensor.
    /// </summary>
    /// <param name="fullData">The full data tensor with shape [samples, totalFeatures].</param>
    /// <param name="columnIndices">The column indices belonging to this party.</param>
    /// <returns>A tensor with shape [samples, partyFeatures] containing only this party's columns.</returns>
    public static Tensor<T> ExtractPartyFeatures(Tensor<T> fullData, IReadOnlyList<int> columnIndices)
    {
        if (fullData is null)
        {
            throw new ArgumentNullException(nameof(fullData));
        }

        if (columnIndices is null || columnIndices.Count == 0)
        {
            throw new ArgumentException("Column indices must not be empty.", nameof(columnIndices));
        }

        int samples = fullData.Shape[0];
        int totalCols = fullData.Rank > 1 ? fullData.Shape[1] : fullData.Shape[0];
        int partyCols = columnIndices.Count;

        var result = new Tensor<T>(new[] { samples, partyCols });
        for (int row = 0; row < samples; row++)
        {
            for (int c = 0; c < partyCols; c++)
            {
                int srcIdx = row * totalCols + columnIndices[c];
                int dstIdx = row * partyCols + c;
                result[dstIdx] = fullData[srcIdx];
            }
        }

        return result;
    }
}
