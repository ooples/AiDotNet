using System.Diagnostics;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Provides benchmarking utilities for evaluating VFL implementations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When developing VFL systems, it's important to test with controlled
/// datasets to verify that the system works correctly and to measure performance. This class
/// generates synthetic benchmark datasets and runs standardized evaluation suites.</para>
///
/// <para><b>Benchmark scenarios:</b></para>
/// <list type="bullet">
/// <item><description><b>Balanced parties:</b> All parties have similar-sized feature sets.</description></item>
/// <item><description><b>Unbalanced parties:</b> One party has many more features than others.</description></item>
/// <item><description><b>High overlap:</b> Most entities are shared across all parties.</description></item>
/// <item><description><b>Low overlap:</b> Few entities are shared (tests missing feature handling).</description></item>
/// <item><description><b>Noisy IDs:</b> Entity IDs have typos/inconsistencies (tests fuzzy matching).</description></item>
/// </list>
///
/// <para><b>Reference:</b> VertiBench (ICLR 2024): "Feature distribution diversity matters for
/// fair and accurate VFL evaluation."</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalFederatedBenchmark<T> : FederatedLearningComponentBase<T>
{
    /// <summary>
    /// Generates a synthetic vertically-partitioned dataset for benchmarking.
    /// </summary>
    /// <param name="totalEntities">Total number of entities in the dataset.</param>
    /// <param name="totalFeatures">Total number of features across all parties.</param>
    /// <param name="numberOfParties">Number of parties to split features across.</param>
    /// <param name="overlapRatio">Fraction of entities shared across all parties (0.0 to 1.0).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>A <see cref="VflBenchmarkDataset{T}"/> ready for use with <see cref="VerticalFederatedTrainer{T}"/>.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates fake data for testing your VFL implementation.
    /// You specify how many entities, features, and parties you want, plus how much overlap
    /// there is. The labels are generated from a simple linear function of the features.</para>
    /// </remarks>
    public static VflBenchmarkDataset<T> GenerateDataset(
        int totalEntities = 1000,
        int totalFeatures = 20,
        int numberOfParties = 2,
        double overlapRatio = 0.8,
        int? seed = null)
    {
        if (totalEntities <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalEntities));
        }

        if (totalFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalFeatures));
        }

        if (numberOfParties <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties), "Need at least 2 parties.");
        }

        if (overlapRatio < 0.0 || overlapRatio > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(overlapRatio));
        }

        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Generate full dataset
        var allEntityIds = new string[totalEntities];
        var fullFeatures = new Tensor<T>(new[] { totalEntities, totalFeatures });

        for (int i = 0; i < totalEntities; i++)
        {
            allEntityIds[i] = $"entity_{i:D6}";
            for (int f = 0; f < totalFeatures; f++)
            {
                fullFeatures[i * totalFeatures + f] = NumOps.FromDouble(random.NextDouble() * 2.0 - 1.0);
            }
        }

        // Generate labels from a linear function with noise
        var allLabels = new Tensor<T>(new[] { totalEntities });
        for (int i = 0; i < totalEntities; i++)
        {
            double labelVal = 0.0;
            for (int f = 0; f < totalFeatures; f++)
            {
                double weight = (f % 2 == 0 ? 1.0 : -1.0) / totalFeatures;
                labelVal += weight * NumOps.ToDouble(fullFeatures[i * totalFeatures + f]);
            }

            labelVal += random.NextDouble() * 0.1 - 0.05; // Small noise
            allLabels[i] = NumOps.FromDouble(labelVal);
        }

        // Partition features across parties
        var featurePartition = VerticalDataPartitioner<T>.PartitionSequential(totalFeatures, numberOfParties);

        // Determine which entities each party has
        int sharedCount = (int)(totalEntities * overlapRatio);
        int uniquePerParty = (totalEntities - sharedCount) / numberOfParties;

        var partyDatasets = new List<VflPartyDataset<T>>();

        for (int p = 0; p < numberOfParties; p++)
        {
            var entityIndices = new List<int>();

            // Shared entities (first sharedCount entities)
            for (int i = 0; i < sharedCount; i++)
            {
                entityIndices.Add(i);
            }

            // Unique entities for this party
            int uniqueStart = sharedCount + p * uniquePerParty;
            int uniqueEnd = Math.Min(uniqueStart + uniquePerParty, totalEntities);
            for (int i = uniqueStart; i < uniqueEnd; i++)
            {
                entityIndices.Add(i);
            }

            // Extract party features
            var partyColumns = featurePartition[p];
            int partyFeatures = partyColumns.Count;
            var partyData = new Tensor<T>(new[] { entityIndices.Count, partyFeatures });

            var partyEntityIds = new string[entityIndices.Count];
            for (int row = 0; row < entityIndices.Count; row++)
            {
                int globalRow = entityIndices[row];
                partyEntityIds[row] = allEntityIds[globalRow];
                for (int c = 0; c < partyFeatures; c++)
                {
                    int srcCol = partyColumns[c];
                    partyData[row * partyFeatures + c] = fullFeatures[globalRow * totalFeatures + srcCol];
                }
            }

            // Label holder is the last party
            Tensor<T>? partyLabels = null;
            if (p == numberOfParties - 1)
            {
                partyLabels = new Tensor<T>(new[] { entityIndices.Count });
                for (int row = 0; row < entityIndices.Count; row++)
                {
                    partyLabels[row] = allLabels[entityIndices[row]];
                }
            }

            partyDatasets.Add(new VflPartyDataset<T>
            {
                PartyId = $"party_{p}",
                Features = partyData,
                EntityIds = partyEntityIds,
                Labels = partyLabels,
                IsLabelHolder = p == numberOfParties - 1,
                FeatureColumnIndices = partyColumns
            });
        }

        return new VflBenchmarkDataset<T>
        {
            Parties = partyDatasets,
            TotalEntities = totalEntities,
            TotalFeatures = totalFeatures,
            OverlapRatio = overlapRatio,
            SharedEntityCount = sharedCount
        };
    }

    /// <summary>
    /// Runs a standardized benchmark suite on a VFL implementation.
    /// </summary>
    /// <param name="options">VFL training options to benchmark.</param>
    /// <param name="dataset">The benchmark dataset to use.</param>
    /// <returns>A <see cref="VflBenchmarkResult"/> with timing and accuracy metrics.</returns>
    public static VflBenchmarkResult RunBenchmark(
        VerticalFederatedLearningOptions options,
        VflBenchmarkDataset<T> dataset)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (dataset is null)
        {
            throw new ArgumentNullException(nameof(dataset));
        }

        var stopwatch = Stopwatch.StartNew();

        // Create trainer and register parties
        var trainer = new VerticalFederatedTrainer<T>(options);

        foreach (var partyData in dataset.Parties)
        {
            IVerticalParty<T> party;
            if (partyData.IsLabelHolder && partyData.Labels is not null)
            {
                party = new VerticalPartyLabelHolder<T>(
                    partyData.PartyId, partyData.Features, partyData.Labels,
                    partyData.EntityIds, options.SplitModel.EmbeddingDimension, options.RandomSeed);
            }
            else
            {
                party = new VerticalPartyClient<T>(
                    partyData.PartyId, partyData.Features, partyData.EntityIds,
                    options.SplitModel.EmbeddingDimension, 0, options.RandomSeed);
            }

            trainer.RegisterParty(party);
        }

        // Run training
        var trainingResult = trainer.Train();

        stopwatch.Stop();

        return new VflBenchmarkResult
        {
            TotalTime = stopwatch.Elapsed,
            AlignmentTime = trainingResult.AlignmentSummary?.AlignmentTime ?? TimeSpan.Zero,
            TrainingTime = trainingResult.TotalTrainingTime,
            FinalLoss = trainingResult.FinalLoss,
            EpochsCompleted = trainingResult.EpochsCompleted,
            AlignedEntities = trainingResult.AlignmentSummary?.AlignedEntityCount ?? 0,
            TotalEntities = dataset.TotalEntities,
            NumberOfParties = dataset.Parties.Count
        };
    }
}

/// <summary>
/// Contains data for a single party in a benchmark dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class VflPartyDataset<T>
{
    /// <summary>Gets or sets the party identifier.</summary>
    public string PartyId { get; set; } = string.Empty;

    /// <summary>Gets or sets the party's feature data [numEntities, numFeatures].</summary>
    public Tensor<T> Features { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>Gets or sets the entity IDs for this party.</summary>
    public IReadOnlyList<string> EntityIds { get; set; } = Array.Empty<string>();

    /// <summary>Gets or sets the labels (only for label holder).</summary>
    public Tensor<T>? Labels { get; set; }

    /// <summary>Gets or sets whether this party is the label holder.</summary>
    public bool IsLabelHolder { get; set; }

    /// <summary>Gets or sets which columns from the full dataset this party holds.</summary>
    public IReadOnlyList<int> FeatureColumnIndices { get; set; } = Array.Empty<int>();
}

/// <summary>
/// Contains a complete benchmark dataset with vertically-partitioned data.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class VflBenchmarkDataset<T>
{
    /// <summary>Gets or sets the per-party datasets.</summary>
    public IReadOnlyList<VflPartyDataset<T>> Parties { get; set; } = Array.Empty<VflPartyDataset<T>>();

    /// <summary>Gets or sets the total entity count.</summary>
    public int TotalEntities { get; set; }

    /// <summary>Gets or sets the total feature count across all parties.</summary>
    public int TotalFeatures { get; set; }

    /// <summary>Gets or sets the entity overlap ratio.</summary>
    public double OverlapRatio { get; set; }

    /// <summary>Gets or sets the count of entities shared across all parties.</summary>
    public int SharedEntityCount { get; set; }
}

/// <summary>
/// Contains results from a VFL benchmark run.
/// </summary>
public class VflBenchmarkResult
{
    /// <summary>Gets or sets the total benchmark time including alignment and training.</summary>
    public TimeSpan TotalTime { get; set; }

    /// <summary>Gets or sets the time spent on entity alignment.</summary>
    public TimeSpan AlignmentTime { get; set; }

    /// <summary>Gets or sets the time spent on model training.</summary>
    public TimeSpan TrainingTime { get; set; }

    /// <summary>Gets or sets the final training loss.</summary>
    public double FinalLoss { get; set; }

    /// <summary>Gets or sets the number of training epochs completed.</summary>
    public int EpochsCompleted { get; set; }

    /// <summary>Gets or sets the number of aligned entities.</summary>
    public int AlignedEntities { get; set; }

    /// <summary>Gets or sets the total entities in the dataset.</summary>
    public int TotalEntities { get; set; }

    /// <summary>Gets or sets the number of parties.</summary>
    public int NumberOfParties { get; set; }
}
