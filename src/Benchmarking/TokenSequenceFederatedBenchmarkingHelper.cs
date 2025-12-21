using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Benchmarking;

internal static class TokenSequenceFederatedBenchmarkingHelper
{
    public const string PadToken = "<PAD>";
    public const string UnknownToken = "<UNK>";

    public static Dictionary<int, FederatedClientDataset<string[][], string[]>> SampleClientDatasets(
        IReadOnlyDictionary<int, FederatedClientDataset<string[][], string[]>> clientData,
        int maxSamplesPerUser,
        int seed)
    {
        if (maxSamplesPerUser <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamplesPerUser), "Max samples per user must be positive.");
        }

        var sampled = new Dictionary<int, FederatedClientDataset<string[][], string[]>>(clientData.Count);

        foreach (var kvp in clientData)
        {
            int clientId = kvp.Key;
            var dataset = kvp.Value;

            if (dataset.SampleCount == 0)
            {
                sampled[clientId] = dataset;
                continue;
            }

            int take = Math.Min(maxSamplesPerUser, dataset.SampleCount);
            if (take == dataset.SampleCount)
            {
                sampled[clientId] = dataset;
                continue;
            }

            int combinedSeed = unchecked((seed * 16777619) ^ clientId);
            var random = RandomHelper.CreateSeededRandom(combinedSeed);

            var indices = Enumerable.Range(0, dataset.SampleCount).ToArray();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            var selected = indices.Take(take).ToArray();
            Array.Sort(selected);

            var x = dataset.Features;
            var y = dataset.Labels;

            var newX = new string[take][];
            var newY = new string[take];

            for (int row = 0; row < take; row++)
            {
                int sourceRow = selected[row];
                newX[row] = x[sourceRow];
                newY[row] = y[sourceRow];
            }

            sampled[clientId] = new FederatedClientDataset<string[][], string[]>(newX, newY, take);
        }

        return sampled;
    }

    public static (IReadOnlyList<string[]> Sequences, IReadOnlyList<string> Labels, int TotalCount) ConcatenateClientData(
        IReadOnlyDictionary<int, FederatedClientDataset<string[][], string[]>> clientData)
    {
        int totalCount = 0;
        foreach (var dataset in clientData.Values)
        {
            if (dataset.SampleCount <= 0)
            {
                continue;
            }

            totalCount += dataset.SampleCount;
        }

        if (totalCount == 0)
        {
            return (Array.Empty<string[]>(), Array.Empty<string>(), 0);
        }

        var sequences = new List<string[]>(totalCount);
        var labels = new List<string>(totalCount);

        foreach (var dataset in clientData.Values)
        {
            for (int i = 0; i < dataset.SampleCount; i++)
            {
                sequences.Add(dataset.Features[i]);
                labels.Add(dataset.Labels[i]);
            }
        }

        return (sequences, labels, totalCount);
    }

    public static int ResolveSequenceLength(int? configuredSequenceLength, IReadOnlyList<string[]> sequences)
    {
        if (configuredSequenceLength.HasValue)
        {
            if (configuredSequenceLength.Value <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(configuredSequenceLength), "Sequence length must be positive when specified.");
            }

            ValidateSequenceLength(sequences, configuredSequenceLength.Value);
            return configuredSequenceLength.Value;
        }

        int inferred = 0;
        foreach (var sequence in sequences)
        {
            if (sequence is null || sequence.Length == 0)
            {
                continue;
            }

            inferred = sequence.Length;
            break;
        }

        if (inferred <= 0)
        {
            return 0;
        }

        ValidateSequenceLength(sequences, inferred);
        return inferred;
    }

    public static Dictionary<string, int> BuildVocabulary(
        IReadOnlyList<string[]> sequences,
        IReadOnlyList<string> labels,
        int maxVocabularySize,
        int maxTrainingSamples)
    {
        if (maxVocabularySize <= 2)
        {
            throw new ArgumentOutOfRangeException(nameof(maxVocabularySize), "Max vocabulary size must be greater than 2.");
        }

        if (maxTrainingSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTrainingSamples), "Max training samples must be positive.");
        }

        var frequencies = new Dictionary<string, int>(StringComparer.Ordinal);

        int sequencesToScan = Math.Min(sequences.Count, maxTrainingSamples);
        for (int i = 0; i < sequencesToScan; i++)
        {
            var sequence = sequences[i];
            if (sequence is null)
            {
                continue;
            }

            foreach (var token in sequence)
            {
                if (string.IsNullOrEmpty(token))
                {
                    continue;
                }

                if (string.Equals(token, PadToken, StringComparison.Ordinal))
                {
                    continue;
                }

                frequencies[token] = frequencies.TryGetValue(token, out var count) ? count + 1 : 1;
            }
        }

        int labelsToScan = Math.Min(labels.Count, maxTrainingSamples);
        for (int i = 0; i < labelsToScan; i++)
        {
            var token = labels[i];
            if (string.IsNullOrEmpty(token) || string.Equals(token, PadToken, StringComparison.Ordinal))
            {
                continue;
            }

            frequencies[token] = frequencies.TryGetValue(token, out var count) ? count + 1 : 1;
        }

        var orderedTokens = frequencies
            .OrderByDescending(kvp => kvp.Value)
            .ThenBy(kvp => kvp.Key, StringComparer.Ordinal)
            .Select(kvp => kvp.Key);

        var tokenToId = new Dictionary<string, int>(Math.Min(maxVocabularySize, frequencies.Count + 2), StringComparer.Ordinal)
        {
            [PadToken] = 0,
            [UnknownToken] = 1
        };

        foreach (var token in orderedTokens)
        {
            if (tokenToId.Count >= maxVocabularySize)
            {
                break;
            }

            if (tokenToId.ContainsKey(token))
            {
                continue;
            }

            tokenToId[token] = tokenToId.Count;
        }

        return tokenToId;
    }

    public static Matrix<T> EncodeSequencesToMatrix<T>(
        IReadOnlyList<string[]> sequences,
        IReadOnlyDictionary<string, int> tokenToId,
        int sequenceLength,
        INumericOperations<T> numOps,
        CancellationToken cancellationToken)
    {
        if (sequences.Count == 0 || sequenceLength <= 0)
        {
            return new Matrix<T>(0, 0);
        }

        int padId = tokenToId.TryGetValue(PadToken, out var pad) ? pad : 0;
        int unkId = tokenToId.TryGetValue(UnknownToken, out var unk) ? unk : 1;

        var matrix = new Matrix<T>(sequences.Count, sequenceLength);

        for (int row = 0; row < sequences.Count; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sequence = sequences[row] ?? Array.Empty<string>();
            for (int col = 0; col < sequenceLength; col++)
            {
                int tokenId;

                if (col >= sequence.Length)
                {
                    tokenId = padId;
                }
                else
                {
                    var token = sequence[col] ?? string.Empty;
                    if (!tokenToId.TryGetValue(token, out tokenId))
                    {
                        tokenId = unkId;
                    }
                }

                matrix[row, col] = numOps.FromDouble(tokenId);
            }
        }

        return matrix;
    }

    public static Vector<T> EncodeLabelsToVector<T>(
        IReadOnlyList<string> labels,
        IReadOnlyDictionary<string, int> tokenToId,
        INumericOperations<T> numOps)
    {
        if (labels.Count == 0)
        {
            return new Vector<T>(0);
        }

        int unkId = tokenToId.TryGetValue(UnknownToken, out var unk) ? unk : 1;
        var vector = new Vector<T>(labels.Count);

        for (int i = 0; i < labels.Count; i++)
        {
            var token = labels[i] ?? string.Empty;
            if (!tokenToId.TryGetValue(token, out var id))
            {
                id = unkId;
            }

            vector[i] = numOps.FromDouble(id);
        }

        return vector;
    }

    private static void ValidateSequenceLength(IReadOnlyList<string[]> sequences, int expectedLength)
    {
        for (int i = 0; i < sequences.Count; i++)
        {
            var sequence = sequences[i];
            if (sequence is null || sequence.Length == 0)
            {
                continue;
            }

            if (sequence.Length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"Token sequence dataset contains inconsistent sequence lengths. Expected {expectedLength} but found {sequence.Length}.");
            }
        }
    }
}

