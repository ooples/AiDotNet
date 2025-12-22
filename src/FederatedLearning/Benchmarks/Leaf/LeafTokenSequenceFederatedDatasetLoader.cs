using System.IO;
using Newtonsoft.Json.Linq;
using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Loads LEAF-style JSON files that store token sequences (<c>x</c>) and next-token labels (<c>y</c>).
/// </summary>
/// <remarks>
/// <para>
/// This loader is intentionally generic: it expects the standard LEAF container shape
/// (<c>users</c>/<c>num_samples</c>/<c>user_data</c>) where each user's <c>x</c> is a list of token sequences
/// (arrays of strings) and each <c>y</c> is a list of label tokens (strings).
/// </para>
/// <para><b>For Beginners:</b> Some federated text benchmarks are easiest to represent as "predict the next token".
/// This loader keeps the per-user splits intact so federated simulations match the benchmark partitioning.
/// </para>
/// </remarks>
public sealed class LeafTokenSequenceFederatedDatasetLoader
{
    /// <summary>
    /// Loads a LEAF token-sequence split (train/test) from a JSON file.
    /// </summary>
    public LeafFederatedSplit<string[][], string[]> LoadSplitFromFile(
        string filePath,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null/empty.", nameof(filePath));
        }

        var json = File.ReadAllText(filePath);
        return LoadSplitFromJson(json, options);
    }

    /// <summary>
    /// Loads a LEAF token-sequence split (train/test) from a JSON string.
    /// </summary>
    public LeafFederatedSplit<string[][], string[]> LoadSplitFromJson(
        string json,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        if (json is null)
        {
            throw new ArgumentNullException(nameof(json));
        }

        options ??= new LeafFederatedDatasetLoadOptions();

        JObject root;
        try
        {
            root = JObject.Parse(json);
        }
        catch (Exception ex)
        {
            throw new InvalidDataException("LEAF split JSON is not a valid JSON object.", ex);
        }

        var usersElement = GetRequiredProperty(root, "users");
        var declaredSamplesElement = GetRequiredProperty(root, "num_samples");
        var userDataElement = GetRequiredProperty(root, "user_data");

        var userIds = ReadUserIds(usersElement);
        var declaredSamples = ReadDeclaredSampleCounts(declaredSamplesElement);

        if (declaredSamples.Count != userIds.Count)
        {
            throw new InvalidDataException(
                $"LEAF split JSON mismatch: users count {userIds.Count} does not match num_samples count {declaredSamples.Count}.");
        }

        if (userDataElement is not JObject userDataObject)
        {
            throw new InvalidDataException("LEAF split JSON property 'user_data' must be an object.");
        }

        int maxUsers = ResolveMaxUsers(options.MaxUsers, userIds.Count);
        var limitedUserIds = userIds.Take(maxUsers).ToList();

        var perUser = new Dictionary<string, FederatedClientDataset<string[][], string[]>>(
            limitedUserIds.Count,
            StringComparer.Ordinal);

        for (int i = 0; i < limitedUserIds.Count; i++)
        {
            var userId = limitedUserIds[i];
            var declared = declaredSamples[i];

            if (!userDataObject.TryGetValue(userId, StringComparison.Ordinal, out var entry))
            {
                throw new InvalidDataException($"LEAF split JSON is missing user_data entry for user '{userId}'.");
            }

            if (entry is not JObject entryObject)
            {
                throw new InvalidDataException($"LEAF user_data entry for '{userId}' must be an object.");
            }

            var xElement = GetRequiredProperty(entryObject, "x");
            var yElement = GetRequiredProperty(entryObject, "y");

            perUser[userId] = ParseUserDataset(userId, xElement, yElement, declared, options.ValidateDeclaredSampleCounts);
        }

        return new LeafFederatedSplit<string[][], string[]>(limitedUserIds, perUser);
    }

    /// <summary>
    /// Loads a LEAF token-sequence train dataset and optional test dataset from files.
    /// </summary>
    public LeafFederatedDataset<string[][], string[]> LoadDatasetFromFiles(
        string trainFilePath,
        string? testFilePath = null,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(trainFilePath))
        {
            throw new ArgumentException("Train file path cannot be null/empty.", nameof(trainFilePath));
        }

        var train = LoadSplitFromFile(trainFilePath, options);
        LeafFederatedSplit<string[][], string[]>? test = null;

        if (!string.IsNullOrWhiteSpace(testFilePath))
        {
            test = LoadSplitFromFile(testFilePath!, options);
        }

        return new LeafFederatedDataset<string[][], string[]>(train, test);
    }

    private static FederatedClientDataset<string[][], string[]> ParseUserDataset(
        string userId,
        JToken xElement,
        JToken yElement,
        int declaredSamples,
        bool validateDeclaredSamples)
    {
        if (xElement is not JArray xArray)
        {
            throw new InvalidDataException($"LEAF user_data['{userId}'].x must be an array.");
        }

        if (yElement is not JArray yArray)
        {
            throw new InvalidDataException($"LEAF user_data['{userId}'].y must be an array.");
        }

        int sampleCount = yArray.Count;
        if (sampleCount != xArray.Count)
        {
            throw new InvalidDataException($"LEAF user_data['{userId}'] length mismatch: x has {xArray.Count} samples but y has {sampleCount} samples.");
        }

        if (validateDeclaredSamples && declaredSamples != sampleCount)
        {
            throw new InvalidDataException($"LEAF user '{userId}' declared num_samples={declaredSamples} but JSON contains {sampleCount} samples.");
        }

        if (sampleCount == 0)
        {
            return new FederatedClientDataset<string[][], string[]>(Array.Empty<string[]>(), Array.Empty<string>(), 0);
        }

        var sequences = new string[sampleCount][];
        var labels = new string[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            sequences[i] = ReadTokenSequence(userId, xArray[i]);
            labels[i] = ReadRequiredString(userId, "y", yArray[i]);
        }

        return new FederatedClientDataset<string[][], string[]>(sequences, labels, sampleCount);
    }

    private static string[] ReadTokenSequence(string userId, JToken token)
    {
        if (token is not JArray tokenArray)
        {
            throw new InvalidDataException($"LEAF token sequence user '{userId}' contains a non-array x sample.");
        }

        var tokens = new string[tokenArray.Count];
        for (int i = 0; i < tokenArray.Count; i++)
        {
            tokens[i] = ReadRequiredString(userId, "x", tokenArray[i]);
        }

        return tokens;
    }

    private static string ReadRequiredString(string userId, string fieldName, JToken token)
    {
        if (token.Type != JTokenType.String)
        {
            throw new InvalidDataException($"LEAF user '{userId}' {fieldName} values must be strings.");
        }

        return token.Value<string>() ?? string.Empty;
    }

    private static JToken GetRequiredProperty(JObject root, string name)
    {
        if (!root.TryGetValue(name, StringComparison.Ordinal, out var token))
        {
            throw new InvalidDataException($"LEAF split JSON is missing required property '{name}'.");
        }

        return token;
    }

    private static List<string> ReadUserIds(JToken users)
    {
        if (users is not JArray usersArray)
        {
            throw new InvalidDataException("LEAF split JSON property 'users' must be an array.");
        }

        var result = new List<string>(usersArray.Count);
        foreach (var element in usersArray)
        {
            if (element.Type != JTokenType.String)
            {
                throw new InvalidDataException("LEAF split JSON property 'users' must contain only strings.");
            }

            var value = element.Value<string>();
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new InvalidDataException("LEAF split JSON property 'users' cannot contain empty user IDs.");
            }

            result.Add(value!);
        }

        return result;
    }

    private static List<int> ReadDeclaredSampleCounts(JToken numSamples)
    {
        if (numSamples is not JArray samplesArray)
        {
            throw new InvalidDataException("LEAF split JSON property 'num_samples' must be an array.");
        }

        var counts = new List<int>(samplesArray.Count);
        foreach (var element in samplesArray)
        {
            if (element.Type != JTokenType.Integer || element.Value<int>() < 0)
            {
                throw new InvalidDataException("LEAF split JSON property 'num_samples' must contain only non-negative integers.");
            }

            counts.Add(element.Value<int>());
        }

        return counts;
    }

    private static int ResolveMaxUsers(int? maxUsers, int availableUsers)
    {
        if (!maxUsers.HasValue)
        {
            return availableUsers;
        }

        if (maxUsers.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxUsers), "MaxUsers must be positive when specified.");
        }

        return Math.Min(maxUsers.Value, availableUsers);
    }
}

