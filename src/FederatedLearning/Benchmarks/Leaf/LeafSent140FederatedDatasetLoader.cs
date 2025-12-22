using System.IO;
using AiDotNet.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Loads the LEAF Sent140 benchmark JSON files into per-client datasets.
/// </summary>
/// <remarks>
/// <para>
/// Sent140 is a federated sentiment classification benchmark derived from tweets. LEAF stores each sample as an array
/// of string fields (id, date, query, user, text) and a numeric label (0/1).
/// </para>
/// <para><b>For Beginners:</b> This loader reads LEAF JSON and returns one dataset per user so federated learning
/// simulations match the benchmark's per-user partitioning.
/// </para>
/// </remarks>
public sealed class LeafSent140FederatedDatasetLoader
{
    /// <summary>
    /// Loads a LEAF Sent140 split (train/test) from a JSON file.
    /// </summary>
    public LeafFederatedSplit<string[], int[]> LoadSplitFromFile(
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
    /// Loads a LEAF Sent140 split (train/test) from a JSON string.
    /// </summary>
    public LeafFederatedSplit<string[], int[]> LoadSplitFromJson(
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

        var perUser = new Dictionary<string, FederatedClientDataset<string[], int[]>>(
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

        return new LeafFederatedSplit<string[], int[]>(limitedUserIds, perUser);
    }

    /// <summary>
    /// Loads a LEAF Sent140 train dataset and optional test dataset from files.
    /// </summary>
    public LeafFederatedDataset<string[], int[]> LoadDatasetFromFiles(
        string trainFilePath,
        string? testFilePath = null,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(trainFilePath))
        {
            throw new ArgumentException("Train file path cannot be null/empty.", nameof(trainFilePath));
        }

        var train = LoadSplitFromFile(trainFilePath, options);
        LeafFederatedSplit<string[], int[]>? test = null;

        if (!string.IsNullOrWhiteSpace(testFilePath))
        {
            test = LoadSplitFromFile(testFilePath!, options);
        }

        return new LeafFederatedDataset<string[], int[]>(train, test);
    }

    private static FederatedClientDataset<string[], int[]> ParseUserDataset(
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
            throw new InvalidDataException($"LEAF user_data['{userId}'] length mismatch: x has {xArray.Count} samples but y has {sampleCount}.");
        }

        if (validateDeclaredSamples && declaredSamples != sampleCount)
        {
            throw new InvalidDataException($"LEAF user '{userId}' declared num_samples={declaredSamples} but JSON contains {sampleCount} samples.");
        }

        if (sampleCount == 0)
        {
            return new FederatedClientDataset<string[], int[]>(Array.Empty<string>(), Array.Empty<int>(), 0);
        }

        var texts = new string[sampleCount];
        var labels = new int[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            texts[i] = ExtractTextFromXSample(userId, xArray[i]);
            labels[i] = ReadLabel(userId, yArray[i]);
        }

        return new FederatedClientDataset<string[], int[]>(texts, labels, sampleCount);
    }

    private static string ExtractTextFromXSample(string userId, JToken sample)
    {
        if (sample.Type == JTokenType.String)
        {
            return sample.Value<string>() ?? string.Empty;
        }

        if (sample is JArray fieldsArray)
        {
            if (fieldsArray.Count == 0)
            {
                return string.Empty;
            }

            var last = fieldsArray[fieldsArray.Count - 1];
            return last.Type == JTokenType.String ? (last.Value<string>() ?? string.Empty) : last.ToString();
        }

        throw new InvalidDataException($"LEAF Sent140 user '{userId}' contains an unsupported x sample type: {sample.Type}.");
    }

    private static int ReadLabel(string userId, JToken label)
    {
        if (label.Type == JTokenType.Integer)
        {
            return label.Value<int>();
        }

        if (label.Type == JTokenType.Float)
        {
            return Convert.ToInt32(label.Value<double>());
        }

        throw new InvalidDataException($"LEAF Sent140 user '{userId}' labels must be numeric (0/1).");
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

