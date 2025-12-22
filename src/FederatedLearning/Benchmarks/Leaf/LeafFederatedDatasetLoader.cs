using System.IO;
using AiDotNet.FederatedLearning.Infrastructure;
using Newtonsoft.Json.Linq;

namespace AiDotNet.FederatedLearning.Benchmarks.Leaf;

/// <summary>
/// Loads LEAF benchmark JSON files into per-client datasets suitable for federated learning simulation.
/// </summary>
/// <typeparam name="T">The numeric type for features/labels (for example, <see cref="double"/>).</typeparam>
/// <remarks>
/// <para>
/// LEAF stores federated datasets in a JSON structure containing:
/// - <c>users</c>: user/client IDs
/// - <c>num_samples</c>: declared sample counts per user
/// - <c>user_data</c>: per-user objects with <c>x</c> (features) and <c>y</c> (labels)
/// </para>
/// <para><b>For Beginners:</b> This loader reads one LEAF JSON file and converts it into a set of
/// "client datasets" — one dataset per user — so federated learning trainers can run simulations.
/// </para>
/// </remarks>
public sealed class LeafFederatedDatasetLoader<T> : FederatedLearningComponentBase<T>
{
    /// <summary>
    /// Loads a LEAF split (train/test) from a JSON file.
    /// </summary>
    /// <param name="filePath">Path to a LEAF JSON split file.</param>
    /// <param name="options">Optional load options (subset, validation).</param>
    /// <returns>A per-user federated split with numeric features and labels.</returns>
    public LeafFederatedSplit<Matrix<T>, Vector<T>> LoadSplitFromFile(
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
    /// Loads a LEAF split (train/test) from a JSON string.
    /// </summary>
    /// <param name="json">The LEAF JSON content.</param>
    /// <param name="options">Optional load options (subset, validation).</param>
    /// <returns>A per-user federated split with numeric features and labels.</returns>
    public LeafFederatedSplit<Matrix<T>, Vector<T>> LoadSplitFromJson(
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

        var perUser = new Dictionary<string, AiDotNet.Models.FederatedClientDataset<Matrix<T>, Vector<T>>>(
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

            var dataset = ParseNumericUserDataset(userId, xElement, yElement, declared, options.ValidateDeclaredSampleCounts);
            perUser[userId] = dataset;
        }

        return new LeafFederatedSplit<Matrix<T>, Vector<T>>(limitedUserIds, perUser);
    }

    /// <summary>
    /// Loads a LEAF train dataset and optional test dataset from files.
    /// </summary>
    /// <param name="trainFilePath">Path to the train JSON split.</param>
    /// <param name="testFilePath">Optional path to the test JSON split.</param>
    /// <param name="options">Optional load options (subset, validation).</param>
    /// <returns>A dataset with train and optional test splits.</returns>
    public LeafFederatedDataset<Matrix<T>, Vector<T>> LoadDatasetFromFiles(
        string trainFilePath,
        string? testFilePath = null,
        LeafFederatedDatasetLoadOptions? options = null)
    {
        var train = LoadSplitFromFile(trainFilePath, options);
        LeafFederatedSplit<Matrix<T>, Vector<T>>? test = null;

        if (!string.IsNullOrWhiteSpace(testFilePath))
        {
            test = LoadSplitFromFile(testFilePath!, options);
        }

        return new LeafFederatedDataset<Matrix<T>, Vector<T>>(train, test);
    }

    private static JToken GetRequiredProperty(JObject obj, string propertyName)
    {
        if (!obj.TryGetValue(propertyName, StringComparison.Ordinal, out var value))
        {
            throw new InvalidDataException($"LEAF split JSON is missing required property '{propertyName}'.");
        }

        return value;
    }

    private static List<string> ReadUserIds(JToken users)
    {
        if (users is not JArray usersArray)
        {
            throw new InvalidDataException("LEAF split JSON property 'users' must be an array.");
        }

        var userIds = new List<string>(usersArray.Count);
        foreach (var element in usersArray)
        {
            if (element.Type != JTokenType.String)
            {
                throw new InvalidDataException("LEAF split JSON property 'users' must contain only strings.");
            }

            var userId = element.Value<string>();
            if (userId is null || string.IsNullOrWhiteSpace(userId))
            {
                throw new InvalidDataException("LEAF split JSON property 'users' cannot contain null/empty user IDs.");
            }

            userIds.Add(userId);
        }

        if (userIds.Count == 0)
        {
            throw new InvalidDataException("LEAF split JSON must contain at least one user.");
        }

        return userIds;
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

    private AiDotNet.Models.FederatedClientDataset<Matrix<T>, Vector<T>> ParseNumericUserDataset(
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
            // Allow empty clients (some benchmarks include very small/filtered users).
            var emptyX = new Matrix<T>(0, 0);
            var emptyY = new Vector<T>(0);
            return new AiDotNet.Models.FederatedClientDataset<Matrix<T>, Vector<T>>(emptyX, emptyY, 0);
        }

        var firstSample = xArray[0];
        var firstFeatures = FlattenNumericSample(userId, firstSample);
        int featureCount = firstFeatures.Count;

        var features = new Matrix<T>(sampleCount, featureCount);
        for (int j = 0; j < featureCount; j++)
        {
            features[0, j] = NumOps.FromDouble(firstFeatures[j]);
        }

        int rowIndex = 1;
        foreach (var sampleFeatures in xArray.Skip(1).Select(sample => FlattenNumericSample(userId, sample)))
        {
            if (sampleFeatures.Count != featureCount)
            {
                throw new InvalidDataException(
                    $"LEAF user '{userId}' has inconsistent feature dimensions. Expected {featureCount} numeric values but found {sampleFeatures.Count}.");
            }

            for (int j = 0; j < featureCount; j++)
            {
                features[rowIndex, j] = NumOps.FromDouble(sampleFeatures[j]);
            }

            rowIndex++;
        }

        var labelValues = yArray
            .Select(label =>
            {
                if (label.Type != JTokenType.Integer && label.Type != JTokenType.Float)
                {
                    throw new InvalidDataException($"LEAF user '{userId}' labels must be numeric for the numeric loader.");
                }

                return NumOps.FromDouble(label.Value<double>());
            })
            .ToArray();

        var labels = new Vector<T>(labelValues);

        return new AiDotNet.Models.FederatedClientDataset<Matrix<T>, Vector<T>>(features, labels, sampleCount);
    }

    private static List<double> FlattenNumericSample(string userId, JToken sample)
    {
        var values = new List<double>();
        FlattenNumericValuesRecursive(userId, sample, values);
        return values;
    }

    private static void FlattenNumericValuesRecursive(string userId, JToken element, List<double> values)
    {
        if (element.Type == JTokenType.Integer || element.Type == JTokenType.Float)
        {
            values.Add(element.Value<double>());
            return;
        }

        if (element is JArray array)
        {
            foreach (var item in array)
            {
                FlattenNumericValuesRecursive(userId, item, values);
            }

            return;
        }

        throw new InvalidDataException($"LEAF user '{userId}' contains non-numeric feature values in x.");
    }
}
