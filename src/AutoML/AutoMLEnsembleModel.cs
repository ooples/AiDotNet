using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Serialization;

namespace AiDotNet.AutoML;

/// <summary>
/// A simple tabular ensemble model used as a facade-safe AutoML final model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This ensemble combines multiple <see cref="IFullModel{T,TInput,TOutput}"/> members by averaging (regression/binary)
/// or voting (multi-class) over their predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of trusting one model, an ensemble uses multiple models and combines their answers.
/// This often improves stability and accuracy.
/// </para>
/// </remarks>
public sealed class AutoMLEnsembleModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    [JsonProperty("Members")]
    private List<IFullModel<T, Matrix<T>, Vector<T>>> _members = new();

    [JsonIgnore]
    private IReadOnlyList<IFullModel<T, Matrix<T>, Vector<T>>> _membersView = Array.Empty<IFullModel<T, Matrix<T>, Vector<T>>>();

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoMLEnsembleModel{T}"/> class.
    /// </summary>
    /// <remarks>
    /// This constructor exists for serialization only. Prefer the overload that accepts members.
    /// </remarks>
    public AutoMLEnsembleModel()
    {
        SetMembers(new List<IFullModel<T, Matrix<T>, Vector<T>>>());
        Weights = Array.Empty<double>();
    }

    public AutoMLEnsembleModel(
        IEnumerable<IFullModel<T, Matrix<T>, Vector<T>>> members,
        PredictionType predictionType,
        IReadOnlyList<double>? weights = null)
    {
        if (members is null)
        {
            throw new ArgumentNullException(nameof(members));
        }

        var list = members.ToList();
        if (list.Count == 0)
        {
            throw new ArgumentException("Ensemble must include at least one member.", nameof(members));
        }

        SetMembers(list);
        PredictionType = predictionType;
        Weights = weights is null ? CreateUniformWeights(list.Count) : NormalizeWeights(weights, list.Count);
    }

    /// <summary>
    /// Gets the member models in the ensemble.
    /// </summary>
    [JsonIgnore]
    public IReadOnlyList<IFullModel<T, Matrix<T>, Vector<T>>> Members => _membersView;

    /// <summary>
    /// Gets or sets the prediction type used to combine outputs (regression vs classification).
    /// </summary>
    public PredictionType PredictionType { get; set; } = PredictionType.Regression;

    /// <summary>
    /// Gets or sets the per-member weights used when combining predictions.
    /// </summary>
    /// <remarks>
    /// Weights are normalized to sum to 1.0.
    /// </remarks>
    public double[] Weights { get; set; }

    public ILossFunction<T> DefaultLossFunction => Members.Count == 0
        ? throw new InvalidOperationException("Ensemble has no members.")
        : Members[0].DefaultLossFunction;

    public void Train(Matrix<T> input, Vector<T> expectedOutput)
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        foreach (var member in Members)
        {
            member.Train(input, expectedOutput);
        }
    }

    public Vector<T> Predict(Matrix<T> input)
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        var predictions = Members.Select(m => m.Predict(input)).ToList();
        if (predictions.Count == 1)
        {
            return predictions[0];
        }

        return PredictionType == PredictionType.MultiClass
            ? Vote(predictions)
            : WeightedAverage(predictions);
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        var metadata = Members.Count == 0
            ? new ModelMetadata<T>()
            : Members[0].GetModelMetadata();

        metadata.Name = string.IsNullOrWhiteSpace(metadata.Name) ? "AutoML Ensemble" : $"{metadata.Name} (Ensemble)";
        metadata.Description = $"AutoML ensemble with {Members.Count} members.";
        metadata.SetProperty("EnsembleSize", Members.Count);
        metadata.SetProperty("PredictionType", PredictionType.ToString());

        return metadata;
    }

    public byte[] Serialize()
    {
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder(),
            Formatting = Formatting.None
        };

        var json = JsonConvert.SerializeObject(this, settings);
        return Encoding.UTF8.GetBytes(json);
    }

    public void Deserialize(byte[] data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        var settings = new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.Auto,
            SerializationBinder = new SafeSerializationBinder()
        };

        var deserialized = JsonConvert.DeserializeObject<AutoMLEnsembleModel<T>>(json, settings);
        if (deserialized is null)
        {
            throw new InvalidOperationException("Failed to deserialize ensemble model.");
        }

        SetMembers(deserialized._members);
        PredictionType = deserialized.PredictionType;
        Weights = deserialized.Weights ?? Array.Empty<double>();
    }

    private void SetMembers(List<IFullModel<T, Matrix<T>, Vector<T>>>? members)
    {
        _members = members ?? new List<IFullModel<T, Matrix<T>, Vector<T>>>();
        _membersView = _members.AsReadOnly();
    }

    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    public void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    public void SaveState(Stream stream)
    {
        if (stream is null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        if (!stream.CanWrite)
        {
            throw new ArgumentException("Stream must be writable.", nameof(stream));
        }

        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    public void LoadState(Stream stream)
    {
        if (stream is null)
        {
            throw new ArgumentNullException(nameof(stream));
        }

        if (!stream.CanRead)
        {
            throw new ArgumentException("Stream must be readable.", nameof(stream));
        }

        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        Deserialize(ms.ToArray());
    }

    public Vector<T> GetParameters()
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        var vectors = Members.Select(m => m.GetParameters()).ToArray();
        return Vector<T>.Concatenate(vectors);
    }

    public void SetParameters(Vector<T> parameters)
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        int expected = ParameterCount;
        if (parameters.Length != expected)
        {
            throw new ArgumentException($"Parameter vector length {parameters.Length} does not match expected {expected}.", nameof(parameters));
        }

        int offset = 0;
        foreach (var member in Members)
        {
            int count = member.ParameterCount;
            var segment = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                segment[i] = parameters[offset + i];
            }

            member.SetParameters(segment);
            offset += count;
        }
    }

    public int ParameterCount => Members.Sum(m => m.ParameterCount);

    public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        copy.SetParameters(parameters);
        return copy;
    }

    public IEnumerable<int> GetActiveFeatureIndices()
    {
        if (Members.Count == 0)
        {
            return Array.Empty<int>();
        }

        var indices = new HashSet<int>();
        foreach (var member in Members)
        {
            foreach (var idx in member.GetActiveFeatureIndices())
            {
                indices.Add(idx);
            }
        }

        return indices.OrderBy(i => i).ToArray();
    }

    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        foreach (var member in Members)
        {
            member.SetActiveFeatureIndices(featureIndices);
        }
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return Members.Any(m => m.IsFeatureUsed(featureIndex));
    }

    public Dictionary<string, T> GetFeatureImportance()
    {
        if (Members.Count == 0)
        {
            return new Dictionary<string, T>(StringComparer.Ordinal);
        }

        var aggregate = new Dictionary<string, (double Sum, int Count)>(StringComparer.Ordinal);

        foreach (var member in Members)
        {
            Dictionary<string, T> importance;
            try
            {
                importance = member.GetFeatureImportance();
            }
            catch (InvalidOperationException)
            {
                continue;
            }
            catch (NotSupportedException)
            {
                continue;
            }

            foreach (var (key, value) in importance)
            {
                double numeric = NumOps.ToDouble(value);
                if (!aggregate.TryGetValue(key, out var entry))
                {
                    aggregate[key] = (numeric, 1);
                }
                else
                {
                    aggregate[key] = (entry.Sum + numeric, entry.Count + 1);
                }
            }
        }

        var result = new Dictionary<string, T>(StringComparer.Ordinal);
        foreach (var (key, entry) in aggregate)
        {
            result[key] = NumOps.FromDouble(entry.Sum / Math.Max(1, entry.Count));
        }

        return result;
    }

    public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        var copiedMembers = Members.Select(m => m.DeepCopy()).ToList();
        return new AutoMLEnsembleModel<T>(copiedMembers, PredictionType, GetSafeNormalizedWeights(copiedMembers.Count));
    }

    public IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clonedMembers = Members.Select(m => m.Clone()).ToList();
        return new AutoMLEnsembleModel<T>(clonedMembers, PredictionType, GetSafeNormalizedWeights(clonedMembers.Count));
    }

    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        var gradients = Members
            .Select(m => m.ComputeGradients(input, target, lossFunction))
            .ToArray();

        return Vector<T>.Concatenate(gradients);
    }

    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (Members.Count == 0)
        {
            throw new InvalidOperationException("Ensemble has no members.");
        }

        int expected = ParameterCount;
        if (gradients.Length != expected)
        {
            throw new ArgumentException($"Gradient vector length {gradients.Length} does not match expected {expected}.", nameof(gradients));
        }

        int offset = 0;
        foreach (var member in Members)
        {
            int count = member.ParameterCount;
            var segment = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                segment[i] = gradients[offset + i];
            }

            member.ApplyGradients(segment, learningRate);
            offset += count;
        }
    }

    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("Ensemble models do not currently support JIT compilation.");
    }

    public bool SupportsJitCompilation => false;

    private Vector<T> WeightedAverage(IReadOnlyList<Vector<T>> predictions)
    {
        int length = predictions[0].Length;
        var output = new Vector<T>(length);

        var weightVector = GetSafeNormalizedWeights(predictions.Count);
        var weightsT = weightVector.Select(NumOps.FromDouble).ToArray();

        for (int i = 0; i < length; i++)
        {
            T sum = NumOps.Zero;
            for (int m = 0; m < predictions.Count; m++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(predictions[m][i], weightsT[m]));
            }

            output[i] = sum;
        }

        return output;
    }

    private Vector<T> Vote(IReadOnlyList<Vector<T>> predictions)
    {
        int length = predictions[0].Length;
        var output = new Vector<T>(length);
        var weightVector = GetSafeNormalizedWeights(predictions.Count);

        for (int i = 0; i < length; i++)
        {
            var counts = new Dictionary<int, double>();

            for (int m = 0; m < predictions.Count; m++)
            {
                int label = NumOps.ToInt32(predictions[m][i]);
                double weight = weightVector[m];

                counts[label] = counts.TryGetValue(label, out var existing)
                    ? existing + weight
                    : weight;
            }

            int bestLabel = counts.OrderByDescending(kvp => kvp.Value).ThenBy(kvp => kvp.Key).First().Key;
            output[i] = NumOps.FromDouble(bestLabel);
        }

        return output;
    }

    private double[] GetSafeNormalizedWeights(int expectedCount)
    {
        if (expectedCount <= 0)
        {
            return Array.Empty<double>();
        }

        if (Weights is null || Weights.Length != expectedCount)
        {
            return CreateUniformWeights(expectedCount);
        }

        double sum = 0.0;
        for (int i = 0; i < Weights.Length; i++)
        {
            double value = Weights[i];
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return CreateUniformWeights(expectedCount);
            }

            sum += value;
        }

        if (sum <= 0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            return CreateUniformWeights(expectedCount);
        }

        if (Math.Abs(sum - 1.0) <= 1e-12)
        {
            return Weights;
        }

        var normalized = new double[expectedCount];
        for (int i = 0; i < expectedCount; i++)
        {
            normalized[i] = Weights[i] / sum;
        }

        return normalized;
    }

    private static double[] CreateUniformWeights(int count)
    {
        if (count <= 0)
        {
            return Array.Empty<double>();
        }

        double w = 1.0 / count;
        var weights = new double[count];
        for (int i = 0; i < count; i++)
        {
            weights[i] = w;
        }

        return weights;
    }

    private static double[] NormalizeWeights(IReadOnlyList<double> weights, int count)
    {
        if (count <= 0)
        {
            return Array.Empty<double>();
        }

        if (weights.Count != count)
        {
            throw new ArgumentException($"Expected {count} weights but received {weights.Count}.", nameof(weights));
        }

        double sum = weights.Sum();
        if (sum <= 0 || double.IsNaN(sum) || double.IsInfinity(sum))
        {
            return CreateUniformWeights(count);
        }

        var normalized = new double[count];
        for (int i = 0; i < count; i++)
        {
            normalized[i] = weights[i] / sum;
        }

        return normalized;
    }
}
