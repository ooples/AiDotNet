using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Clustering.Base;

/// <summary>
/// Provides a base implementation for clustering algorithms that group similar data points together.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public abstract class ClusteringBase<T> : IClustering<T>, IConfigurableModel<T>, IModelShape
{
    /// <summary>
    /// Gets the numeric operations for the specified type T.
    /// </summary>
    protected INumericOperations<T> NumOps { get; private set; }

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Reference to the training data matrix. Used for safe in-sample prediction
    /// checks via ReferenceEquals (not row count which is unreliable).
    /// </summary>
    protected Matrix<T>? TrainingDataRef { get; set; }

    /// <summary>
    /// Gets the clustering options.
    /// </summary>
    protected ClusteringOptions<T> Options { get; private set; }

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => Options;

    /// <summary>
    /// Gets the regularization method.
    /// </summary>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets the default loss function.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <inheritdoc/>
    public int NumClusters { get; protected set; }

    /// <inheritdoc/>
    public Matrix<T>? ClusterCenters { get; protected set; }

    /// <inheritdoc/>
    public Vector<T>? Labels { get; protected set; }

    /// <inheritdoc/>
    public T? Inertia { get; protected set; }

    /// <summary>
    /// Gets or sets whether the model has been trained.
    /// </summary>
    public bool IsTrained { get; protected set; }

    /// <summary>
    /// Gets or sets the number of features.
    /// </summary>
    public int NumFeatures { get; protected set; }

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets the expected parameter count.
    /// </summary>
    protected virtual int ExpectedParameterCount => NumFeatures * NumClusters;

    /// <summary>
    /// Random number generator.
    /// </summary>
    protected Random? Random { get; set; }

    /// <summary>
    /// Initializes a new instance of the ClusteringBase class.
    /// </summary>
    protected ClusteringBase(
        ClusteringOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
    {
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new ClusteringOptions<T>();
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        Random = Options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Fits the model on data (unsupervised).
    /// </summary>
    public virtual void Fit(Matrix<T> x)
    {
        TrainingDataRef = x;
        var y = new Vector<T>(x.Rows);
        Train(x, y);
    }

    /// <summary>
    /// Trains the model on data (unsupervised convenience method).
    /// </summary>
    public virtual void Train(Matrix<T> x)
    {
        TrainingDataRef = x;
        var y = new Vector<T>(x.Rows);
        Train(x, y);
    }

    /// <summary>
    /// Validates that the model has been trained.
    /// </summary>
    protected void ValidateIsTrained()
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before performing this operation.");
        }
    }

    /// <inheritdoc/>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <inheritdoc/>
    public virtual Vector<T> FitPredict(Matrix<T> x)
    {
        Fit(x);
        return Labels ?? Predict(x);
    }

    /// <inheritdoc/>
    public virtual Matrix<T> Transform(Matrix<T> x)
    {
        if (ClusterCenters is null)
        {
            throw new InvalidOperationException("Transform requires cluster centers.");
        }

        var distances = new Matrix<T>(x.Rows, NumClusters);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int k = 0; k < NumClusters; k++)
            {
                distances[i, k] = ComputeDistance(x, i, ClusterCenters, k);
            }
        }
        return distances;
    }

    /// <summary>
    /// Computes distance between a sample and a cluster center.
    /// </summary>
    protected virtual T ComputeDistance(Matrix<T> x, int sampleIndex, Matrix<T> centers, int clusterIndex)
    {
        // Engine.DotProduct is zero-overhead in 0.13.0 — always use it
        int d = x.Columns;
        var diff = new Vector<T>(d);
        for (int j = 0; j < d; j++)
            diff[j] = NumOps.Subtract(x[sampleIndex, j], centers[clusterIndex, j]);
        return NumOps.Sqrt(Engine.DotProduct(diff, diff));
    }

    /// <summary>
    /// Computes squared distance.
    /// </summary>
    protected T ComputeSquaredDistance(Matrix<T> x, int sampleIndex, Matrix<T> centers, int clusterIndex)
    {
        int d = x.Columns;
        var diff = new Vector<T>(d);
        for (int j = 0; j < d; j++)
            diff[j] = NumOps.Subtract(x[sampleIndex, j], centers[clusterIndex, j]);
        return Engine.DotProduct(diff, diff);
    }

    /// <summary>
    /// Computes inertia.
    /// </summary>
    protected virtual T ComputeInertia(Matrix<T> x, Vector<T> labels, Matrix<T> centers)
    {
        T inertia = NumOps.Zero;
        for (int i = 0; i < x.Rows; i++)
        {
            int clusterIdx = (int)NumOps.ToDouble(labels[i]);
            if (clusterIdx >= 0 && clusterIdx < NumClusters)
            {
                T dist = ComputeSquaredDistance(x, i, centers, clusterIdx);
                inertia = NumOps.Add(inertia, dist);
            }
        }
        return inertia;
    }

    /// <summary>
    /// Gets a row from a matrix.
    /// </summary>
    protected Vector<T> GetRow(Matrix<T> matrix, int rowIndex)
    {
        return matrix.GetRow(rowIndex);
    }

    /// <summary>
    /// Sets a row in a matrix.
    /// </summary>
    protected void SetRow(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        for (int j = 0; j < matrix.Columns; j++)
        {
            matrix[rowIndex, j] = values[j];
        }
    }

    #region IFullModel Implementation

    /// <summary>
    /// Returns the model type identifier.
    /// </summary>

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            FeatureCount = NumFeatures,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["NumClusters"] = NumClusters,
                ["HasCenters"] = ClusterCenters is not null
            }
        };
    }

    /// <inheritdoc/>
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var modelData = new Dictionary<string, object?>
        {
            { "NumClusters", NumClusters },
            { "NumFeatures", NumFeatures },
            { "ClusterCenters", ClusterCenters is not null ? MatrixToDoubleArray(ClusterCenters) : null },
            { "Labels", Labels is not null ? VectorToDoubleArray(Labels) : null },
            { "Inertia", Inertia is not null ? NumOps.ToDouble(Inertia) : (double?)null },
            { "RegularizationOptions", Regularization.GetOptions() }
        };

        var modelMetadata = GetModelMetadata();
        modelMetadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelData));

        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelMetadata));
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] modelData)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        var jsonString = Encoding.UTF8.GetString(modelData);
        var modelMetadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);

        if (modelMetadata?.ModelData is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        var modelDataString = Encoding.UTF8.GetString(modelMetadata.ModelData);
        var dataObj = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(modelDataString);

        if (dataObj is null)
        {
            throw new InvalidOperationException("Deserialization failed: Invalid model data.");
        }

        NumClusters = dataObj["NumClusters"]?.ToObject<int>() ?? 0;
        NumFeatures = dataObj["NumFeatures"]?.ToObject<int>() ?? 0;

        var centersToken = dataObj["ClusterCenters"];
        if (centersToken is not null && centersToken.Type != Newtonsoft.Json.Linq.JTokenType.Null)
        {
            var centersArray = centersToken.ToObject<double[][]>();
            if (centersArray is not null && centersArray.Length > 0)
            {
                ClusterCenters = DoubleArrayToMatrix(centersArray);
            }
        }

        var labelsToken = dataObj["Labels"];
        if (labelsToken is not null && labelsToken.Type != Newtonsoft.Json.Linq.JTokenType.Null)
        {
            var labelsArray = labelsToken.ToObject<double[]>();
            if (labelsArray is not null)
            {
                Labels = DoubleArrayToVector(labelsArray);
            }
        }

        var inertiaToken = dataObj["Inertia"];
        if (inertiaToken is not null && inertiaToken.Type != Newtonsoft.Json.Linq.JTokenType.Null)
        {
            Inertia = NumOps.FromDouble(inertiaToken.ToObject<double>());
        }

        var regToken = dataObj["RegularizationOptions"];
        if (regToken is not null)
        {
            var regOptions = JsonConvert.DeserializeObject<RegularizationOptions>(regToken.ToString());
            if (regOptions is not null)
            {
                Regularization = RegularizationFactory.CreateRegularization<T, Matrix<T>, Vector<T>>(regOptions);
            }
        }

        IsTrained = NumClusters > 0 && (ClusterCenters is not null || Labels is not null);
    }

    private double[][] MatrixToDoubleArray(Matrix<T> matrix)
    {
        var result = new double[matrix.Rows][];
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i] = new double[matrix.Columns];
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i][j] = NumOps.ToDouble(matrix[i, j]);
            }
        }
        return result;
    }

    private Matrix<T> DoubleArrayToMatrix(double[][] array)
    {
        var matrix = new Matrix<T>(array.Length, array[0].Length);
        for (int i = 0; i < array.Length; i++)
        {
            for (int j = 0; j < array[i].Length; j++)
            {
                matrix[i, j] = NumOps.FromDouble(array[i][j]);
            }
        }
        return matrix;
    }

    private double[] VectorToDoubleArray(Vector<T> vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = NumOps.ToDouble(vector[i]);
        }
        return result;
    }

    private Vector<T> DoubleArrayToVector(double[] array)
    {
        var vector = new Vector<T>(array.Length);
        for (int i = 0; i < array.Length; i++)
        {
            vector[i] = NumOps.FromDouble(array[i]);
        }
        return vector;
    }

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { NumFeatures };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        // Output dimension equals the number of clusters (e.g., for soft assignment probabilities)
        return new[] { NumClusters > 0 ? NumClusters : 1 };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        string fullPath = Path.GetFullPath(filePath);

        byte[] data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json);

        string? directory = Path.GetDirectoryName(fullPath);
        if (directory is not null && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        string fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"Model file not found: {fullPath}", fullPath);

        byte[] data = File.ReadAllBytes(fullPath);

        // Extract payload from AIMF envelope if present; fall back to raw bytes for legacy files.
        try
        {
            data = ModelFileHeader.ExtractPayload(data);
        }
        catch (InvalidOperationException)
        {
            // File is not in AIMF envelope format — treat raw bytes as the model payload.
        }

        Deserialize(data);
    }

    /// <inheritdoc/>
    public virtual void SaveCheckpoint(string path) => SaveModel(path);

    /// <inheritdoc/>
    public virtual void LoadCheckpoint(string path) => LoadModel(path);

    /// <inheritdoc/>
    public virtual void SaveState(Stream stream)
    {
        byte[] data = Serialize();
        stream.Write(data, 0, data.Length);
    }

    /// <inheritdoc/>
    public virtual void LoadState(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        Deserialize(ms.ToArray());
    }

    /// <inheritdoc/>
    public virtual int ParameterCount => ExpectedParameterCount;

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <summary>
    /// Whether this clustering model supports direct parameter-based initialization.
    /// Hierarchical and density-based models should override to return false.
    /// Returns true by default; override to false for models that don't support it
    /// (e.g., density-based models where parameters aren't centroid coordinates).
    /// </summary>
    public virtual bool SupportsParameterInitialization => true;

    /// <inheritdoc/>
    public virtual Vector<T> GetParameters()
    {
        if (ClusterCenters is null)
        {
            return new Vector<T>(0);
        }

        // Ensure NumFeatures is derived from ClusterCenters if it wasn't set during training.
        // This fixes a bug where subclasses set ClusterCenters but forget to set NumFeatures.
        if (NumFeatures == 0 && ClusterCenters.Columns > 0)
        {
            NumFeatures = ClusterCenters.Columns;
        }

        var parameters = new Vector<T>(NumClusters * NumFeatures);
        int idx = 0;
        for (int k = 0; k < NumClusters; k++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                parameters[idx++] = ClusterCenters[k, j];
            }
        }
        return parameters;
    }

    /// <inheritdoc/>
    public virtual void SetParameters(Vector<T> parameters)
    {
        // If NumFeatures wasn't set but NumClusters is known, derive from parameter vector
        if (NumFeatures == 0 && NumClusters > 0 && parameters.Length > 0 && parameters.Length % NumClusters == 0)
        {
            NumFeatures = parameters.Length / NumClusters;
        }

        // If both are zero (untrained model), try to infer reasonable dimensions
        // This happens when the optimizer initializes a random solution on a cloned untrained model
        if (ExpectedParameterCount == 0 && parameters.Length > 0)
        {
            // Default to 2 clusters if not set, infer features from parameters
            if (NumClusters == 0) NumClusters = 2;
            NumFeatures = parameters.Length / NumClusters;
            if (NumFeatures == 0) NumFeatures = parameters.Length;
        }

        // If parameter count doesn't match, re-derive NumFeatures from the parameters.
        // This handles cloned models where NumFeatures wasn't preserved, and cases where
        // NumClusters changed after construction.
        if (parameters.Length != ExpectedParameterCount && parameters.Length > 0 && NumClusters > 0)
        {
            if (parameters.Length % NumClusters == 0)
            {
                NumFeatures = parameters.Length / NumClusters;
            }
            else
            {
                // Parameters don't divide evenly into NumClusters — try adjusting NumClusters
                // This can happen when the optimizer uses different cluster count than the model
                for (int k = NumClusters; k >= 1; k--)
                {
                    if (parameters.Length % k == 0)
                    {
                        NumClusters = k;
                        NumFeatures = parameters.Length / k;
                        break;
                    }
                }
            }
        }

        if (parameters.Length != ExpectedParameterCount)
        {
            throw new ArgumentException($"Expected {ExpectedParameterCount} parameters, got {parameters.Length}.");
        }

        ClusterCenters = new Matrix<T>(NumClusters, NumFeatures);
        int idx = 0;
        for (int k = 0; k < NumClusters; k++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                ClusterCenters[k, j] = parameters[idx++];
            }
        }
    }

    /// <inheritdoc/>
    public abstract IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        for (int i = 0; i < NumFeatures; i++)
        {
            yield return i;
        }
    }

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Default: no-op
    }

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= NumFeatures)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex));
        }
        return true;
    }

    /// <inheritdoc/>
    public virtual Vector<T> GetFeatureImportances()
    {
        var importances = new Vector<T>(NumFeatures);
        T value = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumFeatures));
        for (int i = 0; i < NumFeatures; i++)
        {
            importances[i] = value;
        }
        return importances;
    }

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.One;
        }
        return result;
    }

    /// <summary>
    /// Creates a new instance of this clustering algorithm.
    /// </summary>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <inheritdoc/>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        byte[] data = Serialize();
        var copy = CreateNewInstance();
        copy.Deserialize(data);
        return copy;
    }

    /// <inheritdoc/>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone() => DeepCopy();

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? _defaultLossFunction;
        var predictions = Predict(input);
        return loss.CalculateDerivative(predictions, target);
    }

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (ClusterCenters is null) return;

        int idx = 0;
        for (int k = 0; k < NumClusters; k++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                if (idx < gradients.Length)
                {
                    T update = NumOps.Multiply(gradients[idx], learningRate);
                    ClusterCenters[k, j] = NumOps.Subtract(ClusterCenters[k, j], update);
                    idx++;
                }
            }
        }
    }

    /// <inheritdoc/>
    public virtual bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("JIT compilation is not supported for this clustering algorithm.");
    }

    #endregion

    /// <summary>
    /// Merges clusters whose centers are within a small distance of each other.
    /// Call this after Train to handle degenerate data (e.g., single natural cluster
    /// split into multiple by a fixed-K algorithm).
    /// </summary>
    /// <param name="x">The training data matrix.</param>
    protected void MergeDegenerateClusters(Matrix<T> x)
    {
        // Iterate merging until stable — single-pass may not merge transitive chains
        // (A close to B, B close to C, but A not close to C after one pass)
        int maxIterations = 10;
        for (int iter = 0; iter < maxIterations; iter++)
        {
            int beforeK = NumClusters;
            MergeDegenerateClustersOnce(x);
            if (NumClusters >= beforeK) break; // No more merges possible
        }
    }

    private void MergeDegenerateClustersOnce(Matrix<T> x)
    {
        if (Labels is null || ClusterCenters is null || NumClusters <= 1) return;

        // Compute data range per feature for relative threshold
        double maxRange = 0;
        for (int j = 0; j < x.Columns; j++)
        {
            double colMin = double.MaxValue, colMax = double.MinValue;
            for (int i = 0; i < x.Rows; i++)
            {
                double v = NumOps.ToDouble(x[i, j]);
                if (v < colMin) colMin = v;
                if (v > colMax) colMax = v;
            }
            double range = colMax - colMin;
            if (range > maxRange) maxRange = range;
        }
        // Merge degenerate clusters: when cluster centers are closer than 10% of the maximum
        // feature range, they likely represent the same region of the data space.
        double mergeThreshold = Math.Max(1e-6, maxRange * 0.1);

        // First: identify which clusters actually have data points
        var clusterPopulations = new int[NumClusters];
        for (int i = 0; i < Labels.Length; i++)
        {
            int c = (int)Math.Round(NumOps.ToDouble(Labels[i]));
            if (c >= 0 && c < NumClusters)
                clusterPopulations[c]++;
        }

        // Build merge map: merge empty clusters into nearest populated one,
        // and merge nearby populated clusters
        var mergedId = Enumerable.Range(0, NumClusters).ToArray();

        // Find first populated cluster
        int firstPopulated = -1;
        for (int c = 0; c < NumClusters; c++)
        {
            if (clusterPopulations[c] > 0)
            {
                firstPopulated = c;
                break;
            }
        }

        // Merge all empty clusters into the first populated one
        if (firstPopulated >= 0)
        {
            for (int c = 0; c < NumClusters; c++)
            {
                if (clusterPopulations[c] == 0)
                    mergedId[c] = firstPopulated;
            }
        }

        // Merge nearby populated clusters
        double minDist = double.MaxValue;
        for (int a = 0; a < NumClusters; a++)
        {
            if (clusterPopulations[a] == 0) continue;
            for (int b = a + 1; b < NumClusters; b++)
            {
                if (clusterPopulations[b] == 0 || mergedId[b] != b) continue;
                double dist = 0;
                for (int j = 0; j < ClusterCenters.Columns; j++)
                {
                    double dd = NumOps.ToDouble(ClusterCenters[a, j]) - NumOps.ToDouble(ClusterCenters[b, j]);
                    dist += dd * dd;
                }
                double eucDist = Math.Sqrt(dist);
                if (eucDist < minDist) minDist = eucDist;
                if (eucDist < mergeThreshold)
                    mergedId[b] = mergedId[a];
            }
        }

        // Compact IDs and remap labels
        var uniqueMerged = mergedId.Distinct().OrderBy(x2 => x2).ToList();
        if (uniqueMerged.Count < NumClusters)
        {
            var compactMap = new Dictionary<int, int>();
            for (int i = 0; i < uniqueMerged.Count; i++)
                compactMap[uniqueMerged[i]] = i;

            for (int i = 0; i < Labels.Length; i++)
            {
                int oldCluster = (int)Math.Round(NumOps.ToDouble(Labels[i]));
                if (oldCluster >= 0 && oldCluster < mergedId.Length)
                    Labels[i] = NumOps.FromDouble(compactMap[mergedId[oldCluster]]);
            }
            NumClusters = uniqueMerged.Count;

            // Recompute cluster centers
            var newCenters = new Matrix<T>(NumClusters, x.Columns);
            var counts = new int[NumClusters];
            for (int i = 0; i < x.Rows; i++)
            {
                int c = (int)Math.Round(NumOps.ToDouble(Labels[i]));
                if (c >= 0 && c < NumClusters)
                {
                    for (int j = 0; j < x.Columns; j++)
                        newCenters[c, j] = NumOps.Add(newCenters[c, j], x[i, j]);
                    counts[c]++;
                }
            }
            for (int c = 0; c < NumClusters; c++)
            {
                if (counts[c] > 0)
                {
                    for (int j = 0; j < x.Columns; j++)
                        newCenters[c, j] = NumOps.Divide(newCenters[c, j], NumOps.FromDouble(counts[c]));
                }
            }
            ClusterCenters = newCenters;
        }
    }
}
