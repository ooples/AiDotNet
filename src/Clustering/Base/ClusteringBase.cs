using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Clustering.Base;

/// <summary>
/// Provides a base implementation for clustering algorithms that group similar data points together.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public abstract class ClusteringBase<T> : IClustering<T>
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
    /// Gets the clustering options.
    /// </summary>
    protected ClusteringOptions<T> Options { get; private set; }

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
        Random = Options.RandomState.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomState.Value)
            : new Random();
    }

    /// <inheritdoc/>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Fits the model on data (unsupervised).
    /// </summary>
    public virtual void Fit(Matrix<T> x)
    {
        var y = new Vector<T>(x.Rows);
        Train(x, y);
    }

    /// <summary>
    /// Trains the model on data (unsupervised convenience method).
    /// </summary>
    public virtual void Train(Matrix<T> x)
    {
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
        T sum = NumOps.Zero;
        for (int j = 0; j < NumFeatures; j++)
        {
            T diff = NumOps.Subtract(x[sampleIndex, j], centers[clusterIndex, j]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Computes squared distance.
    /// </summary>
    protected T ComputeSquaredDistance(Matrix<T> x, int sampleIndex, Matrix<T> centers, int clusterIndex)
    {
        T sum = NumOps.Zero;
        for (int j = 0; j < NumFeatures; j++)
        {
            T diff = NumOps.Subtract(x[sampleIndex, j], centers[clusterIndex, j]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }
        return sum;
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
        var row = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            row[j] = matrix[rowIndex, j];
        }
        return row;
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
    protected abstract ModelType GetModelType();

    /// <inheritdoc/>
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = GetModelType(),
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
    public virtual void SaveModel(string filePath)
    {
        byte[] data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        byte[] data = File.ReadAllBytes(filePath);
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
    public virtual Vector<T> GetParameters()
    {
        if (ClusterCenters is null)
        {
            return new Vector<T>(0);
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
}
