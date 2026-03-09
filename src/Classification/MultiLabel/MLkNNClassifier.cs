using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements ML-kNN (Multi-Label k-Nearest Neighbors) for multi-label classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> ML-kNN extends k-NN to multi-label problems using Bayesian inference.
/// For each label, it estimates the probability that a sample has the label given how many of its
/// k nearest neighbors have that label.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Find k nearest neighbors of the query sample</item>
/// <item>Count how many neighbors have each label</item>
/// <item>Use Bayesian inference with prior probabilities from training data</item>
/// <item>Predict label if P(label=1|neighbors) > P(label=0|neighbors)</item>
/// </list>
/// </para>
///
/// <para><b>Key formula:</b>
/// P(H_l | E_l) = P(E_l | H_l) * P(H_l) / P(E_l)
/// where H_l = label l is present, E_l = count of neighbors with label l</para>
///
/// <para><b>Reference:</b> Zhang &amp; Zhou, "ML-KNN: A lazy learning approach to multi-label learning" (2007)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MLkNNClassifier<T> : MultiLabelClassifierBase<T>
{
    private readonly MLkNNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    private Matrix<T>? _trainFeatures;
    private Matrix<T>? _trainLabels;
    private double[]? _priorProbs; // P(H_l = 1)
    private double[,]? _condProbsPos; // P(E_l = j | H_l = 1) for j = 0..k
    private double[,]? _condProbsNeg; // P(E_l = j | H_l = 0) for j = 0..k

    /// <summary>
    /// Creates a new ML-kNN classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public MLkNNClassifier(MLkNNOptions<T>? options = null)
        : base()
    {
        _options = options ?? new MLkNNOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    protected override void TrainMultiLabelCore(Matrix<T> features, Matrix<T> labels)
    {
        _trainFeatures = features;
        _trainLabels = labels;

        int n = features.Rows;
        int k = _options.KNeighbors;
        double s = _options.Smoothing;

        // Initialize probability arrays
        _priorProbs = new double[NumLabels];
        _condProbsPos = new double[NumLabels, k + 1];
        _condProbsNeg = new double[NumLabels, k + 1];

        // Count arrays for conditional probabilities
        var countPos = new int[NumLabels, k + 1]; // C[l,j] = count of samples with label l having j neighbors with label l
        var countNeg = new int[NumLabels, k + 1];
        var totalPos = new int[NumLabels];
        var totalNeg = new int[NumLabels];

        // Compute prior probabilities and conditional counts
        for (int i = 0; i < n; i++)
        {
            // Find k nearest neighbors (excluding self)
            var neighbors = FindKNearestNeighbors(features, i, k, excludeSelf: true);

            for (int l = 0; l < NumLabels; l++)
            {
                // Count neighbors with label l
                int neighborCount = 0;
                foreach (int neighborIdx in neighbors)
                {
                    if (NumOps.ToDouble(labels[neighborIdx, l]) > 0.5)
                    {
                        neighborCount++;
                    }
                }

                // Check if sample i has label l
                bool hasLabel = NumOps.ToDouble(labels[i, l]) > 0.5;
                if (hasLabel)
                {
                    countPos[l, neighborCount]++;
                    totalPos[l]++;
                }
                else
                {
                    countNeg[l, neighborCount]++;
                    totalNeg[l]++;
                }
            }
        }

        // Compute prior and conditional probabilities with smoothing
        for (int l = 0; l < NumLabels; l++)
        {
            _priorProbs[l] = (totalPos[l] + s) / (n + s * 2);

            for (int j = 0; j <= k; j++)
            {
                _condProbsPos[l, j] = (countPos[l, j] + s) / (totalPos[l] + s * (k + 1));
                _condProbsNeg[l, j] = (countNeg[l, j] + s) / (totalNeg[l] + s * (k + 1));
            }
        }
    }

    /// <inheritdoc />
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> features)
    {
        if (_trainFeatures is null || _priorProbs is null || _condProbsPos is null || _condProbsNeg is null)
        {
            throw new InvalidOperationException("Model must be trained before prediction.");
        }

        int n = features.Rows;
        int k = _options.KNeighbors;
        var probs = new Matrix<T>(n, NumLabels);

        for (int i = 0; i < n; i++)
        {
            // Find k nearest neighbors in training data
            var neighbors = FindKNearestNeighborsInTraining(features, i, k);

            for (int l = 0; l < NumLabels; l++)
            {
                // Count neighbors with label l
                int neighborCount = 0;
                foreach (int neighborIdx in neighbors)
                {
                    if (NumOps.ToDouble(_trainLabels![neighborIdx, l]) > 0.5)
                    {
                        neighborCount++;
                    }
                }

                // Bayesian inference: P(H_l=1|E_l=j) = P(E_l=j|H_l=1)*P(H_l=1) / P(E_l=j)
                double pPos = _condProbsPos[l, neighborCount] * _priorProbs[l];
                double pNeg = _condProbsNeg[l, neighborCount] * (1 - _priorProbs[l]);

                double prob = pPos / (pPos + pNeg + 1e-10);
                probs[i, l] = NumOps.FromDouble(prob);
            }
        }

        return probs;
    }

    private int[] FindKNearestNeighbors(Matrix<T> data, int sampleIdx, int k, bool excludeSelf)
    {
        int n = data.Rows;
        var distances = new List<(int Index, double Distance)>();

        for (int i = 0; i < n; i++)
        {
            if (excludeSelf && i == sampleIdx) continue;

            double dist = ComputeDistance(data, sampleIdx, data, i);
            distances.Add((i, dist));
        }

        return distances
            .OrderBy(x => x.Distance)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();
    }

    private int[] FindKNearestNeighborsInTraining(Matrix<T> query, int queryIdx, int k)
    {
        int n = _trainFeatures!.Rows;
        var distances = new List<(int Index, double Distance)>();

        for (int i = 0; i < n; i++)
        {
            double dist = ComputeDistanceCross(query, queryIdx, _trainFeatures, i);
            distances.Add((i, dist));
        }

        return distances
            .OrderBy(x => x.Distance)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();
    }

    private double ComputeDistance(Matrix<T> data1, int idx1, Matrix<T> data2, int idx2)
    {
        return NumOps.ToDouble(VectorHelper.EuclideanDistance(data1.GetRow(idx1), data2.GetRow(idx2)));
    }

    private double ComputeDistanceCross(Matrix<T> query, int queryIdx, Matrix<T> train, int trainIdx)
    {
        double dist = 0;
        int cols = query.Columns;

        for (int c = 0; c < cols; c++)
        {
            double diff = NumOps.ToDouble(query[queryIdx, c]) - NumOps.ToDouble(train[trainIdx, c]);
            dist += diff * diff;
        }

        return Math.Sqrt(dist);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (_priorProbs is null) return new Vector<T>(0);

        int k = _options.KNeighbors;
        int size = NumLabels + NumLabels * (k + 1) * 2;
        var parameters = new Vector<T>(size);

        int idx = 0;
        for (int l = 0; l < NumLabels; l++)
        {
            parameters[idx++] = NumOps.FromDouble(_priorProbs[l]);
        }

        for (int l = 0; l < NumLabels; l++)
        {
            for (int j = 0; j <= k; j++)
            {
                parameters[idx++] = NumOps.FromDouble(_condProbsPos![l, j]);
            }
        }

        for (int l = 0; l < NumLabels; l++)
        {
            for (int j = 0; j <= k; j++)
            {
                parameters[idx++] = NumOps.FromDouble(_condProbsNeg![l, j]);
            }
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Parameters alone are insufficient - need training data for k-NN
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        int k = _options.KNeighbors;

        // Serialize training features as double[][]
        double[][]? trainFeaturesArr = null;
        if (_trainFeatures is not null)
        {
            trainFeaturesArr = new double[_trainFeatures.Rows][];
            for (int i = 0; i < _trainFeatures.Rows; i++)
            {
                trainFeaturesArr[i] = new double[_trainFeatures.Columns];
                for (int j = 0; j < _trainFeatures.Columns; j++)
                {
                    trainFeaturesArr[i][j] = NumOps.ToDouble(_trainFeatures[i, j]);
                }
            }
        }

        // Serialize training labels as double[][]
        double[][]? trainLabelsArr = null;
        if (_trainLabels is not null)
        {
            trainLabelsArr = new double[_trainLabels.Rows][];
            for (int i = 0; i < _trainLabels.Rows; i++)
            {
                trainLabelsArr[i] = new double[_trainLabels.Columns];
                for (int j = 0; j < _trainLabels.Columns; j++)
                {
                    trainLabelsArr[i][j] = NumOps.ToDouble(_trainLabels[i, j]);
                }
            }
        }

        // Serialize 2D conditional probability arrays as double[][]
        double[][]? condProbsPosArr = null;
        double[][]? condProbsNegArr = null;
        if (_condProbsPos is not null && _condProbsNeg is not null)
        {
            condProbsPosArr = new double[NumLabels][];
            condProbsNegArr = new double[NumLabels][];
            for (int l = 0; l < NumLabels; l++)
            {
                condProbsPosArr[l] = new double[k + 1];
                condProbsNegArr[l] = new double[k + 1];
                for (int j = 0; j <= k; j++)
                {
                    condProbsPosArr[l][j] = _condProbsPos[l, j];
                    condProbsNegArr[l][j] = _condProbsNeg[l, j];
                }
            }
        }

        var modelDict = new Dictionary<string, object?>
        {
            { "NumLabels", NumLabels },
            { "NumFeatures", NumFeatures },
            { "NumClasses", NumClasses },
            { "TaskType", (int)TaskType },
            { "LabelNames", LabelNames },
            { "KNeighbors", k },
            { "Smoothing", _options.Smoothing },
            { "PriorProbs", _priorProbs },
            { "CondProbsPos", condProbsPosArr },
            { "CondProbsNeg", condProbsNegArr },
            { "TrainFeatures", trainFeaturesArr },
            { "TrainLabels", trainLabelsArr }
        };

        var metadata = GetModelMetadata();
        metadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelDict));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(metadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var metadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString)
            ?? throw new InvalidOperationException("Failed to deserialize MLkNNClassifier: invalid metadata.");
        if (metadata.ModelData is null)
            throw new InvalidOperationException("Failed to deserialize MLkNNClassifier: missing model data.");

        var dataString = Encoding.UTF8.GetString(metadata.ModelData);
        var jObj = JsonConvert.DeserializeObject<JObject>(dataString)
            ?? throw new InvalidOperationException("Failed to deserialize MLkNNClassifier: invalid model payload.");

        NumLabels = jObj["NumLabels"]?.ToObject<int>() ?? 0;
        NumFeatures = jObj["NumFeatures"]?.ToObject<int>() ?? 0;
        NumClasses = jObj["NumClasses"]?.ToObject<int>() ?? 2;
        TaskType = (ClassificationTaskType)(jObj["TaskType"]?.ToObject<int>() ?? 0);
        LabelNames = jObj["LabelNames"]?.ToObject<string[]>();
        _options.Smoothing = jObj["Smoothing"]?.ToObject<double>() ?? _options.Smoothing;

        _priorProbs = jObj["PriorProbs"]?.ToObject<double[]>();

        int k = jObj["KNeighbors"]?.ToObject<int>() ?? _options.KNeighbors;

        var condProbsPosArr = jObj["CondProbsPos"]?.ToObject<double[][]>();
        var condProbsNegArr = jObj["CondProbsNeg"]?.ToObject<double[][]>();
        if (condProbsPosArr is not null && condProbsNegArr is not null)
        {
            if (condProbsPosArr.Length < NumLabels)
                throw new InvalidOperationException(
                    $"Failed to deserialize MLkNNClassifier: CondProbsPos has {condProbsPosArr.Length} rows but expected at least {NumLabels}.");
            if (condProbsNegArr.Length < NumLabels)
                throw new InvalidOperationException(
                    $"Failed to deserialize MLkNNClassifier: CondProbsNeg has {condProbsNegArr.Length} rows but expected at least {NumLabels}.");

            _condProbsPos = new double[NumLabels, k + 1];
            _condProbsNeg = new double[NumLabels, k + 1];
            for (int l = 0; l < NumLabels && l < condProbsPosArr.Length; l++)
            {
                for (int j = 0; j <= k && j < condProbsPosArr[l].Length; j++)
                {
                    _condProbsPos[l, j] = condProbsPosArr[l][j];
                    _condProbsNeg[l, j] = condProbsNegArr[l][j];
                }
            }
        }

        var trainFeaturesArr = jObj["TrainFeatures"]?.ToObject<double[][]>();
        if (trainFeaturesArr is not null && trainFeaturesArr.Length > 0)
        {
            int rows = trainFeaturesArr.Length;
            int cols = trainFeaturesArr[0].Length;
            _trainFeatures = new Matrix<T>(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _trainFeatures[i, j] = NumOps.FromDouble(trainFeaturesArr[i][j]);
                }
            }
        }

        var trainLabelsArr = jObj["TrainLabels"]?.ToObject<double[][]>();
        if (trainLabelsArr is not null && trainLabelsArr.Length > 0)
        {
            int rows = trainLabelsArr.Length;
            int cols = trainLabelsArr[0].Length;
            _trainLabels = new Matrix<T>(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _trainLabels[i, j] = NumOps.FromDouble(trainLabelsArr[i][j]);
                }
            }
        }
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.MultiLabelClassifier;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance()
    {
        return new MLkNNClassifier<T>(_options);
    }
}
