using System.Text;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// // Create ML-kNN for multi-label classification with Bayesian inference
/// var options = new MLkNNOptions&lt;double&gt;();
///
/// // Prepare features and multi-label targets
/// var features = new Matrix&lt;double&gt;(4, 2);
/// features[0, 0] = 1.0; features[0, 1] = 2.0;
/// features[1, 0] = 3.0; features[1, 1] = 4.0;
/// features[2, 0] = 5.0; features[2, 1] = 6.0;
/// features[3, 0] = 7.0; features[3, 1] = 8.0;
/// var labels = new Matrix&lt;double&gt;(4, 3);
/// labels[0, 0] = 1; labels[0, 1] = 0; labels[0, 2] = 1;
/// labels[1, 0] = 1; labels[1, 1] = 1; labels[1, 2] = 0;
/// labels[2, 0] = 0; labels[2, 1] = 1; labels[2, 2] = 1;
/// labels[3, 0] = 0; labels[3, 1] = 0; labels[3, 2] = 1;
///
/// // Train by storing instances and computing Bayesian priors
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Matrix&lt;double&gt;&gt;()
///     .ConfigureModel(new MLkNNClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict labels using k-NN neighbor counts and Bayesian inference
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 2.0; newSample[0, 1] = 3.0;
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("ML-KNN: A Lazy Learning Approach to Multi-Label Learning", "https://doi.org/10.1016/j.patcog.2006.12.019", Year = 2007, Authors = "Min-Ling Zhang, Zhi-Hua Zhou")]
public partial class MLkNNClassifier<T> : MultiLabelClassifierBase<T>
{
    private readonly MLkNNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;
    private readonly Random _random;
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _trainFeatures = new Matrix<T>(0, 0);
    [AiDotNet.Attributes.FittedParameter]
    private Matrix<T> _trainLabels = new Matrix<T>(0, 0);
    private double[]? _priorProbs; // P(H_l = 1)
    private double[,] _condProbsPos = new double[0, 0]; // P(E_l = j | H_l = 1) for j = 0..k
    private double[,] _condProbsNeg = new double[0, 0]; // P(E_l = j | H_l = 0) for j = 0..k

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
                    if (NumOps.GreaterThan(labels[neighborIdx, l], NumOps.FromDouble(0.5)))
                    {
                        neighborCount++;
                    }
                }

                // Check if sample i has label l
                bool hasLabel = NumOps.GreaterThan(labels[i, l], NumOps.FromDouble(0.5));
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
                    if (NumOps.GreaterThan(_trainLabels[neighborIdx, l], NumOps.FromDouble(0.5)))
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
        int n = _trainFeatures.Rows;
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
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "bayesian-probabilities",
            new AiDotNet.Models.Parameters.VariableLengthParameterSource<T>(
                GetProbabilityParameterCount,
                GetProbabilityParameters,
                RestoreProbabilityParameters),
            AiDotNet.Models.Parameters.ParameterSlotRole.LearnedState);
    }

    private long GetProbabilityParameterCount()
    {
        if (_priorProbs is null) return 0;
        return checked((long)_priorProbs.Length + _condProbsPos.Length + _condProbsNeg.Length);
    }

    private Vector<T> GetProbabilityParameters()
    {
        if (_priorProbs is null) return new Vector<T>(0);

        int labels = _priorProbs.Length;
        int conditionalWidth = _options.KNeighbors + 1;
        if (_condProbsPos.GetLength(0) != labels ||
            _condProbsNeg.GetLength(0) != labels ||
            _condProbsPos.GetLength(1) != conditionalWidth ||
            _condProbsNeg.GetLength(1) != conditionalWidth)
        {
            throw new InvalidOperationException(
                "ML-kNN probability arrays do not share the fitted label and neighbor dimensions.");
        }

        int size = checked((int)GetProbabilityParameterCount());
        var parameters = new Vector<T>(size);

        int idx = 0;
        for (int l = 0; l < labels; l++)
        {
            parameters[idx++] = NumOps.FromDouble(_priorProbs[l]);
        }

        for (int l = 0; l < labels; l++)
        {
            for (int j = 0; j < conditionalWidth; j++)
            {
                parameters[idx++] = NumOps.FromDouble(_condProbsPos[l, j]);
            }
        }

        for (int l = 0; l < labels; l++)
        {
            for (int j = 0; j < conditionalWidth; j++)
            {
                parameters[idx++] = NumOps.FromDouble(_condProbsNeg[l, j]);
            }
        }

        return parameters;
    }

    private void RestoreProbabilityParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int conditionalWidth = _options.KNeighbors + 1;
        int valuesPerLabel = checked(1 + 2 * conditionalWidth);
        if (parameters.Length % valuesPerLabel != 0)
        {
            throw new ArgumentException(
                $"ML-kNN parameters must contain {valuesPerLabel} values per label; got " +
                $"{parameters.Length} values.", nameof(parameters));
        }

        int labels = parameters.Length / valuesPerLabel;
        NumLabels = labels;
        _priorProbs = new double[labels];
        _condProbsPos = new double[labels, conditionalWidth];
        _condProbsNeg = new double[labels, conditionalWidth];

        int idx = 0;
        for (int l = 0; l < labels; l++)
            _priorProbs[l] = NumOps.ToDouble(parameters[idx++]);
        for (int l = 0; l < labels; l++)
            for (int j = 0; j < conditionalWidth; j++)
                _condProbsPos[l, j] = NumOps.ToDouble(parameters[idx++]);
        for (int l = 0; l < labels; l++)
            for (int j = 0; j < conditionalWidth; j++)
                _condProbsNeg[l, j] = NumOps.ToDouble(parameters[idx++]);
    }

    /// <inheritdoc />
}
