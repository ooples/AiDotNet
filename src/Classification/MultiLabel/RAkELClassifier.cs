using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements RAkEL (Random k-Labelsets) for multi-label classification.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> RAkEL solves multi-label classification by training multiple
/// Label Powerset classifiers on random subsets of k labels each. This captures label correlations
/// (like Label Powerset) while avoiding the exponential explosion of label combinations.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Randomly partition labels into overlapping subsets of size k</item>
/// <item>Train a Label Powerset classifier on each subset</item>
/// <item>Combine predictions by voting across all classifiers that predict each label</item>
/// </list>
/// </para>
///
/// <para><b>Key parameters:</b>
/// <list type="bullet">
/// <item><b>k:</b> Size of each labelset (default: 3). Larger k captures more correlations but has more classes.</item>
/// <item><b>numLabelsets:</b> Number of random labelsets to create (default: 2*numLabels)</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Tsoumakas et al., "Random k-Labelsets for Multilabel Classification" (2011)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RAkELClassifier<T> : MultiLabelClassifierBase<T>
{
    /// <summary>
    /// Gets the size of each labelset (k parameter).
    /// </summary>
    public int LabelsetSize { get; private set; }

    /// <summary>
    /// Gets the number of labelsets to create.
    /// </summary>
    public int NumLabelsets { get; private set; }

    /// <summary>
    /// The random labelsets (each is an array of label indices).
    /// </summary>
    private List<int[]> _labelsets;

    /// <summary>
    /// Weight matrices for each labelset classifier.
    /// </summary>
    private List<Matrix<T>> _labelsetWeights;

    /// <summary>
    /// The unique label combination codes for each labelset classifier.
    /// </summary>
    private List<Dictionary<string, int>> _labelCombinationMaps;

    /// <summary>
    /// Inverse mapping from class index back to label combination.
    /// </summary>
    private List<Dictionary<int, int[]>> _inverseLabelMaps;

    /// <summary>
    /// Random instance for reproducibility.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Creates a new RAkEL classifier.
    /// </summary>
    /// <param name="labelsetSize">Size of each random labelset (default: 3).</param>
    /// <param name="numLabelsets">Number of labelsets (default: 0 = auto-calculate as 2*numLabels).</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <param name="options">Classifier options.</param>
    /// <param name="regularization">Regularization method.</param>
    public RAkELClassifier(
        int labelsetSize = 3,
        int numLabelsets = 0,
        int? seed = null,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        if (labelsetSize < 2)
            throw new ArgumentOutOfRangeException(nameof(labelsetSize), "Labelset size must be at least 2.");

        LabelsetSize = labelsetSize;
        NumLabelsets = numLabelsets;
        _labelsets = new List<int[]>();
        _labelsetWeights = new List<Matrix<T>>();
        _labelCombinationMaps = new List<Dictionary<string, int>>();
        _inverseLabelMaps = new List<Dictionary<int, int[]>>();

        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Core training implementation for RAkEL.
    /// </summary>
    protected override void TrainMultiLabelCore(Matrix<T> features, Matrix<T> labels)
    {
        int effectiveLabelsetSize = Math.Min(LabelsetSize, NumLabels);

        // Auto-calculate number of labelsets if not specified
        if (NumLabelsets <= 0)
            NumLabelsets = Math.Max(2, 2 * NumLabels);

        // Generate random labelsets
        _labelsets.Clear();
        _labelsetWeights.Clear();
        _labelCombinationMaps.Clear();
        _inverseLabelMaps.Clear();

        for (int i = 0; i < NumLabelsets; i++)
        {
            var labelset = GenerateRandomLabelset(effectiveLabelsetSize);
            _labelsets.Add(labelset);
        }

        // Train a Label Powerset classifier for each labelset
        int numSamples = features.Rows;
        int numIterations = 100;
        double learningRate = 0.1;

        foreach (var labelset in _labelsets)
        {
            // Create label combinations for this labelset
            var combinationMap = new Dictionary<string, int>();
            var inverseCombinationMap = new Dictionary<int, int[]>();

            // Extract labels for this labelset and create unique combinations
            for (int i = 0; i < numSamples; i++)
            {
                var combination = new int[labelset.Length];
                for (int j = 0; j < labelset.Length; j++)
                {
                    combination[j] = NumOps.GreaterThan(labels[i, labelset[j]], NumOps.FromDouble(0.5)) ? 1 : 0;
                }

                string key = string.Join(",", combination);
                if (!combinationMap.ContainsKey(key))
                {
                    int classIdx = combinationMap.Count;
                    combinationMap[key] = classIdx;
                    inverseCombinationMap[classIdx] = combination;
                }
            }

            _labelCombinationMaps.Add(combinationMap);
            _inverseLabelMaps.Add(inverseCombinationMap);

            int numClasses = combinationMap.Count;

            // Initialize weights for this labelset classifier
            var weights = new Matrix<T>(NumFeatures + 1, numClasses); // +1 for bias

            // Train with softmax cross-entropy via gradient descent
            for (int iter = 0; iter < numIterations; iter++)
            {
                for (int i = 0; i < numSamples; i++)
                {
                    // Get the true class for this sample
                    var combination = new int[labelset.Length];
                    for (int j = 0; j < labelset.Length; j++)
                    {
                        combination[j] = NumOps.GreaterThan(labels[i, labelset[j]], NumOps.FromDouble(0.5)) ? 1 : 0;
                    }
                    string key = string.Join(",", combination);
                    int trueClass = combinationMap[key];

                    // Compute logits
                    var logits = new double[numClasses];
                    for (int c = 0; c < numClasses; c++)
                    {
                        logits[c] = NumOps.ToDouble(weights[NumFeatures, c]); // bias
                        for (int f = 0; f < NumFeatures; f++)
                        {
                            logits[c] += NumOps.ToDouble(features[i, f]) * NumOps.ToDouble(weights[f, c]);
                        }
                    }

                    // Softmax
                    double maxLogit = logits.Max();
                    var expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
                    double sumExp = expLogits.Sum();
                    var probs = expLogits.Select(e => e / sumExp).ToArray();

                    // Gradient update (cross-entropy)
                    for (int c = 0; c < numClasses; c++)
                    {
                        double target = c == trueClass ? 1.0 : 0.0;
                        double gradient = (probs[c] - target) / numSamples;

                        // Update bias
                        weights[NumFeatures, c] = NumOps.Subtract(
                            weights[NumFeatures, c],
                            NumOps.FromDouble(learningRate * gradient));

                        // Update feature weights
                        for (int f = 0; f < NumFeatures; f++)
                        {
                            double featureGrad = gradient * NumOps.ToDouble(features[i, f]);
                            weights[f, c] = NumOps.Subtract(
                                weights[f, c],
                                NumOps.FromDouble(learningRate * featureGrad));
                        }
                    }
                }
            }

            _labelsetWeights.Add(weights);
        }
    }

    /// <summary>
    /// Generates a random labelset of the specified size.
    /// </summary>
    private int[] GenerateRandomLabelset(int size)
    {
        var available = Enumerable.Range(0, NumLabels).ToList();
        var labelset = new int[size];

        for (int i = 0; i < size; i++)
        {
            int idx = _random.Next(available.Count);
            labelset[i] = available[idx];
            available.RemoveAt(idx);
        }

        return labelset.OrderBy(x => x).ToArray();
    }

    /// <summary>
    /// Predicts multi-label probabilities using RAkEL voting.
    /// </summary>
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        int numSamples = input.Rows;
        var labelVotes = new double[numSamples, NumLabels];
        var labelCounts = new int[NumLabels];

        // Accumulate votes from each labelset classifier
        for (int ls = 0; ls < _labelsets.Count; ls++)
        {
            var labelset = _labelsets[ls];
            var weights = _labelsetWeights[ls];
            var inverseLabelMap = _inverseLabelMaps[ls];
            int numClasses = inverseLabelMap.Count;

            // Count how many times each label appears in labelsets
            foreach (int labelIdx in labelset)
                labelCounts[labelIdx]++;

            for (int i = 0; i < numSamples; i++)
            {
                // Compute logits for this labelset classifier
                var logits = new double[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    logits[c] = NumOps.ToDouble(weights[NumFeatures, c]); // bias
                    for (int f = 0; f < NumFeatures; f++)
                    {
                        logits[c] += NumOps.ToDouble(input[i, f]) * NumOps.ToDouble(weights[f, c]);
                    }
                }

                // Softmax
                double maxLogit = logits.Max();
                var expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
                double sumExp = expLogits.Sum();
                var probs = expLogits.Select(e => e / sumExp).ToArray();

                // Vote for each label based on class probabilities
                for (int c = 0; c < numClasses; c++)
                {
                    var labelCombination = inverseLabelMap[c];
                    for (int j = 0; j < labelset.Length; j++)
                    {
                        int labelIdx = labelset[j];
                        // Weight vote by probability and label value
                        labelVotes[i, labelIdx] += probs[c] * labelCombination[j];
                    }
                }
            }
        }

        // Normalize votes to get probabilities
        var result = new Matrix<T>(numSamples, NumLabels);
        for (int i = 0; i < numSamples; i++)
        {
            for (int l = 0; l < NumLabels; l++)
            {
                double prob = labelCounts[l] > 0 ? labelVotes[i, l] / labelCounts[l] : 0;
                result[i, l] = NumOps.FromDouble(Math.Max(0, Math.Min(1, prob)));
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        if (_labelsetWeights.Count == 0)
            return new Vector<T>(0);

        // Flatten all labelset weights into single vector
        int totalParams = 0;
        foreach (var weights in _labelsetWeights)
            totalParams += weights.Rows * weights.Columns;

        var parameters = new Vector<T>(totalParams);
        int idx = 0;
        foreach (var weights in _labelsetWeights)
        {
            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    parameters[idx++] = weights[i, j];
                }
            }
        }

        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Restore parameters to labelset weights
        int idx = 0;
        foreach (var weights in _labelsetWeights)
        {
            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Columns; j++)
                {
                    if (idx < parameters.Length)
                        weights[i, j] = parameters[idx++];
                }
            }
        }
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        // Serialize weight matrices as double[][][]
        var weightsArr = new List<double[][]>();
        foreach (var weights in _labelsetWeights)
        {
            var matrix = new double[weights.Rows][];
            for (int i = 0; i < weights.Rows; i++)
            {
                matrix[i] = new double[weights.Columns];
                for (int j = 0; j < weights.Columns; j++)
                {
                    matrix[i][j] = NumOps.ToDouble(weights[i, j]);
                }
            }
            weightsArr.Add(matrix);
        }

        // Serialize inverse label maps as Dictionary<string, int[]>[] (JSON needs string keys)
        var inverseMapsArr = new List<Dictionary<string, int[]>>();
        foreach (var map in _inverseLabelMaps)
        {
            var strMap = new Dictionary<string, int[]>();
            foreach (var kvp in map)
            {
                strMap[kvp.Key.ToString()] = kvp.Value;
            }
            inverseMapsArr.Add(strMap);
        }

        var modelDict = new Dictionary<string, object?>
        {
            { "NumLabels", NumLabels },
            { "NumFeatures", NumFeatures },
            { "NumClasses", NumClasses },
            { "TaskType", (int)TaskType },
            { "LabelNames", LabelNames },
            { "LabelsetSize", LabelsetSize },
            { "NumLabelsets", NumLabelsets },
            { "Labelsets", _labelsets },
            { "LabelsetWeights", weightsArr },
            { "LabelCombinationMaps", _labelCombinationMaps },
            { "InverseLabelMaps", inverseMapsArr }
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
            ?? throw new InvalidOperationException("Failed to deserialize RAkELClassifier: invalid metadata.");
        if (metadata.ModelData is null)
            throw new InvalidOperationException("Failed to deserialize RAkELClassifier: missing model data.");

        var dataString = Encoding.UTF8.GetString(metadata.ModelData);
        var jObj = JsonConvert.DeserializeObject<JObject>(dataString)
            ?? throw new InvalidOperationException("Failed to deserialize RAkELClassifier: invalid model payload.");

        NumLabels = jObj["NumLabels"]?.ToObject<int>() ?? 0;
        NumFeatures = jObj["NumFeatures"]?.ToObject<int>() ?? 0;
        NumClasses = jObj["NumClasses"]?.ToObject<int>() ?? 2;
        TaskType = (ClassificationTaskType)(jObj["TaskType"]?.ToObject<int>() ?? 0);
        LabelNames = jObj["LabelNames"]?.ToObject<string[]>();
        var deserializedLabelsetSize = jObj["LabelsetSize"]?.ToObject<int>() ?? LabelsetSize;
        if (deserializedLabelsetSize < 2)
            throw new InvalidOperationException(
                $"Failed to deserialize RAkELClassifier: LabelsetSize ({deserializedLabelsetSize}) must be >= 2.");
        LabelsetSize = deserializedLabelsetSize;
        NumLabelsets = jObj["NumLabelsets"]?.ToObject<int>() ?? NumLabelsets;

        // Deserialize labelsets
        var labelsetsArr = jObj["Labelsets"]?.ToObject<List<int[]>>();
        _labelsets = labelsetsArr ?? new List<int[]>();

        // Deserialize weight matrices
        _labelsetWeights = new List<Matrix<T>>();
        var weightsArr = jObj["LabelsetWeights"]?.ToObject<List<double[][]>>();
        if (weightsArr is not null)
        {
            foreach (var matrixArr in weightsArr)
            {
                int rows = matrixArr.Length;
                int cols = rows > 0 ? matrixArr[0].Length : 0;
                var matrix = new Matrix<T>(rows, cols);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        matrix[i, j] = NumOps.FromDouble(matrixArr[i][j]);
                    }
                }
                _labelsetWeights.Add(matrix);
            }
        }

        // Deserialize label combination maps
        _labelCombinationMaps = jObj["LabelCombinationMaps"]?.ToObject<List<Dictionary<string, int>>>()
            ?? new List<Dictionary<string, int>>();

        // Deserialize inverse label maps (string keys back to int)
        _inverseLabelMaps = new List<Dictionary<int, int[]>>();
        var inverseMapsArr = jObj["InverseLabelMaps"]?.ToObject<List<Dictionary<string, int[]>>>();
        if (inverseMapsArr is not null)
        {
            foreach (var strMap in inverseMapsArr)
            {
                var intMap = new Dictionary<int, int[]>();
                foreach (var kvp in strMap)
                {
                    intMap[int.Parse(kvp.Key)] = kvp.Value;
                }
                _inverseLabelMaps.Add(intMap);
            }
        }
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.RAkEL;

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance()
    {
        return new RAkELClassifier<T>(LabelsetSize, NumLabelsets, null, Options, Regularization);
    }
}
