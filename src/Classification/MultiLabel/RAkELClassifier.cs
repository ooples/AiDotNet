using System.Text;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Matrix&lt;double&gt;(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 }, { 0.0, 0.0 } });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for rakelclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Matrix&lt;double&gt;&gt;()
///     .ConfigureModel(new RAkELClassifier&lt;double&gt;(labelsetSize: 3));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Random k-Labelsets for Multilabel Classification", "https://doi.org/10.1109/TKDE.2010.164", Year = 2011, Authors = "Grigorios Tsoumakas, Ioannis Katakis, Ioannis Vlahavas")]
public partial class RAkELClassifier<T> : MultiLabelClassifierBase<T>
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

        var engine = Engine;
        var augmented = AugmentWithBias(features);
        var labelTensor = Tensor<T>.FromMatrix(labels);

        foreach (var labelset in _labelsets)
        {
            // Create label combinations for this labelset
            var combinationMap = new Dictionary<string, int>();
            var inverseCombinationMap = new Dictionary<int, int[]>();
            var trueClasses = new int[numSamples];

            // Extract labels for this labelset (label value > 0.5) and create unique combinations
            var bits = numSamples == 0 || labelset.Length == 0
                ? Array.Empty<int>()
                : ZeroOneToInt(engine.TensorGreaterThan(
                    engine.TensorIndexSelect(labelTensor, new Tensor<int>(labelset, new[] { labelset.Length }), 1),
                    NumOps.FromDouble(0.5))).ToArray();
            for (int i = 0; i < numSamples; i++)
            {
                var combination = new int[labelset.Length];
                Array.Copy(bits, i * labelset.Length, combination, 0, labelset.Length);

                string key = string.Join(",", combination);
                if (!combinationMap.ContainsKey(key))
                {
                    int classIdx = combinationMap.Count;
                    combinationMap[key] = classIdx;
                    inverseCombinationMap[classIdx] = combination;
                }

                trueClasses[i] = combinationMap[key];
            }

            _labelCombinationMaps.Add(combinationMap);
            _inverseLabelMaps.Add(inverseCombinationMap);

            int numClasses = combinationMap.Count;

            // Weights for this labelset classifier, kept transposed as [numClasses, NumFeatures + 1] (+1 for bias)
            var weightsT = new Tensor<T>(new[] { numClasses, NumFeatures + 1 });
            var targets = numSamples == 0 || numClasses == 0
                ? new Tensor<T>(new[] { numSamples, numClasses })
                : engine.TensorOneHot<T>(new Tensor<int>(trueClasses, new[] { numSamples }), numClasses);
            T stepScale = NumOps.FromDouble(learningRate / numSamples);

            // Train with softmax cross-entropy via per-sample gradient descent
            for (int iter = 0; iter < numIterations && numClasses > 0; iter++)
            {
                for (int i = 0; i < numSamples; i++)
                {
                    weightsT = SoftmaxStep(weightsT, augmented.Slice(0, i, i + 1), targets.Slice(0, i, i + 1), stepScale);
                }
            }

            _labelsetWeights.Add(engine.TensorTranspose(weightsT).ToMatrix());
        }
    }

    /// <summary>
    /// [x | 1]: the first NumFeatures columns of <paramref name="x"/> followed by a bias column.
    /// </summary>
    private Tensor<T> AugmentWithBias(Matrix<T> x)
    {
        var features = Tensor<T>.FromMatrix(x);
        if (x.Columns != NumFeatures)
        {
            features = features.Slice(1, 0, NumFeatures);
        }

        var ones = new Tensor<T>(new[] { x.Rows, 1 });
        Engine.TensorFill(ones, NumOps.One);
        return Engine.TensorConcatenate(new[] { features, ones }, 1);
    }

    /// <summary>
    /// One cross-entropy gradient step on a single sample: W -= scale * (softmax(W x) - target) x^T.
    /// </summary>
    /// <param name="weightsT">The [classes, features + 1] weights.</param>
    /// <param name="sample">The [1, features + 1] augmented sample.</param>
    /// <param name="target">The [1, classes] one-hot target.</param>
    /// <param name="scale">learningRate / numSamples.</param>
    /// <returns>The updated weights.</returns>
    private Tensor<T> SoftmaxStep(Tensor<T> weightsT, Tensor<T> sample, Tensor<T> target, T scale)
    {
        var engine = Engine;
        var logits = engine.TensorMatMul(sample, engine.TensorTranspose(weightsT));
        var residual = engine.TensorSubtract(engine.Softmax(logits, 1), target);
        var update = engine.TensorMatMul(engine.TensorTranspose(residual), sample);
        return engine.TensorSubtract(weightsT, engine.TensorMultiplyScalar(update, scale));
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
        var labelCounts = new int[NumLabels];
        if (numSamples == 0 || NumLabels == 0)
        {
            return new Matrix<T>(numSamples, NumLabels);
        }

        var engine = Engine;
        var augmented = AugmentWithBias(input);
        var labelVotes = new Tensor<T>(new[] { numSamples, NumLabels });

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

            if (numClasses == 0 || labelset.Length == 0)
            {
                continue;
            }

            // Class probabilities: softmax of [x | 1] W
            var probs = engine.Softmax(engine.TensorMatMul(augmented, Tensor<T>.FromMatrix(weights)), 1);

            // Vote for each label: probability-weighted label values of every class combination
            var votes = engine.TensorMatMul(probs, CombinationMatrix(inverseLabelMap, numClasses, labelset.Length));
            labelVotes = engine.TensorScatterAdd(labelVotes,
                new Tensor<int>(labelset, new[] { labelset.Length }), votes, 1);
        }

        // Normalize votes to get probabilities (labels in no labelset stay 0), clamped to [0, 1]
        var counts = engine.TensorGather(engine.TensorArange<T>(0, _labelsets.Count + 1),
            new Tensor<int>(labelCounts, new[] { NumLabels }), 0);
        var denominators = engine.TensorClampMin(counts, NumOps.One).Reshape(1, NumLabels);
        var normalized = engine.TensorDivide(labelVotes, denominators);
        return engine.TensorClamp(normalized, NumOps.Zero, NumOps.One).ToMatrix();
    }

    /// <summary>
    /// The [classes, labelsetSize] 0/1 label values of every class combination.
    /// </summary>
    private Tensor<T> CombinationMatrix(Dictionary<int, int[]> inverseLabelMap, int numClasses, int labelsetSize)
    {
        var flags = new int[numClasses * labelsetSize];
        for (int c = 0; c < numClasses; c++)
        {
            Array.Copy(inverseLabelMap[c], 0, flags, c * labelsetSize, labelsetSize);
        }

        return Engine.TensorGather(Engine.TensorArange<T>(0, 2), new Tensor<int>(flags, new[] { flags.Length }), 0)
            .Reshape(numClasses, labelsetSize);
    }

    /// <inheritdoc />
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "labelset-weights",
            new AiDotNet.Models.Parameters.MatrixCollectionParameterSource<T>(
                () => _labelsetWeights));
    }

    /// <inheritdoc />
}
