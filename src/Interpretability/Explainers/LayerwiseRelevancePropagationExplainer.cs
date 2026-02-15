using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Layer-wise Relevance Propagation (LRP) explainer for neural networks.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LRP explains neural network predictions by propagating
/// "relevance" scores backward from the output to the input.
///
/// Key idea:
/// - Start with the prediction score as the total "relevance"
/// - At each layer, distribute relevance to neurons in the previous layer
/// - Continue until you reach the input features
/// - Each input feature gets a relevance score
///
/// Conservation principle:
/// - The total relevance is conserved at each layer
/// - Relevance in = Relevance out (like energy conservation)
/// - This means attributions sum to the prediction value
///
/// LRP rules (how to distribute relevance):
/// - <b>LRP-0 (Basic)</b>: Distribute proportional to contribution
/// - <b>LRP-ε (Epsilon)</b>: Adds stability with small epsilon
/// - <b>LRP-γ (Gamma)</b>: Emphasizes positive contributions
/// - <b>LRP-αβ</b>: Separately handles positive and negative contributions
///
/// When to use LRP:
/// - You want to understand which inputs were "responsible" for the output
/// - You need attributions that sum exactly to the prediction
/// - You want a principled way to handle different layer types
///
/// Example: For image classification:
/// - LRP shows which pixels contributed positively or negatively
/// - Red = positive relevance (supported the prediction)
/// - Blue = negative relevance (contradicted the prediction)
/// </para>
/// </remarks>
public class LayerwiseRelevancePropagationExplainer<T> : ILocalExplainer<T, LRPExplanation<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>[]>? _getLayerActivations;
    private readonly Func<int, Matrix<T>>? _getLayerWeights;
    private readonly int _numFeatures;
    private readonly int[]? _layerSizes;
    private readonly string[]? _featureNames;
    private readonly LRPRule _rule;
    private readonly double _epsilon;
    private readonly double _gamma;

    /// <inheritdoc/>
    public string MethodName => "LRP";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <summary>
    /// Initializes a new LRP explainer.
    /// </summary>
    /// <param name="predictFunction">Function that makes predictions.</param>
    /// <param name="getLayerActivations">Function that returns activations for all layers.
    /// Returns array where [0] is input, [L] is output.</param>
    /// <param name="getLayerWeights">Function that returns weight matrix for a layer index.
    /// Returns matrix of shape [input_neurons, output_neurons].</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="layerSizes">Sizes of each layer including input and output.</param>
    /// <param name="featureNames">Optional names for features.</param>
    /// <param name="rule">LRP rule to use (default: Epsilon).</param>
    /// <param name="epsilon">Epsilon for LRP-ε rule (default: 1e-4).</param>
    /// <param name="gamma">Gamma for LRP-γ rule (default: 0.25).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>rule</b>: Epsilon rule is most commonly used. Use Gamma rule to suppress noise.
    /// - <b>epsilon</b>: Small value that prevents division by zero. Don't set too high.
    /// - <b>gamma</b>: Higher values emphasize positive contributions more.
    /// </para>
    /// </remarks>
    public LayerwiseRelevancePropagationExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>[]>? getLayerActivations = null,
        Func<int, Matrix<T>>? getLayerWeights = null,
        int numFeatures = 0,
        int[]? layerSizes = null,
        string[]? featureNames = null,
        LRPRule rule = LRPRule.Epsilon,
        double epsilon = 1e-4,
        double gamma = 0.25)
    {
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
        _getLayerActivations = getLayerActivations;
        _getLayerWeights = getLayerWeights;
        _numFeatures = numFeatures;
        _layerSizes = layerSizes;
        _featureNames = featureNames;
        _rule = rule;
        _epsilon = epsilon;
        _gamma = gamma;
    }

    /// <summary>
    /// Computes LRP relevance scores for an input.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <returns>LRP explanation with relevance scores.</returns>
    public LRPExplanation<T> Explain(Vector<T> instance)
    {
        return Explain(instance, outputIndex: -1);
    }

    /// <summary>
    /// Computes LRP relevance scores for a specific output.
    /// </summary>
    /// <param name="instance">The input instance to explain.</param>
    /// <param name="outputIndex">Index of the output to explain (-1 for highest scoring class).</param>
    /// <returns>LRP explanation with relevance scores.</returns>
    public LRPExplanation<T> Explain(Vector<T> instance, int outputIndex)
    {
        int numFeatures = instance.Length;

        // Get prediction
        var prediction = _predictFunction(instance);

        // Determine target output
        if (outputIndex < 0)
        {
            outputIndex = 0;
            double maxVal = double.MinValue;
            for (int i = 0; i < prediction.Length; i++)
            {
                double val = NumOps.ToDouble(prediction[i]);
                if (val > maxVal)
                {
                    maxVal = val;
                    outputIndex = i;
                }
            }
        }

        double outputValue = outputIndex < prediction.Length
            ? NumOps.ToDouble(prediction[outputIndex])
            : 0;

        T[] relevanceScores;

        if (_getLayerActivations != null && _getLayerWeights != null && _layerSizes != null)
        {
            // Full LRP with layer access
            relevanceScores = ComputeFullLRP(instance, outputIndex);
        }
        else
        {
            // Approximate LRP using gradient-weighted input
            relevanceScores = ComputeApproximateLRP(instance, outputIndex);
        }

        // Compute positive and negative relevance
        double posRelevance = 0;
        double negRelevance = 0;
        for (int i = 0; i < numFeatures; i++)
        {
            double rel = NumOps.ToDouble(relevanceScores[i]);
            if (rel > 0) posRelevance += rel;
            else negRelevance += rel;
        }

        return new LRPExplanation<T>
        {
            RelevanceScores = new Vector<T>(relevanceScores),
            Input = instance,
            Prediction = prediction,
            OutputIndex = outputIndex,
            OutputValue = NumOps.FromDouble(outputValue),
            TotalPositiveRelevance = NumOps.FromDouble(posRelevance),
            TotalNegativeRelevance = NumOps.FromDouble(negRelevance),
            FeatureNames = _featureNames ?? Enumerable.Range(0, numFeatures).Select(i => $"Feature {i}").ToArray(),
            Rule = _rule
        };
    }

    /// <inheritdoc/>
    public LRPExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var explanations = new LRPExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            explanations[i] = Explain(instances.GetRow(i));
        }
        return explanations;
    }

    /// <summary>
    /// Computes full LRP using layer activations and weights.
    /// </summary>
    private T[] ComputeFullLRP(Vector<T> input, int outputIndex)
    {
        // Get activations
        var activations = _getLayerActivations!(input);
        int numLayers = activations.Length;

        // Initialize relevance at output layer
        var relevance = new double[activations[numLayers - 1].Length];
        relevance[outputIndex] = NumOps.ToDouble(activations[numLayers - 1][outputIndex]);

        // Propagate backward through layers
        for (int layer = numLayers - 2; layer >= 0; layer--)
        {
            var weights = _getLayerWeights!(layer);
            var lowerActivations = activations[layer];

            var newRelevance = new double[lowerActivations.Length];

            switch (_rule)
            {
                case LRPRule.Basic:
                    PropagateBasicRule(lowerActivations, weights, relevance, newRelevance);
                    break;
                case LRPRule.Epsilon:
                    PropagateEpsilonRule(lowerActivations, weights, relevance, newRelevance);
                    break;
                case LRPRule.Gamma:
                    PropagateGammaRule(lowerActivations, weights, relevance, newRelevance);
                    break;
                case LRPRule.AlphaBeta:
                    PropagateAlphaBetaRule(lowerActivations, weights, relevance, newRelevance);
                    break;
            }

            relevance = newRelevance;
        }

        // Convert to T[]
        var result = new T[relevance.Length];
        for (int i = 0; i < relevance.Length; i++)
        {
            result[i] = NumOps.FromDouble(relevance[i]);
        }

        return result;
    }

    /// <summary>
    /// LRP-0 Basic rule.
    /// </summary>
    private void PropagateBasicRule(Vector<T> lowerAct, Matrix<T> weights, double[] upperRel, double[] lowerRel)
    {
        int lowerSize = lowerAct.Length;
        int upperSize = upperRel.Length;

        for (int j = 0; j < lowerSize; j++)
        {
            double sum = 0;
            double aj = NumOps.ToDouble(lowerAct[j]);

            for (int k = 0; k < upperSize; k++)
            {
                // Compute z_jk = a_j * w_jk
                double wjk = NumOps.ToDouble(weights[j, k]);
                double zjk = aj * wjk;

                // Compute denominator (sum of z for all inputs to neuron k)
                double zk = 0;
                for (int i = 0; i < lowerSize; i++)
                {
                    zk += NumOps.ToDouble(lowerAct[i]) * NumOps.ToDouble(weights[i, k]);
                }

                if (Math.Abs(zk) > 1e-10)
                {
                    sum += zjk / zk * upperRel[k];
                }
            }

            lowerRel[j] = sum;
        }
    }

    /// <summary>
    /// LRP-ε Epsilon rule for numerical stability.
    /// </summary>
    private void PropagateEpsilonRule(Vector<T> lowerAct, Matrix<T> weights, double[] upperRel, double[] lowerRel)
    {
        int lowerSize = lowerAct.Length;
        int upperSize = upperRel.Length;

        for (int j = 0; j < lowerSize; j++)
        {
            double sum = 0;
            double aj = NumOps.ToDouble(lowerAct[j]);

            for (int k = 0; k < upperSize; k++)
            {
                double wjk = NumOps.ToDouble(weights[j, k]);
                double zjk = aj * wjk;

                double zk = 0;
                for (int i = 0; i < lowerSize; i++)
                {
                    zk += NumOps.ToDouble(lowerAct[i]) * NumOps.ToDouble(weights[i, k]);
                }

                // Add epsilon with sign of zk
                double signZk = zk >= 0 ? 1 : -1;
                zk = zk + signZk * _epsilon;

                sum += zjk / zk * upperRel[k];
            }

            lowerRel[j] = sum;
        }
    }

    /// <summary>
    /// LRP-γ Gamma rule emphasizing positive contributions.
    /// </summary>
    private void PropagateGammaRule(Vector<T> lowerAct, Matrix<T> weights, double[] upperRel, double[] lowerRel)
    {
        int lowerSize = lowerAct.Length;
        int upperSize = upperRel.Length;

        for (int j = 0; j < lowerSize; j++)
        {
            double sum = 0;
            double aj = NumOps.ToDouble(lowerAct[j]);

            for (int k = 0; k < upperSize; k++)
            {
                double wjk = NumOps.ToDouble(weights[j, k]);

                // Modify weights: w+ = max(0, w), increase positive weights
                double wjkPlus = Math.Max(0, wjk);
                double wjkMod = wjk + _gamma * wjkPlus;

                double zjk = aj * wjkMod;

                double zk = 0;
                for (int i = 0; i < lowerSize; i++)
                {
                    double wik = NumOps.ToDouble(weights[i, k]);
                    double wikPlus = Math.Max(0, wik);
                    double wikMod = wik + _gamma * wikPlus;
                    zk += NumOps.ToDouble(lowerAct[i]) * wikMod;
                }

                if (Math.Abs(zk) > 1e-10)
                {
                    sum += zjk / zk * upperRel[k];
                }
            }

            lowerRel[j] = sum;
        }
    }

    /// <summary>
    /// LRP-αβ rule separating positive and negative contributions.
    /// </summary>
    private void PropagateAlphaBetaRule(Vector<T> lowerAct, Matrix<T> weights, double[] upperRel, double[] lowerRel, double alpha = 2, double beta = 1)
    {
        int lowerSize = lowerAct.Length;
        int upperSize = upperRel.Length;

        for (int j = 0; j < lowerSize; j++)
        {
            double sum = 0;
            double aj = NumOps.ToDouble(lowerAct[j]);

            for (int k = 0; k < upperSize; k++)
            {
                double wjk = NumOps.ToDouble(weights[j, k]);

                // Positive contribution
                double zjkPlus = aj * Math.Max(0, wjk);
                double zkPlus = 0;
                for (int i = 0; i < lowerSize; i++)
                {
                    zkPlus += NumOps.ToDouble(lowerAct[i]) * Math.Max(0, NumOps.ToDouble(weights[i, k]));
                }

                // Negative contribution
                double zjkMinus = aj * Math.Min(0, wjk);
                double zkMinus = 0;
                for (int i = 0; i < lowerSize; i++)
                {
                    zkMinus += NumOps.ToDouble(lowerAct[i]) * Math.Min(0, NumOps.ToDouble(weights[i, k]));
                }

                double posContrib = Math.Abs(zkPlus) > 1e-10 ? alpha * zjkPlus / zkPlus : 0;
                double negContrib = Math.Abs(zkMinus) > 1e-10 ? beta * zjkMinus / zkMinus : 0;

                sum += (posContrib - negContrib) * upperRel[k];
            }

            lowerRel[j] = sum;
        }
    }

    /// <summary>
    /// Computes approximate LRP using gradient × input (a simple approximation).
    /// </summary>
    private T[] ComputeApproximateLRP(Vector<T> input, int outputIndex)
    {
        int numFeatures = input.Length;
        var relevance = new T[numFeatures];

        // Compute gradients
        var gradients = ComputeNumericalGradient(input, outputIndex);

        // Relevance ≈ gradient × input (a simple approximation to LRP)
        for (int i = 0; i < numFeatures; i++)
        {
            double grad = NumOps.ToDouble(gradients[i]);
            double inputVal = NumOps.ToDouble(input[i]);
            relevance[i] = NumOps.FromDouble(grad * inputVal);
        }

        return relevance;
    }

    /// <summary>
    /// Computes numerical gradient.
    /// </summary>
    private Vector<T> ComputeNumericalGradient(Vector<T> input, int outputIndex)
    {
        int n = input.Length;
        var gradient = new T[n];
        double epsilon = 1e-4;

        for (int i = 0; i < n; i++)
        {
            var inputPlus = new T[n];
            var inputMinus = new T[n];
            for (int j = 0; j < n; j++)
            {
                inputPlus[j] = input[j];
                inputMinus[j] = input[j];
            }

            inputPlus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) + epsilon);
            inputMinus[i] = NumOps.FromDouble(NumOps.ToDouble(input[i]) - epsilon);

            var predPlus = _predictFunction(new Vector<T>(inputPlus));
            var predMinus = _predictFunction(new Vector<T>(inputMinus));

            double valPlus = outputIndex < predPlus.Length ? NumOps.ToDouble(predPlus[outputIndex]) : 0;
            double valMinus = outputIndex < predMinus.Length ? NumOps.ToDouble(predMinus[outputIndex]) : 0;

            gradient[i] = NumOps.FromDouble((valPlus - valMinus) / (2 * epsilon));
        }

        return new Vector<T>(gradient);
    }
}

/// <summary>
/// LRP propagation rules.
/// </summary>
public enum LRPRule
{
    /// <summary>
    /// Basic LRP-0 rule.
    /// </summary>
    Basic,

    /// <summary>
    /// LRP-ε rule with stabilizing epsilon.
    /// </summary>
    Epsilon,

    /// <summary>
    /// LRP-γ rule emphasizing positive contributions.
    /// </summary>
    Gamma,

    /// <summary>
    /// LRP-αβ rule separating positive and negative.
    /// </summary>
    AlphaBeta
}

/// <summary>
/// Represents the result of an LRP analysis.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LRPExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the relevance scores for each input feature.
    /// </summary>
    public Vector<T> RelevanceScores { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the input instance.
    /// </summary>
    public Vector<T> Input { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the model prediction.
    /// </summary>
    public Vector<T> Prediction { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets or sets the output index that was explained.
    /// </summary>
    public int OutputIndex { get; set; }

    /// <summary>
    /// Gets or sets the output value being explained.
    /// </summary>
    public T OutputValue { get; set; } = default!;

    /// <summary>
    /// Gets or sets the sum of positive relevance scores.
    /// </summary>
    public T TotalPositiveRelevance { get; set; } = default!;

    /// <summary>
    /// Gets or sets the sum of negative relevance scores.
    /// </summary>
    public T TotalNegativeRelevance { get; set; } = default!;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[] FeatureNames { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets the LRP rule used.
    /// </summary>
    public LRPRule Rule { get; set; }

    /// <summary>
    /// Gets relevance scores sorted by absolute value (most relevant first).
    /// </summary>
    public List<(string name, T relevance)> GetSortedRelevance()
    {
        var result = new List<(string, T)>();
        for (int i = 0; i < RelevanceScores.Length; i++)
        {
            result.Add((FeatureNames[i], RelevanceScores[i]));
        }
        return result.OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Item2))).ToList();
    }

    /// <summary>
    /// Gets features with positive relevance (supporting the prediction).
    /// </summary>
    public List<(string name, T relevance)> GetPositiveRelevance()
    {
        return GetSortedRelevance()
            .Where(x => NumOps.ToDouble(x.relevance) > 0)
            .ToList();
    }

    /// <summary>
    /// Gets features with negative relevance (contradicting the prediction).
    /// </summary>
    public List<(string name, T relevance)> GetNegativeRelevance()
    {
        return GetSortedRelevance()
            .Where(x => NumOps.ToDouble(x.relevance) < 0)
            .OrderBy(x => NumOps.ToDouble(x.relevance))
            .ToList();
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        double totalRel = NumOps.ToDouble(TotalPositiveRelevance) + NumOps.ToDouble(TotalNegativeRelevance);

        var lines = new List<string>
        {
            $"Layer-wise Relevance Propagation (Rule: {Rule}):",
            $"  Output value: {NumOps.ToDouble(OutputValue):F4}",
            $"  Total relevance: {totalRel:F4}",
            $"  Positive relevance: {NumOps.ToDouble(TotalPositiveRelevance):F4}",
            $"  Negative relevance: {NumOps.ToDouble(TotalNegativeRelevance):F4}",
            "",
            "Top Positive (supporting prediction):"
        };

        var positive = GetPositiveRelevance().Take(5);
        foreach (var (name, rel) in positive)
        {
            lines.Add($"  {name}: +{NumOps.ToDouble(rel):F4}");
        }

        lines.Add("");
        lines.Add("Top Negative (contradicting prediction):");

        var negative = GetNegativeRelevance().Take(5);
        foreach (var (name, rel) in negative)
        {
            lines.Add($"  {name}: {NumOps.ToDouble(rel):F4}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
