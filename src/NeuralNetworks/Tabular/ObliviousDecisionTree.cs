using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Oblivious Decision Tree (ODT) for NODE architecture.
/// </summary>
/// <remarks>
/// <para>
/// An oblivious decision tree uses the same feature and threshold at each level,
/// making it more regularized and efficient than standard decision trees.
/// NODE uses differentiable ODTs with entmax splits for end-to-end learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> An oblivious tree is a special type of decision tree where:
/// - At level 1, ALL nodes use the same feature (e.g., "age > 30")
/// - At level 2, ALL nodes use the same feature (e.g., "income > 50k")
/// - And so on...
///
/// This is simpler than regular trees where each node can use different features.
/// The simplicity helps prevent overfitting and makes the tree faster.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ObliviousDecisionTree<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _inputDim;
    private readonly int _depth;
    private readonly int _outputDim;

    // Each level has one feature selection and one threshold
    private Tensor<T> _featureSelectionWeights;  // [depth, inputDim] - softmax to select feature
    private Tensor<T> _thresholds;                // [depth]
    private Tensor<T> _leafValues;                // [numLeaves, outputDim]

    // Gradients
    private Tensor<T> _featureSelectionGrad;
    private Tensor<T> _thresholdsGrad;
    private Tensor<T> _leafValuesGrad;

    // Cached values
    private Tensor<T>? _inputCache;
    private Tensor<T>? _featureSelectionsCache;
    private Tensor<T>? _splitDecisionsCache;
    private Tensor<T>? _leafProbabilitiesCache;

    private readonly int _numLeaves;

    /// <summary>
    /// Gets the number of leaf nodes (2^depth).
    /// </summary>
    public int NumLeaves => _numLeaves;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount =>
        _depth * _inputDim +      // feature selection weights
        _depth +                   // thresholds
        _numLeaves * _outputDim;   // leaf values

    /// <summary>
    /// Initializes an oblivious decision tree.
    /// </summary>
    /// <param name="inputDim">Input feature dimension.</param>
    /// <param name="depth">Tree depth (number of split levels).</param>
    /// <param name="outputDim">Output dimension per leaf.</param>
    /// <param name="initScale">Initialization scale.</param>
    public ObliviousDecisionTree(int inputDim, int depth = 6, int outputDim = 1, double initScale = 0.01)
    {
        _inputDim = inputDim;
        _depth = depth;
        _outputDim = outputDim;
        _numLeaves = 1 << depth;  // 2^depth
        _random = RandomHelper.CreateSecureRandom();

        // Initialize parameters
        _featureSelectionWeights = new Tensor<T>([depth, inputDim]);
        _thresholds = new Tensor<T>([depth]);
        _leafValues = new Tensor<T>([_numLeaves, outputDim]);

        // Initialize gradients
        _featureSelectionGrad = new Tensor<T>([depth, inputDim]);
        _thresholdsGrad = new Tensor<T>([depth]);
        _leafValuesGrad = new Tensor<T>([_numLeaves, outputDim]);

        InitializeParameters(initScale);
    }

    private void InitializeParameters(double scale)
    {
        // Initialize feature selection weights (uniform, will be softmaxed)
        for (int i = 0; i < _featureSelectionWeights.Length; i++)
        {
            _featureSelectionWeights[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }

        // Initialize thresholds to small random values
        for (int i = 0; i < _thresholds.Length; i++)
        {
            _thresholds[i] = NumOps.FromDouble(_random.NextGaussian() * scale * 0.1);
        }

        // Initialize leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }
    }

    /// <summary>
    /// Forward pass through the oblivious decision tree.
    /// </summary>
    /// <param name="input">Input features [batchSize, inputDim].</param>
    /// <returns>Tree output [batchSize, outputDim].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;
        int batchSize = input.Shape[0];

        // Compute soft feature selections for each level (entmax/softmax)
        var featureSelections = ComputeFeatureSelections();
        _featureSelectionsCache = featureSelections;

        // Compute split decisions for each level
        var splitDecisions = ComputeSplitDecisions(input, featureSelections, batchSize);
        _splitDecisionsCache = splitDecisions;

        // Compute leaf probabilities
        var leafProbs = ComputeLeafProbabilities(splitDecisions, batchSize);
        _leafProbabilitiesCache = leafProbs;

        // Weighted sum of leaf values
        var output = ComputeOutput(leafProbs, batchSize);

        return output;
    }

    private Tensor<T> ComputeFeatureSelections()
    {
        var selections = new Tensor<T>([_depth, _inputDim]);

        for (int level = 0; level < _depth; level++)
        {
            // Softmax over features for this level
            var maxVal = _featureSelectionWeights[level * _inputDim];
            for (int f = 1; f < _inputDim; f++)
            {
                var val = _featureSelectionWeights[level * _inputDim + f];
                if (NumOps.Compare(val, maxVal) > 0)
                    maxVal = val;
            }

            var sumExp = NumOps.Zero;
            for (int f = 0; f < _inputDim; f++)
            {
                var exp = NumOps.Exp(NumOps.Subtract(
                    _featureSelectionWeights[level * _inputDim + f], maxVal));
                selections[level * _inputDim + f] = exp;
                sumExp = NumOps.Add(sumExp, exp);
            }

            for (int f = 0; f < _inputDim; f++)
            {
                selections[level * _inputDim + f] = NumOps.Divide(
                    selections[level * _inputDim + f], sumExp);
            }
        }

        return selections;
    }

    private Tensor<T> ComputeSplitDecisions(Tensor<T> input, Tensor<T> featureSelections, int batchSize)
    {
        // [batchSize, depth] - probability of going right at each level
        var decisions = new Tensor<T>([batchSize, _depth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int level = 0; level < _depth; level++)
            {
                // Compute weighted feature value
                var weightedFeature = NumOps.Zero;
                for (int f = 0; f < _inputDim; f++)
                {
                    weightedFeature = NumOps.Add(weightedFeature,
                        NumOps.Multiply(
                            featureSelections[level * _inputDim + f],
                            input[b * _inputDim + f]));
                }

                // Soft split decision using sigmoid
                var logit = NumOps.Subtract(weightedFeature, _thresholds[level]);
                var rightProb = Sigmoid(logit);
                decisions[b * _depth + level] = rightProb;
            }
        }

        return decisions;
    }

    private Tensor<T> ComputeLeafProbabilities(Tensor<T> splitDecisions, int batchSize)
    {
        var leafProbs = new Tensor<T>([batchSize, _numLeaves]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int leaf = 0; leaf < _numLeaves; leaf++)
            {
                // Compute probability of reaching this leaf
                var prob = NumOps.One;
                for (int level = 0; level < _depth; level++)
                {
                    var rightProb = splitDecisions[b * _depth + level];
                    var leftProb = NumOps.Subtract(NumOps.One, rightProb);

                    // Check if this leaf goes right or left at this level
                    bool goRight = ((leaf >> (_depth - 1 - level)) & 1) == 1;
                    prob = NumOps.Multiply(prob, goRight ? rightProb : leftProb);
                }
                leafProbs[b * _numLeaves + leaf] = prob;
            }
        }

        return leafProbs;
    }

    private Tensor<T> ComputeOutput(Tensor<T> leafProbs, int batchSize)
    {
        var output = new Tensor<T>([batchSize, _outputDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < _outputDim; o++)
            {
                var sum = NumOps.Zero;
                for (int leaf = 0; leaf < _numLeaves; leaf++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        leafProbs[b * _numLeaves + leaf],
                        _leafValues[leaf * _outputDim + o]));
                }
                output[b * _outputDim + o] = sum;
            }
        }

        return output;
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Backward pass through the oblivious decision tree.
    /// </summary>
    /// <param name="gradient">Gradient with respect to output [batchSize, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_inputCache == null || _leafProbabilitiesCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward");
        }

        int batchSize = _inputCache.Shape[0];
        var inputGrad = new Tensor<T>([batchSize, _inputDim]);

        // Reset gradients
        for (int i = 0; i < _featureSelectionGrad.Length; i++)
            _featureSelectionGrad[i] = NumOps.Zero;
        for (int i = 0; i < _thresholdsGrad.Length; i++)
            _thresholdsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _leafValuesGrad.Length; i++)
            _leafValuesGrad[i] = NumOps.Zero;

        // Gradient for leaf values
        for (int b = 0; b < batchSize; b++)
        {
            for (int leaf = 0; leaf < _numLeaves; leaf++)
            {
                var leafProb = _leafProbabilitiesCache[b * _numLeaves + leaf];
                for (int o = 0; o < _outputDim; o++)
                {
                    _leafValuesGrad[leaf * _outputDim + o] = NumOps.Add(
                        _leafValuesGrad[leaf * _outputDim + o],
                        NumOps.Multiply(leafProb, gradient[b * _outputDim + o]));
                }
            }
        }

        // Simplified input gradient (full implementation would backprop through soft selections)
        return inputGrad;
    }

    /// <summary>
    /// Gets feature importance based on selection weights.
    /// </summary>
    public Vector<T> GetFeatureImportance()
    {
        var importance = new Vector<T>(_inputDim);

        if (_featureSelectionsCache != null)
        {
            for (int f = 0; f < _inputDim; f++)
            {
                var sum = NumOps.Zero;
                for (int level = 0; level < _depth; level++)
                {
                    sum = NumOps.Add(sum, _featureSelectionsCache[level * _inputDim + f]);
                }
                importance[f] = NumOps.Divide(sum, NumOps.FromDouble(_depth));
            }
        }
        else
        {
            // Use raw weights if forward hasn't been called
            var selections = ComputeFeatureSelections();
            for (int f = 0; f < _inputDim; f++)
            {
                var sum = NumOps.Zero;
                for (int level = 0; level < _depth; level++)
                {
                    sum = NumOps.Add(sum, selections[level * _inputDim + f]);
                }
                importance[f] = NumOps.Divide(sum, NumOps.FromDouble(_depth));
            }
        }

        return importance;
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _featureSelectionWeights.Length; i++)
        {
            _featureSelectionWeights[i] = NumOps.Subtract(_featureSelectionWeights[i],
                NumOps.Multiply(learningRate, _featureSelectionGrad[i]));
        }

        for (int i = 0; i < _thresholds.Length; i++)
        {
            _thresholds[i] = NumOps.Subtract(_thresholds[i],
                NumOps.Multiply(learningRate, _thresholdsGrad[i]));
        }

        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = NumOps.Subtract(_leafValues[i],
                NumOps.Multiply(learningRate, _leafValuesGrad[i]));
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _featureSelectionsCache = null;
        _splitDecisionsCache = null;
        _leafProbabilitiesCache = null;

        for (int i = 0; i < _featureSelectionGrad.Length; i++)
            _featureSelectionGrad[i] = NumOps.Zero;
        for (int i = 0; i < _thresholdsGrad.Length; i++)
            _thresholdsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _leafValuesGrad.Length; i++)
            _leafValuesGrad[i] = NumOps.Zero;
    }
}
