using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Necks;

/// <summary>
/// Bidirectional Feature Pyramid Network (BiFPN) with weighted feature fusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BiFPN improves upon FPN and PANet by using learnable
/// weights for feature fusion. Instead of simply adding features, it learns how
/// much each input feature should contribute to the fused output.</para>
///
/// <para>Key features:
/// - Bidirectional (top-down + bottom-up) feature flow
/// - Learnable weights for weighted feature fusion
/// - Fast normalized fusion with softmax
/// - Used in EfficientDet for state-of-the-art detection
/// </para>
///
/// <para>Reference: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020</para>
/// </remarks>
public class BiFPN<T> : NeckBase<T>
{
    private readonly int _outputChannels;
    private readonly int[] _inputChannels;
    private readonly int _numLevels;
    private readonly int _numRepeats;
    private readonly double _epsilon;

    // Lateral connections
    private readonly List<Tensor<T>> _lateralWeights;
    private readonly List<Tensor<T>> _lateralBiases;

    // Top-down pathway
    private readonly List<List<Tensor<T>>> _topDownFusionWeights;
    private readonly List<Tensor<T>> _topDownConvWeights;
    private readonly List<Tensor<T>> _topDownConvBiases;

    // Bottom-up pathway
    private readonly List<List<Tensor<T>>> _bottomUpFusionWeights;
    private readonly List<Tensor<T>> _bottomUpConvWeights;
    private readonly List<Tensor<T>> _bottomUpConvBiases;

    /// <inheritdoc/>
    public override string Name => $"BiFPN-{_numRepeats}x";

    /// <inheritdoc/>
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc/>
    public override int NumLevels => _numLevels;

    /// <summary>
    /// Creates a new Bidirectional Feature Pyramid Network.
    /// </summary>
    /// <param name="inputChannels">Channel counts from backbone at each level.</param>
    /// <param name="outputChannels">Output channel count for all levels (default 256).</param>
    /// <param name="numRepeats">Number of BiFPN repeat blocks (default 3).</param>
    public BiFPN(int[] inputChannels, int outputChannels = 256, int numRepeats = 3)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _numLevels = inputChannels.Length;
        _numRepeats = numRepeats;
        _epsilon = 1e-4;

        _lateralWeights = new List<Tensor<T>>();
        _lateralBiases = new List<Tensor<T>>();
        _topDownFusionWeights = new List<List<Tensor<T>>>();
        _topDownConvWeights = new List<Tensor<T>>();
        _topDownConvBiases = new List<Tensor<T>>();
        _bottomUpFusionWeights = new List<List<Tensor<T>>>();
        _bottomUpConvWeights = new List<Tensor<T>>();
        _bottomUpConvBiases = new List<Tensor<T>>();

        InitializeWeightsForNetwork();
    }

    /// <summary>
    /// Creates BiFPN from a configuration object.
    /// </summary>
    /// <param name="config">Neck configuration.</param>
    public BiFPN(NeckConfig config)
        : this(config.InputChannels, config.OutputChannels)
    {
    }

    private void InitializeWeightsForNetwork()
    {
        var random = new Random(42);

        // Lateral connections (only for first repeat)
        for (int i = 0; i < _numLevels; i++)
        {
            var lateralWeight = new Tensor<T>(new[] { _outputChannels, _inputChannels[i] });
            var lateralBias = new Tensor<T>(new[] { _outputChannels });
            InitializeConvWeights(lateralWeight, random);
            _lateralWeights.Add(lateralWeight);
            _lateralBiases.Add(lateralBias);
        }

        // Top-down pathway weights (for each level except the deepest)
        for (int i = 0; i < _numLevels - 1; i++)
        {
            // Fusion weights for combining current level with upsampled deeper level
            // 2 inputs: current lateral, upsampled from below
            var fusionWeights = new List<Tensor<T>>();
            for (int j = 0; j < 2; j++)
            {
                var weight = new Tensor<T>(new[] { 1 });
                weight[0] = NumOps.FromDouble(1.0); // Initialize to equal contribution
                fusionWeights.Add(weight);
            }
            _topDownFusionWeights.Add(fusionWeights);

            var convWeight = new Tensor<T>(new[] { _outputChannels, _outputChannels });
            var convBias = new Tensor<T>(new[] { _outputChannels });
            InitializeConvWeights(convWeight, random);
            _topDownConvWeights.Add(convWeight);
            _topDownConvBiases.Add(convBias);
        }

        // Bottom-up pathway weights (for each level except the highest resolution)
        for (int i = 1; i < _numLevels; i++)
        {
            // Fusion weights for combining: top-down output, original lateral, downsampled from above
            // Intermediate levels have 3 inputs, first level in bottom-up has 2
            int numInputs = (i == 1) ? 2 : 3;
            var fusionWeights = new List<Tensor<T>>();
            for (int j = 0; j < numInputs; j++)
            {
                var weight = new Tensor<T>(new[] { 1 });
                weight[0] = NumOps.FromDouble(1.0);
                fusionWeights.Add(weight);
            }
            _bottomUpFusionWeights.Add(fusionWeights);

            var convWeight = new Tensor<T>(new[] { _outputChannels, _outputChannels });
            var convBias = new Tensor<T>(new[] { _outputChannels });
            InitializeConvWeights(convWeight, random);
            _bottomUpConvWeights.Add(convWeight);
            _bottomUpConvBiases.Add(convBias);
        }
    }

    private void InitializeConvWeights(Tensor<T> weights, Random random)
    {
        double scale = Math.Sqrt(2.0 / weights.Shape[1]);
        for (int i = 0; i < weights.Length; i++)
        {
            double val = random.NextDouble() * 2 * scale - scale;
            weights[i] = NumOps.FromDouble(val);
        }
    }

    /// <inheritdoc/>
    public override List<Tensor<T>> Forward(List<Tensor<T>> features)
    {
        ValidateFeatures(features, _inputChannels);

        // Apply lateral connections to get initial pyramid features
        var pyramidFeatures = new List<Tensor<T>>();
        for (int i = 0; i < _numLevels; i++)
        {
            var lateral = Conv1x1(features[i], _lateralWeights[i], _lateralBiases[i]);
            pyramidFeatures.Add(lateral);
        }

        // Apply BiFPN blocks
        var current = pyramidFeatures;
        for (int repeat = 0; repeat < _numRepeats; repeat++)
        {
            current = ApplyBiFPNBlock(current);
        }

        return current;
    }

    private List<Tensor<T>> ApplyBiFPNBlock(List<Tensor<T>> inputs)
    {
        // Phase 1: Top-down pathway
        var topDownOutputs = new List<Tensor<T>>(_numLevels);

        // Deepest level passes through unchanged
        topDownOutputs.Add(inputs[_numLevels - 1]);

        // Process from second-deepest to highest resolution
        for (int i = _numLevels - 2; i >= 0; i--)
        {
            var current = inputs[i];
            var deeper = topDownOutputs[0]; // Most recent output (deeper level)

            // Upsample deeper level
            var upsampled = Upsample2x(deeper);
            if (upsampled.Shape[2] != current.Shape[2] || upsampled.Shape[3] != current.Shape[3])
            {
                upsampled = ResizeToMatch(upsampled, current);
            }

            // Weighted fusion
            var fusionWeights = _topDownFusionWeights[_numLevels - 2 - i];
            var fused = FastNormalizedFusion(new List<Tensor<T>> { current, upsampled }, fusionWeights);

            // Apply convolution
            var convIdx = _numLevels - 2 - i;
            var output = Conv1x1(fused, _topDownConvWeights[convIdx], _topDownConvBiases[convIdx]);
            output = ApplySwish(output);

            topDownOutputs.Insert(0, output);
        }

        // Phase 2: Bottom-up pathway
        var bottomUpOutputs = new List<Tensor<T>>(_numLevels);

        // Highest resolution passes through
        bottomUpOutputs.Add(topDownOutputs[0]);

        // Process from second-highest to lowest resolution
        for (int i = 1; i < _numLevels; i++)
        {
            var topDownOutput = topDownOutputs[i];
            var shallower = bottomUpOutputs[i - 1]; // Previous bottom-up output

            // Downsample shallower level
            var downsampled = Downsample2x(shallower);
            if (downsampled.Shape[2] != topDownOutput.Shape[2] || downsampled.Shape[3] != topDownOutput.Shape[3])
            {
                downsampled = ResizeToMatch(downsampled, topDownOutput);
            }

            // Build input list for fusion
            var fusionInputs = new List<Tensor<T>> { topDownOutput, downsampled };

            // For intermediate levels, also include original input
            if (i < _numLevels - 1)
            {
                fusionInputs.Add(inputs[i]);
            }

            // Weighted fusion
            var fusionWeights = _bottomUpFusionWeights[i - 1];
            var fused = FastNormalizedFusion(fusionInputs, fusionWeights);

            // Apply convolution
            var output = Conv1x1(fused, _bottomUpConvWeights[i - 1], _bottomUpConvBiases[i - 1]);
            output = ApplySwish(output);

            bottomUpOutputs.Add(output);
        }

        return bottomUpOutputs;
    }

    /// <summary>
    /// Fast normalized fusion with learned weights.
    /// </summary>
    /// <remarks>
    /// Implements: output = sum(w_i * x_i) / (sum(w_i) + epsilon)
    /// where w_i = ReLU(w_i) to ensure non-negative weights.
    /// </remarks>
    private Tensor<T> FastNormalizedFusion(List<Tensor<T>> inputs, List<Tensor<T>> weights)
    {
        if (inputs.Count != weights.Count)
        {
            throw new ArgumentException("Number of inputs must match number of weights");
        }

        // Calculate normalized weights using ReLU
        var normalizedWeights = new double[weights.Count];
        double weightSum = _epsilon;

        for (int i = 0; i < weights.Count; i++)
        {
            double w = Math.Max(0, NumOps.ToDouble(weights[i][0])); // ReLU
            normalizedWeights[i] = w;
            weightSum += w;
        }

        // Normalize
        for (int i = 0; i < normalizedWeights.Length; i++)
        {
            normalizedWeights[i] /= weightSum;
        }

        // Weighted sum
        var result = new Tensor<T>(inputs[0].Shape);
        for (int i = 0; i < result.Length; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputs.Count; j++)
            {
                sum += normalizedWeights[j] * NumOps.ToDouble(inputs[j][i]);
            }
            result[i] = NumOps.FromDouble(sum);
        }

        return result;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = 0;

        // Lateral connections
        for (int i = 0; i < _numLevels; i++)
        {
            count += _inputChannels[i] * _outputChannels + _outputChannels;
        }

        // Top-down pathway
        for (int i = 0; i < _numLevels - 1; i++)
        {
            count += 2; // Fusion weights
            count += _outputChannels * _outputChannels + _outputChannels; // Conv
        }

        // Bottom-up pathway
        for (int i = 1; i < _numLevels; i++)
        {
            int numInputs = (i == 1) ? 2 : 3;
            count += numInputs; // Fusion weights
            count += _outputChannels * _outputChannels + _outputChannels; // Conv
        }

        // Multiply by number of repeats for repeated BiFPN blocks
        return count * _numRepeats;
    }

    private Tensor<T> ResizeToMatch(Tensor<T> source, Tensor<T> target)
    {
        int batch = source.Shape[0];
        int channels = source.Shape[1];
        int targetH = target.Shape[2];
        int targetW = target.Shape[3];
        int sourceH = source.Shape[2];
        int sourceW = source.Shape[3];

        var result = new Tensor<T>(new[] { batch, channels, targetH, targetW });

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        int srcH = Math.Min(h * sourceH / targetH, sourceH - 1);
                        int srcW = Math.Min(w * sourceW / targetW, sourceW - 1);
                        result[n, c, h, w] = source[n, c, srcH, srcW];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplySwish(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double swish = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = NumOps.FromDouble(swish);
        }
        return result;
    }
}
