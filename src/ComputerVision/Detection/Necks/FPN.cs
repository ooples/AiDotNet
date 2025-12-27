using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Necks;

/// <summary>
/// Feature Pyramid Network (FPN) for multi-scale feature fusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FPN creates a feature pyramid by combining features from
/// different backbone levels using a top-down pathway. High-resolution features from
/// earlier layers are enriched with semantic information from deeper layers.</para>
///
/// <para>Key features:
/// - Top-down pathway with lateral connections
/// - 1x1 convolutions to match channel dimensions
/// - Simple element-wise addition for fusion
/// - Fast and memory efficient
/// </para>
///
/// <para>Reference: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017</para>
/// </remarks>
public class FPN<T> : NeckBase<T>
{
    private readonly int _outputChannels;
    private readonly int[] _inputChannels;
    private readonly int _numLevels;
    private readonly List<Tensor<T>> _lateralWeights;
    private readonly List<Tensor<T>> _lateralBiases;
    private readonly List<Tensor<T>> _outputWeights;
    private readonly List<Tensor<T>> _outputBiases;

    /// <inheritdoc/>
    public override string Name => "FPN";

    /// <inheritdoc/>
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc/>
    public override int NumLevels => _numLevels;

    /// <summary>
    /// Creates a new Feature Pyramid Network.
    /// </summary>
    /// <param name="inputChannels">Channel counts from backbone at each level.</param>
    /// <param name="outputChannels">Output channel count for all levels (default 256).</param>
    public FPN(int[] inputChannels, int outputChannels = 256)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _numLevels = inputChannels.Length;

        _lateralWeights = new List<Tensor<T>>();
        _lateralBiases = new List<Tensor<T>>();
        _outputWeights = new List<Tensor<T>>();
        _outputBiases = new List<Tensor<T>>();

        // Initialize lateral connections (1x1 conv to match channels)
        for (int i = 0; i < _numLevels; i++)
        {
            var lateralWeight = new Tensor<T>(new[] { outputChannels, inputChannels[i] });
            var lateralBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(lateralWeight);
            _lateralWeights.Add(lateralWeight);
            _lateralBiases.Add(lateralBias);

            // Output convolutions (3x3 for refinement, simplified as 1x1 here)
            var outputWeight = new Tensor<T>(new[] { outputChannels, outputChannels });
            var outputBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(outputWeight);
            _outputWeights.Add(outputWeight);
            _outputBiases.Add(outputBias);
        }
    }

    /// <summary>
    /// Creates FPN from a configuration object.
    /// </summary>
    /// <param name="config">Neck configuration.</param>
    public FPN(NeckConfig config)
        : this(config.InputChannels, config.OutputChannels)
    {
    }

    private void InitializeWeights(Tensor<T> weights)
    {
        // He initialization
        double scale = Math.Sqrt(2.0 / weights.Shape[1]);
        var random = RandomHelper.CreateSeededRandom(42);

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

        // Build top-down pathway
        var lateralFeatures = new List<Tensor<T>>(_numLevels);
        var outputFeatures = new List<Tensor<T>>(_numLevels);

        // Apply lateral connections (1x1 conv)
        for (int i = 0; i < _numLevels; i++)
        {
            var lateral = Conv1x1(features[i], _lateralWeights[i], _lateralBiases[i]);
            lateralFeatures.Add(lateral);
        }

        // Top-down pathway with lateral connections
        // Start from the deepest level (smallest spatial resolution)
        for (int i = _numLevels - 1; i >= 0; i--)
        {
            Tensor<T> current = lateralFeatures[i];

            // Add upsampled feature from deeper level (if not the deepest)
            if (i < _numLevels - 1)
            {
                // Get the output from the next deeper level and upsample
                var upsampled = Upsample2x(outputFeatures[^1]);

                // Resize if dimensions don't match exactly (due to odd sizes)
                if (upsampled.Shape[2] != current.Shape[2] || upsampled.Shape[3] != current.Shape[3])
                {
                    upsampled = ResizeToMatch(upsampled, current);
                }

                current = Add(current, upsampled);
            }

            // Apply output convolution
            var output = Conv1x1(current, _outputWeights[i], _outputBiases[i]);
            output = ApplyReLU(output);
            outputFeatures.Insert(0, output);
        }

        return outputFeatures;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = 0;

        for (int i = 0; i < _numLevels; i++)
        {
            // Lateral weights and biases
            count += _inputChannels[i] * _outputChannels + _outputChannels;
            // Output weights and biases
            count += _outputChannels * _outputChannels + _outputChannels;
        }

        return count;
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
                        // Nearest neighbor interpolation
                        int srcH = Math.Min(h * sourceH / targetH, sourceH - 1);
                        int srcW = Math.Min(w * sourceW / targetW, sourceW - 1);
                        result[n, c, h, w] = source[n, c, srcH, srcW];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            result[i] = NumOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }
}
