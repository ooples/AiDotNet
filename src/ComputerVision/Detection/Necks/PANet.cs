using System.IO;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Necks;

/// <summary>
/// Path Aggregation Network (PANet) for enhanced multi-scale feature fusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> PANet improves upon FPN by adding a bottom-up pathway
/// after the top-down pathway. This creates a bidirectional flow of information,
/// allowing both high-level semantics to flow down and low-level details to flow up.</para>
///
/// <para>Key features:
/// - FPN-style top-down pathway
/// - Additional bottom-up pathway for better localization
/// - Used in YOLOv4, YOLOv5, and many modern detectors
/// </para>
///
/// <para>Reference: Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018</para>
/// </remarks>
public class PANet<T> : NeckBase<T>
{
    private readonly int _outputChannels;
    private readonly int[] _inputChannels;
    private readonly int _numLevels;

    // Top-down (FPN) pathway weights
    private readonly List<Tensor<T>> _lateralWeights;
    private readonly List<Tensor<T>> _lateralBiases;
    private readonly List<Tensor<T>> _topDownWeights;
    private readonly List<Tensor<T>> _topDownBiases;

    // Bottom-up pathway weights
    private readonly List<Tensor<T>> _bottomUpWeights;
    private readonly List<Tensor<T>> _bottomUpBiases;
    private readonly List<Tensor<T>> _downsampleWeights;
    private readonly List<Tensor<T>> _downsampleBiases;

    /// <summary>
    /// Random number generator for weight initialization (created once, reused for all weights).
    /// </summary>
    private readonly Random _random;

    /// <inheritdoc/>
    public override string Name => "PANet";

    /// <inheritdoc/>
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc/>
    public override int NumLevels => _numLevels;

    /// <summary>
    /// Creates a new Path Aggregation Network.
    /// </summary>
    /// <param name="inputChannels">Channel counts from backbone at each level.</param>
    /// <param name="outputChannels">Output channel count for all levels (default 256).</param>
    public PANet(int[] inputChannels, int outputChannels = 256)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _numLevels = inputChannels.Length;

        _lateralWeights = new List<Tensor<T>>();
        _lateralBiases = new List<Tensor<T>>();
        _topDownWeights = new List<Tensor<T>>();
        _topDownBiases = new List<Tensor<T>>();
        _bottomUpWeights = new List<Tensor<T>>();
        _bottomUpBiases = new List<Tensor<T>>();
        _downsampleWeights = new List<Tensor<T>>();
        _downsampleBiases = new List<Tensor<T>>();

        // Create random generator once for all weight initializations
        // Using a seed for reproducibility, but the same RNG instance ensures different values
        _random = RandomHelper.CreateSeededRandom(42);

        // Initialize top-down pathway weights
        for (int i = 0; i < _numLevels; i++)
        {
            // Lateral connections
            var lateralWeight = new Tensor<T>(new[] { outputChannels, inputChannels[i] });
            var lateralBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(lateralWeight);
            _lateralWeights.Add(lateralWeight);
            _lateralBiases.Add(lateralBias);

            // Top-down output
            var topDownWeight = new Tensor<T>(new[] { outputChannels, outputChannels });
            var topDownBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(topDownWeight);
            _topDownWeights.Add(topDownWeight);
            _topDownBiases.Add(topDownBias);

            // Bottom-up output
            var bottomUpWeight = new Tensor<T>(new[] { outputChannels, outputChannels });
            var bottomUpBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(bottomUpWeight);
            _bottomUpWeights.Add(bottomUpWeight);
            _bottomUpBiases.Add(bottomUpBias);
        }

        // Downsample convolutions for bottom-up pathway
        for (int i = 0; i < _numLevels - 1; i++)
        {
            var downsampleWeight = new Tensor<T>(new[] { outputChannels, outputChannels });
            var downsampleBias = new Tensor<T>(new[] { outputChannels });
            InitializeWeights(downsampleWeight);
            _downsampleWeights.Add(downsampleWeight);
            _downsampleBiases.Add(downsampleBias);
        }
    }

    /// <summary>
    /// Creates PANet from a configuration object.
    /// </summary>
    /// <param name="config">Neck configuration.</param>
    public PANet(NeckConfig config)
        : this(config.InputChannels, config.OutputChannels)
    {
    }

    private void InitializeWeights(Tensor<T> weights)
    {
        double scale = Math.Sqrt(2.0 / weights.Shape[1]);

        for (int i = 0; i < weights.Length; i++)
        {
            double val = _random.NextDouble() * 2 * scale - scale;
            weights[i] = NumOps.FromDouble(val);
        }
    }

    /// <inheritdoc/>
    public override List<Tensor<T>> Forward(List<Tensor<T>> features)
    {
        ValidateFeatures(features, _inputChannels);

        // Phase 1: Top-down pathway (like FPN)
        var lateralFeatures = new List<Tensor<T>>(_numLevels);
        var topDownFeatures = new List<Tensor<T>>(_numLevels);

        // Apply lateral connections
        for (int i = 0; i < _numLevels; i++)
        {
            var lateral = Conv1x1(features[i], _lateralWeights[i], _lateralBiases[i]);
            lateralFeatures.Add(lateral);
        }

        // Top-down fusion
        for (int i = _numLevels - 1; i >= 0; i--)
        {
            Tensor<T> current = lateralFeatures[i];

            if (i < _numLevels - 1)
            {
                var upsampled = Upsample2x(topDownFeatures[^1]);
                if (upsampled.Shape[2] != current.Shape[2] || upsampled.Shape[3] != current.Shape[3])
                {
                    upsampled = ResizeToMatch(upsampled, current);
                }
                current = Add(current, upsampled);
            }

            var output = Conv1x1(current, _topDownWeights[i], _topDownBiases[i]);
            output = ApplyReLU(output);
            topDownFeatures.Insert(0, output);
        }

        // Phase 2: Bottom-up pathway
        var bottomUpFeatures = new List<Tensor<T>>(_numLevels);

        // Start from highest resolution (index 0)
        for (int i = 0; i < _numLevels; i++)
        {
            Tensor<T> current = topDownFeatures[i];

            if (i > 0)
            {
                // Add downsampled feature from previous level
                var downsampled = Downsample2x(bottomUpFeatures[i - 1]);

                // Apply downsample convolution
                downsampled = Conv1x1(downsampled, _downsampleWeights[i - 1], _downsampleBiases[i - 1]);

                if (downsampled.Shape[2] != current.Shape[2] || downsampled.Shape[3] != current.Shape[3])
                {
                    downsampled = ResizeToMatch(downsampled, current);
                }

                current = Add(current, downsampled);
            }

            var output = Conv1x1(current, _bottomUpWeights[i], _bottomUpBiases[i]);
            output = ApplyReLU(output);
            bottomUpFeatures.Add(output);
        }

        return bottomUpFeatures;
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        long count = 0;

        for (int i = 0; i < _numLevels; i++)
        {
            // Lateral
            count += _inputChannels[i] * _outputChannels + _outputChannels;
            // Top-down
            count += _outputChannels * _outputChannels + _outputChannels;
            // Bottom-up
            count += _outputChannels * _outputChannels + _outputChannels;
        }

        // Downsample convolutions
        for (int i = 0; i < _numLevels - 1; i++)
        {
            count += _outputChannels * _outputChannels + _outputChannels;
        }

        return count;
    }

    /// <inheritdoc/>
    public override void WriteParameters(BinaryWriter writer)
    {
        // Write configuration
        writer.Write(_numLevels);
        writer.Write(_outputChannels);
        foreach (int ic in _inputChannels)
        {
            writer.Write(ic);
        }

        // Write top-down pathway weights
        for (int i = 0; i < _numLevels; i++)
        {
            WriteTensor(writer, _lateralWeights[i]);
            WriteTensor(writer, _lateralBiases[i]);
            WriteTensor(writer, _topDownWeights[i]);
            WriteTensor(writer, _topDownBiases[i]);
            WriteTensor(writer, _bottomUpWeights[i]);
            WriteTensor(writer, _bottomUpBiases[i]);
        }

        // Write downsample convolution weights
        for (int i = 0; i < _numLevels - 1; i++)
        {
            WriteTensor(writer, _downsampleWeights[i]);
            WriteTensor(writer, _downsampleBiases[i]);
        }
    }

    /// <inheritdoc/>
    public override void ReadParameters(BinaryReader reader)
    {
        // Read and verify configuration
        int numLevels = reader.ReadInt32();
        int outputChannels = reader.ReadInt32();
        var inputChannels = new int[numLevels];
        for (int i = 0; i < numLevels; i++)
        {
            inputChannels[i] = reader.ReadInt32();
        }

        if (numLevels != _numLevels || outputChannels != _outputChannels)
        {
            throw new InvalidOperationException($"PANet configuration mismatch: expected {_numLevels} levels with {_outputChannels} channels, got {numLevels} levels with {outputChannels} channels.");
        }

        for (int i = 0; i < numLevels; i++)
        {
            if (inputChannels[i] != _inputChannels[i])
            {
                throw new InvalidOperationException($"PANet input channel mismatch at level {i}: expected {_inputChannels[i]}, got {inputChannels[i]}.");
            }
        }

        // Read top-down pathway weights
        for (int i = 0; i < _numLevels; i++)
        {
            ReadTensor(reader, _lateralWeights[i]);
            ReadTensor(reader, _lateralBiases[i]);
            ReadTensor(reader, _topDownWeights[i]);
            ReadTensor(reader, _topDownBiases[i]);
            ReadTensor(reader, _bottomUpWeights[i]);
            ReadTensor(reader, _bottomUpBiases[i]);
        }

        // Read downsample convolution weights
        for (int i = 0; i < _numLevels - 1; i++)
        {
            ReadTensor(reader, _downsampleWeights[i]);
            ReadTensor(reader, _downsampleBiases[i]);
        }
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        // Write shape
        writer.Write(tensor.Rank);
        foreach (int dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        // Write data
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(NumOps.ToDouble(tensor[i]));
        }
    }

    private void ReadTensor(BinaryReader reader, Tensor<T> tensor)
    {
        // Read and verify shape
        int rank = reader.ReadInt32();
        if (rank != tensor.Rank)
        {
            throw new InvalidOperationException($"Tensor rank mismatch: expected {tensor.Rank}, got {rank}.");
        }

        for (int i = 0; i < rank; i++)
        {
            int dim = reader.ReadInt32();
            if (dim != tensor.Shape[i])
            {
                throw new InvalidOperationException($"Tensor shape mismatch at dimension {i}: expected {tensor.Shape[i]}, got {dim}.");
            }
        }

        // Read data
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        }
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
