using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// Detection head used in YOLO family detectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The detection head takes features from the neck and
/// produces the final predictions: bounding box coordinates, objectness score, and
/// class probabilities for each grid cell and anchor.</para>
///
/// <para>Each output tensor has shape [batch, num_anchors * (5 + num_classes), height, width]
/// where 5 = (x, y, w, h, objectness).</para>
/// </remarks>
internal class YOLOHead<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numClasses;
    private readonly int _numAnchors;
    private readonly List<Conv2D<T>> _convLayers;
    private readonly int[] _inputChannels;
    private readonly int _outputPerAnchor;

    /// <summary>
    /// Creates a new YOLO detection head.
    /// </summary>
    /// <param name="inputChannels">Input channels for each feature level.</param>
    /// <param name="numClasses">Number of detection classes.</param>
    /// <param name="numAnchors">Number of anchors per grid cell.</param>
    public YOLOHead(int[] inputChannels, int numClasses, int numAnchors = 3)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _numClasses = numClasses;
        _numAnchors = numAnchors;
        _inputChannels = inputChannels;
        _outputPerAnchor = 5 + numClasses; // (x, y, w, h, obj) + classes

        _convLayers = new List<Conv2D<T>>();

        // Create output convolutions for each feature level
        int outputChannels = numAnchors * _outputPerAnchor;
        for (int i = 0; i < inputChannels.Length; i++)
        {
            _convLayers.Add(new Conv2D<T>(
                inChannels: inputChannels[i],
                outChannels: outputChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                useBias: true
            ));
        }
    }

    /// <summary>
    /// Forward pass through the detection head.
    /// </summary>
    /// <param name="features">Multi-scale features from the neck.</param>
    /// <returns>Raw detection outputs for each scale.</returns>
    public List<Tensor<T>> Forward(List<Tensor<T>> features)
    {
        var outputs = new List<Tensor<T>>();

        for (int i = 0; i < features.Count; i++)
        {
            var output = _convLayers[i].Forward(features[i]);
            outputs.Add(output);
        }

        return outputs;
    }

    /// <summary>
    /// Decodes raw network outputs into detection boxes and scores.
    /// </summary>
    /// <param name="outputs">Raw outputs from Forward pass.</param>
    /// <param name="strides">Stride for each feature level.</param>
    /// <param name="imageHeight">Original image height.</param>
    /// <param name="imageWidth">Original image width.</param>
    /// <returns>Decoded detections (boxes, scores, class IDs).</returns>
    public List<(float[] boxes, float[] scores, int[] classIds)> DecodeOutputs(
        List<Tensor<T>> outputs,
        int[] strides,
        int imageHeight,
        int imageWidth)
    {
        if (outputs.Count == 0)
        {
            return new List<(float[] boxes, float[] scores, int[] classIds)>();
        }

        // Determine batch size from first output tensor
        int batchSize = outputs[0].Shape[0];

        // Initialize per-batch collections
        var batchBoxes = new List<float>[batchSize];
        var batchScores = new List<float>[batchSize];
        var batchClassIds = new List<int>[batchSize];
        for (int bi = 0; bi < batchSize; bi++)
        {
            batchBoxes[bi] = new List<float>();
            batchScores[bi] = new List<float>();
            batchClassIds[bi] = new List<int>();
        }


        for (int levelIdx = 0; levelIdx < outputs.Count; levelIdx++)
        {
            var output = outputs[levelIdx];
            int stride = strides[levelIdx];

            int batch = output.Shape[0];
            int featH = output.Shape[2];
            int featW = output.Shape[3];

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < featH; h++)
                {
                    for (int w = 0; w < featW; w++)
                    {
                        for (int a = 0; a < _numAnchors; a++)
                        {
                            int offset = a * _outputPerAnchor;

                            // Get box predictions
                            double tx = _numOps.ToDouble(output[b, offset + 0, h, w]);
                            double ty = _numOps.ToDouble(output[b, offset + 1, h, w]);
                            double tw = _numOps.ToDouble(output[b, offset + 2, h, w]);
                            double th = _numOps.ToDouble(output[b, offset + 3, h, w]);
                            double objScore = Sigmoid(_numOps.ToDouble(output[b, offset + 4, h, w]));

                            // Get class predictions
                            double maxClassScore = 0;
                            int maxClassId = 0;
                            for (int c = 0; c < _numClasses; c++)
                            {
                                double classScore = Sigmoid(_numOps.ToDouble(output[b, offset + 5 + c, h, w]));
                                if (classScore > maxClassScore)
                                {
                                    maxClassScore = classScore;
                                    maxClassId = c;
                                }
                            }

                            // Combined score
                            double score = objScore * maxClassScore;

                            // Decode box coordinates
                            double cx = (Sigmoid(tx) + w) * stride;
                            double cy = (Sigmoid(ty) + h) * stride;
                            double bw = Math.Exp(tw) * stride;
                            double bh = Math.Exp(th) * stride;

                            // Convert to xyxy format
                            float x1 = (float)Math.Max(0, cx - bw / 2);
                            float y1 = (float)Math.Max(0, cy - bh / 2);
                            float x2 = (float)Math.Min(imageWidth, cx + bw / 2);
                            float y2 = (float)Math.Min(imageHeight, cy + bh / 2);

                            // Add to this batch's collections
                            batchBoxes[b].AddRange(new[] { x1, y1, x2, y2 });
                            batchScores[b].Add((float)score);
                            batchClassIds[b].Add(maxClassId);
                        }
                    }
                }
            }
        }

        // Return one result per batch item
        var results = new List<(float[] boxes, float[] scores, int[] classIds)>();
        for (int b = 0; b < batchSize; b++)
        {
            results.Add((batchBoxes[b].ToArray(), batchScores[b].ToArray(), batchClassIds[b].ToArray()));
        }
        return results;
    }

    /// <summary>
    /// Gets the total parameter count for the head.
    /// </summary>
    public long GetParameterCount()
    {
        long count = 0;
        for (int i = 0; i < _inputChannels.Length; i++)
        {
            int outputChannels = _numAnchors * _outputPerAnchor;
            count += _inputChannels[i] * outputChannels + outputChannels; // 1x1 conv + bias
        }
        return count;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}

/// <summary>
/// Anchor-free detection head used in YOLOv8+.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike earlier YOLO versions that use predefined anchor boxes,
/// YOLOv8+ uses an anchor-free approach where the network directly predicts box sizes
/// relative to each grid cell. This simplifies the architecture and often improves accuracy.</para>
/// </remarks>
internal class YOLOv8Head<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numClasses;
    private readonly int _regMax;
    private readonly List<Conv2D<T>> _clsConvs;
    private readonly List<Conv2D<T>> _regConvs;
    private readonly List<Conv2D<T>> _clsHeads;
    private readonly List<Conv2D<T>> _regHeads;
    private readonly int[] _inputChannels;

    /// <summary>
    /// Creates a new YOLOv8 anchor-free detection head.
    /// </summary>
    /// <param name="inputChannels">Input channels for each feature level.</param>
    /// <param name="numClasses">Number of detection classes.</param>
    /// <param name="regMax">Maximum value for regression distribution (default 16).</param>
    public YOLOv8Head(int[] inputChannels, int numClasses, int regMax = 16)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _numClasses = numClasses;
        _regMax = regMax;
        _inputChannels = inputChannels;

        _clsConvs = new List<Conv2D<T>>();
        _regConvs = new List<Conv2D<T>>();
        _clsHeads = new List<Conv2D<T>>();
        _regHeads = new List<Conv2D<T>>();

        int hiddenChannels = 256;

        for (int i = 0; i < inputChannels.Length; i++)
        {
            // Classification branch
            _clsConvs.Add(new Conv2D<T>(inputChannels[i], hiddenChannels, kernelSize: 3, padding: 1));
            _clsHeads.Add(new Conv2D<T>(hiddenChannels, numClasses, kernelSize: 1));

            // Regression branch (predicts distribution over regMax values for each of 4 box sides)
            _regConvs.Add(new Conv2D<T>(inputChannels[i], hiddenChannels, kernelSize: 3, padding: 1));
            _regHeads.Add(new Conv2D<T>(hiddenChannels, 4 * regMax, kernelSize: 1));
        }
    }

    /// <summary>
    /// Forward pass through the detection head.
    /// </summary>
    /// <param name="features">Multi-scale features from the neck.</param>
    /// <returns>Classification and regression outputs for each scale.</returns>
    public (List<Tensor<T>> clsOutputs, List<Tensor<T>> regOutputs) Forward(List<Tensor<T>> features)
    {
        var clsOutputs = new List<Tensor<T>>();
        var regOutputs = new List<Tensor<T>>();

        for (int i = 0; i < features.Count; i++)
        {
            // Classification branch
            var clsFeat = _clsConvs[i].Forward(features[i]);
            clsFeat = ApplySiLU(clsFeat);
            var clsOut = _clsHeads[i].Forward(clsFeat);
            clsOutputs.Add(clsOut);

            // Regression branch
            var regFeat = _regConvs[i].Forward(features[i]);
            regFeat = ApplySiLU(regFeat);
            var regOut = _regHeads[i].Forward(regFeat);
            regOutputs.Add(regOut);
        }

        return (clsOutputs, regOutputs);
    }

    /// <summary>
    /// Decodes outputs into detection boxes and scores.
    /// </summary>
    public List<(float[] boxes, float[] scores, int[] classIds)> DecodeOutputs(
        List<Tensor<T>> clsOutputs,
        List<Tensor<T>> regOutputs,
        int[] strides,
        int imageHeight,
        int imageWidth)
    {
        if (clsOutputs.Count == 0)
        {
            return new List<(float[] boxes, float[] scores, int[] classIds)>();
        }

        // Determine batch size from first output tensor
        int batchSize = clsOutputs[0].Shape[0];

        // Initialize per-batch collections
        var batchBoxes = new List<float>[batchSize];
        var batchScores = new List<float>[batchSize];
        var batchClassIds = new List<int>[batchSize];
        for (int bi = 0; bi < batchSize; bi++)
        {
            batchBoxes[bi] = new List<float>();
            batchScores[bi] = new List<float>();
            batchClassIds[bi] = new List<int>();
        }


        for (int levelIdx = 0; levelIdx < clsOutputs.Count; levelIdx++)
        {
            var clsOutput = clsOutputs[levelIdx];
            var regOutput = regOutputs[levelIdx];
            int stride = strides[levelIdx];

            int batch = clsOutput.Shape[0];
            int featH = clsOutput.Shape[2];
            int featW = clsOutput.Shape[3];

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < featH; h++)
                {
                    for (int w = 0; w < featW; w++)
                    {
                        // Get max class score
                        double maxScore = 0;
                        int maxClassId = 0;
                        for (int c = 0; c < _numClasses; c++)
                        {
                            double score = Sigmoid(_numOps.ToDouble(clsOutput[b, c, h, w]));
                            if (score > maxScore)
                            {
                                maxScore = score;
                                maxClassId = c;
                            }
                        }

                        // Decode box using DFL (Distribution Focal Loss) approach
                        // Each side of the box has regMax distribution values
                        double left = DecodeDistribution(regOutput, b, 0, h, w);
                        double top = DecodeDistribution(regOutput, b, 1, h, w);
                        double right = DecodeDistribution(regOutput, b, 2, h, w);
                        double bottom = DecodeDistribution(regOutput, b, 3, h, w);

                        // Convert to xyxy format
                        double cx = (w + 0.5) * stride;
                        double cy = (h + 0.5) * stride;

                        float x1 = (float)Math.Max(0, cx - left * stride);
                        float y1 = (float)Math.Max(0, cy - top * stride);
                        float x2 = (float)Math.Min(imageWidth, cx + right * stride);
                        float y2 = (float)Math.Min(imageHeight, cy + bottom * stride);

                        // Add to this batch's collections
                        batchBoxes[b].AddRange(new[] { x1, y1, x2, y2 });
                        batchScores[b].Add((float)maxScore);
                        batchClassIds[b].Add(maxClassId);
                    }
                }
            }
        }

        // Return one result per batch item
        var results = new List<(float[] boxes, float[] scores, int[] classIds)>();
        for (int b = 0; b < batchSize; b++)
        {
            results.Add((batchBoxes[b].ToArray(), batchScores[b].ToArray(), batchClassIds[b].ToArray()));
        }
        return results;
    }

    private double DecodeDistribution(Tensor<T> regOutput, int b, int side, int h, int w)
    {
        // Apply softmax over regMax values and compute weighted sum
        double[] probs = new double[_regMax];
        double maxVal = double.NegativeInfinity;

        for (int i = 0; i < _regMax; i++)
        {
            double val = _numOps.ToDouble(regOutput[b, side * _regMax + i, h, w]);
            maxVal = Math.Max(maxVal, val);
        }

        double sumExp = 0;
        for (int i = 0; i < _regMax; i++)
        {
            double val = _numOps.ToDouble(regOutput[b, side * _regMax + i, h, w]);
            probs[i] = Math.Exp(val - maxVal);
            sumExp += probs[i];
        }

        // Weighted sum (expectation)
        double result = 0;
        for (int i = 0; i < _regMax; i++)
        {
            probs[i] /= sumExp;
            result += probs[i] * i;
        }

        return result;
    }

    /// <summary>
    /// Gets the total parameter count for the head.
    /// </summary>
    public long GetParameterCount()
    {
        long count = 0;
        int hiddenChannels = 256;

        for (int i = 0; i < _inputChannels.Length; i++)
        {
            // Classification branch
            count += _inputChannels[i] * hiddenChannels * 9 + hiddenChannels; // 3x3 conv
            count += hiddenChannels * _numClasses + _numClasses; // 1x1 conv

            // Regression branch
            count += _inputChannels[i] * hiddenChannels * 9 + hiddenChannels; // 3x3 conv
            count += hiddenChannels * (4 * _regMax) + (4 * _regMax); // 1x1 conv
        }

        return count;
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(silu);
        }
        return result;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
