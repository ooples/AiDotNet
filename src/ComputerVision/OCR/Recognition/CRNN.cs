using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR.Recognition;

/// <summary>
/// CRNN (Convolutional Recurrent Neural Network) for text recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CRNN combines CNNs for visual feature extraction with
/// RNNs (specifically LSTM) for sequence modeling. It's trained with CTC loss to
/// handle variable-length text without requiring character-level alignment.</para>
///
/// <para>Key features:
/// - CNN backbone for visual features
/// - Bidirectional LSTM for sequence modeling
/// - CTC (Connectionist Temporal Classification) decoding
/// - Handles variable-length text naturally
/// </para>
///
/// <para>Reference: Shi et al., "An End-to-End Trainable Neural Network for
/// Image-based Sequence Recognition and Its Application to Scene Text Recognition", TPAMI 2017</para>
/// </remarks>
public class CRNN<T> : OCRBase<T>
{
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly Conv2D<T> _conv3;
    private readonly Conv2D<T> _conv4;
    private readonly Conv2D<T> _conv5;
    private readonly Dense<T> _lstm1Forward;
    private readonly Dense<T> _lstm1Backward;
    private readonly Dense<T> _lstm2Forward;
    private readonly Dense<T> _lstm2Backward;
    private readonly Dense<T> _outputLayer;
    private readonly int _hiddenDim;

    /// <inheritdoc/>
    public override string Name => "CRNN";

    /// <summary>
    /// Creates a new CRNN text recognizer.
    /// </summary>
    public CRNN(OCROptions<T> options) : base(options)
    {
        _hiddenDim = 256;

        // CNN backbone for feature extraction
        _conv1 = new Conv2D<T>(3, 64, kernelSize: 3, padding: 1);
        _conv2 = new Conv2D<T>(64, 128, kernelSize: 3, padding: 1);
        _conv3 = new Conv2D<T>(128, 256, kernelSize: 3, padding: 1);
        _conv4 = new Conv2D<T>(256, 256, kernelSize: 3, padding: 1);
        _conv5 = new Conv2D<T>(256, 512, kernelSize: 3, padding: 1);

        // Bidirectional LSTM layers (simplified as Dense for forward pass)
        // In practice, this would be proper LSTM cells
        _lstm1Forward = new Dense<T>(512, _hiddenDim);
        _lstm1Backward = new Dense<T>(512, _hiddenDim);
        _lstm2Forward = new Dense<T>(_hiddenDim * 2, _hiddenDim);
        _lstm2Backward = new Dense<T>(_hiddenDim * 2, _hiddenDim);

        // Output layer to vocabulary
        _outputLayer = new Dense<T>(_hiddenDim * 2, VocabularySize);
    }

    /// <inheritdoc/>
    public override OCRResult<T> Recognize(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        // Preprocess
        var input = PreprocessCrop(image);

        // Forward pass
        var (text, confidence) = RecognizeText(input);

        var result = new OCRResult<T>
        {
            FullText = text,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };

        if (!string.IsNullOrEmpty(text))
        {
            result.TextRegions.Add(new RecognizedText<T>(text, confidence));
        }

        return result;
    }

    /// <inheritdoc/>
    public override (string text, T confidence) RecognizeText(Tensor<T> croppedImage)
    {
        // Forward pass through CNN backbone
        var x = _conv1.Forward(croppedImage);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 2);

        x = _conv2.Forward(x);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 2);

        x = _conv3.Forward(x);
        x = ApplyReLU(x);

        x = _conv4.Forward(x);
        x = ApplyReLU(x);
        x = MaxPool2D(x, 2, 1); // Keep width, reduce height

        x = _conv5.Forward(x);
        x = ApplyReLU(x);

        // Squeeze height dimension and transpose to (batch, width, channels)
        var seqFeatures = SqueezeAndPermute(x);

        // Bidirectional LSTM (simplified)
        var lstmOut = ApplyBidirectionalLSTM(seqFeatures);

        // Output projection
        var logits = ApplyOutputLayer(lstmOut);

        // CTC decoding
        string text = DecodeCTC(logits);
        T confidence = ComputeConfidence(logits, text);

        return (text, confidence);
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        return _conv1.GetParameterCount() +
               _conv2.GetParameterCount() +
               _conv3.GetParameterCount() +
               _conv4.GetParameterCount() +
               _conv5.GetParameterCount() +
               _lstm1Forward.GetParameterCount() +
               _lstm1Backward.GetParameterCount() +
               _lstm2Forward.GetParameterCount() +
               _lstm2Backward.GetParameterCount() +
               _outputLayer.GetParameterCount();
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        throw new NotImplementedException("Weight saving not yet implemented");
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

    private Tensor<T> MaxPool2D(Tensor<T> x, int kernelH, int kernelW)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        int outH = height / kernelH;
        int outW = width / kernelW;

        var result = new Tensor<T>(new[] { batch, channels, outH, outW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outH; h++)
                {
                    for (int w = 0; w < outW; w++)
                    {
                        double maxVal = double.NegativeInfinity;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int srcH = h * kernelH + kh;
                                int srcW = w * kernelW + kw;

                                if (srcH < height && srcW < width)
                                {
                                    maxVal = Math.Max(maxVal, NumOps.ToDouble(x[b, c, srcH, srcW]));
                                }
                            }
                        }

                        result[b, c, h, w] = NumOps.FromDouble(maxVal);
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> SqueezeAndPermute(Tensor<T> x)
    {
        // x: [batch, channels, height, width]
        // Output: [batch, width, channels*height]
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        int featureDim = channels * height;

        var result = new Tensor<T>(new[] { batch, width, featureDim });

        for (int b = 0; b < batch; b++)
        {
            for (int w = 0; w < width; w++)
            {
                int idx = 0;
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        result[b, w, idx++] = x[b, c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyBidirectionalLSTM(Tensor<T> x)
    {
        // Simplified bidirectional processing
        // x: [batch, seq_len, features]
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int features = x.Shape[2];

        // First layer forward/backward
        var fw1 = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });
        var bw1 = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Extract feature vector at time t
                var feat = new Tensor<T>(new[] { 1, features });
                for (int f = 0; f < features; f++)
                {
                    feat[0, f] = x[b, t, f];
                }

                // Forward LSTM
                var fwOut = _lstm1Forward.Forward(feat);
                for (int h = 0; h < _hiddenDim; h++)
                {
                    fw1[b, t, h] = fwOut[0, h];
                }

                // Backward LSTM (process in reverse order)
                int revT = seqLen - 1 - t;
                var featRev = new Tensor<T>(new[] { 1, features });
                for (int f = 0; f < features; f++)
                {
                    featRev[0, f] = x[b, revT, f];
                }

                var bwOut = _lstm1Backward.Forward(featRev);
                for (int h = 0; h < _hiddenDim; h++)
                {
                    bw1[b, revT, h] = bwOut[0, h];
                }
            }
        }

        // Concatenate forward and backward
        var concat1 = new Tensor<T>(new[] { batch, seqLen, _hiddenDim * 2 });
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _hiddenDim; h++)
                {
                    concat1[b, t, h] = fw1[b, t, h];
                    concat1[b, t, _hiddenDim + h] = bw1[b, t, h];
                }
            }
        }

        // Second layer (similar processing)
        var fw2 = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });
        var bw2 = new Tensor<T>(new[] { batch, seqLen, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim * 2 });
                for (int h = 0; h < _hiddenDim * 2; h++)
                {
                    feat[0, h] = concat1[b, t, h];
                }

                var fwOut = _lstm2Forward.Forward(feat);
                for (int h = 0; h < _hiddenDim; h++)
                {
                    fw2[b, t, h] = fwOut[0, h];
                }

                int revT = seqLen - 1 - t;
                var featRev = new Tensor<T>(new[] { 1, _hiddenDim * 2 });
                for (int h = 0; h < _hiddenDim * 2; h++)
                {
                    featRev[0, h] = concat1[b, revT, h];
                }

                var bwOut = _lstm2Backward.Forward(featRev);
                for (int h = 0; h < _hiddenDim; h++)
                {
                    bw2[b, revT, h] = bwOut[0, h];
                }
            }
        }

        // Final concatenation
        var result = new Tensor<T>(new[] { batch, seqLen, _hiddenDim * 2 });
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _hiddenDim; h++)
                {
                    result[b, t, h] = fw2[b, t, h];
                    result[b, t, _hiddenDim + h] = bw2[b, t, h];
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyOutputLayer(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int features = x.Shape[2];

        var result = new Tensor<T>(new[] { batch, seqLen, VocabularySize });

        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var feat = new Tensor<T>(new[] { 1, features });
                for (int f = 0; f < features; f++)
                {
                    feat[0, f] = x[b, t, f];
                }

                var output = _outputLayer.Forward(feat);
                for (int v = 0; v < VocabularySize; v++)
                {
                    result[b, t, v] = output[0, v];
                }
            }
        }

        return result;
    }
}
