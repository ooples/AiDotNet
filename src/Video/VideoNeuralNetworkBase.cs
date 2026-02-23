using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;

namespace AiDotNet.Video;

/// <summary>
/// Base class for video-focused neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide video-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video neural networks process sequences of image frames to perform
/// tasks like super-resolution, frame interpolation, optical flow estimation, denoising,
/// stabilization, and inpainting. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Frame preprocessing utilities (normalization, patch extraction)
/// - Temporal context handling (multi-frame processing)
///
/// You can use derived classes in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
public abstract class VideoNeuralNetworkBase<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the expected frame height for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input frames should be resized to match this height if different.
    /// Common values: 64, 128, 256, 480, 720.
    /// </para>
    /// </remarks>
    public int FrameHeight { get; protected set; } = 256;

    /// <summary>
    /// Gets or sets the expected frame width for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input frames should be resized to match this width if different.
    /// Common values: 64, 128, 256, 640, 1280.
    /// </para>
    /// </remarks>
    public int FrameWidth { get; protected set; } = 256;

    /// <summary>
    /// Gets or sets the number of color channels expected by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 3 (RGB), 1 (grayscale), 4 (RGBA).
    /// </para>
    /// </remarks>
    public int NumChannels { get; protected set; } = 3;

    /// <summary>
    /// Gets or sets the number of frames this model processes at once.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Models process this many frames as temporal context.
    /// Common values: 2 (pair-based), 5-7 (local window), 16+ (full sequence).
    /// </para>
    /// </remarks>
    public int NumFrames { get; protected set; } = 16;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for inference.
    /// When false, the model uses native layers and can be trained.
    /// </para>
    /// </remarks>
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the VideoNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VideoNeuralNetworkBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In ONNX mode, training is not supported - the model is inference-only.
    /// In native mode, training is fully supported.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses raw video frames for model input.
    /// </summary>
    /// <param name="rawFrames">Raw video frames tensor [numFrames, channels, height, width].</param>
    /// <returns>Preprocessed frames suitable for model input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Raw video frames have pixel values from 0-255.
    /// Neural networks typically work better with normalized values (e.g., 0-1 or -1 to 1).
    /// This method converts raw frames into the format the model expects.
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> PreprocessFrames(Tensor<T> rawFrames);

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Runs inference using ONNX model(s).
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Override this method to implement ONNX-specific inference logic
    /// for models with complex encoder-decoder or multi-model architectures.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (OnnxModel is not null)
        {
            return OnnxModel.Run(input);
        }

        if (OnnxEncoder is not null)
        {
            var encoded = OnnxEncoder.Run(input);
            if (OnnxDecoder is not null)
            {
                return OnnxDecoder.Run(encoded);
            }
            return encoded;
        }

        throw new InvalidOperationException("No ONNX model is loaded.");
    }

    /// <summary>
    /// Performs a forward pass through the native neural network layers.
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Normalizes frame pixel values from [0, 255] to [0, 1].
    /// </summary>
    /// <param name="frames">Raw frames tensor.</param>
    /// <returns>Normalized frames tensor.</returns>
    protected Tensor<T> NormalizeFrames(Tensor<T> frames)
    {
        var normalized = new Tensor<T>(frames.Shape);
        for (int i = 0; i < frames.Length; i++)
        {
            double val = NumOps.ToDouble(frames.Data.Span[i]);
            normalized.Data.Span[i] = NumOps.FromDouble(val / 255.0);
        }
        return normalized;
    }

    /// <summary>
    /// Denormalizes frame values from [0, 1] back to [0, 255].
    /// </summary>
    /// <param name="frames">Normalized frames tensor.</param>
    /// <returns>Denormalized frames tensor with values clipped to [0, 255].</returns>
    protected Tensor<T> DenormalizeFrames(Tensor<T> frames)
    {
        var denormalized = new Tensor<T>(frames.Shape);
        for (int i = 0; i < frames.Length; i++)
        {
            double val = NumOps.ToDouble(frames.Data.Span[i]) * 255.0;
            val = Math.Max(0.0, Math.Min(255.0, val));
            denormalized.Data.Span[i] = NumOps.FromDouble(val);
        }
        return denormalized;
    }

    /// <summary>
    /// Extracts a single frame from a multi-frame tensor.
    /// </summary>
    /// <param name="frames">Multi-frame tensor [numFrames, channels, height, width].</param>
    /// <param name="frameIndex">Index of the frame to extract.</param>
    /// <returns>Single frame tensor [channels, height, width].</returns>
    protected Tensor<T> ExtractFrame(Tensor<T> frames, int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= frames.Shape[0])
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"Frame index {frameIndex} is out of range [0, {frames.Shape[0]}).");

        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        var frame = new Tensor<T>([channels, height, width]);
        int frameSize = channels * height * width;
        int srcOffset = frameIndex * frameSize;

        for (int i = 0; i < frameSize; i++)
        {
            frame.Data.Span[i] = frames.Data.Span[srcOffset + i];
        }

        return frame;
    }

    /// <summary>
    /// Stores a single frame into a multi-frame tensor.
    /// </summary>
    /// <param name="output">Target multi-frame tensor [numFrames, channels, height, width].</param>
    /// <param name="frame">Single frame tensor [channels, height, width].</param>
    /// <param name="frameIndex">Index where the frame should be stored.</param>
    protected void StoreFrame(Tensor<T> output, Tensor<T> frame, int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= output.Shape[0])
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"Frame index {frameIndex} is out of range [0, {output.Shape[0]}).");

        int channels = output.Shape[1];
        int height = output.Shape[2];
        int width = output.Shape[3];
        int frameSize = channels * height * width;
        int dstOffset = frameIndex * frameSize;

        for (int i = 0; i < frameSize; i++)
        {
            output.Data.Span[dstOffset + i] = frame.Data.Span[i];
        }
    }

    /// <summary>
    /// Warps a feature map using optical flow via bilinear interpolation.
    /// </summary>
    /// <param name="feature">Feature tensor [channels, height, width] or [batch, channels, height, width].</param>
    /// <param name="flow">Optical flow tensor [2, height, width] (dx, dy).</param>
    /// <returns>Warped feature tensor with the same shape as input.</returns>
    protected Tensor<T> WarpFeature(Tensor<T> feature, Tensor<T> flow)
    {
        bool hasBatch = feature.Rank == 4;
        int batch = hasBatch ? feature.Shape[0] : 1;
        int channels = hasBatch ? feature.Shape[1] : feature.Shape[0];
        int height = hasBatch ? feature.Shape[2] : feature.Shape[1];
        int width = hasBatch ? feature.Shape[3] : feature.Shape[2];

        var warped = new Tensor<T>(feature.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + h * width + w
                        : h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + height * width + h * width + w
                        : height * width + h * width + w;

                    double dx = NumOps.ToDouble(flow.Data.Span[flowIdxX]);
                    double dy = NumOps.ToDouble(flow.Data.Span[flowIdxY]);

                    double srcX = w + dx;
                    double srcY = h + dy;

                    for (int c = 0; c < channels; c++)
                    {
                        T value = BilinearSample(feature, b, c, srcY, srcX, hasBatch, height, width, channels);
                        int outIdx = hasBatch
                            ? b * channels * height * width + c * height * width + h * width + w
                            : c * height * width + h * width + w;
                        warped.Data.Span[outIdx] = value;
                    }
                }
            }
        }

        return warped;
    }

    /// <summary>
    /// Performs bilinear sampling from a feature tensor.
    /// </summary>
    protected T BilinearSample(Tensor<T> tensor, int b, int c, double h, double w, bool hasBatch, int height, int width, int channels)
    {
        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        h0 = Math.Max(0, Math.Min(h0, height - 1));
        h1 = Math.Max(0, Math.Min(h1, height - 1));
        w0 = Math.Max(0, Math.Min(w0, width - 1));
        w1 = Math.Max(0, Math.Min(w1, width - 1));

        double hWeight = h - Math.Floor(h);
        double wWeight = w - Math.Floor(w);

        T v00 = GetFeatureValue(tensor, b, c, h0, w0, hasBatch, height, width, channels);
        T v01 = GetFeatureValue(tensor, b, c, h0, w1, hasBatch, height, width, channels);
        T v10 = GetFeatureValue(tensor, b, c, h1, w0, hasBatch, height, width, channels);
        T v11 = GetFeatureValue(tensor, b, c, h1, w1, hasBatch, height, width, channels);

        T top = NumOps.Add(
            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
        T bottom = NumOps.Add(
            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));

        return NumOps.Add(
            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));
    }

    /// <summary>
    /// Gets a value from a feature tensor at the specified position.
    /// </summary>
    protected T GetFeatureValue(Tensor<T> tensor, int b, int c, int h, int w, bool hasBatch, int height, int width, int channels)
    {
        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return tensor.Data.Span[idx];
    }

    /// <summary>
    /// Concatenates two feature tensors along the channel dimension.
    /// </summary>
    /// <param name="feat1">First feature tensor.</param>
    /// <param name="feat2">Second feature tensor.</param>
    /// <returns>Concatenated tensor with combined channels.</returns>
    protected Tensor<T> ConcatenateFeatures(Tensor<T> feat1, Tensor<T> feat2)
    {
        bool hasBatch = feat1.Rank == 4;
        int batch = hasBatch ? feat1.Shape[0] : 1;
        int c1 = hasBatch ? feat1.Shape[1] : feat1.Shape[0];
        int c2 = hasBatch ? feat2.Shape[1] : feat2.Shape[0];
        int height = hasBatch ? feat1.Shape[2] : feat1.Shape[1];
        int width = hasBatch ? feat1.Shape[3] : feat1.Shape[2];

        var outShape = hasBatch
            ? new[] { batch, c1 + c2, height, width }
            : new[] { c1 + c2, height, width };
        var output = new Tensor<T>(outShape);

        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < c1; c++)
            {
                int srcOffset = hasBatch ? b * c1 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch ? b * (c1 + c2) * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data.Span[dstOffset + i] = feat1.Data.Span[srcOffset + i];
                }
            }

            for (int c = 0; c < c2; c++)
            {
                int srcOffset = hasBatch ? b * c2 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch ? b * (c1 + c2) * pixelsPerChannel + (c1 + c) * pixelsPerChannel : (c1 + c) * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data.Span[dstOffset + i] = feat2.Data.Span[srcOffset + i];
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
