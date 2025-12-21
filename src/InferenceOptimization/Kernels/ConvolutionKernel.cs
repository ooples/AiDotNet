using System;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.InferenceOptimization.Kernels
{
    /// <summary>
    /// Optimized convolution kernels including depthwise and group convolutions
    /// </summary>
    public class ConvolutionKernel : ICustomOperator<float>
    {
        public string Name => "Convolution";
        public string Version => "1.0.0";
        public int Priority => 100;

        public bool IsSupported()
        {
            return true;
        }

        public double EstimatedSpeedup()
        {
            var caps = PlatformDetector.Capabilities;
            if (caps.HasAVX2) return 2.5;
            if (caps.HasNeon) return 2.0;
            return 1.5;
        }

        /// <summary>
        /// Executes convolution on the provided inputs.
        /// Expects 2-3 inputs: input tensor, kernel tensor, and optional config tensor.
        /// Config tensor format: [stride, padding] (defaults to stride=1, padding=0)
        /// </summary>
        public Tensor<float> Execute(params Tensor<float>[] inputs)
        {
            if (inputs == null || inputs.Length < 2)
            {
                throw new ArgumentException(
                    "ConvolutionKernel requires at least 2 inputs: input tensor and kernel tensor. " +
                    "Optional 3rd input for config [stride, padding].");
            }

            var input = inputs[0];
            var kernel = inputs[1];

            // Extract stride and padding from optional config tensor or use defaults
            int stride = 1;
            int padding = 0;

            if (inputs.Length >= 3 && inputs[2] != null && inputs[2].Data.Length >= 2)
            {
                stride = Math.Max(1, (int)inputs[2].Data[0]);
                padding = Math.Max(0, (int)inputs[2].Data[1]);
            }

            // Determine convolution type based on kernel shape
            // Standard: kernel[out_channels, in_channels, kH, kW]
            // Depthwise: kernel[channels, 1, kH, kW]
            if (kernel.Shape.Length == 4 && kernel.Shape[1] == 1)
            {
                // Depthwise convolution (kernel has 1 in_channel dimension)
                return DepthwiseConv2D(input, kernel, stride, padding);
            }

            // Default to standard 2D convolution
            return Conv2D(input, kernel, stride, padding);
        }

        /// <summary>
        /// Standard 2D convolution
        /// </summary>
        public Tensor<float> Conv2D(
            Tensor<float> input,
            Tensor<float> kernel,
            int stride = 1,
            int padding = 0)
        {
            // Input: [batch, in_channels, height, width]
            // Kernel: [out_channels, in_channels, kernel_h, kernel_w]

            if (input.Shape.Length != 4 || kernel.Shape.Length != 4)
                throw new ArgumentException("Conv2D requires 4D tensors");

            if (stride <= 0)
                throw new ArgumentOutOfRangeException(nameof(stride), $"stride must be positive, but got {stride}");

            if (padding < 0)
                throw new ArgumentOutOfRangeException(nameof(padding), $"padding must be non-negative, but got {padding}");

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            int outChannels = kernel.Shape[0];
            int kernelH = kernel.Shape[2];
            int kernelW = kernel.Shape[3];

            if (kernelH <= 0 || kernelW <= 0)
                throw new ArgumentException($"Kernel dimensions must be positive, but got {kernelH}x{kernelW}");

            if (kernel.Shape[1] != inChannels)
                throw new ArgumentException($"Conv2D requires kernel.Shape[1] == inChannels ({inChannels}), but got {kernel.Shape[1]}");

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

            if (outHeight <= 0 || outWidth <= 0)
                throw new ArgumentException(
                    $"Invalid output dimensions ({outHeight}x{outWidth}). " +
                    $"Check stride ({stride}), padding ({padding}), and kernel size ({kernelH}x{kernelW}).");
            var output = new Tensor<float>(new[] { batchSize, outChannels, outHeight, outWidth });

            // Parallelize over batch and output channels
            Parallel.For(0, batchSize * outChannels, idx =>
            {
                int b = idx / outChannels;
                int oc = idx % outChannels;

                Conv2DSingleOutput(input, kernel, output, b, oc,
                    inChannels, inHeight, inWidth,
                    kernelH, kernelW, stride, padding,
                    outHeight, outWidth);
            });

            return output;
        }

        private void Conv2DSingleOutput(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int outChannel,
            int inChannels, int inHeight, int inWidth,
            int kernelH, int kernelW, int stride, int padding,
            int outHeight, int outWidth)
        {
            var inputData = input.Data;
            var kernelData = kernel.Data;
            var outputData = output.Data;

            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    float sum = 0.0f;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int inputIdx = ((batch * inChannels + ic) * inHeight + ih) * inWidth + iw;
                                    int kernelIdx = ((outChannel * inChannels + ic) * kernelH + kh) * kernelW + kw;
                                    sum += inputData[inputIdx] * kernelData[kernelIdx];
                                }
                            }
                        }
                    }

                    int outputIdx = ((batch * output.Shape[1] + outChannel) * outHeight + oh) * outWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        }

        /// <summary>
        /// Depthwise separable convolution (more efficient for mobile architectures)
        /// </summary>
        public Tensor<float> DepthwiseConv2D(
            Tensor<float> input,
            Tensor<float> kernel,
            int stride = 1,
            int padding = 0)
        {
            // Input: [batch, channels, height, width]
            // Kernel: [channels, 1, kernel_h, kernel_w]

            if (input.Shape.Length != 4 || kernel.Shape.Length != 4)
                throw new ArgumentException("DepthwiseConv2D requires 4D tensors");

            if (stride <= 0)
                throw new ArgumentOutOfRangeException(nameof(stride), $"stride must be positive, but got {stride}");

            if (padding < 0)
                throw new ArgumentOutOfRangeException(nameof(padding), $"padding must be non-negative, but got {padding}");

            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            int kernelH = kernel.Shape[2];
            int kernelW = kernel.Shape[3];

            if (kernelH <= 0 || kernelW <= 0)
                throw new ArgumentException($"Kernel dimensions must be positive, but got {kernelH}x{kernelW}");

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

            if (outHeight <= 0 || outWidth <= 0)
                throw new ArgumentException(
                    $"Invalid output dimensions ({outHeight}x{outWidth}). " +
                    $"Check stride ({stride}), padding ({padding}), and kernel size ({kernelH}x{kernelW}).");

            if (kernel.Shape[1] != 1)
                throw new ArgumentException(
                    $"Depthwise convolution requires kernel.Shape[1] == 1, but got {kernel.Shape[1]}");

            if (kernel.Shape[0] != channels)
                throw new ArgumentException(
                    $"Depthwise convolution requires kernel.Shape[0] == channels ({channels}), but got {kernel.Shape[0]}");

            var output = new Tensor<float>(new[] { batchSize, channels, outHeight, outWidth });

            Parallel.For(0, batchSize * channels, idx =>
            {
                int b = idx / channels;
                int c = idx % channels;

                DepthwiseConv2DSingleChannel(input, kernel, output, b, c,
                    inHeight, inWidth, kernelH, kernelW,
                    stride, padding, outHeight, outWidth);
            });

            return output;
        }

        private void DepthwiseConv2DSingleChannel(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int channel,
            int inHeight, int inWidth, int kernelH, int kernelW,
            int stride, int padding, int outHeight, int outWidth)
        {
            var inputData = input.Data;
            var kernelData = kernel.Data;
            var outputData = output.Data;

            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    float sum = 0.0f;

                    for (int kh = 0; kh < kernelH; kh++)
                    {
                        for (int kw = 0; kw < kernelW; kw++)
                        {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                            {
                                int inputIdx = ((batch * input.Shape[1] + channel) * inHeight + ih) * inWidth + iw;
                                int kernelIdx = (channel * kernelH + kh) * kernelW + kw;
                                sum += inputData[inputIdx] * kernelData[kernelIdx];
                            }
                        }
                    }

                    int outputIdx = ((batch * output.Shape[1] + channel) * outHeight + oh) * outWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        }

        /// <summary>
        /// Group convolution (reduces parameters and computation)
        /// </summary>
        public Tensor<float> GroupConv2D(
            Tensor<float> input,
            Tensor<float> kernel,
            int groups,
            int stride = 1,
            int padding = 0)
        {
            if (input.Shape.Length != 4 || kernel.Shape.Length != 4)
                throw new ArgumentException("GroupConv2D requires 4D tensors");

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int inHeight = input.Shape[2];
            int inWidth = input.Shape[3];

            int outChannels = kernel.Shape[0];
            int kernelH = kernel.Shape[2];
            int kernelW = kernel.Shape[3];

            if (groups <= 0)
                throw new ArgumentOutOfRangeException(nameof(groups), "groups must be positive.");

            if (inChannels % groups != 0 || outChannels % groups != 0)
                throw new ArgumentException("Channels must be divisible by groups");

            int inChannelsPerGroup = inChannels / groups;
            int outChannelsPerGroup = outChannels / groups;

            if (kernel.Shape[1] != inChannelsPerGroup)
                throw new ArgumentException(
                    $"Group convolution requires kernel.Shape[1] == inChannelsPerGroup ({inChannelsPerGroup}), " +
                    $"but got {kernel.Shape[1]}");

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

            if (outHeight <= 0 || outWidth <= 0)
                throw new ArgumentException(
                    $"Invalid output dimensions ({outHeight}x{outWidth}). " +
                    $"Check stride ({stride}), padding ({padding}), and kernel size ({kernelH}x{kernelW}).");

            var output = new Tensor<float>(new[] { batchSize, outChannels, outHeight, outWidth });

            // Process each group independently
            Parallel.For(0, groups, g =>
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int oc = 0; oc < outChannelsPerGroup; oc++)
                    {
                        int globalOutChannel = g * outChannelsPerGroup + oc;

                        GroupConv2DSingleOutput(input, kernel, output, b, globalOutChannel, g,
                            inChannelsPerGroup, inHeight, inWidth,
                            kernelH, kernelW, stride, padding,
                            outHeight, outWidth);
                    }
                }
            });

            return output;
        }

        private void GroupConv2DSingleOutput(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int outChannel, int group,
            int inChannelsPerGroup, int inHeight, int inWidth,
            int kernelH, int kernelW, int stride, int padding,
            int outHeight, int outWidth)
        {
            int inChannelStart = group * inChannelsPerGroup;
            var inputData = input.Data;
            var kernelData = kernel.Data;
            var outputData = output.Data;

            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    float sum = 0.0f;

                    for (int ic = 0; ic < inChannelsPerGroup; ic++)
                    {
                        int globalInChannel = inChannelStart + ic;

                        for (int kh = 0; kh < kernelH; kh++)
                        {
                            for (int kw = 0; kw < kernelW; kw++)
                            {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth)
                                {
                                    int inputIdx = ((batch * input.Shape[1] + globalInChannel) * inHeight + ih) * inWidth + iw;
                                    int kernelIdx = ((outChannel * inChannelsPerGroup + ic) * kernelH + kh) * kernelW + kw;
                                    sum += inputData[inputIdx] * kernelData[kernelIdx];
                                }
                            }
                        }
                    }

                    int outputIdx = ((batch * output.Shape[1] + outChannel) * outHeight + oh) * outWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        }
    }
}
