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

        public Tensor<float> Execute(params Tensor<float>[] inputs)
        {
            throw new NotImplementedException("Use specific convolution methods");
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

            if (input.Dimensions.Length != 4 || kernel.Dimensions.Length != 4)
                throw new ArgumentException("Conv2D requires 4D tensors");

            int batchSize = input.Dimensions[0];
            int inChannels = input.Dimensions[1];
            int inHeight = input.Dimensions[2];
            int inWidth = input.Dimensions[3];

            int outChannels = kernel.Dimensions[0];
            int kernelH = kernel.Dimensions[2];
            int kernelW = kernel.Dimensions[3];

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

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

        private unsafe void Conv2DSingleOutput(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int outChannel,
            int inChannels, int inHeight, int inWidth,
            int kernelH, int kernelW, int stride, int padding,
            int outHeight, int outWidth)
        {
            fixed (float* pInput = input.Data, pKernel = kernel.Data, pOutput = output.Data)
            {
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

                                        sum += pInput[inputIdx] * pKernel[kernelIdx];
                                    }
                                }
                            }
                        }

                        int outputIdx = ((batch * output.Dimensions[1] + outChannel) * outHeight + oh) * outWidth + ow;
                        pOutput[outputIdx] = sum;
                    }
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

            if (input.Dimensions.Length != 4 || kernel.Dimensions.Length != 4)
                throw new ArgumentException("DepthwiseConv2D requires 4D tensors");

            int batchSize = input.Dimensions[0];
            int channels = input.Dimensions[1];
            int inHeight = input.Dimensions[2];
            int inWidth = input.Dimensions[3];

            int kernelH = kernel.Dimensions[2];
            int kernelW = kernel.Dimensions[3];

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

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

        private unsafe void DepthwiseConv2DSingleChannel(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int channel,
            int inHeight, int inWidth, int kernelH, int kernelW,
            int stride, int padding, int outHeight, int outWidth)
        {
            fixed (float* pInput = input.Data, pKernel = kernel.Data, pOutput = output.Data)
            {
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
                                    int inputIdx = ((batch * input.Dimensions[1] + channel) * inHeight + ih) * inWidth + iw;
                                    int kernelIdx = (channel * kernelH + kh) * kernelW + kw;

                                    sum += pInput[inputIdx] * pKernel[kernelIdx];
                                }
                            }
                        }

                        int outputIdx = ((batch * output.Dimensions[1] + channel) * outHeight + oh) * outWidth + ow;
                        pOutput[outputIdx] = sum;
                    }
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
            if (input.Dimensions.Length != 4 || kernel.Dimensions.Length != 4)
                throw new ArgumentException("GroupConv2D requires 4D tensors");

            int batchSize = input.Dimensions[0];
            int inChannels = input.Dimensions[1];
            int inHeight = input.Dimensions[2];
            int inWidth = input.Dimensions[3];

            int outChannels = kernel.Dimensions[0];
            int kernelH = kernel.Dimensions[2];
            int kernelW = kernel.Dimensions[3];

            if (inChannels % groups != 0 || outChannels % groups != 0)
                throw new ArgumentException("Channels must be divisible by groups");

            int inChannelsPerGroup = inChannels / groups;
            int outChannelsPerGroup = outChannels / groups;

            int outHeight = (inHeight + 2 * padding - kernelH) / stride + 1;
            int outWidth = (inWidth + 2 * padding - kernelW) / stride + 1;

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

        private unsafe void GroupConv2DSingleOutput(
            Tensor<float> input, Tensor<float> kernel, Tensor<float> output,
            int batch, int outChannel, int group,
            int inChannelsPerGroup, int inHeight, int inWidth,
            int kernelH, int kernelW, int stride, int padding,
            int outHeight, int outWidth)
        {
            int inChannelStart = group * inChannelsPerGroup;

            fixed (float* pInput = input.Data, pKernel = kernel.Data, pOutput = output.Data)
            {
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
                                        int inputIdx = ((batch * input.Dimensions[1] + globalInChannel) * inHeight + ih) * inWidth + iw;
                                        int kernelIdx = ((outChannel * inChannelsPerGroup + ic) * kernelH + kh) * kernelW + kw;

                                        sum += pInput[inputIdx] * pKernel[kernelIdx];
                                    }
                                }
                            }
                        }

                        int outputIdx = ((batch * output.Dimensions[1] + outChannel) * outHeight + oh) * outWidth + ow;
                        pOutput[outputIdx] = sum;
                    }
                }
            }
        }
    }
}
