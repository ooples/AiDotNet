using System;
using AiDotNet.InferenceOptimization.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class ConvolutionKernelValidationTests
{
    [Fact]
    public void Conv2D_Throws_WhenKernelInChannelsMismatch()
    {
        var kernel = new ConvolutionKernel();

        var input = new Tensor<float>(new[] { 1, 3, 5, 5 });
        var badKernel = new Tensor<float>(new[] { 2, 2, 3, 3 });

        var ex = Assert.Throws<ArgumentException>(() => kernel.Conv2D(input, badKernel));
        Assert.Contains("kernel.Shape[1] == inChannels", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}

