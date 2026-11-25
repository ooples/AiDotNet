using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Empty inputs on the CPU engine and on the OpenCL engine: an empty tensor in gives an empty tensor of the same
/// shape out, and an empty vector gives a 0 x 0 Jacobian. A zero-length device buffer cannot be created
/// (<c>CL_INVALID_BUFFER_SIZE</c>), so these paths must return before they reach the engine.
/// </summary>
[Collection("NonParallelIntegration")]
public class ActivationEmptyInputEngineTests
{
    private static IEnumerable<ActivationFunctionBase<float>> Elementwise()
    {
        yield return new ReLUActivation<float>();
        yield return new ReLU6Activation<float>();
        yield return new LeakyReLUActivation<float>(0.1);
        yield return new ELUActivation<float>(1.3);
        yield return new HardSwishActivation<float>();
        yield return new SwishActivation<float>();
        yield return new GELUActivation<float>();
        yield return new BinarySpikingActivation<float>(1f, 2f, 0.5f);
        yield return new SignActivation<float>();
    }

    private static IEnumerable<ActivationFunctionBase<float>> VectorJacobians()
    {
        foreach (var activation in Elementwise())
        {
            if (activation is not SignActivation<float>) yield return activation;
        }

        yield return new SoftmaxActivation<float>();
        yield return new SoftminActivation<float>();
        yield return new LogSoftmaxActivation<float>();
        yield return new LogSoftminActivation<float>();
        yield return new TaylorSoftmaxActivation<float>(3);
        yield return new SparsemaxActivation<float>();
        yield return new SphericalSoftmaxActivation<float>();
        yield return new SquashActivation<float>();
        yield return new MaxoutActivation<float>(2);
        yield return new GumbelSoftmaxActivation<float>(1.0, 3);
    }

    private static void AssertEmptyInputsGiveEmptyResults()
    {
        foreach (int[] shape in new[] { new[] { 0 }, new[] { 2, 0 }, new[] { 0, 3 } })
        {
            foreach (var activation in Elementwise())
            {
                string name = activation.GetType().Name;
                // The two forward passes this sweep moved onto the engine. The other forward passes are the
                // engine calls they were before, and on the OpenCL engine those reject an empty tensor.
                if (activation is SignActivation<float> or BinarySpikingActivation<float>)
                {
                    var forward = activation.Activate(new Tensor<float>(shape));
                    Assert.True(forward.Length == 0, name + " forward length");
                    Assert.Equal(shape, forward.Shape.ToArray());
                }

                if (activation is SignActivation<float>) continue;
                var derivative = activation.Derivative(new Tensor<float>(shape));
                Assert.True(derivative.Length == 0, name + " derivative length");
                Assert.Equal(shape, derivative.Shape.ToArray());
            }
        }

        foreach (var activation in VectorJacobians())
        {
            string name = activation.GetType().Name;
            var jacobian = activation.Derivative(new Vector<float>(0));
            Assert.True(jacobian.Rows == 0 && jacobian.Columns == 0, name + " jacobian of an empty vector");
            Assert.True(activation.Activate(new Vector<float>(0)).Length == 0, name + " forward of an empty vector");
        }
    }

    [Fact]
    public void EmptyInputs_OnTheCpuEngine_GiveEmptyResults()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            AssertEmptyInputsGiveEmptyResults();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void EmptyInputs_OnTheOpenClEngine_GiveEmptyResults()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "no OpenCL device");
        using var direct = new DirectGpuEngine(backend);
        using var gpu = new DirectGpuTensorEngine(direct);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            AssertEmptyInputsGiveEmptyResults();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
